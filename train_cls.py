import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.yolo import Model
from test_cls import test as validate
from models.experimental import attempt_load
# from models.yolo import ClassificationModel, DetectionModel
from utils.datasets import create_classification_dataloader
from utils.general import (WorkingDirectory, check_git_status, check_requirements, colorstr,
                           increment_path, init_seeds)
from utils.plots import imshow_cls
from utils.torch_utils import (ModelEMA, model_info, reshape_classifier_output, select_device, smart_DDP,
                               smart_optimizer, smartCrossEntropyLoss, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(opt, device):
    init_seeds(opt.seed + 1 + RANK)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        Path(opt.save_dir), Path(opt.data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
            opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Save run settings
    # yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Logger
    # logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    data_dir = data
    # Dataloaders
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes

    if opt.fullresize:
        trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                       imgsz=imgsz,
                                                       batch_size=bs // WORLD_SIZE,
                                                       augment=False,
                                                       cache=opt.cache,
                                                       rank=LOCAL_RANK,
                                                       workers=nw,
                                                       fullresize=True)
    else:
        trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                       imgsz=imgsz,
                                                       batch_size=bs // WORLD_SIZE,
                                                       augment=False,
                                                       cache=opt.cache,
                                                       rank=LOCAL_RANK,
                                                       workers=nw,
                                                       fullresize=False)

    test_dir = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val
    if RANK in {-1, 0}:
        if opt.fullresize:
            testloader = create_classification_dataloader(path=test_dir,
                                                          imgsz=imgsz,
                                                          batch_size=bs // WORLD_SIZE * 2,
                                                          augment=False,
                                                          cache=opt.cache,
                                                          rank=-1,
                                                          workers=nw,
                                                          fullresize=True)
        else:
            testloader = create_classification_dataloader(path=test_dir,
                                                          imgsz=imgsz,
                                                          batch_size=bs // WORLD_SIZE * 2,
                                                          augment=False,
                                                          cache=opt.cache,
                                                          rank=-1,
                                                          workers=nw,
                                                          fullresize=False)

    # Model
    if Path(opt.weight).is_file() or opt.weight.endswith('.pt'):
        model = attempt_load(opt.weight, map_location=device)
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create
        # model = ClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 10)  # convert to classification model
    model.training = True
    reshape_classifier_output(model, nc)  # update class count

    for p in model.parameters():
        p.requires_grad = True  # for training
    for m in model.modules():
        if not pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout
    model = model.to(device)
    names = trainloader.dataset.classes  # class names
    model.names = names  # attach class names

    # Info
    if RANK in {-1, 0}:
        model_info(model)
        if opt.verbose:
            print(model)
        images, labels = next(iter(trainloader))
        file = imshow_cls(images[:25], labels[:25], names=names, f=save_dir / 'train_images.jpg')
        # logger.log_images(file, name='Train Examples')
        # logger.log_graph(model, imgsz)  # log model

    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=5e-5)

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Train
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    val = test_dir.stem  # 'val' or 'test'
    print(f'Image sizes {imgsz} train, {imgsz} test\n'
          f'Using {nw * WORLD_SIZE} dataloader workers\n'
          f"Logging results to {colorstr('bold', save_dir)}\n"
          f'Starting {opt.weight} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
          f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                loss = criterion(model(images)[0], labels)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36

                # Test
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = validate(model=ema.ema,
                                                 dataloader=testloader,
                                                 criterion=criterion,
                                                 pbar=pbar)  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy

        # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]['lr']}  # learning rate
            # logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    'ema': None,  # deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': None,  # optimizer.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        print(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
              f"\nResults saved to {colorstr('bold', save_dir)}")

        # # Plot examples
        # images, labels = (x[:25] for x in next(iter(testloader)))  # first 25 images and labels
        # pred = torch.max(ema.ema((images.half() if cuda else images.float()).to(device)), 1)[1]
        # file = imshow_cls(images, labels, pred, names, verbose=False, f=save_dir / 'test_images.jpg')
        #
        # # Log results
        # meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        # logger.log_images(file, name='Test Examples (true-predicted)', epoch=epoch)
        # logger.log_model(best, epochs, metadata=meta)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=r'', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='',
                        help='model.yaml path')
    parser.add_argument('--data', type=str, default='', help='path of dataset fold.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--fullresize', action='store_true',
                        help='Stretching images without maintaining aspect ratio')  # 不保持宽高比拉伸图像
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train_cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', nargs='?', const=True, default=True, help='start from i.e. --pretrained False')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='AdamW', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--cutoff', type=int, default=None, help='Model layer cutoff index for Classify() head')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout (fraction)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)


def run(**kwargs):
    # Usage: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
