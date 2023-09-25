import argparse
import logging
from test_seg import cal_miou
from utils.datasets import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from pathlib import Path
import os
from utils.torch_utils import select_device
from torch.utils.tensorboard import SummaryWriter
from models.yolo import Model
from datetime import datetime
from utils.general import increment_path,set_logging
from utils.loss import SoftDiceLoss
import shutil
import yaml

logger = logging.getLogger(__name__)
def train_net(opt):
    device = opt.device
    weight_path = opt.weights
    epochs = opt.epochs
    batch_size =opt.batch_size
    save_dir = Path(opt.save_dir)
    lr = opt.lr
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    train_dataset = ISBI_Loader(data_dict["data_path"],opt.img_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,num_workers=opt.workers)
    set_logging(0)
    logger.info(f'Image sizes {opt.img_size} \n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)


    net = Model(opt.cfg, ch=3, nc=2).to(device)  # create
    if weight_path:
        if os.path.exists(weight_path):
            net.load_state_dict(torch.load(weight_path, map_location=device))
            print('successful load weight！')
        else:
            print('not successful load weight')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    if net.loss_funtion == "unet-loss-SoftDice":
        criterion = SoftDiceLoss()
    else:
        criterion = nn.BCELoss()
    best_loss = float('inf')

    # lossFile = open("results/loss.txt", 'a', newline='')
    # tensorboard --logdir logs
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter('./logs/' + TIMESTAMP)
    miou_max = 86.00
    for epoch in range(epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        num_correct = 0
        mloss = torch.zeros(1, device=device)
        pbar = enumerate(train_loader)
        s = ('\n' + '%10s' * 5) % ('Epoch', 'gpu_mem', 'loss', 'labels', 'img_size')
        logger.info(s)

        nb = len(train_loader)
        pbar = tqdm(pbar, total=nb)  # progress bar
        for i, (image, label) in pbar:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)
            # print(pred.shape)
            if isinstance(pred,list):
                pred = pred[0]
            loss = criterion(torch.sigmoid(pred), label)
            # lossFile.write('Loss/train'+str(loss.item())+"\n")
            # print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())
            # print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), best)
            torch.save(net.state_dict(), last)
            loss.backward()

            mloss = (mloss * i + loss) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 3) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, label.shape[0], image.shape[-1])
            pbar.set_description(s)

            optimizer.step()
            train_loss += float(loss.item())

        miou = cal_miou(opt,epoch=str(epoch),weight_path=last,net=net,save_path=opt.save_dir)
        if miou > miou_max:
            miou_max = miou
            shutil.copy(last, best)
        writer.add_scalar("loss", train_loss / nb, epoch)
        writer.add_scalar("miou", miou, epoch)




    # lossFile.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[512, 512], help='[train, test] image sizes')
    parser.add_argument('--lr', nargs='+', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train_seg', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()
    opt.device = select_device(opt.device, batch_size=opt.batch_size)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    train_net(opt)
