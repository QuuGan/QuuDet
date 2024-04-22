# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a classification model on a dataset

Usage:
    $ python classify/val.py --weights yolov5s-cls.pt --data ../datasets/imagenet
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.datasets  import create_classification_dataloader
from utils.general import  check_img_size, colorstr, increment_path
from utils.torch_utils import select_device,time_sync

def test(
    data='',  # dataset dir
    weights='',  # model.pt path(s)
    batch_size=128,  # batch size
    imgsz=224,  # inference size (pixels)
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    project=ROOT / 'runs/val-cls',  # save to project/name
    name='exp',  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    criterion=None,
    pbar=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run

        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # stride = max(int(model.stride.max()), 32)  # model stride
        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half()
        # self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        # model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)

        #pt, jit, engine =  model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=32)  # check image size
        #half = model.fp16  # FP16 supported on limited backends with CUDA
        #batch_size = model.batch_size
        # Dataloader
        data = Path(data)
        test_dir = data / 'test' if (data / 'test').exists() else data / 'val'  # data/test or data/val
        if opt.fullresize:
            dataloader = create_classification_dataloader(path=test_dir,
                                                          imgsz=imgsz,
                                                          batch_size=batch_size,
                                                          augment=False,
                                                          rank=-1,
                                                          workers=workers,
                                                          fullresize=True)
        else:
            dataloader = create_classification_dataloader(path=test_dir,
                                                          imgsz=imgsz,
                                                          batch_size=batch_size,
                                                          augment=False,
                                                          rank=-1,
                                                          workers=workers)

    model.eval()
    pred, targets, loss, dt = [], [], 0, [0.0, 0.0, 0.0]
    n = len(dataloader)  # number of batches
    action = 'validating' if dataloader.dataset.root.stem == 'val' else 'testing'
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0)
    with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
        for images, labels in bar:
            t1 = time_sync()
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            t2 = time_sync()
            dt[0] += t2 - t1

            y = model(images)[0]
            t3 = time_sync()
            dt[1] += t3 - t2

            pred.append(y.argsort(1, descending=True)[:, :5])
            targets.append(labels)
            if criterion:
                loss += criterion(y, labels)
            dt[2] += time_sync() - t3

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()
    txt_result = []

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    if verbose:  # all classes
        s = f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}"
        txt_result.append(s)
        s = f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}"
        txt_result.append(s)
        for i, c in enumerate(model.names):
            aci = acc[targets == i]
            top1i, top5i = aci.mean(0).tolist()
            s = f"{c:>24}{aci.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}"
            txt_result.append(s)

        # Print results
        t = tuple(x / len(dataloader.dataset.samples) * 1E3 for x in dt)  # speeds per image
        shape = (1, 3, imgsz, imgsz)
        s = f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t
        txt_result.append(s)
        with open(save_dir / 'test_cls_result.txt', 'w') as file:
            # 将列表中的每个元素写入文件
            for txt in txt_result:
                print(txt)
                file.write(txt + '\n')
        print(f"Results saved to {colorstr('bold', save_dir)}")

    return top1, top5, loss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default=r'', help='dataset path')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--fullresize', action='store_true',help='Stretching images without maintaining aspect ratio')  # 不保持宽高比拉伸图像
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--verbose', nargs='?', const=True, default=True, help='verbose output')
    parser.add_argument('--project', default=ROOT / 'runs/test_cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    #print_args(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    test(**vars(opt))
