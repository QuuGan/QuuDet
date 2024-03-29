# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run classification inference on images

Usage:
    $ python classify/predict.py --weights yolov5s-cls.pt --source im.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch.nn.functional as F

from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from train_cls import imshow_cls
#from models.common import DetectMultiBackend
from utils.datasets import classify_transforms
from utils.general import   check_requirements, colorstr, increment_path 
from utils.torch_utils import select_device, smart_inference_mode, time_sync


@smart_inference_mode()
def run(
        weights='',  # model.pt path(s)
        source='',  # file/dir/URL/glob, 0 for webcam
        imgsz=224,  # inference size
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        show=True,
        project=ROOT / 'runs/detect_cls',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
):
    file = str(source)
    seen, dt = 1, [0.0, 0.0, 0.0]
    device = select_device(device)

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Transforms
    transforms = classify_transforms(imgsz)

    # Load model
    model = attempt_load(weights, map_location=device)   
    model.half()
    #model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup

    # Image
    t1 = time_sync()
    im = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    im = transforms(im).unsqueeze(0).to(device)
    im = im.half()
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    results = model(im)[0]
    t3 = time_sync()
    dt[1] += t3 - t2

    p = F.softmax(results, dim=1)  # probabilities
    i = p.argsort(1, descending=True)[:, :5].squeeze()  # top 5 indices
    dt[2] += time_sync() - t3

    txt_result = []

    s = f"image {file}: {imgsz}x{imgsz} {', '.join(f'{model.names[j]} {p[0, j]:.2f}' for j in i)}"
    txt_result.append(s)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    shape = (1, 3, imgsz, imgsz)
    s = f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t
    txt_result.append(s)

    with open(save_dir / 'detect_cls_result.txt', 'w') as file:
        # 将列表中的每个元素写入文件
        for txt in txt_result:
            print(txt)
            file.write(txt + '\n')
    # imshow_cls(im, f=save_dir / Path(file).name, verbose=True)
    print(f"Results saved to {colorstr('bold', save_dir)}")
    return p


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='', help='model path(s)')
    parser.add_argument('--source', type=str, default='', help='file')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/detect_cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
