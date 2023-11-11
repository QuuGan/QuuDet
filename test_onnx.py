import argparse
import onnxruntime
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.general import check_dataset, check_img_size, \
    box_iou, non_max_suppression, scale_coords,  xywh2xyxy,xyxy2xywh, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized
import onnx

def post_processing(output, height, width):
    box_array = output[0]
    confs = output[1]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]
    for i in range(box_array.shape[0]):
        box_array[i] = xyxy2xywh(box_array[i])

        box_array[i][:, 1] = box_array[i][:, 1] * height
        box_array[i][:, 3] = box_array[i][:, 3] * height

        # 修改第1、3个数字
        box_array[i][:, 0] = box_array[i][:, 0] * width
        box_array[i][:, 2] = box_array[i][:, 2] * width

    max_conf = np.max(confs, axis=2, keepdims=True)
    return np.concatenate((box_array, max_conf, confs), axis=2)
def count_parameters(model_path):
    model = onnx.load(model_path)
    total_params = 0
    for initializer in model.graph.initializer:
        shape = initializer.dims
        param_size = np.prod(shape)
        total_params += param_size
    return total_params
def test_onnx(opt):
    set_logging()

    #loadmodel
    if opt.tensorrt:
        model = onnxruntime.InferenceSession(opt.weights, providers=['TensorrtExecutionProvider'])
    else:
        model = onnxruntime.InferenceSession(opt.weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    num_params = count_parameters(opt.weights)
    print("Total parameters: ", num_params)
    model_inputs = model.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape
    batch_size = input_shape[0]
    img_size =  input_shape[3]

    model_outputs = model.get_outputs()
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]


    device = select_device(opt.device, batch_size=batch_size)


    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    gs = 32
    imgsz = check_img_size(img_size, s=gs)  # check img_size

    # Configure
    if isinstance(opt.data, str):
        with open(opt.data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc =  int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    opt.single_cls=False

    dataloader = create_dataloader(data[opt.task] , imgsz, batch_size, gs, opt, pad=0.5, rect=False,
                                   prefix=colorstr(f'{"val"}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = data["names"]

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time_synchronized()
        out = model.run(output_names, {input_names[0]: img.cpu().numpy()})  # inference and training outputs
        if opt.darknet:
            out = post_processing(out,width,height)
            out = torch.tensor(out).to(device)
        else:
            out = torch.tensor(out[0]).to(device)
        t0 += time_synchronized() - t

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb, multi_label=True,v6=opt.v6)
        t1 += time_synchronized() - t


        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (nc < 50 ) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple

    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_onnx.py')
    parser.add_argument('--weights', type=str, default='', help='model.onnx path(s)')
    parser.add_argument('--data', type=str, default='', help='*.data path')
    parser.add_argument('--conf-thres', type=float, default=0.005, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--project', default='runs/test_onnx', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--darknet', action='store_true', help='darknet_onnx need process')
    parser.add_argument('--v6', action='store_true', help='v6_onnx')
    parser.add_argument('--tensorrt', action='store_true', help='use tensorrt')
    opt = parser.parse_args()

    print(opt)

    test_onnx(opt)
