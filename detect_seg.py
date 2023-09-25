import argparse
from utils.torch_utils import select_device
from utils.general import increment_path,set_logging
from models.yolo import Model
from utils.datasets import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='detect_seg.py')
    parser.add_argument('--weights', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', type=str, default=r'', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--project', default='runs/detect_seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    opt = parser.parse_args()
    device = select_device(opt.device, batch_size=1)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)  # make dir
    net = Model(opt.cfg, ch=3, nc=2).to(device)  # create
    net.to(device=device)
    net.load_state_dict(torch.load(opt.weights, map_location=device))
    net.eval()
    test_dir = opt.source
    img_names = os.listdir(test_dir)
    image_ids = [image_name.split(".")[0] for image_name in img_names]
    for image_id in tqdm(image_ids):
        image_path = os.path.join(test_dir, image_id + ".jpg")

        img1 = Image.open(image_path)
        img = img1
        origin_shape = img.size

        img = keep_image_size_open_rgb(img)
        img = np.transpose(img, [2, 0, 1])
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        if isinstance(pred, list):
            pred = pred[0]

        pred = np.array(pred.data.cpu()[0])

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        pred = onehot2mask(pred)

        pred = pred * 255
        # print(pred)
        pred = Image.fromarray(pred)
        pred = imge_size_return(pred, origin_shape)

        pred.save(os.path.join(save_dir, image_id + ".png"))



