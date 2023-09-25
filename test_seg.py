import argparse
from utils.utils_metrics import compute_mIoU, show_results
from utils.datasets import *
from utils.torch_utils import select_device
from utils.general import increment_path
from models.yolo import Model
import yaml


def cal_miou(opt,epoch="",weight_path="",net=None,save_path=""):
    device = opt.device
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    data_path = data_dict["data_path"]
    if weight_path=="":
        weight_path = opt.weights

    # ---------------------------------------------------------------------------#
    #   miou_ mode is used to specify the content to be calculated when the file is run
    #   miou_ The mode of 0 represents the entire miou calculation process, including obtaining the prediction result and calculating miou.
    #   miou_ Mode 1 represents obtaining only the prediction result.
    #   miou_ Mode 2 represents only calculating the miou.
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    num_classes = data_dict["nc"]
    name_classes = data_dict["names"]


    test_dir = data_path + "/Testing_Images"
    gt_dir = data_path + "/Testing_Labels"
    pred_dir = data_path + "/results"
    img_names = os.listdir(test_dir)
    image_ids = [image_name.split(".")[0] for image_name in img_names]

    if miou_mode == 0 or miou_mode == 1:
        if net==None:
            net = Model(opt.cfg, ch=3, nc=2).to(device)  # create
            net.to(device=device)
            net.load_state_dict(torch.load(weight_path, map_location=device))
        net.eval()

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
            if isinstance(pred,list):
                pred = pred[0]

            pred = np.array(pred.data.cpu()[0])

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            pred = onehot2mask(pred)

            pred = pred * 255
            pred = Image.fromarray(pred)
            pred = imge_size_return(pred, origin_shape)

            pred.save(os.path.join(pred_dir, image_id + ".png"))

    if miou_mode == 0 or miou_mode == 2:
        hist, IoUs, PA_Recall, Precision,txt_list = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        with open(save_path+"/result.txt", 'a', newline='') as f:
            f.write("epoch = " + epoch + " miou=" + str(np.nanmean(IoUs) * 100) + "\n")
            for txt in txt_list :
                f.write(txt + "\n")
                print(txt)
            f.write("\n")


    return np.nanmean(IoUs) * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_seg.py')
    parser.add_argument('--weights', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='', help='*.data path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/test_seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    opt = parser.parse_args()
    opt.device = select_device(opt.device, batch_size=1)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)  # make dir
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    cal_miou(opt,save_path=opt.save_dir)