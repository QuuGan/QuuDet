import os
import os.path as osp
import cv2
import shutil
import numpy as np

def json2png(json_folder, png_save_folder):
    if osp.isdir(png_save_folder):
        shutil.rmtree(png_save_folder)
    os.makedirs(png_save_folder)
    json_files = os.listdir(json_folder)
    for json_file in json_files:
        json_path = osp.join(json_folder, json_file)
        os.system("labelme_json_to_dataset {}".format(json_path))
        label_path = osp.join(json_folder, json_file.split(".")[0] + "_json/label.png")
        png_save_path = osp.join(png_save_folder, json_file.split(".")[0] + ".png")
        label_png = cv2.imread(label_path, 0)
        label_png[label_png > 0] = 255
        cv2.imwrite(png_save_path, label_png)


if __name__ == '__main__':
    json_dir = r"D:\Desktop\json"	# The folder where json is located
    label_dir = r"D:\Desktop\label"	# The folder where the generated labels are located
    json2png(json_folder=json_dir,png_save_folder=label_dir)