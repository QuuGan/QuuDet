# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd
import random
import shutil

def make_dir(resule_path):
    if not os.path.exists(resule_path):
        os.mkdir(resule_path)
    if not os.path.exists(resule_path+"/images"):
        os.mkdir(resule_path+"/images")
    if not os.path.exists(resule_path+"/images/train"):
        os.mkdir(resule_path+"/images/train")
    if not os.path.exists(resule_path+"/images/val"):
        os.mkdir(resule_path+"/images/val")
    if not os.path.exists(resule_path+"/labels"):
        os.mkdir(resule_path+"/labels")
    if not os.path.exists(resule_path+"/labels/train"):
        os.mkdir(resule_path+"/labels/train")
    if not os.path.exists(resule_path+"/labels/val"):
        os.mkdir(resule_path+"/labels/val")

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id,is_train,classes,label_path):
    xp = label_path+'/%s.xml' % (image_id)
    if not os.path.exists(xp):
        return False
    in_file = open(xp,encoding='utf-8')
    out_file = ''
    if is_train:
        out_file = open(resule_path+r'/labels/train/%s.txt' % (image_id), 'w',
                        encoding='utf-8')
    else:
        out_file = open(resule_path+r'/labels/val/%s.txt' % (image_id), 'w',
                        encoding='utf-8')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    h = int(size.find('width').text)
    w = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return True


if __name__ == '__main__':


    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    image_path = r'D:\voc2012\Images'
    label_path = r'D:\voc2012\Annotations'
    resule_path = r'D:\dataset\voc'  # Path for saving results
    file_name = "voc"  # The name of the folder in train.txt
    train_test_ratio = 4  # Ratio of training set to test set

    make_dir(resule_path)

    wd = getcwd()

    train_txt = open(resule_path+r'\train.txt', mode='w', encoding='utf-8')
    val_txt = open(resule_path+r'\val.txt', mode='w', encoding='utf-8')

    for root, dirs, names in os.walk(image_path):
        for name in names:
            num = random.randint(0, train_test_ratio)
            is_train = True
            DestinationName = ''
            prefix = os.path.splitext(name)[0]
            suffix = os.path.splitext(name)[1]
            labelPath = label_path+'/' + prefix+'.xml'
            if not os.path.exists(labelPath):
                continue

            if num == 0:
                val_txt.write(file_name +'/images/val/%s\n' % (name))
                is_train = False
                DestinationName = resule_path+'/images/val/' +name
            else:
                train_txt.write(file_name +'/images/train/%s\n' % (name))
                DestinationName = resule_path+'/images/train/' +name

            RecourseName = os.path.join(root, name)
            prefix = os.path.splitext(name)[0]
            suffix = os.path.splitext(name)[1]
            if convert_annotation(prefix,is_train,classes,label_path):
                shutil.copy(RecourseName, DestinationName)









