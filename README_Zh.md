
# QuuDet

1.支持构建全系列yolo算法模型

2.yolo组件可自由拆分组合，方便构建自定义模型

## 环境配置



``` shell
pip install -r requirements.txt 
```

 

## 网络配置
 
### 组件配置
 * 组件配置文件在文件夹cfg/component中,组件分为backbone、neck和head这3类。
 * 每个组件的层数都是从0层开始计算。 
 * 参数out_layer为默认指定组件输出的层数，参数activate_funtion为默认指定组件的激活函数。
 * 每一层中，第一个参数代表这一层的数据来源的层数，例如-1代表上一层，-2代表上一层的上一层，以此类推。
 * 第二个参数代表该模块重复的次数 。
 * 第三个参数为模块名称，模块的定义和实现在models/common.py中。
 * 第四个参数为模块的入参，模型运行时会自动计算模块的输入通道数，若模块的第一个入参为输入通道数，在配置文件中可省略。

 
在backbone组件中模型逐层构建，以yolov7-tiny的backbone组件为例，模型默认的激活函数为nn.LeakyReLU(0.1)，第0层的的输出将会成为第1层的输入，
第1层的的输出将成为第2、3层的输入，第6层的Concat模块将会将它之前的4层进行拼接，即第2到5层，最后在第14、21、28层分别输出计算结果。

```
# yolov7-tiny-backbone
out_layer: [14,21,28]
activate_funtion: nn.LeakyReLU(0.1)
backbone:
  # [from, number, module, args] c2, k=1, s=1, p=None, g=1, act=True
  [[-1, 1, Conv, [32, 3, 2]],  # 0-P1/2
   [-1, 1, Conv, [64, 3, 2]],  # 1-P2/4
   [-1, 1, Conv, [32, 1, 1]],  # 2
   [-2, 1, Conv, [32, 1, 1]],  # 3
   [-1, 1, Conv, [32, 3, 1]],  # 4
   [-1, 1, Conv, [32, 3, 1]],  # 5
   [[-1, -2, -3, -4], 1, Concat, [1]],  # 6
   [-1, 1, Conv, [64, 1, 1]],  # 7
     ……
  ]
```

在neck组件中使用 in 来指定模型的接收输入的位置，其具体数值根据backbone的输出逆序得到，
以yolov7-tiny-neck为例，其中第0层的 in 表示接收backbone中最后一个输出层数即28，
第11层的 in 表示接收backbone中第2个输出层数即21。

``` 
# yolov7-tiny-neck
out_layer: [29,38,47]
activate_funtion: nn.LeakyReLU(0.1)
neck:
  [[in, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, SP, [5]],
   [-2, 1, SP, [9]],
   [-3, 1, SP, [13]],
   [[-1, -2, -3, -4], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],
   [[-1, -7], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 8
   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [in, 1, Conv, [128, 1, 1]], #11 route backbone P4
   [[-1, -2], 1, Concat, [1]],
      ……
  ]
```
在head组件中，仅需配置一层，运行过程中根据neck组件的out_layer自动连接，其中参数anchors默认为coco数据集产生的anchors，
若默认anchor与当前数据集不符，程序将会自动计算新的anchor，以yolov7-head的组件为例，
IDetect代表检测头的类型，其实现定义在models/yolo.py中，
模块入参中的 nc 表示数据集中检测的种类，运行过程中将自动替换为当前检测种类数。

``` 
# yolov7-head
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

head:
  [
    [[in], 1, IDetect, [nc, anchors]]
  ]
```

### 模型配置
 
 * 模型配置可根据cfg/component中的组件对网络进行组合，可参考cfg/model中已有配置。
 * 在backbone和neck也可配置参数out_layer和activate_funtion，head可配置参数anchors，并以此处为准。
 * 参数loss_funtion可配置所使用的计算损失的方法：
 * * 若使用anchor-base的检测头，则可配置为 "anchor-loss"或"SimOTA-anchor-loss"。
 * * 若使用anchor-free的检测头则可配置为"TalOTA-anchor-free-loss"。

以yolov7-tiny模型为例，其中yolov7-tiny-backbone、yolov7-tiny-neck、yolov7-head组件可以替换为当前已有的组件来获得新的模型。
``` 
# yolov7-tiny
model:
  backbone:
    name: yolov7-tiny-backbone
    out_layer: [ 29,38,47 ]
    activate_funtion: nn.LeakyReLU(0.1)
  neck:
    name: yolov7-tiny-neck
    out_layer: [ 29,38,47 ]
    activate_funtion: nn.LeakyReLU(0.1)
  head:
    name: yolov7-head
    anchors:
      - [ 19,27,  44,40,  38,94 ]  # P3/8
      - [ 96,68,  86,152,  180,137 ]  # P4/16
      - [ 140,301,  303,264,  238,542 ]  # P5/32

loss_funtion: SimOTA-anchor-loss

```
 
## 训练

### 数据准备 

* 数据集采用yolo格式的txt文件作为标签，标签内容分别为：
* 类型索引，归一化的检测框中心点x,y坐标，检测框的宽高。
```  
1 0.683 0.611 0.586 0.244
 ```

* 训练和验证的图片分别在根据train.txt和val.txt中的图片路径划分。
* 图片的文件夹名称可为'images'或'JPEGImages'。
* 标签的文件夹名称可为'labels'。
* 保证图片和标签的文件名相同，并且标签的文件夹路径是基于对应图片的文件夹路径把'images'或'JPEGImages'改成了'labels'，例如：
```  
在train.txt中图片路径为
/home/work/voc/images/train/2007_000005.jpg 
则标签路径应该为
/home/work/voc/labels/train/2007_000005.txt
 ```
### 数据配置

* 数据配置可参照data/voc.yaml进行配置。
* 参数train和val分别为train.txt和val.txt的路径。
* 参数nc为检测类别的数量。
* 参数names为检测的类别名称列表。

``` 
# voc.yaml
train: voc/train.txt
val: voc/val.txt 
nc: 20   
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

### 训练配置

* 在train.py的main函数中可对默认参数进行配置。 
* --weights为预训练模型的路径，可为空。
* --cfg为指定网络模型配置文件所在的路径，可添加缩放比例，例如使用“yolov8n.yaml”将调用缩放比例为“n”的yolov8.yaml。
* --data为数据配置文件所在的路径。
* --epoch为训练的轮数。
* --batch-size为训练时，每一批传入图片的数量，需按照显存大小进行设定。
* --conf-thres为置信度阈值，过滤置信度小于该值的检测框。
* --iou-thres为NMS的iou阈值，当两个检测框的iou大于此值时认为是同一物体，执行NMS保留置信度较高的检测框。
* 如果要恢复前面被中断的训练，那么可以直接加上一个参数--resume。
* 如果加载了已训练的模型，想从0 epoch开始训练，可以加参数--clear。
* 可加参数--warm-restart，启用热重启训练策略，将训练按照epoch分为4个阶段，每个阶段减少数据增强方式。
* 可加参数--fabric，启用混合精度加速训练。
* 
``` shell
# 可带参数运行训练命令
python train.py  --batch-size 16 --data data/voc.yaml  --cfg cfg/custom/yolov7-tiny.yaml --conf-thres 0.25 --iou-thres 0.45 --epoch 600 --img-size 416
```

* 训练结果将会保存在runs/train/exp*/目录下，主要包含了训练的产生的模型和训练过程中所记录的数值。

 ## 测试
### 测试配置
* 在test.py的main函数中可对默认参数进行配置。
* --weights为模型的路径，不可为空。 
* --data为数据配置文件所在的路径。 
* --conf-thres为置信度阈值，过滤置信度小于该值的检测框。
* --iou-thres为NMS的iou阈值，当两个检测框的iou大于此值时认为是同一物体，执行NMS保留置信度较高的检测框。
``` shell
# 可带参数运行检测命令
python test.py  --weights runs/train/exp1/weights/best.pt --data data/voc.yaml 
```
* 测试完成后在控制台输出结果，包含准确率(Pression)，召回率(Recall)，平均精度(mAP)等指标，
* 测试结果保存在runs/test/exp*/目录下。


 ## 检测 
### 检测配置
* 在detect.py的main函数中可对默认参数进行配置。
* --weights为模型的路径，不可为空。 
* --source为需要检测的图片文件所在的路径。 
* --conf-thres为置信度阈值，过滤置信度小于该值的检测框。
* --iou-thres为NMS的iou阈值，当两个检测框的iou大于此值时认为是同一物体，执行NMS保留置信度较高的检测框。
``` shell
# 可带参数运行检测命令
python test.py  --weights runs/train/exp1/weights/best.pt --source inference/images 
```
* 检测结果保存在runs/detect/exp*/目录下。
 
 ## 导出onnx模型

### 导出配置
* 在export.py的main函数中可对默认参数进行配置。
* --weights为模型的路径，不可为空。  
``` shell
# 可带参数运行导出命令
python export.py  --weights runs/train/exp1/weights/best.pt   
```
* 导出结果保存在模型的路径同级目录下。

## 参考项目

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/yjh0410/PyTorch_YOLOv1](https://github.com/yjh0410/PyTorch_YOLOv1)
* [https://github.com/longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3) 
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
* [https://github.com/ultralytics/yolov8](https://github.com/ultralytics/yolov8)
* [https://github.com/AlanLi1997/Slim-neck-by-GSConv](https://github.com/AlanLi1997/Slim-neck-by-GSConv)

## 参考文献

* [[1] Redmon, Joseph, Santosh Divvala, Ross Girshick, and Ali Farhadi. "You only look once: Unified, real-time object detection." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 779-788. 2016.  ](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)
* [[2] Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7263-7271. 2017.](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)
* [[3] Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).](https://arxiv.org/abs/1804.02767)
* [[4] Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934 (2020).](https://arxiv.org/abs/2004.10934)
* [[5] Li, Chuyi, Lulu Li, Hongliang Jiang, Kaiheng Weng, Yifei Geng, Liang Li, Zaidan Ke et al. "YOLOv6: A single-stage object detection framework for industrial applications." arXiv preprint arXiv:2209.02976 (2022).](https://arxiv.org/abs/2209.02976)
* [[6] Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7464-7475. 2023.](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html)
* [[7] Terven, Juan, and Diana Cordova-Esparza. "A comprehensive review of YOLO: From YOLOv1 to YOLOv8 and beyond." arXiv preprint arXiv:2304.00501 (2023).](https://arxiv.org/abs/2304.00501) 
* [[8] Tu, Peng, Xu Xie, Ming Ling, Min Yang, Guo AI, Yawen Huang, and Yefeng Zheng. "FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs." arXiv preprint arXiv:2301.06719 (2023).](https://arxiv.org/abs/2301.06719)
* [[9] Ganesh, Prakhar, Yao Chen, Yin Yang, Deming Chen, and Marianne Winslett. "YOLO-ReT: Towards high accuracy real-time object detection on edge GPUs." In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 3267-3277. 2022.](https://arxiv.org/abs/2110.13713)
* [[10] Li, Hulin, Jun Li, Hanbing Wei, Zheng Liu, Zhenfei Zhan, and Qiliang Ren. "Slim-neck by GSConv: A better design paradigm of detector architectures for autonomous vehicles." arXiv preprint arXiv:2206.02424 (2022).](https://arxiv.org/abs/2206.02424)
* [[11] Ding, Xiaohan, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. "Repvgg: Making vgg-style convnets great again." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13733-13742. 2021.](https://arxiv.org/abs/2101.03697) 
* [[12] Ghiasi, Golnaz, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, and Barret Zoph. "Simple copy-paste is a strong data augmentation method for instance segmentation." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2918-2928. 2021.](https://arxiv.org/abs/2012.07177)
* [[13] Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).](https://arxiv.org/abs/2010.11929) 
* [[14] Tan, Mingxing, Ruoming Pang, and Quoc V. Le. "Efficientdet: Scalable and efficient object detection." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10781-10790. 2020.](https://arxiv.org/abs/1911.09070) 
* [[15] Zheng, Zhaohui, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. "Distance-IoU loss: Faster and better learning for bounding box regression." In Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 07, pp. 12993-13000. 2020.](https://arxiv.org/abs/1911.08287v1) 
* [[16] Li, Xiang, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. "Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection." Advances in Neural Information Processing Systems 33 (2020): 21002-21012.](https://arxiv.org/abs/2006.04388)
* [[17] Zhang, Zhi, Tong He, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. "Bag of freebies for training object detection neural networks." arXiv preprint arXiv:1902.04103 (2019).](https://arxiv.org/abs/1902.04103) 
* [[18] Rezatofighi, Hamid, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. "Generalized intersection over union: A metric and a loss for bounding box regression." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 658-666. 2019.](https://arxiv.org/abs/1902.09630)
* [[19] Tan, Mingxing, and Quoc V. Le. "Mixconv: Mixed depthwise convolutional kernels." arXiv preprint arXiv:1907.09595 (2019).](https://arxiv.org/abs/1907.09595)
* [[20] He, Tong, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. "Bag of tricks for image classification with convolutional neural networks." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 558-567. 2019.](https://arxiv.org/abs/1812.01187) 
* [[21] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).](https://arxiv.org/abs/1710.09412) 
* [[22] Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. "Focal loss for dense object detection." In Proceedings of the IEEE international conference on computer vision, pp. 2980-2988. 2017.](https://arxiv.org/abs/1708.02002)
* [[23] DeVries, Terrance, and Graham W. Taylor. "Improved regularization of convolutional neural networks with cutout." arXiv preprint arXiv:1708.04552 (2017).](https://arxiv.org/abs/1708.04552)
* [[24] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Spatial pyramid pooling in deep convolutional networks for visual recognition." IEEE transactions on pattern analysis and machine intelligence 37, no. 9 (2015): 1904-1916.](https://arxiv.org/abs/1406.4729) 
 


