### [YOLOV3数据集制作+测试](https://www.cnblogs.com/pprp/p/9525508.html) + （https://blog.csdn.net/just_sort/article/details/81389571?utm_source=blogxgwz0）+(https://blog.csdn.net/john_bh/article/details/80625220?utm_source=blogxgwz3)

### [云服务器上配置一下](https://blog.csdn.net/u012966194/article/details/80004647)

### [可能会用到的一些代码](https://www.jianshu.com/p/3c980b3bf60f)

### 下载YOLOv3工程项目

```
git clone https://github.com/pjreddie/darknet  
cd darknet  
```

### 修改Makefile配置，使用GPU训练，修改如下：

```python
GPU=1 #如果使用GPU设置为1，CPU设置为0
CUDNN=1  #如果使用CUDNN设置为1，否则为0
OPENCV=0 #如果调用摄像头，还需要设置OPENCV为1，否则为0
OPENMP=0  #如果使用OPENMP设置为1，否则为0
DEBUG=0  #如果使用DEBUG设置为1，否则为0

CC=gcc
NVCC=/home/user/cuda-9.0/bin/nvcc   #NVCC=nvcc 修改为自己的路径
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
...
ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/home/hebao/cuda-9.0/include/  #修改为自己的路径
CFLAGS+= -DGPU
LDFLAGS+= -L/home/hebao/cuda-9.0/lib64 -lcuda -lcudart -lcublas -lcurand  #修改为自己的路径
endif
```

## VOCdevkit文件夹

数据集下载后解压得到一个名为VOCdevkit的文件夹，该文件夹结构如下：

```python
.
└── VOCdevkit     #根目录
    └── VOC2012   #不同年份的数据集，这里只下载了2012的，还有2007等其它年份的
        ├── Annotations        #存放xml文件，与JPEGImages中的图片一一对应，解释图片的内容等等
        ├── ImageSets          #该目录下存放的都是txt文件，txt文件中每一行包含一个图片的名称，末尾会加上±1表示正负样本
        │   ├── Action
        │   ├── Layout
        │   ├── Main           # 文件中每一行包含一个图片的名称，末尾会加上±1表示正负样本
        │   └── Segmentation
        ├── JPEGImages         #存放源图片
        ├── SegmentationClass  #存放的是图片，分割后的效果，见下文的例子
        └── SegmentationObject #存放的是图片，分割后的效果，见下文的例子
```

这里大概介绍一下各个文件夹的内容，更细节的介绍将在后文给出： 
- Annotation文件夹存放的是xml文件，该文件是对图片的解释，每张图片都对于一个同名的xml文件。 
- ImageSets文件夹存放的是txt文件，这些txt将数据集的图片分成了各种集合。如Main下的train.txt中记录的是用于训练的图片集合 
- JPEGImages文件夹存放的是数据集的原图片 
- SegmentationClass以及SegmentationObject文件夹存放的都是图片，且都是图像分割结果图（楼主没用过，所以不清楚）
```xml
<annotation>
    <folder>VOC2012</folder>  #表明图片来源
    <filename>2007_000027.jpg</filename> #图片名称
    <source>                  #图片来源相关信息
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
    </source>
    <size>     #图像尺寸
        <width>486</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented> #是否用于分割
    <object>  #包含的物体
        <name>person</name> #物体类别
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>  #物体的bbox
            <xmin>174</xmin>
            <ymin>101</ymin>
            <xmax>349</xmax>
            <ymax>351</ymax>
        </bndbox>
        <part> #物体的头
            <name>head</name>
            <bndbox>
                <xmin>169</xmin>
                <ymin>104</ymin>
                <xmax>209</xmax>
                <ymax>146</ymax>
            </bndbox>
        </part>
        <part>   #物体的手
            <name>hand</name>
            <bndbox>
                <xmin>278</xmin>
                <ymin>210</ymin>
                <xmax>297</xmax>
                <ymax>233</ymax>
            </bndbox>
        </part>
        <part>
            <name>foot</name>
            <bndbox>
                <xmin>273</xmin>
                <ymin>333</ymin>
                <xmax>297</xmax>
                <ymax>354</ymax>
            </bndbox>
        </part>
        <part>
            <name>foot</name>
            <bndbox>
                <xmin>319</xmin>
                <ymin>307</ymin>
                <xmax>340</xmax>
                <ymax>326</ymax>
            </bndbox>
        </part>
    </object>
</annotation>
```

Layout和Main文件夹所需text文档。
 制作VOC2007数据集中的trainval.txt， train.txt ， test.txt ， val.txt
 trainval占总数据集的50%，test占总数据集的50%；train占trainval的50%，val占trainval的50%；

*_train中存放的是训练集的图片编号。 
*_val中存放的是验证集的图片编号。 
*_trainval是上面两者的合并集合。 
train和val包含的图片没有交集。

生成了Annotation文件夹下的xml之后，就可以生成Main下的4个txt文件，这四个文件夹中存储的时上一步中xml文件的文件名。trainval和 test内容相加为所有xml文件，train和val内容相加为trainval。代码如下：

```python
import os
import random

trainval_percent = 0.5
train_percent = 0.5
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open(txtsavepath+'/trainval.txt', 'w')
ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
```

最后一步是生成YOLO要用的VOC标签格式，首先下载格式转化文件：`wget https://pjreddie.com/media/files/voc_label.py`，打开voc_label.py，进行修改

```python
# 因为没有用到VOC2012的数据，要修改年份
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# 修改检测的物体种
classes = ["safetyhat"]
```

### cfg文件修改

修改pascal数据的cfg文件，打开cfg/voc.data文件，进行如下修改：

```python
classes= 1  # 自己数据集的类别数
train  = /home/xxx/darknet/train.txt  # train文件的路径
valid  = /home/xxx/darknet/2007_test.txt   # test文件的路径
names = /home/xxx/darknet/data/voc.names #用绝对路径
backup = backup #模型保存的文件夹 这里保存是的weight
```

下面也是

![image-20181021202347111](/Users/yunqingqi/Desktop/note/image-20181021202347111.png)

### 最后，打开data/voc.names文件，对应自己的数据集修改类别。

因为darknet下的data文件下自带 voc.names，我们直接改就好了

![image-20181021202423341](/Users/yunqingqi/Desktop/note/image-20181021202423341.png)

修改cfg/yolov3-voc.cfg，首先修改分类数为自己的分类数，然后注意开头部分训练的batchsize和subdivisions被注释了，如果需要自己训练的话就需要去掉，测试的时候需要改回来，最后可以修改动量参数为0.99和学习率改小，这样可以避免训练过程出现大量nan的情况，最后把每个[yolo]层前的conv层中的filters =（类别+5）* 3

change line batch to batch=64
change line subdivisions to subdivisions=8
change line classes=80 to your number of objects in each of 3 [yolo]-layers 

change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer

注意：yolo层里的class要改， yolo层前的conv里的filter要改

![image-20181021203608441](/Users/yunqingqi/Desktop/note/image-20181021203608441.png)

### Download Pretrained Convolutional Weights

```
wget https://pjreddie.com/media/files/darknet53.conv.74
```

### 改完之后就可以训练我们的模型了

`./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74` 

```
nohup ./darknet detector train cfg/voc_birds.data cfg/yolov3-voc-birds.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3_birds.log &
```

nohup $ 是防止因为ssh断开而中断服务器的进程（如果出现“找不到nohup文件”的错误，去掉命令中的“nohup ... &”）

2>1 | tee visualization/train_yolov3_birds.log 是为了保留训练中的log，为了后续绘制loss曲线。

在训练过程中也可以调用一下训练时候阶段输出的权重来测试一下。 使用在/backup中找到每百次的训练权重，然后使用命令测试一下：(正式测试的时候 thresh用的是0.25)

```
./darknet detect  cfg/yolov3.cfg backup/yolov3_801.weights -thresh 0
```

### 训练过程参数的意义

Region xx: cfg文件中yolo-layer的索引；
Avg IOU:当前迭代中，预测的box与标注的box的平均交并比，越大越好，期望数值为1；
Class: 标注物体的分类准确率，越大越好，期望数值为1；
obj: 越大越好，期望数值为1；
No obj: 越小越好；
.5R: 以IOU=0.5为阈值时候的recall; recall = 检出的正样本/实际的正样本
0.75R: 以IOU=0.75为阈值时候的recall;
count:正样本数目。

将训练得到的weights文件拷贝到darknet/weights文件夹下面（参考）

---------------------
![image-20181021204833180](/Users/yunqingqi/Desktop/note/image-20181021204833180.png)

