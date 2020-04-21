# -*- coding: utf-8 -*-  
#此脚本需要执行两次，在91行train_number执行一次，test_number执行一次
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import os
sets = []
classes = []
cnt = 0
f = open('../data/classes/voc.names', 'r').readlines()
for line in f:
    line = line.strip()
    classes.append(line)
    cnt += 1


#原样保留。size为图片大小
# 将ROI的坐标转换为yolo需要的坐标  
# size是图片的w和h  
# box里保存的是ROI的坐标（x，y的最大值和最小值）  
# 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例  
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

#对于单个xml的处理
def convert_annotation(xml_path, out_file, num):
    in_file = open(xml_path)

    print(xml_path)
    
    tree=ET.parse(in_file)

    root = tree.getroot()
    '''
    #加入我的预处理<name>代码：
    for obj in root.findall("object"):
        #obj.append("number") = obj.find('name').text
        obj.find('name').text = "box"
        print(obj.find('name').text)
    tree.write('/root/darknet/hebing2/Annotations/'+ image_add + '.xml')
    '''


    size = root.find('size')
    # <size>
    #     <width>500</width>
    #     <height>333</height>
    #     <depth>3</depth>
    # </size>
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    string_txt = str(num) + " " + str(image) + ' ' + str(w) + ' ' + str(h)
    #在一个XML中每个Object的迭代
    for obj in root.iter('object'):
        #iter()方法可以递归遍历元素/树的所有子元素
        '''
        <object>
        <name>car</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>141</xmin>
            <ymin>50</ymin>
            <xmax>500</xmax>
            <ymax>330</ymax>
        </bndbox>
        </object>
        '''
        #找到所有的标记框
        cls = obj.find('name').text
        #如果训练标签中的品种不在程序预定品种，跳过此object
        if cls not in classes:
            continue
        #cls_id 类别编号
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        #b是每个Object中，一个bndbox上下左右像素的元组
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), 
            int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        string_txt = string_txt + ' ' + str(cls_id) + ' ' + " ".join([str(a) for a in b])
    out_file.write(string_txt + '\n')
    
voc_028 = '../VOC2028'
voc_019 = '../VOC2019'

anno_path = [os.path.join(voc_028, 'Annotations/'), os.path.join(voc_019, 'Annotations/')]

trainval_path = [os.path.join(voc_028, 'ImageSets/Main/trainval.txt'), \
                 os.path.join(voc_019, 'ImageSets/Main/trainval.txt')]

train_image = []
xml_path = []
train_image28 = open(trainval_path[0])
for image in train_image28:
    image = image.strip()
    xml_path.append(anno_path[0] + image + '.xml')

train_image19 = open(trainval_path[1])
for image in train_image19:
    image = image.strip()
    xml_path.append(anno_path[1] + image + '.xml')

out_file = open('train.txt', 'w')
num = 0
for xml_i in xml_path:
    convert_annotation(xml_i, out_file, num)
    num += 1
out_file.close()


'''voc
wd = getcwd()
for year, image_set in sets:
    #如果lebal文件夹中不存在年份文件夹则创建
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    #000034的这种id
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    #2007_test.txt的这种文件
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        #/home/yuxiang/Desktop/darknet/VOCdevkit/VOC2007/JPEGImages/000001.jpg
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        #/VOC%s/ImageSets/Main/train.txt中每一个image_id都要被执行这个方法
        convert_annotation(year, image_id)
    list_file.close()
'''
