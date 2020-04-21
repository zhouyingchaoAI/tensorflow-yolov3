# -*- coding: utf-8 -*-  
import os
import os.path

def write_txt(image_path, train_or_eval):
    for filenames in os.walk(image_path):
        filenames = list(filenames)
        filenames = filenames[2]
        for filename in filenames:
            print(filename)
            with open ('path_'+train_or_eval+'.txt','a') as f:
                f.write(image_path+filename+'\n')


train_or_eval = 'train'
image_path = "/warehouse/ma_yolov3/image/" + train_or_eval + 'image/'
write_txt(image_path, train_or_eval)

