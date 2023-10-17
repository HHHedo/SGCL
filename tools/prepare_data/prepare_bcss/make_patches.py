#!usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

def clip_image_by_order(image_path,label_path, clip_dir, clip_size):
    """
    按照从左到右，从上到下，裁剪图像
    :param image_path: 大图像路径
    :param label: 大图像标签路径
    :param clip_dir: 切割后的小图像和标签存储路径
    :return:
    """
    window_size = 0
    image_array=np.array(Image.open(image_path))
    label_array=np.array(Image.open(label_path))
    (H,W,_)=image_array.shape
    (H_label,W_label)=label_array.shape

    assert (H==H_label) and (W==W_label),\
        "{}：图像和标签尺寸不一致，请检查图像和标签：".format(image_path)

    #图像名
    image_name=os.path.basename(image_path).split(".")[0]

    #创建切割图像、标签文件夹
    clip_image_dir=os.path.join(clip_dir,"img")
    os.makedirs(clip_image_dir,exist_ok=True)
    clip_label_dir=os.path.join(clip_dir,"label")
    os.makedirs(clip_label_dir,exist_ok=True)

    curr_H=0 #起始行

    while curr_H<H-1:
        curr_W=0 #起始列

        if curr_H+clip_size>H:
            clip_height=H-curr_H
        else:
            clip_height=clip_size

        while curr_W<W:
            if curr_W+clip_size>W:
                clip_width=W-curr_W
            else:
                clip_width=clip_size

            #切割图片名
            clip_name="{img_name}_{row}_{col}_size{size}.png".format(
                img_name=image_name,row=curr_H,col=curr_W,size=clip_size)

            #切割图像、标签存储路径
            clip_image_path=os.path.join(clip_image_dir,clip_name)
            clip_label_path=os.path.join(clip_label_dir,clip_name)

            out_img_array=np.ones((clip_height,clip_width,3))*255
            out_label_array=np.ones((clip_height,clip_width)) * 255
            out_img_array = out_img_array.astype(np.uint8)
            out_label_array = out_label_array.astype(np.uint8)

            out_img_array[:clip_height,:clip_width,:]=image_array[curr_H:curr_H+clip_height,curr_W:curr_W+clip_width,:]
            out_label_array[:clip_height,:clip_width]=label_array[curr_H:curr_H+clip_height,curr_W:curr_W+clip_width]

            out_img = Image.fromarray(out_img_array)
            out_img.save(clip_image_path)
            out_label = Image.fromarray(out_label_array)
            out_label.save(clip_label_path)
            curr_W = curr_W + clip_size - window_size

        curr_H = curr_H + clip_size - window_size



if __name__ == '__main__':
    image_dir= '/remote-home/share/DATA/BCSS/0_Public-data-Amgad2019_0.25MPP/rgbs_colorNormalized'
    label_dir= '/remote-home/share/DATA/BCSS/0_Public-data-Amgad2019_0.25MPP/masks'
    clip_dir= '/remote-home/share/DATA/BCSS/Patches_nooverlap'
    os.makedirs(clip_dir,exist_ok=True)

    image_names=os.listdir('/remote-home/share/DATA/BCSS/0_Public-data-Amgad2019_0.25MPP/rgbs_colorNormalized')

    val_institutes = ['OL', 'LL', 'E2', 'EW', 'GM', 'S3']
    train_list = []
    val_list = []
    for image_name in tqdm(image_names):
        institute = image_name[5:7]
        image_file=os.path.join(image_dir,image_name)
        label_file=os.path.join(label_dir,image_name)
        clip_image_by_order(image_file,label_file,clip_dir, 512)
        if institute in val_institutes:
          val_list.append(image_name)
        else:
          train_list.append(image_name)
          
