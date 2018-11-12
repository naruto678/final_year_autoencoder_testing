#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:12:42 2018

@author: arnab
"""

main_dir='/media/arnab/E0C2EDF9C2EDD3B6/'
test_dir=main_dir+'/large tampered/'
original_dir=test_dir+'/original_cv/'
tampered_dir=test_dir+'/tempered_cv/'
results_dir=test_dir+'/final_results 8X8/'
model_dir=main_dir+'/final_year/'




import numpy as np
import os
from PIL import Image
from keras.models import load_model
model=load_model(model_dir+'8_bilinear_v1.h5')
from skimage.filters import threshold_mean


def find_diff_util(org_image,temp_image):
    '''
    org_image is the iamges from original_cv
    temp_image is the images from tampered_cv
      '''
    org_map=np.array(model.predict(np.asarray(Image.open(original_dir+org_image))[None,:,:,1,None]))
    print(org_map.shape)
    temp_map=np.array(model.predict(np.asarray(Image.open(tampered_dir+org_image))[None,:,:,1,None]))
    print(temp_map.shape)
    diff_map=((abs(org_map-temp_map)).astype(np.uint8))
    threshold=threshold_mean(diff_map)
    bin_map=(diff_map>threshold)
    neg_bin=diff_map<=threshold
    print(bin_map.shape)
    bin_image=np.zeros((1,512,512,1))
    bin_image[bin_map]=255
    bin_image[neg_bin]=0
    bin_image=bin_image.reshape((512,512))
    return bin_image
def concat_images_util(_imga, _imgb):
    imga=_imga
    imgb=_imgb
    if(type(_imga)==str):
        imga=np.asarray(Image.open(original_dir+_imga))
    if(type(_imgb)==str):
        imgb=np.asarray(Image.open(tampered_dir+_imgb))
    
    assert(type(imga)==np.ndarray)
    assert(type(imgb)==np.ndarray)
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb+10#randomly added
    new_img = np.zeros(shape=(max_height, total_width,3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa+10:wa+wb+10]=imgb
    return new_img

def concat_images(imga,imgb,imgc):
    
    new_image=concat_images_util(imga,imgb)
    assert(type(new_image)==np.ndarray)
    return concat_images_util(new_image,imgc)


def find_diff(org_image,temp_image):
    bin_image=find_diff_util(org_image,temp_image)
    final_res=concat_images(org_image,temp_image,bin_image)
    return Image.asarray(final_res)

def getRes(org_dir,temp_dir):
    '''
    org_dir-original_dir
    temp_dir-tampered_dir
    '''
    org_images=os.listdir(original_dir)
    #print(org_images)
    for i,org_image in enumerate(org_images):
        res_image=find_diff(org_image,org_image)
        res_image.save(res_dir+'/'+org_image)
        print('Finished printing {}/{} image'.format(i+1,len(org_images)))
        
        
if __name__=='__main__':
    getRes(original_dir,tampered_dir)