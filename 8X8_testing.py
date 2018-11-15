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
#threshold=6


import time
import numpy as np
import os
from PIL import Image
from keras.models import load_model
model=load_model(model_dir+'8_bilinear_v1.h5')
from skimage.filters import threshold_minimum,threshold_mean


def find_diff_util(org_image,temp_image,debug=False):
    '''
    org_image is the iamges from original_cv
    temp_image is the images from tampered_cv
      '''
    org_map=np.array(model.predict(np.asarray(Image.open(original_dir+org_image))[None,:,:,1,None]))
    #Image.open((original_dir+org_image)).show()
    #print(org_map.shape)
    temp_map=np.array(model.predict(np.asarray(Image.open(tampered_dir+org_image))[None,:,:,1,None]))
    #Image.open((tampered_dir+org_image)).show()
    #print(temp_map.shape)
    diff_map=((abs(org_map-temp_map)).astype(np.uint8))
    threshold=threshold_mean(diff_map)
    bin_map=(diff_map>threshold)
    neg_bin=diff_map<=threshold
    #print(bin_map.shape)
    bin_image=np.zeros((1,512,512,1))
    bin_image[bin_map]=(np.asarray(Image.open(original_dir+org_image))[None,:,:,1,None])[bin_map]
    bin_image[neg_bin]=0
    bin_image=bin_image.reshape((512,512))

    if debug:
        Image.fromarray(bin_image).show()
        print('The current image is the difference image')
        #time.sleep(10)


    return bin_image
def concat_images_util(_imga, _imgb,debug=False):
    imga=_imga
    imgb=_imgb
    if(type(_imga)==str):
        imga=np.asarray(Image.open(original_dir+_imga))
    if(type(_imgb)==str):
        imgb=np.asarray(Image.open(tampered_dir+_imgb))
    
    assert(type(imga)==np.ndarray)
    assert(type(imgb)==np.ndarray)
    if  imga.ndim!=3:
        imga=imga[:,:,None]
        #print(imga.shape)
        
    if imgb.ndim!=3:
        imgb=imgb[:,:,None]
        #print(imgb.shape)
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb+10#randomly added
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha,:wa]=imga[:,:,0]
    new_img[:hb,wa+10:wa+wb+10]=imgb[:,:,0]
    if debug:
        if(new_img.shape[1]>1034):

            Image.fromarray(new_img).show()
            print('This is the image that should have been saved')
            #time.sleep(10)

    return new_img

def concat_images(imga,imgb,imgc,debug=False):
    
    new_image=concat_images_util(imga,imgb,debug)
    assert(type(new_image)==np.ndarray)
    return concat_images_util(new_image,imgc,debug)


def find_diff(org_image,temp_image,debug=False):
    bin_image=find_diff_util(org_image,temp_image,debug)
    final_res=concat_images(org_image,temp_image,bin_image,debug)
    return Image.fromarray(final_res)

def getRes(org_dir,temp_dir,debug=True):
    '''
    org_dir-original_dir
    temp_dir-tampered_dir
    '''
    org_images=os.listdir(original_dir)
    #print(org_images)
    if not debug:

        for i,org_image in enumerate(org_images):
            res_image=find_diff(org_image,org_image)

            save_PIL(res_image,org_image,results_dir)
            print('Finished printing {}/{} image'.format(i+1,len(org_images)))
        
    else:
        res_image=find_diff(org_images[0],org_images[0],debug)
        res_image.show()
        save_PIL(res_image,org_images[0],results_dir)
        #time.sleep(5)
        print('This is the final image')
        exit()

def save_PIL(image_file,image_name,saving_dir):
    image_file=image_file.convert('RGB')
    image_file.save(saving_dir+image_name[:-4]+'.jpg')


if __name__=='__main__':
    getRes(original_dir,tampered_dir,debug=False)

    