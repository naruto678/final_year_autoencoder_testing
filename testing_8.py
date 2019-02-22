#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:12:42 2018

@author: arnab
"""

main_dir='/media/arnab/E0C2EDF9C2EDD3B6/'
test_dir=main_dir+'large tampered/'
original_dir=test_dir+'original_cv/'
tampered_dir=test_dir+'tempered_cv/'
results_dir=test_dir+'final_results 8X8/'
model_dir=main_dir+'final_year/'
hash_dist_dir=model_dir+'8X8_hash_new_large_2.txt'
#threshold=6


import time
import numpy as np
import os
from PIL import Image,ImageFilter
from PIL.ImageOps import autocontrast
from keras.models import load_model
model_list=[load_model(model_dir+'8_bilinear_v2_new.h5'),load_model(model_dir+'8_bilinear_v1_new.h5'),load_model(model_dir+'8_bilinear_v3.h5'),load_model(model_dir+'8_bilinear_v4.h5')]
def get_hash_layer_index(model_list=model_list):
	n=[]

	for model in model_list:
		for i,layer in enumerate(model.layers):
			a,b,c,d=(x.value for x in layer.output.shape)
			if b==8 and c==8 and d==16:
				n.append(i)
				break
	

	return n









n=get_hash_layer_index()
from skimage.filters import threshold_minimum,threshold_mean
import sys
import matplotlib.pyplot as plt







def find_diff_util(org_image,temp_image,debug=False):
    '''
    org_image is the iamges from original_cv
    temp_image is the images from tampered_cv
      '''
    org_map=0
    temp_map=0
    for model in model_list:

        org_map+=np.array(model.predict(np.asarray((Image.open(original_dir+org_image)))[None,:,:,1,None]))
        temp_map+=np.array(model.predict(np.asarray((Image.open(tampered_dir+org_image)))[None,:,:,1,None]))
    org_map/=len(model_list)
    temp_map/=len(model_list)

    diff_map=((abs(org_map-temp_map)).astype(np.uint8))
    threshold=threshold_minimum(diff_map)
    bin_map=(diff_map>threshold)
    neg_bin=diff_map<=threshold
    #print(bin_map.shape)
    bin_image=np.zeros((1,512,512,1))
    bin_image[bin_map]=(np.asarray(Image.open(tampered_dir+org_image))[None,:,:,1,None])[bin_map]
    bin_image[neg_bin]=0
    bin_image=bin_image.reshape((512,512))

    if debug:
        #Image.fromarray(bin_image).show()

        _image=(Image.fromarray(diff_map[0,:,:,0]))
        _image.show()
        

        print('The current image is the difference image')
        


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
        with open(hash_dist_dir,'a') as hash_file:

            for i,org_image in enumerate(org_images):
                try:
                    res_image=find_diff(org_image,org_image)
                    hash_dist,image_name=find_hash_dist(org_image,org_image,model_list,debug)
                    hash_file.write(image_name+'-->'+str(hash_dist)+'\n')

                    save_PIL(res_image,org_image,results_dir)
                    print('Finished printing {}/{} image'.format(i+1,len(org_images)))
                except:
                    continue
    else:
        res_image=find_diff(org_images[0],org_images[0],debug)
        res_image.show()
        save_PIL(res_image,org_images[0],results_dir)
        hash_dist,image_name=find_hash_dist(org_images[0],org_images[0],model_list,debug)
        print('The hash correlation is {}'.format(hash_dist))

        #time.sleep(5)
        print('This is the final image')
        exit(0)

def save_PIL(image_file,image_name,saving_dir):
    image_file=image_file.convert('RGB')
    image_file.save(saving_dir+image_name[:-4]+'.jpg')

def find_output_util(image,n,model):
    from keras import backend as K

    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

# Testing
    test =image.reshape(1,512,512,1)
    layer_outs = functor([test, 1.])
    return layer_outs[n]



def find_output(image,n,model_list=model_list):
	avg_layer_output=np.zeros((1,8,8,16))
	for i,model in enumerate(model_list):
		temp=find_output_util(image,n[i],model)
		assert(temp.shape==(1,8,8,16))
		avg_layer_output+=temp

	avg_layer_output/=len(model_list)
	return avg_layer_output

def find_hash_dist(org_image,temp_image,model_list=model_list,debug=True):

    '''
    returns the hashdistance alongwith the name of the image it is hashing
    takes org_image and the temp_image as the names
    '''
    first_image=np.array(Image.open(original_dir+org_image))[:,:,0].reshape(512,512)
    second_image=np.array(Image.open(tampered_dir+org_image))[:,:,0].reshape(512,512)
    org_hash=find_output(first_image,n,model_list).flatten().astype('uint64')# 14 is the index 
    temp_hash=find_output(second_image,n,model_list).flatten().astype('uint64')
    if debug:
        print(org_hash.shape)
        print(temp_hash.shape)
    return np.corrcoef(org_hash,temp_hash)[0][1],org_image



def make_graph_from_text(hash_file_text,graph_name):
    l1=[]
    with open(hash_file_text,'r') as hash_file:
        
        for lines in hash_file:
            _line=lines.split('-->')
            
            l1.append(float(_line[1]))
    l1=np.array(l1)
    plt.plot(range(len(l1)),l1)
    plt.xlabel('image_number')
    plt.ylabel('hash_correlation')
    plt.savefig(graph_name)
    plt.show()
    print('The figure is saved')

if __name__=='__main__':

    getRes(original_dir,tampered_dir,debug=False)
    #make_graph_from_text(hash_dist_dir,'hash_graph_small.png')
    