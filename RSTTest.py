from keras.models import load_model
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import glob
import re
import math
from pop import find_file_name,Data
from functools import partial
import keras.backend as K
import gc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from random import choice
import random
from testing import RotationTest



class RSTTest(RotationTest):
    def __init__(self,original_dir,tampered_dir,model_dir,threshold,results_dir):
        super().__init__(original_dir,tampered_dir,model_dir,threshold,results_dir)

    


    
    def concat_images_util(self,_imga, _imgb,debug=False):
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
            #logging.debug(imga.shape)
            
        if imgb.ndim!=3:
            imgb=imgb[:,:,None]
            #logging.debug(imgb.shape)
        """
        Combines two color image ndarrays side-by-side.
        """
        if imga.ndim!=imgb.ndim:
            raise ValueError('Cannot combine two images with different number of channels')
        ndim=imga.ndim
        ha,wa = imga.shape[:2]
        hb,wb = imgb.shape[:2]
        max_height = np.max([ha, hb])
        total_width = wa+wb+10#randomly added
        new_img = np.zeros(shape=(max_height, total_width,ndim))
        new_img[:ha,:wa]=imga
        new_img[:hb,wa+10:wa+wb+10]=imgb
        if debug:
            if(new_img.shape[1]>1034):

    
                logging.debug('This is the image that should have been saved')
                #time.sleep(10)

        return new_img

    def concat_images(self,imga,imgb,imgc,debug=False,):
    
        new_image=self.concat_images_util(imga,imgb,debug)
        new_image=self.concat_images_util(new_image,imgc,debug)
        return new_image
    def save_PIL(self,image_file,image_name):
        image_file=image_file.convert('RGB')
        logging.debug(os.path.join(self.results_dir,image_name))
        image_file.save(os.path.join(self.results_dir,image_name))

    def __call__(self,n_samples=10,save=True,show=False):
        for i in tqdm(range(n_samples)):
            index=choice(range(len(self.image_pairs)))
            tampered_image_name,original_image_name=self.image_pairs[index]
            tampered_image=self.image(tampered_image_name)
            corrected_image=self.detect_RST(tampered_image,separate=True)            
            if corrected_image is None:
                corrected_image=self.image(original_image_name)
                
            else:
                rc_image=self.image_np(corrected_image[0])
                rs_image=self.image_np(corrected_image[1])
                final_image=self.concat_images(self.image_np(tampered_image),rc_image,rs_image)
                if  not os.path.exists(self.results_dir):
                    os.mkdir(self.results_dir)
                final_image=final_image.astype(np.uint8)
                
                self.save_PIL(Image.fromarray(final_image),'RST_corrected {}'.format(original_image_name))



if __name__=='__main__':
    original_dir=[ '/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_indonesia' , '/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_japan']
    tampered_dir=['/media/arnab/E0C2EDF9C2EDD3B6/lena/test/indonesia','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/japan']
    model_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/8_bilinear_v6_128.h5'
    results_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/ModelTest'
    different_dir='/media/arnab/E0C2EDF9C2EDD3B6/different/different_cv'
    test=RSTTest(original_dir,tampered_dir,model_dir,0.98,results_dir)
    test(20)