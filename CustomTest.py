
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
from testing import LocalizationTest

class CustomTest(LocalizationTest):
    def __init__(self,original_dir,tampered_dir,model_dir,results_dir,threshold):
        super().__init__(original_dir,tampered_dir,results_dir,model_dir,threshold)

    def detect_RST(self,img):
        '''
        This takes in a PIL image and detects rotation on it and returns the rotated and rescaled image if possible else returns None
        '''
         
        
        img=img.resize((128,128))
        img_array=np.asarray(img)
        non_points=np.argwhere(img_array>0)
        x,y=non_points[:,0],non_points[:,1]
        s1=(max(x),max(y[x==min(x)]))
        s2=(min(x),min(y[x==max(x)]))
        s3=(min(x[y==max(y)]),min(y))
        s4=(min(x[y==min(y)]),max(y))
        l1=[s1,s2,s3,s4]
                                                                             
        def compute_slope(p1,p2):                                           
            return math.degrees(math.atan(abs((p1[1]-p2[1])/(p1[0]-p2[0]))))
                                                                    
        def euqlidean(p1,p2):                                       
            return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)                                                             
        three_point=[i for i in l1 if i[1]<127]
        for i in three_point:
            if i[1]==0:
                second_point=i
            elif i[0]==127:
                third_point=i
            elif i[0]==0:
                first_point=i

        for i in l1:
            if i not in three_point:
                fourth_point=i
                break
        try:

            
            
            if second_point[0]>63:
                first_slope=compute_slope(first_point,second_point)
                second_slope=compute_slope(fourth_point,third_point)
                rotation_degree=min(first_slope,second_slope)
                img=img.rotate(-rotation_degree)
                img=img.crop(img.getbbox()).resize((128,128))
                
                return img    
            elif second_point[0]<63:
                first_slope=compute_slope(first_point,second_point)
                second_slope=compute_slope(fourth_point,third_point)
                rotation_degree=90-max(first_slope,second_slope)
                img=img.rotate(rotation_degree)
                img=img.crop(img.getbbox()).resize((128,128))
                
                return img
            else:
                return None
        except:
            return None

    def __call__(self,rotation_degree:list,write=False,n_samples=10,sample_list=None):
        _x=list(os.path.join(self.tampered_dir,i) for i in os.listdir(self.tampered_dir))
        _y=list(os.path.join(self.original_dir,i) for i in os.listdir(self.original_dir))
        f1_scores=[]
        correlation_coefficients=[]
        map_threshold=10
        org_threshold=30
        self.first_number=1000
        self.second_number=300
        for i in tqdm(range(n_samples)):
            index=choice(range(len(_x)))
            original_image=self.image(_x[index]) 
            tampered_image=self.image(_y[index])
            for degree in tqdm(rotation_degree):
                rotated_original_image=original_image.rotate(degree)
                corrected_original_image=self.detect_RST(rotated_original_image)
                if corrected_original_image is None:
                    corrected_original_image=rotated_original_image
                original_image_np=self.image_np(corrected_original_image)
                tampered_image_np=self.image_np(tampered_image)
                original_map=self.predict(original_image_np)[0,:,:,:]*255
                tampered_map=self.predict(tampered_image_np)[0,:,:,:]*255
            
                diff_map=original_map-tampered_map
                diff_map[diff_map<0]=-diff_map[diff_map<0]
                org_diff_map=original_image_np-tampered_image_np
                org_diff_map[org_diff_map<=0]=-org_diff_map[org_diff_map<=0]

                result_img=np.zeros(original_map.shape)
                compare_img=np.zeros(original_map.shape)
            
                result_img[diff_map[:,:,0]>=map_threshold]=self.second_number# this is the tampered part
                result_img[diff_map[:,:,1]>=map_threshold]=self.second_number# this is the tampered part
                result_img[diff_map[:,:,2]>=map_threshold]=self.second_number# this is the tampered part
                result_img[diff_map[:,:,0]<map_threshold]=0# this is the not  tampered part
                result_img[diff_map[:,:,1]<map_threshold]=0# this is the not tampered part
                result_img[diff_map[:,:,2]<map_threshold]=0# this is the not tampered part
                
                    
                final_image=self.concat_images(tampered_image_np,original_image_np,result_img/self.second_number*original_image_np)
                final_image=final_image.astype(np.uint8)
                self.save_PIL(Image.fromarray(final_image),'rotated {}'.format(degree)+_x[index][_x[index].rfind('/')+1:])
                compare_img[org_diff_map[:,:,0]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,1]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,2]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,0]<org_threshold]=0
                compare_img[org_diff_map[:,:,1]<org_threshold]=0
                logging.debug('set all the values to the first number')
                compare_img[org_diff_map[:,:,2]<org_threshold]=0
                logging.debug('This is the original_diff_map')
                logging.debug(compare_img)
                #f1=self.f1_score(result_img,compare_img)
                #logging.debug(f1)
                original_hash=self.find_output(original_image_np)
                tampered_hash=self.find_output(tampered_image_np)
                correlation=np.corrcoef(original_hash,tampered_hash)[0][1]
                self.classifier(correlation)
                correlation_coefficients.append(correlation)
                # f1_scores.append(f1)
                corrected_original_image.close()
                rotated_original_image.close()    

            return correlation_coefficients

def customTest(original_dir,tampered_dir,model_dir,results_dir,threshold):
    for i in range(len(original_dir)):
        org_dir,tamp_dir,res_dir=original_dir[i],tampered_dir[i],results_dir[i]
    
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        test=CustomTest(org_dir,tamp_dir,model_dir,res_dir,threshold)
        correlation_coefficient=test(rotation_degree=[0,5,10,15,20,25,30,35,40,42,-5,-10,-15,-20,-25,-30,-35,-40,-42],n_samples=10)    
        print(correlation_coefficient)


if __name__=='__main__':
    original_dir=[ '/media/arnab/E0C2EDF9C2EDD3B6/large tampered/original_cv' , '/media/arnab/E0C2EDF9C2EDD3B6/medium tampered/original_cv','/media/arnab/E0C2EDF9C2EDD3B6/small tampered/original_cv']
    tampered_dir=['/media/arnab/E0C2EDF9C2EDD3B6/large tampered/tampered_cv','/media/arnab/E0C2EDF9C2EDD3B6/medium tampered/tampered_cv','/media/arnab/E0C2EDF9C2EDD3B6/small tampered/tampered_cv']
    model_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/8_bilinear_v6_128.h5'
    results_dir=['/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/rotated large tampered','/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/rotated medium tampered','/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/rotated small tampered']
    customTest(original_dir,tampered_dir,model_dir,results_dir,threshold=0.98)    
 






