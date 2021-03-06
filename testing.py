#!/usr/bin/python
'''
@author:arnab
'''
from keras.models import load_model
import os
import sys
import time
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
class LocalizationTest:
    '''
    The main purpose of this class is to get the original_dir and the test_dir and then save the results in the results_dir
    This class should contain all the tests and any extra test should be added here in the future and the correspondig results 
    will be written in the results directory
    '''
    def __init__(self,original_dir,tampered_dir,results_dir,model_dir,threshold:int):
        '''
        Do not hardcode anything
        '''
        self.original_dir=original_dir
        self.tampered_dir=tampered_dir
        self.results_dir=results_dir
        self.threshold=threshold
        self.model_dir=model_dir
        self.model=load_model(model_dir)
        self.hash_layer_index=[i for i,layer in enumerate(self.model.layers) if layer.output_shape==(None,8,8,16)][-1]

        self.input_shape=self.model.layers[0].get_config()['batch_input_shape'][1:-1]
        self.image=lambda image_name:Image.open(image_name).resize(self.input_shape)
        self.image_np=lambda image:np.asarray(image) # returns a numpy array of shape 128X128X3 expects a image as input
        self.predict=lambda image:self.model.predict(image[None,:,:,:]/255) # model needs 1X128X128X3
        self.false_counter=0

    

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



    def find_output(self,image):
        outputs=[layer.output for layer in self.model.layers]
        functor=K.function([self.model.input]+[K.learning_phase()],outputs)
        
        if image.ndim!=4:
            image=image[None,:,:,:]
        layer_outs=functor([image,0.])
        hash_output=layer_outs[self.hash_layer_index]

        if hash_output.ndim==4:
            hash_shape=hash_output.shape
            hash_output=hash_output[0,:,:,:].reshape(-1)
            logging.debug(hash_output.shape)    
        return hash_output


    def classifier(self,correlation):
        if correlation > self.threshold:
            self.false_counter+=1

    def f1_score(self,result_img,org_diff_map):
        '''
        both of these shits are numpy arrays so go nuts
        the compare_img is set at 300 and the result_img is set at 1000 hence just get the region having the 700 part
        '''
        # result_img=result_img.reshape(-1,self.input_shape[0],self.input_shape[1])
        # org_diff_map=org_diff_map.reshape(-1,self.input_shape[0],self.input_shape[1])
 
        logging.debug(result_img)
        logging.debug(org_diff_map)
        
        diff=result_img-org_diff_map
         


        return (len(diff[diff[:,:,0]==700])+len(diff[diff[:,:,1]==700])+len(diff[diff[:,:,2]==700]))/(len(org_diff_map[org_diff_map[:,:,0]==300])+len(org_diff_map[org_diff_map[:,:,1]==300])+len(org_diff_map[org_diff_map[:,:,2]==300]))        
        

    
    def __call__(self,v=False,write=False,write_with_name=False,concat_image=True,image_name=None,results_dir=None,map_threshold=None,org_threshold=None,tamper_percent=False):
        '''
        This takes the original_image and the tampered image and saves the difference image
        '''

        _x=(os.path.join(self.tampered_dir,i) for i in os.listdir(self.tampered_dir))
        _y=(os.path.join(self.original_dir,i) for i in os.listdir(self.original_dir))
        avg_tamper_percent=[]
        if image_name is not None:
            if not isinstance(image_name,list):
                image_name=[image_name]
            _x=(os.path.join(self.tampered_dir,i) for i in image_name)
            _y=(os.path.join(self.original_dir,i) for i in image_name)
        
            
            if results_dir is None:
                raise ValueError('Provide the value of the results_dir buddy')
            self.results_dir=results_dir

        print('Found {} images corresponding to this'.format(len(os.listdir(self.original_dir))))
        correlation_coefficients=[]
        f1_scores=[]
        if map_threshold is None:
            map_threshold=0
        if org_threshold is None:
            org_threshold=30
        self.first_number=300
        self.second_number=1000
        if image_name:
            min_length=len(image_name)
        else:
            min_length=sys.maxsize

        for i in tqdm(range(min(len(os.listdir(self.original_dir)),min_length)),ascii=True,desc='Tampering Localization'):

            tampered_image_name=next(_x)

            original_image_name=next(_y)
            original_image=self.image(original_image_name)
            tampered_image=self.image(tampered_image_name)

            original_image_np=self.image_np(original_image)
            tampered_image_np=self.image_np(tampered_image)
            org_diff_map=original_image_np-tampered_image_np
            org_diff_map[org_diff_map<=0]=-org_diff_map[org_diff_map<=0]
            if tamper_percent:
                nz=list(filter(None,org_diff_map.flatten()!=0))
                avg_tamper_percent.append(len(nz)/len(org_diff_map.flatten()))
                continue

            original_map=self.predict(original_image_np)[0,:,:,:]*255
            tampered_map=self.predict(tampered_image_np)[0,:,:,:]*255
            
            diff_map=original_map-tampered_map
            diff_map[diff_map<0]=-diff_map[diff_map<0]
            
            
            logging.debug(original_image_name)
            logging.debug(original_image_np.shape)
            logging.debug(tampered_image_np.shape)
            logging.debug(original_map.shape)
            logging.debug(tampered_map.shape)
            logging.debug(org_diff_map)

            '''
            Where there is difference show black 
            rest show the entire image
            
            '''
             
            
            if not write :
                result_img=np.zeros(original_map.shape)
                compare_img=np.zeros(original_map.shape)
            
                result_img[diff_map[:,:,0]>=map_threshold]=self.second_number# this is the tampered part
                result_img[diff_map[:,:,1]>=map_threshold]=self.second_number# this is the tampered part
                result_img[diff_map[:,:,2]>=map_threshold]=self.second_number# this is the tampered part
                result_img[diff_map[:,:,0]<map_threshold]=0# this is the not  tampered part
                result_img[diff_map[:,:,1]<map_threshold]=0# this is the not tampered part
                result_img[diff_map[:,:,2]<map_threshold]=0# this is the not tampered part
                
                if concat_image:

                    final_image=self.concat_images(tampered_image_np,original_image_np,result_img/self.second_number*original_image_np)
                    final_image=final_image.astype(np.uint8)
                else:
                    final_image=result_img/self.second_number*original_image_np
                    final_image=final_image.astype(np.uint8)    
                self.save_PIL(Image.fromarray(final_image),original_image_name[original_image_name.rfind('/')+1:])
                compare_img[org_diff_map[:,:,0]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,1]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,2]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,0]<org_threshold]=0
                compare_img[org_diff_map[:,:,1]<org_threshold]=0
                logging.debug('set all the values to the first number')
                compare_img[org_diff_map[:,:,2]<org_threshold]=0
                logging.debug('This is the original_diff_map')
                logging.debug(compare_img)
                f1=self.f1_score(result_img,compare_img)
                logging.debug(f1)
                original_hash=self.find_output(original_image_np)
                tampered_hash=self.find_output(tampered_image_np)
                correlation=np.corrcoef(original_hash,tampered_hash)[0][1]
                self.classifier(correlation)
                correlation_coefficients.append(correlation)
                f1_scores.append(f1)
                if v:
                    Image.fromarray(final_image).show()
                    sys.exit()
            else:
                original_hash=self.find_output(original_image_np)
                tampered_hash=self.find_output(tampered_image_np)
                correlation=np.corrcoef(original_hash,tampered_hash)[0][1]
                correlation_coefficients.append(correlation)
            original_image.close()
            tampered_image.close()
            del original_image_np,tampered_image_np;
            time.sleep(0.01)
            gc.collect();


        if write_with_name:
            return correlation_coefficients,f1_scores,self.false_counter,os.listdir(self.original_dir)
        if tamper_percent:
            return sum(avg_tamper_percent)/(len(avg_tamper_percent)*3)

        return correlation_coefficients,f1_scores,self.false_counter



class ModelTest(Data):
    def __init__(self,x_dir:list,y_dir:list,model_dir,threshold:int,results_dir:str):
        '''
            The class inherits from the data class we made in the training code            
            x_train contains all the operations_images and y_train contains all the 
            original_images

            this class will evaluate the model performance on a dataset 
            The initialier should give a threshold using which the class will evaluate the tpr and the fpr values
            The tpr and the fpr values should be in a table format shown below for which we will use pandas library

            dataset     operations         tpr     fpr mean_f1_score mean_hash_correlaton
            indonesia    
            italy 
            japan

            this table will be repeated for model with and without regularization and for model having inputs of both 512 and 128
            
            a good value of threshold needs to be decided by checking the histogram of hash_correlation array
            a typical vlaue od 0.98 is selcted as threshold

        '''

        super().__init__(x_dir,y_dir,128,128)
        self.model_dir=model_dir
        self.model=load_model(model_dir)
        self.input_shape=self.model.layers[0].input_shape[1:-1]
    
        self.hash_layer_index=[i for i,layer in enumerate(self.model.layers) if layer.output_shape==(None,8,8,16)][-1]
        self.threshold=threshold
        self.results_dir=results_dir
        self.operation_filter=lambda operation_name,image_name:image_name.find(operation_name)!=-1
        self.image=lambda image_name:Image.open(image_name).resize(self.input_shape)
        self.image_np=lambda image:np.asarray(image)
        self.predict=lambda image:self.model.predict(image[None,:,:,:]/255) 
                
    
    def find_output(self,image):
        outputs=[layer.output for layer in self.model.layers]
        functor=K.function([self.model.input]+[K.learning_phase()],outputs)
        
        if image.ndim!=4:
            image=image[None,:,:,:]
        layer_outs=functor([image,0.])
        hash_output=layer_outs[self.hash_layer_index]

        if hash_output.ndim==4:
            hash_shape=hash_output.shape
            hash_output=hash_output[0,:,:,:].reshape(-1)
            
            return hash_output

    
    def create_histogram(self,input_array:np.array):
        pass

    
    def hash_correlation(self,operation)->list:
        '''
        returns the  hash_correlation_coefficient  array for the operation
        '''
        __x=self.x_train[self.indexes]
        __y=self.y_train[self.indexes]
        assert(len(__x)==len(__y))
        x_test=(self.image(image_name) for image_name in __x)
        y_test=(self.image(image_name) for image_name in __y)

        correlation_coefficients=[]
        for i in tqdm(__x,ascii=True,desc='ModelTest'):
            _original_image=next(x_test)
            tampered_image=next(y_test)
            original_image=self.detect_RST(_original_image)
            if original_image is None:
                original_image=_original_image
            else:
                _original_image.close()
            original_hash=self.find_output(self.image_np(original_image))
            tampered_hash=self.find_output(self.image_np(tampered_image))
            c=np.corrcoef(original_hash,tampered_hash)
            correlation_coefficients.append(c[0][1])
            original_image.close()
            tampered_image.close()
            gc.collect()
        assert(correlation_coefficients is not None)
        #logging.debug('The correlation coefficient for this operation is {}'.format(correlation_coefficients))
        return correlation_coefficients

    def detect_RST(self,img,separate=False):
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
        if s2==s3 or s1==s2 or s1==s4 or s2==s4 or s3==s4 or s1==s3:
            logging.debug('The image is not rotated')
            return None                                                                     
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
                if abs(first_slope-second_slope)<3:
                    logging.debug('the image is rotated clckwise')
                    rotation_degree=min(first_slope,second_slope)
                    img=img.rotate(-rotation_degree)
                    
                    img1=img.crop(img.getbbox()).resize((128,128))
                    if separate:
                        return img,img1
                    
                    return img1
                else:
                    return None    
            elif second_point[0]<63:
                first_slope=compute_slope(first_point,second_point)
                second_slope=compute_slope(fourth_point,third_point)
                if abs(first_slope-second_slope)<3:
                    logging.debug('the image is counterclockwise')
                    rotation_degree=90-max(first_slope,second_slope)
                    img=img.rotate(rotation_degree)
                    img1=img.crop(img.getbbox()).resize((128,128))
                    if separate:
                        return img,img1
                    return img1
                else:
                    return None

            else:
                return None
        except:
            return None





    def tpr(self,correlation_coefficients):
        '''
        returns the true positive  score for the image_data
        '''
        tpr=0
        return len(list(filter(lambda x:x>self.threshold,correlation_coefficients)))/len(correlation_coefficients)


        
    def fpr(self,correlation_coefficients):
        '''
            returns the false positive score rate
        '''
        return 1-self.tpr(correlation_coefficients)
    
    def __call__(self,operation,n_samples=None,threshold=None,write=False)->list:
        
        self.indexes=[i for i,image_name in enumerate(self.x_train) if operation in image_name]
        if n_samples is None:
            self.indexes=self.indexes
        else:
            self.indexes=self.indexes[:n_samples]

        logging.debug('Found {} number of images corresponding to this {} in model test'.format(len(self.indexes),operation))
        if len(self.indexes)==0:
            return None,None,None
        if threshold is not None:
            self.threshold=threshold

        correlation=self.hash_correlation(operation)
        if write:
            with open(os.path.join(results_dir,'same_correlation'),'a') as fp:
                for values in np.array(correlation).flatten():
                    if values is not None:
                        fp.write(str(values)+'\n')
            logging.debug('savinf the reslts to a file')
        tpr_value=self.tpr(correlation)
        fpr_value=1-tpr_value
        
        return correlation,tpr_value,fpr_value
class Visualize:
    def __init__(self,results_dir,org_dir,operations=None):
        self.results_dir=results_dir
        self.org_dir=org_dir
        self.dataset_name=org_dir[org_dir.rfind('_')+1:]
        logging.debug(org_dir)
        self.operations=operations
        self.dictionary=None
        if operations is not None:

            self.dictionary=dict([((org_dir,b),[]) for b in operations]) 
            logging.debug(self.dictionary)

    def plot(self,data,operation,debug=False,instance=None):
        fig,ax=plt.subplots(1,2)
        N,bins,patches=ax[0].hist(data,bins=100)
        fracs=N/N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())


        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        ax[0].set_title('Histogram : {}'.format(operation))
        ax[0].set_xlabel('Data')
        ax[0].set_ylabel('Frequency')
        # ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))


        ax[1].plot(data)
        ax[1].set_title('Graph {}'.format(operation))
        ax[1].set_xlabel('Data')
        if instance is None:

            ax[1].set_ylabel('hash correlation')
        else:
            ax[1].set_ylabel('f1 scores')
        fig.tight_layout()
        if debug:
            plt.show()
            exit()
        fig.savefig('{}-{}.png'.format(os.path.join(self.results_dir,self.dataset_name),operation),dpi=120)
        plt.close('all')
    
    
    def add(self,index,column,data:list):
        self.dictionary[(index,column)].append(data)
        
    def __call__(self,correlation,tpr_value,fpr_value,operation,instance=None):

        '''
        On being called it needs to do two things 
        1.plot the histogram of the correlation coefficients for the given operation and save the file in the results_dir
        2.add the correspoinding line to the dataframe
        '''
        if self.dictionary is not  None:
            self.add(self.org_dir,operation,[tpr_value,fpr_value,sum(correlation)/len(correlation)])
        self.plot(correlation,operation,instance=instance)
class RotationTest(ModelTest):
    def __init__(self,x_dir:list,y_dir:list,model_dir,threshold:int,results_dir:str):
        '''
        this gives me how the hash codes generated are robust against the roation operations and to what degree are they allowed
        self.x_train :contains all the names of the operations_images 
        self.y_train: contains all the images of the real_images

        should contain a dictionary showing the rate of detection for each degree of rotation of an image .The threshold is taken the same
        above

        '''
        super().__init__(x_dir,y_dir,model_dir,threshold,results_dir)
        self.model_dir=model_dir
            
        self.image_pairs=[(image,self.y_train[i]) for i,image in enumerate(self.x_train) if 'rotation' in image]
        self.threshold=threshold
        self.results_dir=results_dir
        self.angles_list=[-42,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,42]
        self.total_dict={i:0 for i in self.angles_list}
        self.detect_dict={i:0 for i in self.angles_list}

    def hash_correlation(self,tampered_image_name:str,original_image_name:str,write=False,correction=True)->None:
        '''
        gets the original iamge and the tampered image name and sends the correlation coefficient 
        '''

        original_image=self.image(original_image_name)
        _tampered_image=self.image(tampered_image_name)
        if correction:
            tampered_image=self.detect_RST(_tampered_image)
        else:
            tampered_image=_tampered_image
        if tampered_image is None:
            tampered_image=_tampered_image


        original_hash=self.find_output(self.image_np(original_image))
        tampered_hash=self.find_output(self.image_np(tampered_image))
        c=np.corrcoef(original_hash,tampered_hash)[0][1]
        original_image.close()
        tampered_image.close()
        gc.collect()
        
        rotation_degree=int(tampered_image_name[tampered_image_name.find('ROT_')+4:-4])
        
        

        if rotation_degree not in self.angles_list:
            raise ValueError('{} degree is not present in the angles list'.format(rotation_degree))
        self.total_dict[rotation_degree]+=1
        if c>=0.98:
            self.detect_dict[rotation_degree]+=1
        
    def detect_RST(self,img,separate=False):
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

            
            
            if second_point[0]>fourth_point[0]:
                first_slope=compute_slope(first_point,second_point)
                second_slope=compute_slope(fourth_point,third_point)
                rotation_degree=min(first_slope,second_slope)
                img=img.rotate(-rotation_degree)
                img1=img.crop(img.getbbox()).resize((128,128))
                if separate:
                    return img,img1
                return img1    
            elif second_point[0]<fourth_point[0]:
                first_slope=compute_slope(first_point,second_point)
                second_slope=compute_slope(fourth_point,third_point)
                rotation_degree=90-max(first_slope,second_slope)
                img=img.rotate(rotation_degree)
                img1=img.crop(img.getbbox()).resize((128,128))
                if separate:
                    return img,img1
                return img1
            else:
                return None
        except:
            return None


    




    def __call__(self,n_samples=None,correction=True):
        print('Threshold is set at {}'.format(self.threshold))
        if n_samples is None:
            n_samples=len(self.image_pairs)
        else:
            n_samples=n_samples
        for tampered_image,original_image in tqdm(self.image_pairs[:n_samples],desc='Rotation Test'):
            self.hash_correlation(tampered_image,original_image,correction=correction)
        assert(len(self.total_dict)==len(self.detect_dict))
        self.detect_dict={a:self.detect_dict[a]/b for (a,b) in self.total_dict.items()}
        
        
        x=sorted(self.detect_dict.keys())
        y=[self.detect_dict[i] for i in x]
        
        plt.rcParams['font.weight']='bold'
        plt.rcParams['axes.labelweight']='bold'

        plt.bar(x,y,align='center')
        plt.locator_params(axis='x', nbins=20)
        plt.xlabel('Degrees of rotation')
        plt.ylabel('True positive rate')
        #plt.title('tpr vs degree')
        plt.savefig('tpr_vs_degree.png')
        print(str(self.detect_dict))
class DiscernibiltyTest:
    def __init__(self,x_dir:str,model_dir:str,threshold:int,results_dir):
        self.x_dir=x_dir
        self.model_dir=model_dir
        self.model=load_model(model_dir)
        self.input_shape=self.model.layers[0].input_shape[1:-1]
        self.load_image=lambda image_name:Image.open(image_name).resize(self.input_shape)
        self.image_np=lambda image:np.asarray(image)
        self.fpr_counter=0
        self.threshold=threshold
        self.results_dir=results_dir
        self.x_train=[os.path.join(self.x_dir,image_name) for image_name in os.listdir(x_dir)]
        self.hash_layer_index=[i for i,layer in enumerate(self.model.layers) if layer.output_shape==(None,8,8,16)][-1]
        assert(len(self.x_train)>0)
        self.mean_hash_correlation=0
        self.std_hash_correlation=0

    def find_output(self,image):
        outputs=[layer.output for layer in self.model.layers]
        functor=K.function([self.model.input]+[K.learning_phase()],outputs)
        
        if image.ndim!=4:
            image=image[None,:,:,:]
        layer_outs=functor([image,0.])
        hash_output=layer_outs[self.hash_layer_index]

        if hash_output.ndim==4:
            hash_shape=hash_output.shape
            hash_output=hash_output[0,:,:,:].reshape(-1)
            
            return hash_output

    def __call__(self,n_samples=None,threshold=None,write=False):
        '''
        n_samples is the number of samples for which the discerniblity test will be performed
        '''
        print('the  number of samples is {}'.format(n_samples))
        if threshold is not None:
            self.threshold=threshold
        if n_samples is  None:
            total_samples=len(self.x_train)
        else:
           total_samples=n_samples
        l1=np.zeros(int(total_samples/2*(total_samples-1)))
        if write:
            with open(os.path.join(self.results_dir,'different_correlation'),'a') as fp:
                for i in tqdm(range(int(total_samples*(total_samples-1)/2)),ascii=True,desc='Rotation Test with write'):
                    first_image_name=choice(self.x_train)
                    second_image_name=choice(self.x_train)
                    while  first_image_name==second_image_name:
                        second_image_name=choice(self.x_train)
                    first_image=self.load_image(first_image_name)
                    second_image=self.load_image(second_image_name)
                    first_image_np=self.image_np(first_image)
                    second_image_np=self.image_np(second_image)
                    logging.debug(first_image_np.shape)
                    logging.debug(second_image_np.shape)
                    first_hash=self.find_output(first_image_np/255)
                    second_hash=self.find_output(second_image_np/255)
                    corr=np.corrcoef(first_hash,second_hash)[0][1]
                    logging.debug('correlation coefficient is {}'.format(corr))
                    first_image.close()
                    second_image.close()
                    del first_image_np,first_hash
                    del second_image_np,second_hash
                    gc.collect()

                    if corr>=self.threshold:
                        self.fpr_counter+=1
                    fp.write(str(corr)+'\n')
            return 

        else:
            l1=np.zeros((total_samples*(total_samples-1))//2)
            for i in tqdm(range(int(total_samples*(total_samples-1)/2)),ascii=True,desc='Rotation Test without write'):
                        first_image_name=choice(self.x_train)
                        second_image_name=choice(self.x_train)
                        while  first_image_name==second_image_name:
                            second_image_name=choice(self.x_train)
                        first_image=self.load_image(first_image_name)
                        second_image=self.load_image(second_image_name)
                        first_image_np=self.image_np(first_image)
                        second_image_np=self.image_np(second_image)
                        logging.debug(first_image_np.shape)
                        logging.debug(second_image_np.shape)
                        first_hash=self.find_output(first_image_np/255)
                        second_hash=self.find_output(second_image_np/255)
                        corr=np.corrcoef(first_hash,second_hash)[0][1]
                        logging.debug('correlation coefficient is {}'.format(corr))
                        first_image.close()
                        second_image.close()
                        
                        del first_image_np,first_hash
                        del second_image_np,second_hash
                        gc.collect()

                        if corr>=self.threshold:
                            self.fpr_counter+=1
                        l1[i]=corr



        
        return l1,self.fpr_counter/n_samples
class ComparisonTest(ModelTest):
    def __init__(self,x_dir,y_dir,model_dir,threshold,results_dir):
        super().__init__(x_dir,y_dir,model_dir,threshold,results_dir)
    def __call__(self,folder_name=None,image_name=None,operation_name=None,plot=False,show=False):
        if folder_name is  None:
            raise ValueError('You need to specify the folder name')
        if image_name is None:
            raise ValueError('You need to specify the image name')
        if operation_name is None:
            raise ValueError('You need to specify the operation name')
        for _image_name in image_name:

            self.indexes=[i for i,image  in enumerate(self.x_train) if folder_name in image and _image_name in image and operation_name in image]
            print(_image_name,len(image_name))
            correlation_coefficients=self.hash_correlation(operation_name)
            assert('ModelTest' in results_dir)
            if plot:
                if len(correlation_coefficients)<=0:
                    return 
                assert(len(correlation_coefficients)>0)
                plt.plot(range(len(correlation_coefficients)),correlation_coefficients,'xb-',label='Hash correlation for {}'.format(operation_name))
                plt.plot(range(len(correlation_coefficients)),[self.threshold]*(len(correlation_coefficients)),label='Threshold Line')
                plt.ylabel('Hash correlation for {}'.format(operation_name))
                plt.xlabel('Image Number')
                plt.legend()
                if show:
                    plt.show()
                    sys.exit()
                plt.savefig(os.path.join(results_dir,'{0} {1} Comparison_{2}.jpg'.format(folder_name,image_name,operation_name)))
            
        plt.close()


class CustomTest(LocalizationTest):
    def __init__(self,original_dir,tampered_dir,model_dir,results_dir,threshold):
        super(LocalizationTest).__init__(original_dir,tampered_dir,model_dir,results_dir,threshold)

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
        for i in tqdm(range(n_samples)):
            index=choice(len(_x))
            original_image=self.image(self_x[index]) 
            tampered_image=self.image(self._y[index])
            for degree in tqdm(rotation_degree):
                rotated_original_image=original_image.rotate(degree)
                corrected_original_image=self.detect_RST(rotated_original_image)
                assert(corrected_original_image is not None)
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
                self.save_PIL(Image.fromarray(final_image),'rotated {}'.format(degree)+original_image_name[original_image_name.rfind('/')+1:])
                compare_img[org_diff_map[:,:,0]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,1]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,2]>=org_threshold]=self.first_number
                compare_img[org_diff_map[:,:,0]<org_threshold]=0
                compare_img[org_diff_map[:,:,1]<org_threshold]=0
                logging.debug('set all the values to the first number')
                compare_img[org_diff_map[:,:,2]<org_threshold]=0
                logging.debug('This is the original_diff_map')
                logging.debug(compare_img)
                f1=self.f1_score(result_img,compare_img)
                logging.debug(f1)
                original_hash=self.find_output(original_image_np)
                tampered_hash=self.find_output(tampered_image_np)
                correlation=np.corrcoef(original_hash,tampered_hash)[0][1]
                self.classifier(correlation)
                correlation_coefficients.append(correlation)
                f1_scores.append(f1)
                corrected_original_image.close()
                rotated_original_image.close()    

            return correlation_coefficients,f1_scores
class ComparisonTest(ModelTest):
    def __init__(self,x_dir,y_dir,model_dir,threshold,results_dir):
        super().__init__(x_dir,y_dir,model_dir,threshold,results_dir)
    def __call__(self,folder_name=None,image_name=None,operation_name=None,plot=False,show=False):
        color_list=list('bgrcmykw')
        assert(len(color_list)>=len(image_name))
        if folder_name is  None:
            raise ValueError('You need to specify the folder name')
        if image_name is None:
            raise ValueError('You need to specify the image name')
        if operation_name is None:
            raise ValueError('You need to specify the operation name')
        for i,_image_name in enumerate(image_name):

            self.indexes=[i for i,image  in enumerate(self.x_train) if folder_name in image and _image_name in image and operation_name in image]
            print(_image_name,len(image_name))
            correlation_coefficients=self.hash_correlation(operation_name)
            assert('ModelTest' in results_dir)
            if plot:
                if len(correlation_coefficients)<=0:
                    return 
                assert(len(correlation_coefficients)>0)
                plt.plot(range(len(correlation_coefficients)),correlation_coefficients,'x{}-'.format(color_list[i]),label='Image {:02d}'.format(i))
                plt.ylabel('image :{:02d}'.format(i))
                plt.xlabel('Image Pair Number')
                plt.title('Hash correlation vs image number for {}'.format(operation_name))
                
                if show:
                    plt.show()
                    sys.exit()
                
        plt.plot(range(len(correlation_coefficients)),[self.threshold]*(len(correlation_coefficients)),'k',label='Threshold Line')
        plt.legend()
        plt.savefig(os.path.join(results_dir,'{0} {1} Comparison_{2}.jpg'.format(folder_name,image_name,operation_name)))
        plt.close()




def modelTest1(original_dir,tampered_dir,model_dir,results_dir):
    '''
    this is for generating tpr and fpr rates for different operations in diffrent datasets
    '''
    print('Model Testing Phase')
    operations=['brightness','compression','contrast','gamma','gaussian','rotation','salt and pepper','scaling','speckle','watermark']
    for i in range(2,len(original_dir)):
        org_dir,tamp_dir=original_dir[i],tampered_dir[i]
        logging.debug(org_dir)
        logging.debug(tamp_dir)
        test=ModelTest([org_dir],[tamp_dir],model_dir,0.98,results_dir)
        visualize=Visualize(results_dir,org_dir[org_dir.rfind('_')+1:],operations)
        print('Currently doing {} folder for model test'.format(org_dir[org_dir.rfind('/')+1:]))
        for operation in operations:
            if 'rotation' in operation:

                print('Operation {}'.format(operation))
                correlation,tpr_value,fpr_value=test(operation)
                if correlation is None:
                    continue
                print('Now plotting the results and saving them')
                visualize(correlation,tpr_value,fpr_value,operation)

                with open(os.path.join(results_dir,org_dir[org_dir.rfind('_')+1:]),'a') as f:
                    for (a,b),c in visualize.dictionary.items():
                        f.write('{} {} {}'.format(a,b,c))
                    print('Saved the results in the file')
def localTest(oiginal_dir,tampered_dir,model_dir,results_dir,write=False,plot=False,hash_corr_with_image_no=None,image_name=None,custom_results_dir=None,tamper_percent=False):
    if image_name:
        if custom_results_dir is None:
            raise ValueError('you did not pass the value of the results directory')

        assert(os.path.exists(custom_results_dir))
        
    print('Doing the localization test')
    for i in range(len(original_dir)):
        org_dir,tamp_dir,res_dir=original_dir[i],tampered_dir[i],results_dir[i]
        if not os.path.exists(results_dir[i]):
            os.mkdir(results_dir[i])
        test=LocalizationTest(org_dir,tamp_dir,res_dir,model_dir,0.98)
        if hash_corr_with_image_no:

            correlation,f1_scores,false_counter,image_names=test(concat_image=False)
        if tamper_percent:
            avg_tamper_percent=test(tamper_percent=True)
            print(avg_tamper_percent)         
        else:
            correlation,f1_scores,false_counter=test(concat_image=False,image_name=image_name,results_dir='/home/arnab/Downloads',map_threshold=5,org_threshold=30)
        if write:
            with open(os.path.join(res_dir,'f1_scores'),'a') as fp:
                for score in f1_scores:
                    fp.write(str(score)+'\n')
            with open(os.path.join(res_dir,'fpr'),'a') as fp:
                fp.write(str(fp)) 
            with open(os.path.join(res_dir,'tampred_hash_correlation'),'a') as fp:
                for corr in correlation:
        
                    fp.write(str(corr)+'\n')
        
        elif plot:
            visualize=Visualize(res_dir,org_dir)
            visualize(correlation,0,false_counter,'tampering {} '.format(i))
            visualize(f1_scores,0,0,'f1_scores {}'.format(l1[i]),instance='something')
        if hash_corr_with_image_no:
            with open(os.path.join(res_dir,'tampred_hash_correlation.txt'),'a') as fp:
                for i in range(len(correlation)):
                    fp.write(str(correlation[i])+' '+' '+image_names[i]+'\n')
            print('Finished writing results to tampred_hash_correlation.txt')
            with open(os.path.join(res_dir,'f1_scores.txt'),'a') as fp:
                for i in range(len(correlation)):
                    fp.write(str(f1_scores[i])+' '+image_names[i]+'\n')
            print('Finished writing results to f1_scores.txt')
def rotationTest(original_dir,tampered_dir,mdoel_dir,threshold,results_dir,correction=True):

    print('Currently doing rotation test')
    rotation=RotationTest(original_dir,tampered_dir,model_dir,threshold,results_dir)
    d1=rotation(correction=correction,n_samples=100)
    with open(os.path.join(results_dir,'rotation.txt'),'a') as fp:
        fp.write(str(d1))

def discernibilityTest(results_dir,threshold=0.98,plot=False):
    print('Currently doing discernibility test')
    with open(os.path.join(results_dir,'different_correlation'),'r') as fp:
        lines=[]
        _lines=fp.readlines()
        for line in _lines:
            try:
                lines.append(float(line[:-3]))
            except:
                continue
        assert(len(lines)>0)
        logging.debug('the total number of correlation coefficients is {}'.format(len(lines)))

        if plot:
            plt.rcParams['font.weight']='bold'
            plt.rcParams['axes.labelweight']='bold'
            plt.scatter(range(len(lines)),lines,c='red',label='Hash Correlation of the point')
            
            plt.plot(range(len(lines)),[threshold]*(len(lines)),color='k',label='Threshold line')
            plt.legend()
            plt.xlabel('Image Pair Number')
            plt.title('Hash correlation coefficient vs image number ')            
            plt.ylabel('Hash Correlation Coefficient')
            plt.xlim(0,len(lines))
            plt.ylim(0,1)
            plt.savefig(os.path.join(results_dir,'Discernibility.jpg'))
        fpr_counter=len(list(filter(lambda x:x>threshold,lines)))
        return lines,fpr_counter/len(lines)
def modelTest2(results_dir,threshold=0.98,plot=False,show=True):
    '''
    just find the hash correlation for images undergoing content preserving operations and images having totally different content
    here we are doing this only for aerials dataset as bechmarks are only available for this one
    the different_dir wll conttain the path of all the different images whose combinations we are to take

    and remember always thakuria keyword arguments always come after positional arguments you stupid fuck
    ''' 
    print('Currently doing the distribution for same and the different graph')
    with open(os.path.join(results_dir,'same_correlation')) as fp:
        same_corrleation_coefficients=[float(line) for line in fp.readlines()]
        tpr_counter=len(list(filter(lambda x: x>threshold,same_corrleation_coefficients)))
    tpr_rate=tpr_counter/len(same_corrleation_coefficients)
        
    different_correlation_coefficients,fpr_rate=discernibilityTest(results_dir,threshold)
    with open(os.path.join(results_dir,'tampered_correlation')) as fp:
        tampered_correlation_coefficients=[float(line) for line in fp.readlines()]
    logging.debug(len(list(filter(lambda x : x> 0.98 , different_correlation_coefficients)))/len(different_correlation_coefficients))
    logging.debug(len(list(filter(lambda x:x<0.98, same_corrleation_coefficients)))/len(same_corrleation_coefficients))
    if plot:
        same_corrleation_coefficients=np.array(same_corrleation_coefficients).flatten()
        counts1,bins1=np.histogram(same_corrleation_coefficients,bins=100)
        counts2,bins2=np.histogram(np.array(different_correlation_coefficients).flatten(),bins=100)
        counts3,bins3=np.histogram(np.array(tampered_correlation_coefficients).flatten(),bins=100)
        counts1=np.array(counts1)/sum(counts1)
        counts2=np.array(counts2)/sum(counts2)
        counts3=np.array(counts3)/sum(counts3)
            
        
        logging.debug(counts1)
        logging.debug(counts2)
        plt.rcParams['font.weight']='bold'
        plt.rcParams['axes.labelweight']='bold'
        markerline1,stemline1,baseline1=plt.stem(bins1[:-1],counts1,label='semantically similar pairs',markerfmt='D',use_line_collection=True)
        markerline2,stemline2,baseline2=plt.stem(bins2[:-1],counts2,label='dissimilar image pairs',use_line_collection=True)
        markerline3,stemline3,baseline3=plt.stem(bins3[:-1],counts3,label='tampered image pairs',use_line_collection=True)
        plt.setp(stemline1,linestyle='-',color='blue')
        plt.setp(stemline2,color='green')
        plt.setp(baseline1,visible=True)
        plt.setp(baseline2,visible=True)
        plt.setp(baseline3,visible=True)
        plt.setp(stemline3,color='red')
        plt.setp(markerline3,color='red')
        plt.setp(markerline2,color='green')
        markerline1.set_markerfacecolor('none')
        markerline2.set_markerfacecolor('none')
        plt.ylim(0,min(max(counts1),max(counts2),max(counts3)))
        plt.xlabel('Hash Correlation coefficient')
        plt.ylabel('Frequency * {}'.format(min(len(same_corrleation_coefficients),len(different_correlation_coefficients),len(tampered_correlation_coefficients))))
        
        plt.plot([threshold]*(15),range(0,15),'y',label='Threshold line')
        #plt.legend()
        #plt.title('Hash Robustness and discernibility performance')
        if show:
            plt.show()
            sys.exit()
        plt.savefig(os.path.join(results_dir,'Distribution.jpg'))

    return same_corrleation_coefficients,different_correlation_coefficients,tpr_rate,fpr_rate
def modelTest3(results_dir,threshold_start=0,threshold_end=0.9999,threshold_step=0.0001,write=False):
    tpr_value=[]
    fpr_value=[]
    print('currently doint modelTest3')
    while threshold_start<=threshold_end:
        print('The threshold is set at {}'.format(threshold_start))
        _,_,_tpr_value,_fpr_value=modelTest2(results_dir,threshold_start)
        tpr_value.append(_tpr_value)
        fpr_value.append(_fpr_value)
        threshold_start+=threshold_step
    if write:
        print(tpr_value)
        print(fpr_value)
        return 
    plt.plot(fpr_value,tpr_value,'xb-') # x markers and blue lines
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('TPR vs FPR rates for different thresholds')
    plt.savefig(os.path.join(results_dir,'Diff_threshold.jpg'))

def writeResults(original_dir,tampered_dir,model_dir,results_dir,different_dir):
    '''
    this is for writing the results to the text file
    '''
    # same_test=ModelTest(original_dir,tampered_dir,model_dir,0.98,results_dir)
    different_test=DiscernibiltyTest(different_dir,model_dir,0.98,results_dir=results_dir)
    # operations=['brightness','compression','contrast','gamma','gaussian','rotation','salt and pepper','scaling','speckle','watermark']
    # for operation in operations[8:]:
    #     _,_,_=same_test(operation,write=True)
    
    different_test(n_samples=100,write=True)

def modelTest5(results_dir,threshold=0.98):
    lines=[]
    for i in range(0,len(results_dir),3):
        with open(os.path.join(results_dir[i],'tampred_hash_correlation.txt'),'r') as fp:
            for line in fp.readlines():
                try:
                    lines.append(float(line.split(' ')[0]))
                except:
                    continue
            assert(len(lines)>0)
        assert(len(lines)>0)

    plt.scatter(range(len(lines)),lines,c='blue',label='Hash correlation')

    plt.plot(range(len(lines)),[threshold]*len(lines),label='Threshold_line')
    plt.xlabel('tampered pair number')
    plt.ylabel('Hash correlation')
    plt.title('Hash correlation vs tampered pair')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(os.getcwd(),'Tampered_scatter.jpg'))

def modelTest6(results_dir,threshold=0.98):
    with open(os.path.join(results_dir,'same_correlation')) as fp:
        lines=[float(line) for line in fp.readlines()]
    plt.scatter(range(len(lines)),lines,c='red',label='Hash correlation')
    plt.plot(range(len(lines)),[threshold]*len(lines),label='Threshold line')
    plt.ylabel('Hash correlation ')
    plt.xlabel('Similar image pairs')
    plt.title('Hash correlation vs similar image pairs')
    plt.legend()
    plt.savefig(os.path.join(results_dir,'similar_pairs.jpg'))
def comparisonTest(original_dir,tampered_dir,model_dir,threshold,results_dir):
    operations=['brightness','compression','contrast','gamma','gaussian','rotation','salt and pepper','scaling','speckle','watermark']
    test=ComparisonTest(original_dir,tampered_dir,model_dir,threshold,results_dir)
    for operation in operations:
        test(operation_name=operation,folder_name='operations_indonesia',image_name=['Image01','Image02','Image03','Image04','Image05'],plot=True)



if __name__=='__main__':
    '''
    These directories are for testing the model for the graphs
    modelTest1 is just for computing the tpr and the fpr scores
    modelTest2 is for plotting the distirbution of similarity and dissimilarity 
    modelTest3 is for plotting the distribution of tpr and fpr scores for different thresholds
    '''
    original_dir=[ '/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_indonesia' , '/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_japan']
    tampered_dir=['/media/arnab/E0C2EDF9C2EDD3B6/lena/test/indonesia','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/japan']
    model_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/8_bilinear_v6_128.h5'
    results_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/ModelTest'
    different_dir='/media/arnab/E0C2EDF9C2EDD3B6/different/different_cv'





    logging.basicConfig(level=logging.DEBUG)
    logging.disable(logging.DEBUG)
    
    # discernibilityTest(results_dir,plot=True,threshold=0.98)
    # modelTest1(original_dir,tampered_dir,model_dir,results_dir)
    #rotationTest(original_dir,tampered_dir,model_dir,0.98,results_dir,correction=False)
    #modelTest2(results_dir,threshold=0.98,plot=True,show=True)
    # modelTest3(results_dir,write=True)
    # writeResults(original_dir,tampered_dir,model_dir,results_dir,different_dir)
    # modelTest6(results_dir,0.98)
    #comparisonTest(original_dir,tampered_dir,model_dir,0.98,results_dir)
    



    original_dir=[ '/media/arnab/E0C2EDF9C2EDD3B6/large tampered/original_cv' , '/media/arnab/E0C2EDF9C2EDD3B6/medium tampered/original_cv','/media/arnab/E0C2EDF9C2EDD3B6/small tampered/original_cv']
    tampered_dir=['/media/arnab/E0C2EDF9C2EDD3B6/large tampered/tampered_cv','/media/arnab/E0C2EDF9C2EDD3B6/medium tampered/tampered_cv','/media/arnab/E0C2EDF9C2EDD3B6/small tampered/tampered_cv']
    results_dir=['/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/large tampered','/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/medium tampered','/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/small tampered']
    model_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/8_bilinear_v6_128.h5'
    #localTest(original_dir,tampered_dir,model_dir,results_dir,image_name=['342.jpg'],custom_results_dir='/home/arnab/Downloads')   
    localTest(original_dir,tampered_dir,model_dir,results_dir,tamper_percent=True)
    #modelTest5(results_dir,0.98)

    # results_dir=['/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/rotated large tampered','/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/rotated medium tampered','/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered/rotated small tampered']
    # customTest(original_dir,tampered_dir,model_dir,results_dir,threshold=0.98)    
 


