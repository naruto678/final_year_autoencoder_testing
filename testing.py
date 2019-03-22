


from keras.models import load_model
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import glob
import re
from pop import find_file_name,Data
from functools import partial
import keras.backend as K
import gc
import pandas as pd
import matplotlib.pyplot as plt
class LocalizationTest:
	'''
	The main purpose of this class is to get the original_dir and the test_dir and then save the results in the results_dir
	This class should contain all the tests and any extra test should be added here in the future and the correspondig results 
	will be written in the results directory
	'''
	def __init__(self,original_dir,tampered_dir,results_dir,model_dir):
		'''
		Do not hardcode anything
		'''
		self.original_dir=original_dir
		self.tampered_dir=tampered_dir
		self.results_dir=results_dir
		self.model_dir=model_dir
		self.model=load_model(model_dir)
		self.input_shape=self.model.layers[0].get_config()['batch_input_shape'][1:-1]
		self.image=lambda image_x:(np.asarray(Image.open(image_x).resize(self.input_shape))) # returns a numpy array of shape 128X128X3
		self.predict=lambda image_x:self.model.predict(self.image(image_x)[None,:,:,:]/255) # model needs 1X128X128X3
				

	

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
	        #print(imga.shape)
	        
	    if imgb.ndim!=3:
	        imgb=imgb[:,:,None]
	        #print(imgb.shape)
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

	
	            print('This is the image that should have been saved')
	            #time.sleep(10)

	    return new_img

	def concat_images(self,imga,imgb,imgc,imgd,debug=False,):
    
	    new_image=self.concat_images_util(imga,imgb,debug)
	    new_image=self.concat_images_util(new_image,imgc,debug)
	    return self.concat_images_util(new_image,imgd,debug)
	def save_PIL(self,image_file,image_name):
	    image_file=image_file.convert('RGB')
	    image_file.save(self.results_dir+image_name[:-4]+'.jpg')

	
	def __call__(self,original_image_name,tampered_image_name,v=False):
		'''
		This takes the original_image and the tampered image and saves the difference image
		'''
		original_image=self.image(os.path.join(self.original_dir,original_image_name))
		tampered_image=self.image(os.path.join(self.tampered_dir,tampered_image_name))
		original_map=self.predict(os.path.join(self.original_dir,original_image_name))[0,:,:,:]
		tampered_map=self.predict(os.path.join(self.tampered_dir,tampered_image_name))[0,:,:,:]
		diff_map=(original_map-tampered_map)
		diff_map[diff_map<0]=-diff_map[diff_map<0]
		org_diff_map=original_image-tampered_image
		org_diff_map[org_diff_map<0]=-org_diff_map[org_diff_map<0]

		'''
		Where there is difference show black 
		rest show the entire image
		
		'''
		 
		result_img=np.zeros(original_map.shape)
		
		
		result_img[diff_map>0.05]=0 # this is the tampered part
		result_img[diff_map<=0.1]=255 # this is not the tampered part

		final_image=self.concat_images(original_image,tampered_image,org_diff_map,result_img)
		final_image=final_image.astype('uint8')
		if v:
			Image.fromarray(final_image).show()
		self.save_PIL(Image.fromarray(final_image),original_image_name)
		return Image.fromarray(final_image)
class ModelTest(Data):
	def __init__(self,x_dir:list,y_dir:list,model_dir,threshold:int,results_dir:str):
		'''
			The class inherits from the data class we made in the training code			
			x_train contains all the operations_images and y_train contains all the 
			original_images

			this class will evaluate the model performance on a dataset 
			The initialier should give a threshold using which the class will evaluate the tpr and the fpr values
			The tpr and the fpr values should be in a table format shown below for which we will use pandas library

			dataset 	operations 		tpr 	fpr mean_f1_score mean_hash_correlaton
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
		self.image=lambda image_name:np.asarray(Image.open(image_name).resize(self.input_shape))
		self.predict=lambda image_name:self.model.predict(self.image(image_name)[None,:,:,:]/255) 
		

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
		assert(len(self.__x)==len(self.__y))
		x_test=(self.image(image_name) for image_name in __x)
		y_test=(self.image(image_name) for image_name in __y)

		correlation_coefficients=[]
		for i in tqdm(self.__x):
			original_hash=self.find_output(next(x_test))
			tampered_hash=self.find_output(next(y_test))
			c=np.corrcoef(original_hash,tampered_hash)
			correlation_coefficients.append(c[0][1])


		#print('The correlation coefficient for this operation is {}'.format(correlation_coefficients))
		return correlation_coefficients





	def tpr(self,correlation_coefficients):
		'''
		returns the true positive  score for the image_data
		'''
		tpr=0
		return len(filter(lambda x:x>self.threshold,correlation_coefficients))/len(correlation_coefficients)


		
	def fpr(self,correlation_coefficients):
		'''
			returns the false positive score rate
		'''
		return 1-tpr(correlation_coefficients)
	
	def __call__(self,operation):
		self.indexes=[i for i,image_name in enumerate(self.x_train) if operation in image_name]
		print('Found {} number of images corresponding to this {}'.format(len(self.indexes),operation))
		if len(self.indexes)==0:
			return None
		
		correlation=self.hash_correlation(operation)
		tpr_value=self.tpr(correlation)
		fpr_value=self.fpr(correlation)
		
		return sum(correlation)/len(correlation),tpr_value,fpr_value



	
if __name__=='__main__':
	original_dir=['/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_indonesia','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_japan']
	tampered_dir=['/media/arnab/E0C2EDF9C2EDD3B6/lena/test/indonesia','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/japan']
	model_dir=['/media/arnab/E0C2EDF9C2EDD3B6/final_year/8_bilinear_v5.h5']
	results_dir=['/media/arnab/E0C2EDF9C2EDD3B6/large tampered/final_results 8X8/']
	logging=logging.getLogger()
	operations=['brightness','compression','contrast','gamma','gaussian','rotation','salt and pepper','scaling','speckle','watermark']
	for i in range(len(original_dir)):
		
	for operation in operations:

		correlation,tpr_value,fpr_value,f1_score=test(operation)
		

		
		
		

