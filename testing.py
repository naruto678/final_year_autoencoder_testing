#!/usr/bin/python
'''
@author:arnab
'''
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
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

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

	def concat_images(self,imga,imgb,imgc,imgd,debug=False,):
    
	    new_image=self.concat_images_util(imga,imgb,debug)
	    new_image=self.concat_images_util(new_image,imgc,debug)
	    return self.concat_images_util(new_image,imgd,debug)
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

	def f1_score(self,diff_map,original_diff_map):
		x1=np.argwhere(diff_map>0.05)
		x2=np.argwhere(original_diff_map>0)
		logging.debug(x1.shape)
		
		
		return len(x1)/len(x2)
	
	def __call__(self,v=False):
		'''
		This takes the original_image and the tampered image and saves the difference image
		'''

		_x=(os.path.join(self.tampered_dir,i) for i in os.listdir(self.tampered_dir))
		_y=(os.path.join(self.original_dir,i) for i in os.listdir(self.original_dir))
		print('Found {} images corresponding to this'.format(len(os.listdir(self.original_dir))))
		correlation_coefficients=[]
		f1_scores=[]
		for i in tqdm(range(len(os.listdir(self.original_dir)))):
			original_image_name=next(_x)

			tampered_image_name=next(_y)
			original_image=self.image(original_image_name)
			tampered_image=self.image(tampered_image_name)

			original_image_np=self.image_np(original_image)
			tampered_image_np=self.image_np(tampered_image)
			original_map=self.predict(original_image_np)[0,:,:,:]
			tampered_map=self.predict(tampered_image_np)[0,:,:,:]
			diff_map=original_map-tampered_map
			diff_map[diff_map<0]=-diff_map[diff_map<0]
			org_diff_map=original_image_np-tampered_image_np
			org_diff_map[org_diff_map<0]=-org_diff_map[org_diff_map<0]
			
			logging.debug(original_image_name)
			logging.debug(original_image_np.shape)
			logging.debug(tampered_image_np.shape)
			logging.debug(original_map.shape)
			logging.debug(tampered_map.shape)

			'''
			Where there is difference show black 
			rest show the entire image
			
			'''
			 
			result_img=np.zeros(original_map.shape)
			
			
			result_img[diff_map>=0.05]=tampered_image_np[diff_map>=0.05] # this is the tampered part
			result_img[diff_map<0.1]=0 # this is not the tampered part

			final_image=self.concat_images(original_image_np,tampered_image_np,org_diff_map,result_img)
			final_image=final_image.astype(np.uint8)
			if v:
				Image.fromarray(final_image).show()
				exit()

			self.save_PIL(Image.fromarray(final_image),original_image_name[original_image_name.rfind('/')+1:])
			f1=self.f1_score(diff_map,org_diff_map)
			original_hash=self.find_output(original_image_np)
			tampered_hash=self.find_output(tampered_image_np)
			correlation=np.corrcoef(original_hash,tampered_hash)[0][1]
			self.classifier(correlation)
			correlation_coefficients.append(correlation)
			f1_scores.append(f1)
			original_image.close()
			tampered_image.close()
			
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
		for i in tqdm(__x):
			original_image=next(x_test)
			tampered_image=next(y_test)
			original_hash=self.find_output(self.image_np(original_image))
			tampered_hash=self.find_output(self.image_np(tampered_image))
			c=np.corrcoef(original_hash,tampered_hash)
			correlation_coefficients.append(c[0][1])
			original_image.close()
			tampered_image.close()
			gc.collect()

		#logging.debug('The correlation coefficient for this operation is {}'.format(correlation_coefficients))
		return correlation_coefficients





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
	
	def __call__(self,operation)->list:
		self.indexes=[i for i,image_name in enumerate(self.x_train) if operation in image_name]
		self.indexes=self.indexes[:len(self.indexes)//2]
		logging.debug('Found {} number of images corresponding to this {}'.format(len(self.indexes),operation))
		if len(self.indexes)==0:
			return None,None,None
		
		correlation=self.hash_correlation(operation)
		tpr_value=self.tpr(correlation)
		fpr_value=self.fpr(correlation)
		
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
		self.total_dict={}
		self.detect_dict={}

	def hash_correlation(self,tampered_image_name:str,original_image_name:str)->None:
		'''
		gets the original iamge and the tampered image name and sends the correlation coefficient 
		'''

		original_image=self.image(original_image_name)
		tampered_image=self.image(tampered_image_name)
		original_hash=self.find_output(self.image_np(original_image))
		tampered_hash=self.find_output(self.image_np(tampered_image))
		c=np.corrcoef(original_hash,tampered_hash)[0][1]
		original_image.close()
		tampered_image.close()
		gc.collect()
		rotation_degree=original_image_name[original_image_name.rfind('ROT_')+1:-4]
		if rotation_degree not in self.total_dict:
			self.total_dict[rotation_degree]=1
		else:
			self.total_dict[rotation_degree]+=1
		if c >0.98:
			if rotation_degree not in self.detect_dict:
				self.detect_dict[rotation_degree]=1
			else:
				self.detect_dict[rotation_degree]+=1
		else:
			if rotation_degree not in self.detect_dict:
				self.detect_dict[rotation_degree]=0




	def __call__(self):
		for tampered_image,original_image in tqdm(self.image_pairs):
			self.hash_correlation(tampered_image,original_image)
		assert(len(self.total_dict)==len(self.detect_dict))
		self.detect_dict={a:self.detect_dict[a]/b for (a,b) in self.total_dict.items()}
		return self.detect_dict


def modelTest(original_dir,tampered_dir,model_dir,results_dir):

	print('Model Testing Phase')
	operations=['brightness','compression','contrast','gamma','gaussian','rotation','salt and pepper','scaling','speckle','watermark'][:2]
	for i in range(len(original_dir)):
		org_dir,tamp_dir=original_dir[i],tampered_dir[i]

		test=ModelTest([org_dir],[tamp_dir],model_dir,0.98,results_dir)
		visualize=Visualize(results_dir,org_dir[org_dir.rfind('_')+1:],operations)
		print('Currently doing {} folder'.format(org_dir[org_dir.rfind('/')+1:]))
		for operation in operations:

			correlation,tpr_value,fpr_value=test(operation)
			if correlation is None:
				continue
			print('Now plotting the results and saving them')
			visualize(correlation,tpr_value,fpr_value,operation)

		with open(os.path.join(results_dir,org_dir[org_dir.rfind('_')+1:]),'a') as f:
			for (a,b),c in visualize.dictionary.items():
				f.write('{} {} {}'.format(a,b,c))
		print('Saved the results in the file')
def localTest(oiginal_dir,tampered_dir,model_dir,results_dir):
	l1={0:'large tampered',1:'medium tampered',2:"small tampered"}
	print('Doing the localization test')
	for i in range(len(original_dir)):
		org_dir,tamp_dir=original_dir[i],tampered_dir[i]
		test=LocalizationTest(org_dir,tamp_dir,results_dir,model_dir,0.98)
		correlation,f1_scores,false_counter=test()		 
		visualize=Visualize(results_dir,org_dir)
		#visualize(correlation,0,false_counter,'tampering {} '.format(i))
		visualize(f1_scores,0,0,'f1_scores {}'.format(l1[i]),instance='something')
		with open(os.path.join(results_dir,'tampering_text_{}'.format(i)),'a') as f:
			 
			f.write('The f1 scores are: '+str(f1_scores))
			f.write('The false counter is '+str(false_counter))
def rotationTest(original_dir,tampered_dir,mdoel_dir,threshold,results_dir):
	rotation=RotationTest(original_dir,tampered_dir,model_dir,threshold,results_dir)
	d1=rotation()
	print(d1)
	
if __name__=='__main__':
	'''
	These directories are for testing the model for the graphs
	'''
	original_dir=[ '/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_indonesia' , '/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/operations_japan']
	tampered_dir=['/media/arnab/E0C2EDF9C2EDD3B6/lena/test/indonesia','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/italy','/media/arnab/E0C2EDF9C2EDD3B6/lena/test/japan']
	model_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/8_bilinear_v5.h5'
	results_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/ModelTest'
	





	logging.basicConfig(level=logging.DEBUG)
	logging.disable(logging.CRITICAL)
	# modelTest(original_dir,tampered_dir,model_dir,results_dir)
	rotationTest(original_dir,tampered_dir,model_dir,0.98,results_dir)
	# original_dir=[ '/media/arnab/E0C2EDF9C2EDD3B6/large tampered/original_cv' , '/media/arnab/E0C2EDF9C2EDD3B6/medium tampered/original_cv','/media/arnab/E0C2EDF9C2EDD3B6/small tampered/original_cv']
	# tampered_dir=['/media/arnab/E0C2EDF9C2EDD3B6/large tampered/tampered_cv','/media/arnab/E0C2EDF9C2EDD3B6/medium tampered/tampered_cv','/media/arnab/E0C2EDF9C2EDD3B6/small tampered/tampered_cv']
	# results_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/Results/Tampered'
	# localTest(original_dir,tampered_dir,model_dir,results_dir)	


