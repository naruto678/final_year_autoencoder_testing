from keras.models import load_model
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging





class Test:
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
				

	def f1(self,org_image,tamp_image):
		'''
		returns the f1_score
		'''
		pass
	def hash_correlation(self,org_image_map,temp_image_map):
		'''
		returns the hash_correlation_coefficien
		'''
		pass
	def tpr(self):
		'''
		returns the true positive  score for the image_data
		'''
		pass
	def fpr(self):
		'''
			returns the false positive score rate
		'''
		pass

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

	
if __name__=='__main__':
	original_dir='/media/arnab/E0C2EDF9C2EDD3B6/large tampered/original_cv'
	tampered_dir='/media/arnab/E0C2EDF9C2EDD3B6/large tampered/tampered_cv'
	model_dir='/media/arnab/E0C2EDF9C2EDD3B6/final_year/8_bilinear_v5.h5'
	results_dir='/media/arnab/E0C2EDF9C2EDD3B6/large tampered/final_results 8X8/'
	logging=logging.getLogger()

	test=Test(original_dir,tampered_dir,results_dir,model_dir)
	for image in tqdm(os.listdir(original_dir)):
		res_image=test(image,image,v=False)
		
		

		
		
		

