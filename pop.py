# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from keras import Model,Input,layers,optimizers
import numpy as np
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 04:27:03 2018

@author: arnabr
"""





from keras import Model,Input,layers,optimizers

input_img=Input(shape=(512,512,1))
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(input_img)
x=layers.BatchNormalization()(x);
x=layers.MaxPool2D((2,2),padding='same')(x)
x=layers.Dropout(0.3)(x)
#256
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
x=layers.BatchNormalization()(x);
x=layers.MaxPool2D((2,2),padding='same')(x)
x=layers.Dropout(0.3)(x)
#128
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
x=layers.BatchNormalization()(x);
x=layers.MaxPool2D((2,2),padding='same')(x)
x=layers.Dropout(0.3)(x)
#64
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
x=layers.BatchNormalization()(x);
x=layers.MaxPool2D((2,2),padding='same')(x)
x=layers.Dropout(0.3)(x)
#32
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
x=layers.BatchNormalization()(x);
x=layers.MaxPool2D((2,2),padding='same')(x)
x=layers.Dropout(0.3)(x)
#16

# x=layers.Dense(16*16*16,activation='relu')(x)
# x=layers.Dropout(0.3)(x)
# x=layers.Dense(16*16*16,activation='relu')(x)
# x=layers.Dense(8*8*16,activation='relu')(x)
# encoder_output=layers.Reshape((8,8,16))(x)
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
x=layers.BatchNormalization()(x);
x=layers.MaxPool2D((2,2),padding='same')(x)
x=layers.Dropout(0.3)(x)
#8
# x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
# encoder_output=layers.MaxPool2D((2,2),padding='same')(x)


# # now the decoder part



# #decoder_input=Input(shape=(4,4,16))
# x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(encoder_output)
# x1=layers.UpSampling2D((2,2))(x1)


x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x)
x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x)
x1=layers.Dropout(0.3)(x1)
#16


x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
x1=layers.Dropout(0.3)(x1)
#32

x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
x1=layers.Dropout(0.3)(x1)
#64

x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
x1=layers.Dropout(0.3)(x1)
#128
x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
x1=layers.Conv2D(16,(3,3),activation='relu',padding='same')(x1)
x1=layers.Dropout(0.3)(x1)
#256

x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
x1=layers.Conv2D(1,(3,3),activation='relu',padding='same')(x1)
x1=layers.Dropout(0.3)(x1)
#512
x1=layers.UpSampling2D((2,2),interpolation='bilinear')(x1)
decoder_output=layers.Conv2D(1,(3,3),activation='relu',padding='same')(x1)




#512
#encoder_model=Model(input_img,encoded_output)
#decoder_model=Model(decoder_input,decoded_output) 
final_model=Model(input_img,decoder_output)
#final_model.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['acc'])
final_model.compile(loss='mean_squared_error',optimizer=optimizers.Adagrad(lr=1e-4),metrics=['acc'])
final_model.summary()



from sklearn.model_selection import train_test_split
import glob
from PIL import Image
import os
import numpy as np
import re
import gc
x_dir=['../input/aerials/aerials/operations_aerials','../input/operations_animals/operations_animals','../input/operations_scenery/operations_scenery']
y_dir=['../input/aerials/aerials/modified_aerials','../input/animals_bmp/animals_bmp','../input/scenery_bmp/scenery_bmp']
#print(os.listdir(x_train_dir));
#print(os.listdir(y_train_dir));
#x_train_dir='aerials/operations_aerials'
#y_train_dir=''


def find_file_name(image):
    image_name=re.sub(r"(.npy)|(.tiff)|(.jpg)|(.bmp)|(.tif)","",image)     
    return image_name

def make_data_util(image,x_train_dir,y_train_dir,instance=None):
    #y_images=os.listdir(y_train_dir)
    #x_images=os.listdir(x_train_dir)
    
    x_train=[]
    y_train=[]
    
    image_name=find_file_name(image)
    # print(image_name)
    # print(x_train_dir)
    # print(y_train_dir)
    operations_image=glob.glob(x_train_dir+'/*/'+image_name+'/*')
    assert(isinstance(operations_image,list))
    #operations_image=operations_image[:len(operations_image)/2]
    print('{} numberof images have been found'.format(len(operations_image)))
    if instance==None:
        # print('first instance is running')
        for images in operations_image[:len(operations_image)//2]:
            x_image=np.asarray((Image.open(images)).resize((512,512)))[:,:,0];
            if  not image.endswith('.npy'):
                y_image=np.asarray((Image.open(images)).resize((512,512)))[:,:,0]
            
            else: y_image=np.load(y_train_dir+'/'+image)
            x_train.append(x_image)
            y_train.append(y_image)
    
    else:
        #assert(instance=='second')
        #print('second instance is running')
        for images in operations_image[len(operations_image)//2:]:
            x_image=np.asarray((Image.open(images)).resize((512,512)))[:,:,0];
            if  not image.endswith('.npy'):
                y_image=np.asarray((Image.open(images)).resize((512,512)))[:,:,0]
            
            else: y_image=np.load(y_train_dir+'/'+image)
            x_train.append(x_image)
            y_train.append(y_image)
        
        
        
         
            
    x_train=np.array(x_train)
    
    
    y_train=np.array(y_train)
    
     
    return x_train,y_train

#x_train,y_train=make_data('2.1.01.tiff.npy',x_train_dir[0],y_train_dir[0])
#x_train_1,y_train_1=make_data('2.1.02.tiff.npy')


def make_data(image,x_train_dir,y_train_dir,instance=None):
    x_train=np.zeros((1,512,512))
    y_train=np.zeros((1,512,512))
    
    _x1,_y1=make_data_util(image,x_train_dir,y_train_dir,instance)
        
    x_train=np.concatenate((x_train,_x1))
    y_train=np.concatenate((y_train,_y1))
        
        
    return np.array(x_train)[1:,:,:],np.array(y_train)[1:,:,:]
        
 
 
def preprocess(image,x_train_dir,y_train_dir,instance=None):
     
    
    x_train,y_train=make_data(image,x_train_dir,y_train_dir,instance)
    x_train=x_train.reshape((-1,512,512,1))/255
    y_train=y_train.reshape((-1,512,512,1))/255
    return x_train,y_train

def randomize(x_train,y_train):
    assert(len(x_train)==len(y_train))
    instances=np.arange(len(x_train))
    np.random.shuffle(instances)
    _x1=[]
    _y1=[]
    _x1[np.arange(len(x_train))]=x_train[instances]
    _y1[np.arange(len(y_train))]=y_train[instances]
    del x_train,y_train
    return np.array(_x1),np.array(_y1)
    
def train(iterator,x_train_dir,y_train_dir):
	count1=0
	count=0
	prev_iter=next(iterator)
	while True:
	    number=len(os.listdir(y_train_dir))
	    #count1=0
	    
	    try:
	        if count%2==0:
	            
	            
	            x2_train,y2_train=preprocess(prev_iter,x_train_dir,y_train_dir)
	           
	            count=count+1
	            
	        else :
	             
	            x2_train,y2_train=preprocess(prev_iter,x_train_dir,y_train_dir,instance='second')
	            prev_iter=next(iterator)
	            count=count+1
	            count1+=1    
	        
	        final_model.fit(x2_train,y2_train,epochs=5,validation_split=0.2)
	        del x2_train,y2_train
	        print('collecting garbage')
	        gc.collect()
	       
	        print('Finished doing {}/{} image '.format(count1,number))                
	        
	            
	        
	    except StopIteration:
	        print('Thank you cunty for fucking up')
	        #final_model.save('8_bilinear_v2_new.h5')

	        break

if __name__=='__main__':

	for i in range(len(y_dir)):
	    iterator=iter(os.listdir(y_dir[i]))
	    
	    print('Currently doing {} folder'.format(y_dir[i][y_dir[i].rfind('/')+1:]))
	    #print(x_dir[i],y_dir[i])
	    train(iterator,x_dir[i],y_dir[i])
	    gc.collect()
	
	final_model.save('8_bilinear_v3.h5')



 
