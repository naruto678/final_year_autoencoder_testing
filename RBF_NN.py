import matplotlib.pyplot as plt

from matplotlib import colors
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import time


def make_data(num_samples=100,n_features=10,n_centres=10):
    return make_blobs(n_samples=num_samples,centers=n_centres,n_features=n_features)


def plot_data(x_train,y_train,cluster_centres=None):
    num_classes=np.unique(y_train)
    #color_dict={0:'r',1:'k',2:'b',3:'y',4:'g',5:'c',6:'r+',7:'ro',8:'ko',9:'k+',10:'bo'}
    color_dict=colors.CSS4_COLORS
    color_dict_keys=list(color_dict.keys())
    if len(num_classes) > len(color_dict_keys):
        raise Exception('Too many classes .The plot is fucked')
    for i in range(len(num_classes)):
        _x=x_train[y_train==i] # picking all the values that belong to this class
        plt.scatter(_x[:,0],_x[:,1],color=color_dict[color_dict_keys[i]],label=color_dict_keys[i])
        if len(cluster_centres)!=0:
            plt.scatter(cluster_centres[i,0],cluster_centres[i,1],color='k')
    plt.legend()
    
    plt.show()



class NN:
    def __init__(self,kmeans,x_train,y_train,center_label):
        '''
        center-label is the name of the class
        first pick only those which belong to the same centre
        after picking find the sigma 
        to find the sigma we need the co-cordinates of the center found using k means
        kmeans  returns the centroid,label,inertia,_

        '''
        self.flag=-1
        self.x_train=x_train
        self.y_train=np.zeros(len(y_train))
        self.y_train[y_train==center_label]=1
        self.center_label=center_label

        self.x=x_train[y_train==center_label]
        self.centroid_cordinates=kmeans.cluster_centers_[center_label]
        self.euclidean=lambda x,y:np.sqrt(np.sum(np.square(x-y),axis=1))
        self.sigma=(1/len(self.x))*np.sum(self.euclidean(self.x,self.centroid_cordinates))
        self.beta=-1/(2*(self.sigma**2))
        if self.sigma.shape!=():
            print('The shape came out to be {}'.format(self.sigma.shape))
            raise Exception('You fucked up cunty')
        self.weight=np.random.randn(1)
        
        
        self.phi=lambda x,y:np.exp((self.euclidean(x,y)**2)*self.beta)
        
    def __repr__(self):

        #str1='Center-Cordinates->{}'.format(self.centroid_cordinates)+'\n'+'Sigma->{}'.format(self.sigma) \
        #+'\n'+'Weights->{}'.format(self.weight)
        #return str1
        attributes=[attrib for attrib in dir(self) if  not (attrib in ['x_train','x','y_train','update','phi','output'] or attrib.startswith('__'))]
        
        str1=''
        for attrib in attributes:
            str1=str1+attrib+'->'+str(self.__dict__[attrib])+'\n'
        return str1

    def update(self,learning_rate,arr_loss):
        '''
        this is the fuction that will update the weights of the variable
        it will update sigma and the weight and also the center co-ordinates

        '''

        #print(self.phi(self.x_train,self.centroid_cordinates).shape)
        if self.flag==-1:
            self.prev_weight=self.weight
            self.weight=self.weight+learning_rate*np.sum(arr_loss)
            
            self.flag=1
        if self.flag==1:


            self.weight=self.weight+learning_rate*np.sum(arr_loss*self.phi(self.x_train,self.centroid_cordinates))
        self.prev_sigma=self.sigma
        self.sigma=self.sigma+learning_rate*np.sum((arr_loss*self.prev_weight*self.phi(self.x_train,self.centroid_cordinates)*np.sum((self.x_train-self.centroid_cordinates)**2,axis=1)*self.prev_sigma**-3))
        
        self.centroid_cordinates=self.centroid_cordinates+learning_rate*np.sum(arr_loss*self.prev_weight*self.phi(self.x_train,self.centroid_cordinates)*np.sum((self.x_train-self.centroid_cordinates),axis=1)*self.prev_sigma**-2)        
        


    def output(self):
        '''
        we apply gradient descent here
        first compute the loss then find the gradients of the loss with respect to the weights,sigma and the centr
        
        '''

        x1=self.weight*self.phi(self.x_train,self.centroid_cordinates)
        

        return x1
    

def computeLoss(nn1,nn2,y_train):
    output=nn1.output()+nn2.output()
    arr_loss=abs((y_train-output))
    loss=arr_loss**2
    return (arr_loss),1/2*np.sum((loss)),output

def test(nn1,nn2):
	x_test,y_test=make_data(100,2,2)
	y_pred=(nn1.output()+nn2.output())/2
	
	return y_pred,y_test
    
if __name__=='__main__':
    x_train,y_train=make_data(100,2,2)
    kmeans=KMeans(n_clusters=2,random_state=0).fit(x_train)
    #plot_data(x_train,y_train,kmeans.cluster_centers_)
    nn1=NN(kmeans,x_train,y_train,0)
    nn2=NN(kmeans,x_train,y_train,1)
    learning_rate=0.4
    print(nn1)
    arr_loss,loss,output=computeLoss(nn1,nn2,y_train)
    count=0
    max_iteration=100
    prev_loss=loss
    while(loss>10 and count<max_iteration):

        nn1.update(learning_rate,arr_loss)
        nn2.update(learning_rate,arr_loss)
        arr_loss,loss,output=computeLoss(nn1,nn2,y_train)

        count+=1
        if prev_loss==loss:
            learning_rate=learning_rate/10
        
        print('The loss after {} iterations is {}'.format(count,loss))
    y_pred,y_test=test(nn1,nn2)
    for i in range(len(y_pred)):

   		print(str(y_pred[i])+" "+str(y_test[i]))




        
        
        