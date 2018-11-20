import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import numpy as np



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
        self.x_train=x_train
        self.y_train=np.zeros(len(y_train))
        self.y_train[y_train==center_label]=1
        self.center_label=center_label

        self.x=x_train[y_train==center_label]
        self.centroid_cordinates=kmeans.cluster_centers_[center_label]
        self.sigma=1/len(self.x)*np.sum(np.sqrt(np.sum(np.square((self.x-self.centroid_cordinates)),axis=1)))
        if self.sigma.shape!=():
            print('The shape came out to be {}'.format(self.sigma.shape))
            raise Exception('You fucked up cunty')
        self.weight=np.random.randn(1)
    def __repr__(self):

        #str1='Center-Cordinates->{}'.format(self.centroid_cordinates)+'\n'+'Sigma->{}'.format(self.sigma) \
        #+'\n'+'Weights->{}'.format(self.weight)
        #return str1
        attributes=[attrib for attrib in dir(self) if  not attrib.startswith('__')]
        return str(self.__dict__)

    def update(self):
        '''
        this is the fuction that will update the weights of the variable
        it will update sigma and the weight
        '''
    def train(self):
        '''
        we apply gradient descent here

        
        '''
        pass

    

    
if __name__=='__main__':
    x_train,y_train=make_data(100,2,4)
    kmeans=KMeans(n_clusters=4,random_state=0).fit(x_train)
    #plot_data(x_train,y_train,kmeans.cluster_centers_)
    nn1=NN(kmeans,x_train,y_train,0)
    nn2=NN(kmeans,x_train,y_train,1)
    nn3=NN(kmeans,x_train,y_train,2)
    nn4=NN(kmeans,x_train,y_train,3)
    print(nn1)