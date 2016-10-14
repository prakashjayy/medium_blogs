# Tensorflow -- TFlearn 

# Building VGGnet 11 layer 

import tflearn
from tflearn.layers.core import input_data,dropout, fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np 
import pandas as pd #for creating and working on dataframes
import glob # for creating a list of filenames from a directory
from skimage import io

filelist = glob.glob("/Users/Satish/Downloads/MNIST/Train/Images/train/*.png") #create a list of all filenames.
data= np.array([io.imread(filelist[i]) for i in np.arange(len(filelist))]) # Four Dimensional Array with [Batch,height,width,channels]

data = data[:,:,:,0:3] #Taking only the first 3 layers and ignoring the 3rd layer
data = data.astype(np.float32)

indexlist = list()
for i in range(len(filelist)):
    m = filelist[i]
    indexlist.append(m)
# Dependent Variable 
# Call the Input labels data 
labels = pd.read_table("/Users/Satish/Downloads/MNIST/Train/train.csv",sep=",",index_col = ["filename"])
# Only keep those datapoints which are in the above list 
labels = labels.ix[list(map(int, ([x.rsplit('/',1)[-1].rsplit('.',1)[0] for x in indexlist])))]

print (data.shape)

labels = pd.get_dummies(labels['label'])
print (labels.head()) # TO make sure everything worked 


data = data[:,:,:,0:3] #Taking only the first 3 layers and ignoring the 3rd layer
inputs_data = data.astype(np.float32)
labels_data = labels.as_matrix()



X,Y = inputs_data,labels_data
num_classes = 10


network = input_data(shape=[None,28,28,3])

network = conv_2d(network,64,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
network = max_pool_2d(network,kernel_size=2,strides=2)

network = conv_2d(network,128,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
network = max_pool_2d(network,kernel_size=2,strides=2)

network = conv_2d(network,256,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
network = conv_2d(network,256,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
network = max_pool_2d(network,kernel_size=2,strides=2)

network = conv_2d(network,512,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
network = conv_2d(network,512,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
network = max_pool_2d(network,kernel_size=2,strides=2)

#network = conv_2d(network,512,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
#network = conv_2d(network,512,3,weights_init="truncated_normal",bias_init="zeros",activation="relu")
#network = max_pool_2d(network,kernel_size=2,strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')




network = regression(network,optimizer = "adam",
	loss = "categorical_crossentropy",learning_rate=0.001,batch_size=64,restore = False) #http://tflearn.org/layers/estimator/

# learning 
model = tflearn.DNN(network,checkpoint_path="/tflearn/model.vgg/",
	max_checkpoints=1,tensorboard_verbose=0,tensorboard_dir="/tmp/tflearn_logs/")


model.fit(X, Y, n_epoch=1, shuffle=True,
          show_metric=True, batch_size=1000, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_mnist')

model.save("model.vgg.mnist")

