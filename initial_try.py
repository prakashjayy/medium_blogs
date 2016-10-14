import tensorflow as tf 
import numpy as np 
import pandas as pd #for creating and working on dataframes
import glob # for creating a list of filenames from a directory
from skimage import io
import sys
from sklearn.cross_validation import train_test_split

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Image Recognition Algorithm
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


data = data[:,:,:,0] #Taking only the first 3 layers and ignoring the 3rd layer
inputs_data = data.astype(np.float32)
labels_data = labels.as_matrix()

# Print input data and output data Size and shape 
print (data.size,data.shape)
print (labels_data.size,labels_data.shape)


# Divide the data into train,valid and test 

trX,valX,trY, valY = train_test_split(inputs_data,labels_data,
	test_size = 0.40, random_state = 20)


# Parameters 
learning_rate = 0.001 


# Configuration variables 
n_input = 28 # MNIST data input 
n_steps = 28 # timesteps 
n_hidden = 128 # hidden layer num of features 
n_classes = 10 # MNIST total number of classes 

w_oh = tf.Variable(tf.random_normal([n_hidden,n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# Prepare the data to match RNN function requirements 
def RNN(x,weights,biases): 
    x = tf.transpose(x,[1,0,2]) # Permuting batch_size and n_steps
    x = tf.reshape(x,[-1,n_input])
    x = tf.split(0,n_steps,x)
        

    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True)
    outputs,states = rnn.rnn(lstm_cell,x,dtype = tf.float32)

    return tf.add(tf.matmul(outputs[-1],weights),biases)


mnist_graph = tf.Graph()
with mnist_graph.as_default():
    X = tf.placeholder("float",[None,n_steps,n_input])
    Y = tf.placeholder("float",[None,10])
    
   
    w_oh = tf.Variable(tf.random_normal([n_hidden,10]),"float")
    b = tf.Variable(tf.random_normal([n_classes]),"float")

    with tf.name_scope("Model"):
        pred = RNN(X,w_oh,b)

    with tf.name_scope("Loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,Y))
    
    
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    with tf.name_scope("accuracy"):
    	predict_op = tf.argmax(pred,1)
    	correct_prediction = tf.equal(tf.argmax(Y,1),predict_op)
    	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



    trainop = tflearn.TrainOp(loss=cost,optimizer=optimizer,metric=accuracy,
    	batch_size=128)  #http://tflearn.org/helpers/trainer/ --> go to this link for more details


    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0,tensorboard_dir="/tmp/tflearn_logs/example1",
    	checkpoint_path = "model.lstm",
    	max_checkpoints = 2)

    trainer.fit({X: trX, Y: trY}, val_feed_dicts={X: valX, Y: valY},
                n_epoch=2, show_metric=True)

    #tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2



save("model_lstm_mnist")







