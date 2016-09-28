#Convolution Neural Network 

import tensorflow as tf 
import numpy as np 
import pandas as pd #for creating and working on dataframes
import glob # for creating a list of filenames from a directory
from skimage import io
import sys
from sklearn.cross_validation import train_test_split

# Image Recognition Algorithm
filelist = glob.glob("/Users/PrakashR/Downloads/CIFAR/train/*.png") #create a list of all filenames.
x = np.random.randint(len(filelist), size=10000)
data = np.array([io.imread(filelist[i]) for i in x]) # Four Dimensional Array with [Batch,height,width,channels]


indexlist = list()
for i in range(len(x)):
    m = filelist[x[i]]
    indexlist.append(m)

# Dependent Variable 
# Call the Input labels data 
labels = pd.read_table("/Users/PrakashR/Downloads/CIFAR/trainLabels.csv",sep=",",index_col = ["id"])
# Only keep those datapoints which are in the above list 
labels = labels.ix[list(map(int, ([x.rsplit('/',1)[-1].rsplit('.',1)[0] for x in indexlist])))]

print (data.shape)

labels = pd.get_dummies(labels['label'])
print (labels.head()) # TO make sure everything worked 

inputs_data = data.astype(float)
labels_data = labels.as_matrix()

# Print input data and output data Size and shape 
print (data.size,data.shape)
print (labels_data.size,labels_data.shape)


# Divide the data into train,valid and test 

trX,valX,trY, valY = train_test_split(inputs_data,labels_data,
	test_size = 0.40, random_state = 20)

valX,testX,valY,testY = train_test_split(valX,valY,
	test_size = 0.40, random_state = 20)


# The data is prepared. Lets begin the Algorithm.


def init_weights(shape):
	return tf.Variable(tf.random_normal(shape,stddev=0.01))


def model(X,w,w1,w2,w_o,p_keep_conv,p_keep_hidden):
	l1a = tf.nn.conv2d(X,w,strides=[1,1,1,1],padding="SAME")
	l1 = tf.nn.max_pool(l1a,ksize=[1,2,2,1],strides=[1,2,2,1],
		padding ="SAME")
	l1 = tf.nn.dropout(l1,p_keep_conv)

	l2a = tf.nn.conv2d(l1,w1,strides=[1,1,1,1],padding="SAME")
	l2  = tf.nn.max_pool(l2a,ksize=[1,3,3,1],strides=[1,3,3,1],
		padding = "SAME")
	Shape = l2.get_shape().as_list()
	l3 = tf.reshape(l2,[-1,Shape[1]*Shape[2]*Shape[3]])
	l3 = tf.nn.dropout(l3,p_keep_conv)

	l4 = tf.nn.relu(tf.matmul(l3,w2))
	l4 = tf.nn.dropout(l4,p_keep_hidden)

	pyx = tf.matmul(l4,w_o)
	return pyx


X = tf.placeholder(tf.float32,[None,32,32,3])
Y = tf.placeholder(tf.float32,[None,10])


valid_X = tf.Variable(valX,dtype=tf.float32)
valid_Y = tf.Variable(valY,dtype=tf.float32)

test_X = tf.Variable(testX,dtype=tf.float32)
test_Y = tf.Variable(testY,dtype=tf.float32)



w = init_weights([5,5,3,10])
w1 = init_weights([3,3,10,20])
w2 = init_weights([720,100])
w_o = init_weights([100,10])

p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

py_x = model(X,w,w1,w2,w_o,p_keep_conv,p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,Y))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(py_x,1)

validy_x = tf.argmax(tf.nn.softmax(model(valid_X,w,w1,w2,w_o,p_keep_conv,p_keep_hidden),name = "Valid"),1) # No need to keep probabilities here 
test_pyx = tf.nn.softmax(model(test_X,w,w1,w2,w_o,p_keep_conv,p_keep_hidden),name="test") # No need to keep the probabilities here. We will use the entire network.



with tf.Session() as sess:
	tf.initialize_all_variables().run()

	n_iterations = 10000
	batch_size = 500
	for i in range(n_iterations):
		idxs = np.random.permutation(range(len(trX)))
		n_batches = len(idxs)//batch_size
		for batch_i in range(n_batches):
			idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
			sess.run(train_op,feed_dict= {X:trX[idxs_i],Y:trY[idxs_i],p_keep_conv: 0.8, p_keep_hidden: 0.5})
		print (i,"Accuracy:",np.mean(np.argmax(trY,axis=1) == sess.run(predict_op,feed_dict={X:trX,Y:trY,p_keep_conv: 1.0, p_keep_hidden: 1.0})), 
			"Valid_Acc:",np.mean(np.argmax(valY,axis=1) == sess.run(validy_x,feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0})))

# Testing Accuarcy 
	correct_prediction = tf.equal(tf.argmax(test_Y,1),tf.argmax(sess.run(test_pyx,feed_dict={p_keep_conv: 1.0, p_keep_hidden: 1.0}),1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print("Accuracy:",sess.run(accuracy))



# On a 4 GB RAM, MAC OX Yosemite, The Algorithm is taking 20-30 min to start and there
# after it is taking 10 sec for each iteration.










