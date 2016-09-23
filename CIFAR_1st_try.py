import tensorflow as tf 
import numpy as np 
import pandas as pd #for creating and working on dataframes
import glob # for creating a list of filenames from a directory
from skimage import io
import sys

# Image Recognition Algorithm
print (sys.version)

data = pd.DataFrame()
filelist = glob.glob("/Users/PrakashR/Downloads/CIFAR/train/*.png") #create a list of all filenames.

x = np.random.randint(len(filelist), size=10000)
for i in range(len(x)): #run a loop to read each and every image. Do necessary transformation and convert
    #the n-dimensional array into a row. Append that row to the dataframe.
    img = io.imread(filelist[x[i]]) #read the image
    flattener = img.reshape(1,3072) #flatten the image C - 
    flattener = pd.DataFrame(data=flattener )
    data = data.append(flattener) #appened it to the dataframe
    if( i % 500 ==0):
        print (i)

indexlist = list()
for i in range(len(x)):
    m = filelist[x[i]]
    indexlist.append(m)

# Index all the tupules with relavent filenames 
data.index = [x.rsplit('/',1)[-1].rsplit('.',1)[0] for x in indexlist]


# Dependent Variable 
# Call the Input labels data 
labels = pd.read_table("/Users/PrakashR/Downloads/CIFAR/trainLabels.csv",sep=",",index_col = ["id"])
# Only keep those datapoints which are in the above list 
labels = labels.ix[list(map(int, ([x.rsplit('/',1)[-1].rsplit('.',1)[0] for x in indexlist])))]

print (data.head()) # TO make sure everything worked 

labels = pd.get_dummies(labels['label'])
print (labels.head()) # TO make sure everything worked 

inputs_data = data.as_matrix()
labels_data = labels.as_matrix()
print (inputs_data.size)
print (labels_data.shape)

trX = inputs_data
trY = labels_data




def init_weights(shape):
	return tf.Variable(tf.random_normal(shape,stddev=0.01))


def model(X,w_h,w_o):
	h = tf.nn.sigmoid(tf.matmul(X,w_h))
	return tf.matmul(h,w_o)


# Call the data 

X= tf.placeholder(tf.float32,[None,3072])
Y = tf.placeholder(tf.float32,[None,10])

w_h1 = init_weights([3072,1024])
w_h2 = init_weights([1024,32])
w_o = init_weights([32,10])

py_x1 = model (X,w_h1,w_h2)
py_x = model(py_x1,w_h2,w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,Y))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
predict_op = tf.argmax(py_x,1)


with tf.Session() as sess:
	tf.initialize_all_variables().run()

	n_iterations = 50
	batch_size = 500
	for i in range(n_iterations):
		idxs = np.random.permutation(range(len(trX)))
		n_batches = len(idxs)//batch_size
		for batch_i in range(n_batches):
			idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
			sess.run(train_op,feed_dict= {X:trX[idxs_i],Y:trY[idxs_i]})
		print (i, np.mean(np.argmax(trY,axis=1) == sess.run(predict_op,feed_dict={X:trX,Y:trY})))

	correct_prediction = tf.equal(tf.argmax(trY,1),tf.argmax(py_x,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print("Accuracy:",sess.run(accuracy,feed_dict={X:trX,Y:trY}))



