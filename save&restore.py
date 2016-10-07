import tensorflow as tf 
import numpy as np 
import pandas as pd #for creating and working on dataframes
import glob # for creating a list of filenames from a directory
from skimage import io
import sys
from sklearn.cross_validation import train_test_split

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


data = data[:,:,:,0:3] #Taking only the first 3 layers and ignoring the 3rd layer
inputs_data = data.astype(np.float32)
labels_data = labels.as_matrix()

# Print input data and output data Size and shape 
print (data.size,data.shape)
print (labels_data.size,labels_data.shape)


# Divide the data into train,valid and test 

trX,valX,trY, valY = train_test_split(inputs_data,labels_data,
	test_size = 0.40, random_state = 20)

valX,testX,valY,testY = train_test_split(valX,valY,
	test_size = 0.40, random_state = 20)

# Building 

#Architecture 
#Input - 28*28
#C1 - 6@28*28
#S2 - 6@14*14
#C3 - 16@10*10
#S4 - 16@5*5
#C5 - 120@1*1
#F6 - 84
#Output - 10

#1# 6 [5,5,3] conv - Valid Padding + bias and squasing fuction 
#2# 6 [2*2] pool - Same Padding + bias and squasing function 
#3# 16 [5,5,6] conv - valid padding + bias and squasing function 
#4# 16 [2*2] pool - Same Padding + bias and squasing function 
#5# 120 [5,5,16] conv - valid padding +bias and squasing function
#6# 84 [120,84] FC + bias and squasing function 
#7# 10 [84,10] FC + bias and squasing function

#weights 
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape,stddev=0.01),dtype=tf.float32)

def bias(shape):
	return tf.Variable(tf.zeros(shape,dtype=tf.float32))

def model(X,w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7):
    #Layer1 
    with tf.name_scope("conv1"):
    	l1a = tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding="VALID")
    	l1_op = tf.nn.relu(tf.nn.bias_add(l1a,b1))
	
	# Layer2
	with tf.name_scope("avg_pool1"):
		l2 = tf.nn.avg_pool(l1_op,ksize=[1,2,2,1],strides=[1,2,2,1],
		padding ="SAME")
		l2 = tf.mul(l2,tf.constant([4],dtype=tf.float32)) # since there is only average option and we would like to add all the four numbers and  multiply it with trainable co-efficient . We are adding a weight layer
		l2_op = tf.nn.relu(tf.nn.bias_add(l2,b2))
	
	# Layer3 
	with tf.name_scope("conv2"):
		l3a = tf.nn.conv2d(l2_op,w3,strides=[1,1,1,1],padding="VALID")
		l3_op = tf.nn.relu(tf.nn.bias_add(l3a,b3))
	
	# Layer4
	with tf.name_scope("avg_pool2"):
		l4 = tf.nn.avg_pool(l3_op,ksize=[1,2,2,1],strides=[1,2,2,1],
		padding ="SAME")
		l4 = tf.mul(l4,tf.constant([4],dtype=tf.float32)) # since there is only average option and we would like to add all the four numbers and multiply it with trainable co-efficient . We are adding a weight layer
		l4_op = tf.nn.relu(tf.nn.bias_add(l4,b4))
	
	# Layer5
	with tf.name_scope("conv3"):
		l5a = tf.nn.conv2d(l4_op,w5,strides=[1,1,1,1],padding="VALID")
		l5_op = tf.nn.relu(tf.nn.bias_add(l5a,b5))
		Shape = l5_op.get_shape().as_list()
		l5 = tf.reshape(l5_op,[-1,Shape[1]*Shape[2]*Shape[3]])
	
	# Layer 6 
	with tf.name_scope("FC1"):
		l6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l5,w6),b6))
	
	# Layer 7
	return tf.nn.bias_add(tf.matmul(l6,w7),b7) # We will add the activation(squasing function later)


# Calculate the loss functions 
mnist_graph = tf.Graph()
with mnist_graph.as_default():
    # Generate placeholders for the images and labels.
    w1 = init_weights([5,5,3,6]) # 6 features with size [5,5,3]
    b1 = bias([6])
    w2 = init_weights([6])
    b2 = bias([6])
    w3 = init_weights([5,5,6,16])
    b3 = bias([16])
    w4 = init_weights([16])
    b4 = bias([16])
    w5 = init_weights([4,4,16,120])
    b5 = bias([120])
    w6 = init_weights([120,84])
    b6 = bias([84])
    w7 = init_weights([84,10])
    b7 = bias([10])

    X = tf.placeholder(tf.float32,[None,28,28,3],name = "InputData")
    Y = tf.placeholder(tf.float32,[None,10],name = "InputLables")
    
    valid_X = tf.Variable(valX,dtype=tf.float32)
    valid_Y = tf.Variable(valY,dtype=tf.float32)
    
    test_X = tf.Variable(testX,dtype=tf.float32)
    test_Y = tf.Variable(testY,dtype=tf.float32)
    
    tf.add_to_collection("images", X)  # Remember this Op.
    tf.add_to_collection("labels", Y)  # Remember this Op.


    with tf.name_scope("Model"):
    	py_x = model(X,w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7)

    # Build a Graph that computes predictions from the inference model.
    tf.add_to_collection("py_x", py_x)  # Remember this Op.

    with tf.name_scope("Loss"):
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,Y))
    tf.scalar_summary("Loss",cost)
    
    with tf.name_scope("optimizer"):
    	train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    
    with tf.name_scope("accuracy"):
    	predict_op = tf.argmax(py_x,1)
    	correct_prediction = tf.equal(tf.argmax(Y,1),predict_op)
    	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.scalar_summary("accuracy",accuracy)


    validy_x = tf.argmax(tf.nn.softmax(model(valid_X,w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7),name = "Valid"),1) # No need to keep probabilities here 
    test_pyx = tf.nn.softmax(model(test_X,w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7),name="test") # No need to keep the probabilities here. We will use the entire network.

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    merged_summary_op = tf.merge_all_summaries()

logs_path = "tmp/tensorflow_logs/example"
with tf.Session(graph=mnist_graph) as sess:
	sess.run(init)
	summary_writer = tf.train.SummaryWriter(logs_path,sess.graph)
	n_iterations = 20
	batch_size = 100
	for i in range(n_iterations):
		idxs = np.random.permutation(range(len(trX)))
		n_batches = len(idxs)//batch_size
		for batch_i in range(n_batches):
			idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
			_,c,summary=sess.run([train_op,cost,merged_summary_op],feed_dict= {X:trX[idxs_i],Y:trY[idxs_i]})
			summary_writer.add_summary(summary,batch_i+1)
		if i%10 ==0:
			save_path = saver.save(sess,"model.mnist1")
			print("Model is saved in file: %s" % save_path)
			print (i,"Accuracy:",np.mean(np.argmax(trY,axis=1) == sess.run(predict_op,feed_dict={X:trX,Y:trY})), 
			"Valid_Acc:",np.mean(np.argmax(valY,axis=1) == sess.run(validy_x)))
	print ("Optimization Finished")
	print ("Run the command line:\n"\
		"--> tensorboard --logdir=/tmp/tensorflow_logs "\
		"\nThen open http://0.0.0.0.6006/ into your web browser")



# Testing Accuarcy 
	correct_prediction = tf.equal(tf.argmax(test_Y,1),tf.argmax(sess.run(test_pyx),1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print("Accuracy:",sess.run(accuracy))


# Should add drop-outs to make sure that the model is not overfit
# 


# Restoring the model and test on external data


# This is test data set
test_filelist = glob.glob("/Users/Satish/Downloads/MNIST/Train/Images/test/*.png") #create a list of all filenames.
x = np.arange(len(test_filelist)) #approx 29000
test_data = np.array([io.imread(test_filelist[i]) for i in x]) # Four Dimensional Array with [Batch,height,width,channels]

test_data = test_data[:,:,:,0:3] #Taking only the first 3 layers and ignoring the 3rd layer
test_data = test_data.astype(np.float32)

test_indexlist = list()
for i in range(len(x)):
    m = test_filelist[x[i]]
    test_indexlist.append(m)

# Dependent Variable 
# Call the Input labels data 
indexlist_new = [x1.rsplit('/',1)[-1] for x1 in test_indexlist]


mnist_graph = tf.Graph()
with mnist_graph.as_default():
    # Generate placeholders for the images and labels.
    w1 = init_weights([5,5,3,6]) # 6 features with size [5,5,3]
    b1 = bias([6])
    w2 = init_weights([6])
    b2 = bias([6])
    w3 = init_weights([5,5,6,16])
    b3 = bias([16])
    w4 = init_weights([16])
    b4 = bias([16])
    w5 = init_weights([4,4,16,120])
    b5 = bias([120])
    w6 = init_weights([120,84])
    b6 = bias([84])
    w7 = init_weights([84,10])
    b7 = bias([10])

    X = tf.placeholder(tf.float32,[None,28,28,3],name = "InputData")
    Y = tf.placeholder(tf.float32,[None,10],name = "InputLables")
    
    valid_X = tf.Variable(valX,dtype=tf.float32)
    valid_Y = tf.Variable(valY,dtype=tf.float32)
    
    test_X = tf.Variable(testX,dtype=tf.float32)
    test_Y = tf.Variable(testY,dtype=tf.float32)
    
    tf.add_to_collection("images", X)  # Remember this Op.
    tf.add_to_collection("labels", Y)  # Remember this Op.


    with tf.name_scope("Model"):
    	py_x = model(X,w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7)

    # Build a Graph that computes predictions from the inference model.
    with tf.name_scope("Loss"):
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,Y))
    
    
    with tf.name_scope("optimizer"):
    	train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    
    with tf.name_scope("accuracy"):
    	predict_op = tf.argmax(py_x,1)
    	correct_prediction = tf.equal(tf.argmax(Y,1),predict_op)
    	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    test_pyx1 = tf.nn.softmax(model(test_data,w1,w2,w3,w4,w5,w6,w7,b1,b2,b3,b4,b5,b6,b7),name="test_set")
    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    
with tf.Session(graph=mnist_graph) as sess:
	sess.run(init)
	saver.restore(sess,"model.mnist1")
	print ("Model Restored!")
	result= sess.run((tf.argmax(sess.run(test_pyx1),1)))
	d = {"filename":indexlist_new,"label":result}
	df  = pd.DataFrame(data=d)
	df.to_csv("/Users/Satish/Downloads/MNIST/submission4.csv")
	print ("Done!")

	