import tensorflow as tf 
import numpy as np 
import pandas as pd 

data = pd.read_csv("/Users/PrakashR/Desktop/Rdirectory/class1al/Auto.csv")
data_req = data[["displacement","horsepower","weight",
"acceleration","mpg"]]

# Call a Data frame and choose the necessary columns you need

print (data_req.shape)

# Drop the NA value rows 
data_req = data_req.dropna()
print (data_req.shape)

# Convert Data frame to Matrix
data_req = np.matrix(data_req).astype(np.float32)
trX,trY = data_req[:,0:4],data_req[:,4] # Divide independent and dependent Varaiables.

# PlaceHolders 
X = tf.placeholder(tf.float32,[None,4])
Y = tf.placeholder(tf.float32,[None,1])


w = tf.Variable(np.zeros([4,1]),dtype = tf.float32,name="weights")
b = tf.Variable(np.zeros([1]),dtype = tf.float32,name= "bias")
y_model = tf.add(tf.matmul(X,w),b)

# Use square error for cost fuction
cost = tf.reduce_sum(tf.pow(y_model-Y, 2))/(2*50)

train_op = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost) # Construct an optimizer to minimize the cost and fit line to my data


# launch the graph 
with tf.Session() as sess:
	tf.initialize_all_variables().run()
	print (sess.run(w))
	print (sess.run(b))
	n_iterations = 50
	batch_size = 50
	for i in range(n_iterations):
		idxs = np.random.permutation(range(len(trX)))
		n_batches = len(idxs)//batch_size
		for batch_i in range(n_batches):
			idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
			sess.run(train_op,feed_dict= {X:trX[idxs_i],Y:trY[idxs_i]})

			if batch_i % 10 == 0:
				print (sess.run(cost,feed_dict= {X:trX[idxs_i],Y:trY[idxs_i]}))
	print (sess.run(w))
	print (sess.run(b))

# For some reason the Gradient Descent is reaching Inf value very easily.
# i.e the only reason why learning rate is kept very very low.


