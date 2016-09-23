import tensorflow as tf 

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y = tf.mul(a,b)

with tf.Session() as sess:
	print ("%f should equal 2.0" % sess.run(y, feed_dict={a:1,b:2}))
	print (sess.run(y,feed_dict = {a:1,b:6}))