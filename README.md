# TensorFlow
This Repository contains all tensorflow tutorials . There might be some replicas from other repositories.


Working with Graphs 

    import tensorflow as tf  
    slim = tf.contrib.slim 
    
    # Define a basic graph:
    def ciresan(X):
         with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.tanh, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(X, 32, [4, 4], scope="conv1", padding = "SAME")
        net = slim.max_pool2d(net,[2, 2], scope="pool1", padding = "SAME")
        net = slim.conv2d(X, 48, [5, 5], scope="conv2", padding = "SAME")
        net = slim.max_pool2d(net,[3, 3], scope="pool2", padding = "SAME")
        net = slim.fully_connected(net, 150, scope="fc3")
        net = slim.fully_connected(net, 5, scope="fc4", activation_fn = tf.nn.softmax)
    return net
    
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = ciresan(X)
    
How to save a graph?

    import os
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, os.getcwd(), "model.pb", as_text = False)
  
To see what variable names does the model contain
    
    [v.name for v in slim.get_model_variables()]
    out:
    ['conv1/weights:0',
    'conv1/biases:0',
    'conv2/weights:0',
    'conv2/biases:0',
    'fc3/weights:0',
    'fc3/biases:0',
    'fc4/weights:0',
    'fc4/biases:0']
  
To just see the model variables

    [<tensorflow.python.ops.variables.Variable at 0x1150f6f60>,
    <tensorflow.python.ops.variables.Variable at 0x1150f6e80>,
    <tensorflow.python.ops.variables.Variable at 0x1150f6a20>,
    <tensorflow.python.ops.variables.Variable at 0x1150f6e48>,
    <tensorflow.python.ops.variables.Variable at 0x1150f68d0>,
    <tensorflow.python.ops.variables.Variable at 0x1150f6b38>,
    <tensorflow.python.ops.variables.Variable at 0x1150f62e8>,
    <tensorflow.python.ops.variables.Variable at 0x1151a2ef0>]
    
To see the operations in the graph (You will see many, so to cut short I have given here only the first string.
    
    sess = tf.Session()
    op = sess.graph.get_operations()
    [m.values() for m in op][1]
    out:
    (<tf.Tensor 'conv1/weights:0' shape=(4, 4, 3, 32) dtype=float32_ref>,)
    
How to internally access a layer (an op ) . We will dicuss about the graph layer

    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
    with gfile.FastGFile("model.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        po3 = sess.graph.get_tensor_by_name("conv1/weights:0")
        print (po3)
        
    out:
    Tensor("conv1/weights:0", shape=(4, 4, 3, 32), dtype=float32_ref)

Instead of print(po3) . You can actually print weights by using 


    print (sess.run(po3, feed_dict={X:images}) # This is how transfer learning works 
  
 
