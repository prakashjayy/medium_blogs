Transfer learning is very useful. Trained models have robust features which can be generalized to any batch of  
images easily. 

Here we will be doing only one thing:  
1) When u get a new set of images and you want to classify them, you can send these images through the imagenet trained inception network of your choice and extract features or weights of which every layer we like 
2) With these extracted features as dependent variables we can apply any machine learning models and do classification 

Import Tensorflow:

    import tensorflow as tf
    slim = tf.contrib.slim
    
2. Do a git pull to the [link](https://github.com/tensorflow/models) https://github.com/tensorflow/models.git. There is a folder called slim which again has the following folders 
- datasets: used to download the required dataset
- nets: All the network architecutres are present
- preprocessing: Required preprocessing for the network to efficiently train is present here 
    
3. Now to download the trained model run the following commands (it downloads 200MB file for inception netowork and 500MB for VGGNet

    from datasets import dataset_utils (required to download the trained model checkpoints)
    url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
    checkpoints_dir = '/tmp/checkpoints'
    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)
    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

4. Now import the model folder . Since we are using inception here, we will import inception file 
    
    from nets import inception
    image_size = inception.inception_v1.default_image_size
    from preprocessing import inception_preprocessing #import the preprocessing file also 
   
5. Now download your dataset, Here it is the list of image location that I need.

    import cv2, os, glob
    filelist  = glob.glob("/kaggle-fish/data/ALB/*")
    filelist[0]
    
    Out[5]: '/kaggle-fish/data/ALB/img_00003.jpg'
    
6. Now create a placeholder and  define the Graph:
    
    with tf.Graph().as_default():
        with slim.arg_scope(inception.inception_v1_arg_scope()):
        model = inception.inception_v1(X, num_classes=1001 , is_training = False) #(since ware not training this network, it is 1001 out classes) check the inception_v1.py file for more clarity
                
7. Check all the tensor names available in the graph and choose the filters(tensors) which you want to extract from it 

    sess = tf.Session()
    op = sess.graph.get_operations()
    [m.values() for m in op]
       
     #This is one of the tensor name (First conv layer filters) available
     Out[5]: (<tf.Tensor 'InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution:0' shape=(?, 112, 112, 64) dtype=float32>,)  
   
8. Now Here is a long length code. which will actually extract the features which you need, please follow through the code

    import numpy as np
    
    #initialize all the variables 
    init = tf.initialize_all_variables()
       
    # store the variables 
    restorer = tf.train.Saver()
       
    # create a session and initiate the network 
     with tf.Session() as sess:
            sess.run(init)
       
       #restore the network from the last checkpoint. In our case it is the downloaded file above
            restorer.restore(sess, "/checkpoints/inception_v1.ckpt")
       
       # Now this is the main part, Pull the tensor by name of which ever features you want to access. In this case here I randomly have taken one of the layer and extracted the features to make sure everything is working from end to end. 
       
            p3 = sess.graph.get_tensor_by_name('InceptionV1/InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/convolution:0')
       
The wait is over . Now we can pull all the features from the particular layer and store it in a list or dictionary. :)

       X_pool3 = []
    for i in range(len(filelist[0:10])):
        img = cv2.imread(filelist[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.constant(img, tf.float32)
        
        # Preprocess the image 
        processed_image = inception_preprocessing.preprocess_image(img, image_size, image_size, is_training=False)
        
        # convert tensorshape from [224,224,3] to [1,224,224,3] incase if you want to send a batch of images concat it. 
        processed_images  = tf.expand_dims(processed_image, 0)
        
        # this step is required because placeholders doesn't accept tensor object but instead accept strings, list, dict, numpy arrays etc.
        image = sess.run(processed_images)
        
        # send the image through the network 
        features = sess.run(p3,{X:image})
        
        #append the features to list
        X_pool3.append(features) 
        
 
 How to do machine leaning after getten the features?
 1) Now since we get the features of each image in numpy array format. 
      - Flatten it and create a dataframe
 2) Apply multi-class SVM or XGBoost or Neural Networks of your choice.
 3) Do hyperparameter tuning and select the best params (Check NeuPy for NN )
 
 I will write another post on my findings
