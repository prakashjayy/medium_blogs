Transfer learning is very useful. Trained models have robust features which can be generalized to any batch of  
images easily. 

Here we will be doing only one thing:  
1) When u get a new set of images and you want to classify them, you can send these images through the  
imagenet trained inception network of your choice and extract features or weights of which every layer we like 
2) With these extracted features as dependent variables we can apply any machine learning models and do classification 

First Import Tensorflow:

    import tensorflow as tf 
    slim = tf.contrib.slim
