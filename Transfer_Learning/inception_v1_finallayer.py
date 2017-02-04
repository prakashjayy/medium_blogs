import tensorflow as tf
slim = tf.contrib.slim

from models import inception
from datasets import dataset_utils

from nets import inception
from preprocessing import inception_preprocessing
import cv2, glob, os, random, itertools
import numpy as np
import pandas as pd
from tqdm import tqdm





tf.app.flags.DEFINE_string(
    'train_data', '/Users/Satish/Downloads/kaggle-fish/data/*',
    'Directory where the image files are present.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/Users/Satish/Downloads/tensorflow-2017/checkpoints/inception_v1.ckpt',
    'location of the checkpoint files')

tf.app.flags.DEFINE_string(
    'tensor_name', 'InceptionV1/Logits/MaxPool_0a_7x7/AvgPool:0',
    'The name of the tensor which you want to access')

FLAGS = tf.app.flags.FLAGS



image_size = inception.inception_v1.default_image_size
X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])

filelist = glob.glob(FLAGS.train_data)
filelist = [glob.glob(filelist[i]+"/*") for i in range(len(filelist))]
filelist = list(itertools.chain(*filelist))

# random.seed(42)
# random.shuffle(filelist)
# x_train = filelist[0:int(0.6*len(filelist))]
# x_valid = filelist[int(0.6*len(filelist)):]

print("[INFO] the filelist has been created")

with tf.Graph().as_default():
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        model = inception.inception_v1(X, num_classes = 1001, is_training = False)


init = tf.initialize_all_variables()
restorer = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    restorer.restore(sess, FLAGS.checkpoint_dir)
    p3 = sess.graph.get_tensor_by_name(FLAGS.tensor_name)
    X_pool3 = []
    print("[INFO] feature collection has begun")
    for i in tqdm(range(len(filelist))):
        img= cv2.imread(filelist[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.constant(img, tf.float32)
        processed_image = inception_preprocessing.preprocess_image(img, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)
        image = sess.run(processed_images)
        features = sess.run(p3,{X:image})
        X_pool3.append(features)

print("[INFO] All the features have been collected . Saving it to a dataframe now")
tags = [filelist[i].rsplit("/")[-2] for i in range(len(filelist))]
images = np.vstack([np.reshape(X_pool3[i],-1) for i in range(len(X_pool3))])
numpy.savetxt("fish.csv", np.c_[images, tags], delimiter=",")
