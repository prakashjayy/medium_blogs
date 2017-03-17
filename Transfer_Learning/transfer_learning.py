import tensorflow as tf
slim = tf.contrib.slim

from datasets import dataset_utils

from nets import inception
from preprocessing import inception_preprocessing
import cv2, glob, os, random, itertools
import numpy as np
from tqdm import tqdm

from ise.indexer.featureindexer import FeatureIndexer




tf.app.flags.DEFINE_string(
    'train_data', '/home/prakash/Downloads/test_stg1/*.jpg',
    'Directory where the image files are present.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/home/prakash/Downloads/pre-trained/checkpoints/inception_v1.ckpt',
    'location of the checkpoint files')

tf.app.flags.DEFINE_string(
    'tensor_name', 'InceptionV1/Logits/MaxPool_0a_7x7/AvgPool:0',
    'The name of the tensor which you want to access')

tf.app.flags.DEFINE_string(
'features_db', "test_features.hdf5",
"the name of features file which will save the data")

tf.app.flags.DEFINE_integer("approx_images", 1000, "The Approximate number of Images in the data")

tf.app.flags.DEFINE_integer("max_buffer_size", 100, "The maximum buffer size to save the data")


FLAGS = tf.app.flags.FLAGS



image_size = inception.inception_v1.default_image_size
x = tf.placeholder(tf.float32,[None,None,3])
processed_image = inception_preprocessing.preprocess_image(x, image_size, image_size, is_training=False)
processed_images  = tf.expand_dims(processed_image, 0)
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

fi = FeatureIndexer(FLAGS.features_db,estNumImages=FLAGS.approx_images, maxBufferSize = FLAGS.max_buffer_size)
print("[INFO] the database has been created")


with tf.Session() as sess:
    sess.run(init)
    restorer.restore(sess, FLAGS.checkpoint_dir)
    p3 = sess.graph.get_tensor_by_name(FLAGS.tensor_name)
    #X_pool3 = []
    print("[INFO] feature collection has begun")
    for i in tqdm(range(len(filelist))):
        img= cv2.imread(filelist[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = sess.run(processed_images,feed_dict={x:np.float32(img)})
        features = sess.run(p3,{X:m})
        fi.add(filelist[i].rsplit("/")[-2],features.reshape([1,features.shape[3]]))


fi.finish()

print("[INFO] All the features have been collected . Saving it to a dataframe now")
# tags = [filelist[i].rsplit("/")[-2] for i in range(len(filelist))]
# images = np.vstack([np.reshape(X_pool3[i],-1) for i in range(len(X_pool3))])
# numpy.savetxt("fish.csv", np.c_[images, tags], delimiter=",")
