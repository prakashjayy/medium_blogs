"""
Tensorflow MLP optimizers benchmark with the same model:

"""


import numpy as np
import tensorflow as tf

class BO():
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        """The class builds a MLP network and trains a model with the input data

        """
        self.x_train = x_train / 255.0
        self.y_train =y_train
        self.x_valid = x_valid / 255.0
        self.y_valid = y_valid
        self.x_test = x_test / 255.0
        self.y_test = y_test
        self.input = tf.placeholder("float", [None, 784])
        self.output = tf.placeholder("float", [None, 10])
        self.layers = {}
        self.train_step = None

    def tf_optimizer(self, optimize, learning_rate = 0.001):

        if optimize == 'adadelta':
           optimizer = tf.train.AdadeltaOptimizer(
               learning_rate)
        elif optimize == 'adagrad':
           optimizer = tf.train.AdagradOptimizer(
               learning_rate)
        elif optimize == 'adam':
           optimizer = tf.train.AdamOptimizer(
               learning_rate)
        elif optimize == 'ftrl':
           optimizer = tf.train.FtrlOptimizer(
               learning_rate)
        elif optimize == 'momentum':
           optimizer = tf.train.MomentumOptimizer(
               learning_rate,
               momentum = 0.9,
               name='Momentum')
        elif optimize == 'rmsprop':
           optimizer = tf.train.RMSPropOptimizer(
               learning_rate)
        elif optimize == 'sgd':
           optimizer = tf.train.GradientDescentOptimizer(
                learning_rate)

        return optimizer

    def build_graph(self):
        self.layers["layer1"] = tf.contrib.layers.fully_connected(inputs = self.input, num_outputs = 256, activation_fn = tf.nn.relu, scope = "input")
        self.layers["layer2"] = tf.contrib.layers.fully_connected(inputs = self.layers["layer1"], num_outputs = 128, activation_fn = tf.nn.relu, scope = "fc2")
        self.layers["layer3"] = tf.contrib.layers.fully_connected(inputs = self.layers["layer2"],
        num_outputs = 10, activation_fn = tf.nn.softmax, scope = "output")
        tf.Print(self.layers["layer3"], [tf.argmax(self.layers["layer3"], 1)],
        "argmax(out) = ", summarize= 20, first_n= 7)
        return self

    def compile_graph(self, optimize, learning_rate = 0.001):
        print ("[Using optimizer]:", optimize)
        with tf.name_scope("cross_entropy"):
            self._output = tf.reduce_mean(-tf.reduce_sum(self.output * tf.log(self.layers["layer3"]+0.00001),[1]))
        tf.summary.scalar("cross_entropy", self._output)

        with tf.name_scope("accuracy"):
            self._prediction = tf.equal(tf.argmax(self.layers["layer3"], 1), tf.argmax(self.output,1))
            self._accuracy = tf.reduce_mean(tf.cast(self._prediction, tf.float32))
        tf.summary.scalar("accuracy", self._accuracy)

        with tf.name_scope("gradient_calculations"):
          self.train_step = self.tf_optimizer(optimize).minimize(self._output)
        
        return self

    def train(self,  batch_size = 32, epochs = 100, summary_dir = "/tmp/mnist"):
        # saver = tf.train.Saver(tf.global_variables())
        merged = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

            iterations = int(self.x_train.shape[0]/batch_size)
            for ep in range(epochs):
                for i in range(iterations):
                    batch_x , batch_y = self.x_train[i*batch_size:(i+1)*batch_size], self.y_train[i*batch_size:(i+1)*batch_size]
                    summary, _, c = sess.run([merged , self.train_step, self._output ], feed_dict = {self.input: batch_x, self.output: batch_y})
                    summary_writer.add_summary(summary, ep*iterations+i)
                    if i%500 ==0:
                         train_cost, train_acc = sess.run([self._output, self._accuracy], feed_dict={self.input: self.x_train, self.output: self.y_train})
                         valid_cost, valid_acc = sess.run([self._output, self._accuracy], feed_dict={self.input: self.x_valid, self.output: self.y_valid})
                         print ("epoch: ", ep, "train_cost: ", train_cost , "Train_Accuracy: ", train_acc, "valid_cost: ", valid_cost, "Validation_Accuracy: ", valid_acc)
            print("optimization_finished")
            test_accuracy = sess.run([self._accuracy],feed_dict={self.input: self.x_test, self.output: self.y_test} )
            print("Test Accuracy:", test_accuracy)
