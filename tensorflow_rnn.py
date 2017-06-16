import tensorflow as tf

batch_size = 128
num_steps = 30 # Max length of a sentence
state_size = 3
output_classes = 2 #positive or negitive --> use 1 if using sigmoid
n_words = 400 # Number of words in the corpus

x = tf.placeholder(tf.int32, [batch_size, num_steps], name = "input_placeholder")
y = tf.placeholder(tf.int32, [batch_size, output_classes], name = "labels_placeholder")
init_state = tf.zeros([batch_size, state_size])
# init_state = <tf.Tensor 'zeros:0' shape=(128, 3) dtype=float32>

x_one_hot = tf.one_hot(x, n_words) # [128, 400, 30]
# <tf.Tensor 'one_hot:0' shape=(128, 30, 400) dtype=float32>

rnn_inputs = tf.unstack(x_one_hot, axis = 1) # [128, 400]
# rnn_inputs[0] shape is <tf.Tensor 'unstack:0' shape=(128, 400) dtype=float32>

# Define the weights and bias
with tf.variable_scope("Wih"):
    wih = tf.get_variable("W", [n_words , state_size])
    b = tf.get_variable("b", [state_size], initializer = tf.constant_initializer(0.0))

with tf.variable_scope("Whh"):
    whh = tf.get_variable("W", [state_size, state_size])


def rnn_cell(rnn_input, state):
    with tf.variable_scope("Wih", reuse = True):
        wih = tf.get_variable("W", [n_words , state_size])
        bh = tf.get_variable("b", [state_size], initializer = tf.constant_initializer(0.0))

    with tf.variable_scope("Whh", reuse = True):
        whh = tf.get_variable("W", [state_size, state_size])

    new_state = tf.tanh(tf.nn.bias_add(tf.matmul(rnn_input, wih)+tf.matmul(state, whh), bh))
    return new_state

state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)

final_state = rnn_outputs[-1]
# final state = <tf.Tensor 'Tanh_29:0' shape=(128, 3) dtype=float32>

# Adding a dense layer
output_layer1 = 4
with tf.variable_scope("Dense_Layer"):
    w1 = tf.get_variable("W1", [state_size , output_layer1])
    b1 = tf.get_variable("b1", [output_layer1], initializer = tf.constant_initializer(0.0))

final_state = tf.tanh(tf.nn.bias_add(tf.matmul(final_state, w1), b1))
# <tf.Tensor 'Tanh_30:0' shape=(128, 4) dtype=float32>

output_layer2 = 6
with tf.variable_scope("Dense_Layer"):
    w2 = tf.get_variable("W2", [output_layer1 , output_layer2])
    b2 = tf.get_variable("b2", [output_layer2], initializer = tf.constant_initializer(0.0))

final_state = tf.tanh(tf.nn.bias_add(tf.matmul(final_state, w2), b2))
#  <tf.Tensor 'Tanh_31:0' shape=(128, 6) dtype=float32>

with tf.variable_scope("output_layer"):
    wo = tf.get_variable("W", [output_layer2 , output_classes])
    bo= tf.get_variable("b", [output_classes], initializer = tf.constant_initializer(0.0))

final_state = tf.nn.bias_add(tf.matmul(final_state, wo), bo)
#<tf.Tensor 'BiasAdd_32:0' shape=(128, 2) dtype=float32>
final_state = tf.nn.softmax(final_state)
