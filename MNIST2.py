from __future__ import print_function, division

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import operator


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape, stddev = 0.1, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=stddev), name=name)



# data input
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784], name='X')
Y = tf.placeholder("float", [None, 10], name='Y')
# model parameters
w_h1 = init_weights([784, 50], name='hidden_weights_1') # create symbolic variables
b_1 = init_weights([50], name='hidden_bias_1')

w_h2 = init_weights([50, 30], name='hidden_weights_2')
b_2 = init_weights([30], name='hidden_bias_2')

w_h3 = init_weights([30, 30], name='hidden_weights_3')
b_3 = init_weights([30], name='hidden_bias_3')

w_o = init_weights([30, 10], name='output_weights')

with tf.variable_scope('classification_model'):
	# model definition
	layer1_pre = tf.nn.relu(tf.matmul(X, w_h1))
	layer1 = tf.add(layer1_pre, b_1)

	layer2_pre = tf.nn.relu(tf.matmul(layer1, w b_2)
	layer2 = tf.add(layer2_pre, b_1)

	layer3_pre = tf.nn.relu(tf.matmul(layer2, w b_3)
	layer3 = tf.add(layer3_pre, b_1)

	py_x = tf.matmul(layer3, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


# cost, ompitmizer, and train ops
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y), name='cost') # compute costs

optimizer = tf.train.GradientDescentOptimizer(0.015, name='gradients_descent_optimizer')
grads_and_vars = optimizer.compute_gradients(cost, )
train_op = optimizer.apply_gradients(grads_and_vars)

predict_op = tf.argmax(py_x, 1, name='predict_op')

# put the layer inputs into a list so we can use them as input for the synthetic gradient
layer_inputs = [X, layer1_pre, layer1, layer2_pre, layer2, layer3_pre, layer3]

for i, (grad, var) in enumerate(grads_and_vars):
	print(i, var.get_shape(), grad)


batch_size = 1

def prod(factors):
    return reduce(operator.mul, factors, 1)


# gradient predictor

def synthetic_gradient_learner(layer_input, real_gradient):
	""" given the input, guess the gradient, and train to improve
		works in weights (2d) and biases (1d)
	"""
	# input is [batches,size]
	in_size = layer_input.get_shape()[1]
	# gradient is [layer input size, layer output size]
	out_size = prod(real_gradient.get_shape())
	
	# create random weights and make a prediction
	w_s = init_weights(in_size, out_size)
	prediction = np.matmul(layer_input, w_s)

	# compare it against the real gradient
	grad_flat = tf.reshape(read_gradient, [1, out_size], name='flattened_gradient')
	cost_s = tf.losses.mean_squared_error(grad_prediction, grad_flat)
	train_op_s = tf.train.GradientDescentOptimizer(0.05).minimize(cost_s)

	# reshape to batches + 2d, and then average down to 2d
	synthetic_gradient_reduce_mean = tf.reduce_mean(synthetic_gradient_all_batches, axis=0)
	synthetic_gradient = tf.reshape(grad_flat, real_gradient.get_shape())
	return synthetic_gradient, train_op_s

# iterate through real gradients and make networks to predict them
synthetic_gradients_and_vars = list()
synthetic_train_ops = list()

for layer_input, (grad, var) in zip(layer_inputs, grads_and_vars):
	synthetic_gradient, train_op_s = synthetic_gradient_learner(layer_input, grad)
	synthetic_gradients_and_vars.append((synthetic_gradient, var))
	synthetic_train_ops.append(train_op_s)
	

	

# now to define a custom training op where we apply the calculated gradient.
train_op_decoupled = optimizer.apply_gradients(synthetic_gradients_and_vars)

# easy to use summary graphs
tf.summary.scalar("synthetic_gradient_cost_function", cost_s)
tf.summary.scalar("cost_function", cost)


merged_summary_op = tf.summary.merge_all()




# Launch the graph in a session
with tf.Session() as sess:
	# write stuff to tensorboard
	summary_writer = tf.summary.FileWriter("/home/max/logs/synth1", sess.graph)



	# you need to initialize all variables
	tf.global_variables_initializer().run()

	for i in range(100):
		for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
			if i < 1:
				sess.run([train_op, train_op_s], 
					feed_dict={X: trX[start:end], Y: trY[start:end]})
			else:
				sess.run([train_op_decoupled], 
					feed_dict={X: trX[start:end], Y: trY[start:end]})
			
#			sess.run([train_op, train_op_s], 
#					feed_dict={X: trX[start:end], Y: trY[start:end]})
			if start % 1000 == 0:
				summary_str, predictions, synthgrad_cost = sess.run(
							[merged_summary_op, predict_op, cost_s], 
							feed_dict={X: teX[:], Y:teY[:]})
				print(i, start, np.mean(np.argmax(teY, axis=1) == predictions), synthgrad_cost, sep='\t')

			summary_writer.add_summary(summary_str, i*len(trX) + start)

	writer.close()
