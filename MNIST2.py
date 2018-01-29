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
	layer1_pre = tf.matmul(X, w_h1)
	layer1 = tf.add(tf.nn.relu(layer1_pre), b_1)

	layer2_pre = tf.nn.relu(tf.matmul(layer1, w_h2))
	layer2 = tf.add(layer2_pre, b_2)

	layer3_pre = tf.nn.relu(tf.matmul(layer2, w_h3))
	layer3 = tf.add(layer3_pre, b_3)

	py_x = tf.matmul(layer3, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


# cost, ompitmizer, and train ops
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y), name='cost') # compute costs

optimizer = tf.train.GradientDescentOptimizer(0.015, name='gradients_descent_optimizer')
grads_and_vars = optimizer.compute_gradients(cost, )
train_op = optimizer.apply_gradients(grads_and_vars)

predict_op = tf.argmax(py_x, 1, name='predict_op')

# put the layer outputs into a list so we can use them as input for the synthetic gradient
layer_outputs = [layer1_pre, layer1, layer2_pre, layer2, layer3_pre, layer3, py_x]

for i, (grad, var) in enumerate(grads_and_vars):
	print(i, var.get_shape(), grad)


batch_size = 1

def prod(factors):
	prod = 1
	for factor in factors:
		prod = prod * int(factor)
	return prod

# gradient predictor

def synthetic_gradient_learner(layer_output, real_gradient):
	""" given the layer's output output, guess the gradient, and train to improve
		works in weights (2d) and biases (1d) by flattening all to 1d
	"""
	# input is [batches,size]
	in_size = int(layer_output.get_shape()[1])
	# gradient is [layer input size, layer output size]
	out_size = prod(real_gradient.get_shape())
	
	print("in and out sizes: ", in_size, out_size)
	# create random weights and make a prediction
	w_s = init_weights([in_size, out_size])
	print('debug1: ', w_s, '\ndebug', layer_output)
	grad_prediction = tf.matmul(tf.stop_gradient(layer_output), w_s)


	# compare it against the real gradient
	grad_real_flat = tf.stop_gradient(tf.reshape(real_gradient, [batch_size, out_size], name='flattened_gradient'))
	cost_s = tf.losses.mean_squared_error(grad_real_flat, grad_prediction)
	train_op_s = tf.train.GradientDescentOptimizer(0.05).minimize(cost_s)

	# reshape to batches + 2d, and then average down to 2d
	grad_prediction_flat = tf.reduce_mean(grad_prediction, axis=0)
	synthetic_gradient = tf.reshape(grad_prediction_flat, real_gradient.get_shape())
	return synthetic_gradient, train_op_s, cost_s

# iterate through real gradients and make networks to predict them
synthetic_gradients_and_vars = list()
synthetic_train_ops = list()
synthetic_costs = list()

for layer_output, (grad, var) in zip(layer_outputs, grads_and_vars):
	synthetic_gradient, train_op_s, cost_s = synthetic_gradient_learner(layer_output, grad)
	synthetic_gradients_and_vars.append((synthetic_gradient, var))
	synthetic_train_ops.append(train_op_s)
	synthetic_costs.append(cost_s)
	
cost_s_sum = sum(synthetic_costs)
	

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

	# labels for tab columns printed out
	print('i\tstart\tacc\tsyn l\ttime S')

	for i in range(100):
		for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
			if start < 2000:
				# train the network and DNI subnetworks
				sess.run(synthetic_train_ops + [train_op],
					feed_dict={X: trX[start:end], Y: trY[start:end]})
			else:
				# use the DNI subnetworks
				sess.run(train_op_decoupled,
					feed_dict={X: trX[start:end], Y: trY[start:end]})
			
			# periodically print training info
			if start % 1000 == 0 or (start >= 2000 and start%5==0):
				summary_str, predictions, synthgrad_cost = sess.run(
							[merged_summary_op, predict_op, cost_s_sum], 
							feed_dict={X: teX[:], Y:teY[:]})
				print(i, start, np.mean(np.argmax(teY, axis=1) == predictions), synthgrad_cost, sep='\t')

			summary_writer.add_summary(summary_str, i*len(trX) + start)

	writer.close()
