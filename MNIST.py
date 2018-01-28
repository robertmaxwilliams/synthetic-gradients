from __future__ import print_function, division

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
	layer1 = tf.add(tf.nn.relu(tf.matmul(X, w_h1)), b_1)
	layer2 = tf.add(tf.nn.relu(tf.matmul(layer1, w_h2)), b_2)
	layer3 = tf.add(tf.nn.relu(tf.matmul(layer2, w_h3)), b_3)
	py_x = tf.matmul(layer3, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


# cost, ompitmizer, and train ops
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y), name='cost') # compute costs

optimizer = tf.train.GradientDescentOptimizer(0.015, name='gradients_descent_optimizer')
grads_and_vars = optimizer.compute_gradients(cost, )
train_op = optimizer.apply_gradients(grads_and_vars)

predict_op = tf.argmax(py_x, 1, name='predict_op')


for i, (grad, var) in enumerate(grads_and_vars):
	print(i, var.get_shape(), grad)


batch_size = 1
# gradient predictor

# let's try to play with the gradient
# here is the gradient with respect to w_h2, the (50, 30) matrix
partial_grad = grads_and_vars[2][0]#tf.gradients(cost, [w2])[0]

def linear_network(pixels, W):
	with tf.variable_scope("synthetic-gradient-predictor"):
		return tf.matmul(pixels, W)

# linear netowork, converts 50 input vector to 50*30 gradient (flattened)
w_s = init_weights([50, 50*30], name='synthetic_weights')
grad_prediction = linear_network(layer1, w_s)


# reshape the synthetic gradient and put it to use
synthetic_gradient_all_batches = tf.reshape(grad_prediction, [batch_size,50,30], name="flat_synthetic_for_use_all_batches")
synthetic_gradient = tf.reduce_mean(synthetic_gradient_all_batches, axis=0)
print("synthetic gradient: ", synthetic_gradient)

# iterate through real gradient and vars and make gradient zeroe except where
# our model predicts it
synthetic_gradients = list()
for i, (grad, var) in enumerate(grads_and_vars):
	if i == 2:
		synthetic_gradients.append((synthetic_gradient, var))
	else:
		synthetic_gradients.append((tf.zeros(grad.get_shape()), var))
	
# now to define a custom training op where we apply the calculated gradient.
train_op_decoupled = optimizer.apply_gradients(synthetic_gradients)

# find how bad the guess was this time and train to improve it
with tf.variable_scope('train_synthetic_network'):
	grad_flat = tf.reshape(partial_grad, [1, 50*30], name='flattened_gradient')
	cost_s = tf.losses.mean_squared_error(grad_flat, grad_prediction)
	train_op_s = tf.train.GradientDescentOptimizer(0.05).minimize(cost_s)

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
