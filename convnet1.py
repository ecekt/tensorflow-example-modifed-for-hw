import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.load('data/ORL_faces.npz')
#r = data['trainX'][0].reshape(112,92)
#from scipy.misc import imshow
#imshow(r)

cl = np.unique(data['trainY']).size

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 10304])
y_ = tf.placeholder(tf.float32, shape=[None, 20])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def convert_to_one_hot(array, classno):

	s = np.size(array)
	o = np.zeros(s*classno).reshape(s,classno)

	for i in range(np.size(array)):
		o[i][array[i]] = 1

	return o

def next_batch(data, sets='', batch_size=10):

	minibatch = []
	x = []
	y = []

	per = np.random.permutation(len(data[sets+'X']))
	
	for i in range(batch_size):
		x.append(data[sets+'X'][per[i]])
		y.append(data[sets+'Y'][[per[i]]])

	minibatch.append(np.transpose(x))
	minibatch.append(y)
	return minibatch

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,112,92,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([28 * 23 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 28*23*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 20])
b_fc2 = bias_variable([20])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

acc = []

epochs = 200
for i in range(epochs):
  batch = next_batch(data, 'train', 20)
  batch[1] = convert_to_one_hot(batch[1], cl)
  print i
  train_accuracy = accuracy.eval(feed_dict={x:batch[0].T, y_: batch[1]})
  acc.append(1- train_accuracy)
  print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0].T, y_: batch[1]})


test_accuracy = accuracy.eval(feed_dict={x:data['testX'], y_: convert_to_one_hot(data['testY'],cl)})
print "test accuracy ", test_accuracy

iters = []
for i in range(epochs):
  iters.append(i + 1)

plt.plot(iters,acc)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

