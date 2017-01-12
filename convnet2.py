import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data = np.load('data/ORL_faces.npz')
#plt.imshow(data['trainX'][0].reshape(112,92),cmap=plt.get_cmap('gray'))

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

def visualize(row, col, data, title, type):
	
  fig = plt.figure()
  gs = gridspec.GridSpec(row, col)
  gs.update(wspace=0.025, hspace=0.025) 
  
  ax = [plt.subplot(gs[i]) for i in range(row*col)]

  fig.suptitle(title, fontsize=16)

  for f in range(row*col):
    if(type == 'filter'):
      ax[f].imshow(sess.run(data[:,:,0,f]),cmap=plt.get_cmap('gray'))
    elif(type == 'activation'):
      ax[f].imshow(data[0,:,:,f],cmap=plt.get_cmap('gray'))
    ax[f].axis('off')

  plt.show()


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

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 20])
b_fc2 = bias_variable([20])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

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
  train_accuracy = accuracy.eval(feed_dict={x:batch[0].T, y_: batch[1], keep_prob: 1.0})
  acc.append(1 - train_accuracy)
  print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0].T, y_: batch[1], keep_prob: 0.5})
  
test_accuracy = accuracy.eval(feed_dict={x:data['testX'], y_: convert_to_one_hot(data['testY'],cl), keep_prob: 1.0})
print "test accuracy ", test_accuracy

visualize(4,8,W_conv1,'Filters for Conv Layer 1', 'filter')

visualize(8,8,W_conv2,"Filters for Conv Layer 2", 'filter')

activations1 = h_conv1.eval(session=sess,feed_dict={x:data['trainX'], y_: convert_to_one_hot(data['trainY'],cl), keep_prob: 0.5})
activations2 = h_conv2.eval(session=sess,feed_dict={x:data['trainX'], y_: convert_to_one_hot(data['trainY'],cl), keep_prob: 0.5})

visualize(4,8,activations1,"RELU Activations after Conv Layer 1",'activation')

visualize(8,8,activations2,"RELU Activations after Conv Layer 2",'activation')


iters = []
for i in range(epochs):
  iters.append(i + 1)

plt.plot(iters,acc)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

