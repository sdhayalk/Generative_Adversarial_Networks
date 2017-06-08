import tensorflow as tf
import numpy as np

def get_dataset():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    dataset_train_feautres = mnist.train.images
    dataset_train_labels = mnist.train.labels
    dataset_test_feautres = mnist.test.images
    dataset_test_labels = mnist.test.labels

    print('dataset_train_feautres.shape: ', dataset_train_feautres.shape)
    print('dataset_train_labels.shape: ', dataset_train_labels.shape)
    print('dataset_test_feautres.shape: ', dataset_test_feautres.shape)
    print('dataset_test_labels.shape: ', dataset_test_labels.shape)

    return dataset_train_feautres, dataset_train_labels, dataset_test_feautres, dataset_test_labels

#Xavier initialization: https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
def xavier_init(shape):
    input_nodes = shape[0]
    stddev_xavier = 1. / tf.sqrt(input_nodes/2.)
    return tf.random_normal(shape=shape, stddev=stddev_xavier)

def generator(z):
    layer_1_weights = tf.Variable(xavier_init([100, 256]))
    layer_1_biases = tf.Variable(tf.zeros(shape=[256]))

    layer_2_weights = tf.Variable(xavier_init([256, 512]))
    layer_2_biases = tf.Variable(tf.zeros(shape=[512]))

    layer_3_weights = tf.Variable(xavier_init([512, 784]))
    layer_3_biases = tf.Variable(tf.zeros(shape=[784]))

    layer_1_output = tf.nn.relu(tf.matmul(z, layer_1_weights) + layer_1_biases)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, layer_2_weights) + layer_2_biases)
    layer_3_output = tf.nn.sigmoid(tf.matmul(layer_2_output, layer_3_weights) + layer_3_biases)

    return layer_3_output

def discriminator(x):
    layer_1_weights = tf.Variable(xavier_init([784, 256]))
    layer_1_biases = tf.Variable(tf.zeros(shape=[256]))

    layer_2_weights = tf.Variable(xavier_init([256, 512]))
    layer_2_biases = tf.Variable(tf.zeros(shape=[512]))

    layer_3_weights = tf.Variable(xavier_init([512, 1]))
    layer_3_biases = tf.Variable(tf.zeros(shape=[1]))

    layer_1_output = tf.nn.relu(tf.matmul(x, layer_1_weights) + layer_1_biases)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, layer_2_weights) + layer_2_biases)
    layer_3_grid = tf.matmul(layer_2_output, layer_3_weights) + layer_3_biases
    layer_3_output = tf.nn.sigmoid(layer_3_grid)

    return layer_3_output, layer_3_grid

get_dataset()
