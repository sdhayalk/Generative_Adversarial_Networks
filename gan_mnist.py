import tensorflow
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

get_dataset()
