import tensorflow as tf
import numpy as np

def get_dataset():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    dataset_train_features = mnist.train.images
    dataset_train_labels = mnist.train.labels
    dataset_test_features = mnist.test.images
    dataset_test_labels = mnist.test.labels

    print('dataset_train_features.shape: ', dataset_train_features.shape)
    print('dataset_train_labels.shape: ', dataset_train_labels.shape)
    print('dataset_test_features.shape: ', dataset_test_features.shape)
    print('dataset_test_labels.shape: ', dataset_test_labels.shape)

    return dataset_train_features, dataset_train_labels, dataset_test_features, dataset_test_labels

#Xavier initialization: https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
def xavier_init(shape):
    input_nodes = shape[0]
    stddev_xavier = 1. / tf.sqrt(input_nodes/2.)
    return tf.random_normal(shape=shape, stddev=stddev_xavier)


def generate_noise(batch_size):
    return np.random.uniform(-1., 1., size=[batch_size, 100])


def generator(z):
    layer_1_weights = tf.Variable(xavier_init([100, 256]))
    layer_1_biases = tf.Variable(tf.zeros(shape=[256]))

    layer_2_weights = tf.Variable(xavier_init([256, 512]))
    layer_2_biases = tf.Variable(tf.zeros(shape=[512]))

    layer_3_weights = tf.Variable(xavier_init([512, 784]))
    layer_3_biases = tf.Variable(tf.zeros(shape=[784]))

    var_list = [layer_1_weights, layer_2_weights, layer_3_weights, layer_1_biases, layer_2_biases, layer_3_biases]

    layer_1_output = tf.nn.relu(tf.matmul(z, layer_1_weights) + layer_1_biases)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, layer_2_weights) + layer_2_biases)
    layer_3_output = tf.nn.sigmoid(tf.matmul(layer_2_output, layer_3_weights) + layer_3_biases)

    return layer_3_output, var_list


def discriminator(x):
    layer_1_weights = tf.Variable(xavier_init([784, 256]))
    layer_1_biases = tf.Variable(tf.zeros(shape=[256]))

    layer_2_weights = tf.Variable(xavier_init([256, 512]))
    layer_2_biases = tf.Variable(tf.zeros(shape=[512]))

    layer_3_weights = tf.Variable(xavier_init([512, 1]))
    layer_3_biases = tf.Variable(tf.zeros(shape=[1]))

    var_list = [layer_1_weights, layer_2_weights, layer_3_weights, layer_1_biases, layer_2_biases, layer_3_biases]

    layer_1_output = tf.nn.relu(tf.matmul(x, layer_1_weights) + layer_1_biases)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, layer_2_weights) + layer_2_biases)
    layer_3_grid = tf.matmul(layer_2_output, layer_3_weights) + layer_3_biases
    layer_3_output = tf.nn.sigmoid(layer_3_grid)

    return layer_3_output, layer_3_grid, var_list

#main:

dataset_train_features, dataset_train_labels, dataset_test_features, dataset_test_labels = get_dataset()

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])

generator_sample, generator_var_list = generator(Z)
discriminator_real, discriminator_grid_real, discriminator_var_list_real = discriminator(X)
discriminator_fake, discriminator_grid_fake, discriminator_var_list_fake = discriminator(generator_sample)

generator_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_grid_fake, labels=tf.ones_like(discriminator_grid_fake))
                    )
generator_optimizer = tf.train.AdamOptimizer().minimize(generator_loss, var_list=generator_var_list)

discriminator_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_grid_real, labels=tf.ones_like(discriminator_grid_real))
                        ) + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_grid_fake, labels=tf.zeros_like(discriminator_grid_fake))
                        )
discriminator_optimizer = tf.train.AdamOptimizer().minimize(discriminator_loss, var_list=discriminator_var_list_real)   #can use any one of discriminator_var_list_real or discriminator_var_list_fake

sess = tf.Session()
sess.run(tf.global_variables_initializer())

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
temp, _ = mnist.train.next_batch(128)
print(temp.shape)

batch_size = 550
for epoch in range(0, 1000):
    for batch in range(0, dataset_train_features.shape[0]):
        noise = generate_noise(batch_size)
        dataset_train_features_batch = dataset_train_features[batch : batch+batch_size]
        # print('noise.shape:', noise.shape)
        # print('dataset_train_features_batch.shape:', dataset_train_features_batch.shape)

        _, discriminator_loss_temp = sess.run([discriminator_optimizer, discriminator_loss], feed_dict={X: dataset_train_features, Z: noise})
        _, generator_loss_temp = sess.run([generator_optimizer, generator_loss], feed_dict={Z: noise})

        batch = batch + batch_size

        print('Epoch:', epoch, '\tbatch:', batch)
        print('\tgenerator loss:', generator_loss_temp)
        print('\tdiscriminator loss:', discriminator_loss_temp)
