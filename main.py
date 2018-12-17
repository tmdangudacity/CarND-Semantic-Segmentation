#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    img_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep      = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3    = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4    = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7    = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return img_input, keep, layer3, layer4, layer7

print("Testing load_vgg")
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 Convolution from VGG layer 7
    conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                  kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-3))

    # Upscaling by 2
    output = tf.layers.conv2d_transpose(conv_1x1_7, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-3))

    # Adding 1x1 Convolution from VGG layer 4 with the same number of classes
    conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                  kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-3))

    output = tf.add(output, conv_1x1_4)

    # Upscaling by 2
    output = tf.layers.conv2d_transpose(output, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-3))

    # Adding 1x1 Convolution from VGG layer 3 with the same number of classes
    conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                  kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-3))

    output = tf.add(output, conv_1x1_3)

    # Upscaling by 8
    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides= (8, 8), padding= 'same',
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    return output

print("Testing layers function")
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    #Reshaping from 4D to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    reshaped_label = tf.reshape(correct_label, (-1,num_classes))

    #Loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=reshaped_label))

    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)

    #Training operation
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

print("Testing optimize function")
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    keep_prob_value = 0.5
    learning_rate_value = 0.00085

    sess.run(tf.global_variables_initializer())

    print("Start training, Epochs: ", epochs, ", Batch size: ", batch_size, ", Keep prob: ", keep_prob_value, ", Learning rate: ", learning_rate_value)

    start_training_time = time.time()

    for epoch in range(epochs):
        start_epoch_time = time.time()
        batch_loss = []
        for image, label in get_batches_fn(batch_size):
            _ , loss = sess.run([train_op, cross_entropy_loss],
                                 feed_dict={ input_image: image,
                                             correct_label: label,
                                             keep_prob: keep_prob_value,
                                             learning_rate: learning_rate_value
                                           }
                               )

            batch_loss.append(loss)

        print("Epoch, ", (epoch + 1), ", Run time: ", (time.time() - start_epoch_time),
              ", Mean loss, ", np.mean(batch_loss), ", Min loss, ", np.amin(batch_loss),
              ", Max loss, ", np.amax(batch_loss), ", Stdev, ", np.std(batch_loss))


    print("Training completed in ", (time.time() - start_training_time))

print("Testing training function")
tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'

    print("Testing kitti_dataset")
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Hyperparameters
    EPOCHS     = 24
    BATCH_SIZE = 5

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Building NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        #print("Loading VGG")
        input_image, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        #print("Building Decoder layers")
        output_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)

        #print("Preparing Logits, Training Operation and Loss function")
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        # Training NN using the train_nn function
        #print("Training NN")
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        #print("Saving Inference samples")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
