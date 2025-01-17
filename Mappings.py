""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import BreastCalcMatrix as BreastMatrix
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scikitplot as skplt

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('epoch_size', 60, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 4, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', '60', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 256, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 128, """the dimensions fed into the network""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.998, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'testing/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Val1_128/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")


def eval():

    # Create tensorflow graph for evaluation
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):

            # Get a dictionary of our images, id's, and labels here
            _, validation = BreastMatrix.inputs(skip=True)

            # Define phase of training
            phase_train = tf.placeholder(tf.bool)

            # To retreive labels
            labels = tf.one_hot(tf.cast(validation['label'], tf.uint8), 2)
            images = validation['data']

            # Build a graph that computes the prediction from the inference model (Forward pass)
            logits, _, conv = BreastMatrix.forward_pass(validation['data'], phase_train=phase_train)

            # Calculate the cost from the normalized logits
            prob = tf.nn.softmax(logits)
            cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(prob)), axis=1)

            # gradient for partial linearization. We only care about target visualization class.
            y_c = tf.reduce_sum(tf.multiply(logits, labels), axis=1)

            # Get last convolutional layer and gradients. tf.gradients outputs derivatives of y_c wrt x in xs
            target_conv_layer = conv
            target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

            # Guided backpropagtion back to input layer
            gb_grad = tf.gradients(cost, images)[0]

            # # Restore moving average of the variables
            var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
            var_restore = var_ema.variables_to_restore()

            # Initialize the saver
            saver = tf.train.Saver(var_restore, max_to_keep=3)

    with tf.Session(graph=eval_graph) as sess:

        # Initialize the variables
        sess.run(var_init)

        # Retreive the checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)
        if ckpt and ckpt.model_checkpoint_path:

            # Restore the learned variables
            restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

            # Restore the graph
            restorer.restore(sess, ckpt.model_checkpoint_path)
            print('Graph Restored: ', ckpt.model_checkpoint_path)

        # Start queues
        with slim.queues.QueueRunners(sess):

            # Retreive the images and softmax predictions
            prob, batch_img = sess.run([prob, images], feed_dict={phase_train: False})
            print ('Softmax: ', prob[0])

            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run(
                [gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={phase_train: False})

            # Create save dir
            tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.RunInfo + 'Visualizations/')

            for i in range(FLAGS.batch_size):
                output_class = utils.print_prob(prob[i])
                utils.visualize(batch_img[i], target_conv_layer_value[i], target_conv_layer_grad_value[i],
                                gb_grad_value[i], i, True)


def main(argv=None):
    eval()


if __name__ == '__main__':
    tf.app.run()
