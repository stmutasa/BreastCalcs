""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time

import BreastCalcMatrix as BreastMatrix
import tensorflow as tf
import SODTester as SDT
import tensorflow.contrib.slim as slim


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scikitplot as skplt

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('epoch_size', 60, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 60, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', '60', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 256, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 32, """the dimensions fed into the network""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.998, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'testing/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Val1/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")


def eval():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default():

        # Get a dictionary of our images, id's, and labels here
        _, validation = BreastMatrix.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss = BreastMatrix.forward_pass(validation['data'], phase_train=phase_train)

        # To retreive labels
        labels = validation['label']

        # Just to print out the loss
        _ = BreastMatrix.total_loss(logits, labels)

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize the handle to the summary writer in our training directory
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir + 'Test_' + FLAGS.RunInfo)

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Retreive the checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)

            # Initialize the variables
            mon_sess.run(var_init)

            time.sleep(5)

            if ckpt and ckpt.model_checkpoint_path:

                # Restore the learned variables
                print ('Restoring model: ', ckpt.model_checkpoint_path)
                restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                # Restore the graph
                restorer.restore(mon_sess, ckpt.model_checkpoint_path)

                # Extract the epoch
                Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

            # Initialize the step counter
            step = 0

            # Set the max step count
            max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

            # Define tester class instance
            sdt = SDT.SODTester(True, False)

            # Use slim to handle queues:
            with slim.queues.QueueRunners(mon_sess):

                for step in range(max_steps):

                    # Load some metrics for testing
                    lbl1, logtz, serz = mon_sess.run([labels, logits, validation['series']], feed_dict={phase_train: False})
                    examples, labba, logga = sdt.combine_predictions(lbl1, logtz, serz, FLAGS.batch_size)

                    # Increment step
                    print ('Step: ', step)
                    step += 1

                # Plot the ROC
                skplt.metrics.plot_roc_curve(labba, sdt.calc_softmax(logga), title='ROC Curve', curves=('macro'))
                plt.show()


def main(argv=None):
    eval()


if __name__ == '__main__':
    tf.app.run()
