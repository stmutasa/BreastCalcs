""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import os
import time                                 # to retreive current time
import numpy as np

import BreastCalcMatrix as BreastMatrix
import tensorflow as tf
import SODTester as SDT
import tensorflow.contrib.slim as slim

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('num_epochs', 605, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', 'Test', """Files for testing have this name""")

tf.app.flags.DEFINE_integer('box_dims', 256, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 128, """the dimensions fed into the network""")

# inv Epoch sizes: 0: 264, 1: 254, 2:266, 3:260, 4:268
tf.app.flags.DEFINE_integer('epoch_size', 697, """How many images were loaded""")
tf.app.flags.DEFINE_integer('print_interval', 5, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 15, """How many epochs to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")

# Regularizers
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.998, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")

# Hyperparameters to control the learning rate
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Shuffled_New/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

# Define a custom training class
def train():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default():

        # Get a dictionary of our images, id's, and labels here
        images, _ = BreastMatrix.inputs(skip=True, data_type='INV')

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss, _ = BreastMatrix.forward_pass(images['data'], phase_train=phase_train)

        # Labels
        labels = images['label2']

        # Calculate the objective function loss
        SCE_loss = BreastMatrix.total_loss(logits, labels)

        # Add in L2 Regularization
        loss = tf.add(SCE_loss, l2loss, name='loss')

        # Update the moving average batch norm ops
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Retreive the training operation with the applied gradients
        with tf.control_dependencies(extra_update_ops): train_op = BreastMatrix.backward_pass(loss)

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(0.998)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Tester instance
        sdt = SDT.SODTester(True, False)

        # Set the intervals
        max_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs)
        print_interval = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.print_interval)+1
        checkpoint_interval = int((FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.checkpoint_interval)+1
        print('Max Steps: %s, Print Interval: %s, Checkpoint: %s' % (max_steps, print_interval, checkpoint_interval))

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Initialize the variables
            mon_sess.run(var_init)

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir + FLAGS.RunInfo, mon_sess.graph)

            # Initialize the trackers
            accuracy, step = 0, 0

            # Use slim to handle queues:
            with slim.queues.QueueRunners(mon_sess):

                for step in range(max_steps):

                    # Run an iteration
                    start_time = time.time()
                    mon_sess.run(train_op, feed_dict={phase_train: True})
                    duration = time.time() - start_time
                    Epoch = int((step * FLAGS.batch_size) / FLAGS.epoch_size)

                    if step % print_interval == 0:  # This statement will print loss, step and other stuff

                        # Load some metrics
                        lbl1, logtz, loss1, loss2, tot = mon_sess.run([labels, logits, SCE_loss, l2loss, loss], feed_dict={phase_train: True})

                        # Calculate examples per second
                        eg_s = FLAGS.batch_size / duration

                        # use numpy to print only the first sig fig
                        np.set_printoptions(precision=2)

                        # Print the data
                        print ('-'*70)
                        print('Epoch %d, L2 Loss: = %.3f (%.1f eg/s;), Total Loss: %.3f SCE: %.4f' % (Epoch, loss2, eg_s, tot, loss1))

                        # Retreive and print the labels and logits
                        print('Labels: %s' % np.squeeze(lbl1.astype(np.int8))[:20])
                        print('Logits: %s' % np.squeeze(np.argmax(logtz.astype(np.float), axis=1))[:20])

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries, feed_dict={phase_train: True})

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, step)


                    if (step % checkpoint_interval == 0) or (int(Epoch)==300):

                        print('-' * 70, '\n %s: Saving... Epoch: %s, GPU: %s, File:%s' % (time.time(), Epoch, FLAGS.GPU, FLAGS.RunInfo[:-1]))

                        # Define the filename
                        file = ('Epoch_%s' % Epoch)

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join(FLAGS.train_dir + FLAGS.RunInfo, file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)

                        # Sleep an amount of time to let testing catch up
                        if Epoch > 1: time.sleep(60)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.RunInfo)
    tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.RunInfo)
    train()

if __name__ == '__main__':
    tf.app.run()