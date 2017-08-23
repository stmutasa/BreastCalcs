""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import os
import time                                 # to retreive current time
import numpy as np

import BreastCalcMatrix as BreastMatrix
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
# tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('num_epochs', 1200, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', '4', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 128, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('cross_validations', 5, """Save this number of buffers for cross validation""")

# >5k example lesions total
tf.app.flags.DEFINE_integer('epoch_size', 186, """How many images were loaded""")
tf.app.flags.DEFINE_integer('print_interval', 6, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 59, """How many STEPS to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")

# Regularizers
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.998, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('loss_factor', 1.46, """Penalty for missing a class is this times more severe""")

# Hyperparameters to control the learning rate
tf.app.flags.DEFINE_float('learning_rate', 3e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")


# Define a custom training class
def train():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default():

        # Get a dictionary of our images, id's, and labels here
        images, validation = BreastMatrix.inputs(skip=True)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss = BreastMatrix.forward_pass(images['image'], phase_train1=True)

        # Labels
        labels = images['label2']

        # Calculate the objective function loss
        SCE_loss = BreastMatrix.total_loss(logits, labels)

        # Add in L2 Regularization
        loss = tf.add(SCE_loss, l2loss, name='loss')

        # Build the backprop graph to train the model with one batch and update the parameters (Backward pass)
        train_op = BreastMatrix.backward_pass(loss)

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(0.998)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=6)

        # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
        with tf.Session() as mon_sess:

            # Initialize the variables
            mon_sess.run(var_init)

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter('training/Train', mon_sess.graph)

            # Initialize the thread coordinator
            coord = tf.train.Coordinator()

            # Start the queue runners
            threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

            # Initialize the step counter
            step = 0

            # Accuracy tracker
            accuracy = 0

            # Set the max step count
            max_steps = (FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs

            try:
                while step <= max_steps:

                    # Start the timer
                    start_time = time.time()

                    # Run an iteration
                    mon_sess.run(train_op)

                    # Calculate Duration
                    duration = time.time() - start_time


                    if step % FLAGS.print_interval == 0:  # This statement will print loss, step and other stuff

                        # Load some metrics
                        lbl1, logtz, loss1, loss2, tot = mon_sess.run([labels, logits, SCE_loss, l2loss, loss])

                        # Calculate examples per second
                        eg_s = FLAGS.batch_size / duration

                        # use numpy to print only the first sig fig
                        np.set_printoptions(precision=2)

                        # Print the data
                        print ('-'*70)
                        print('Step %d, L2 Loss: = %.3f (%.1f eg/s;), Total Loss: %.3f SCE: %.4f'
                              % (step, loss2, eg_s, tot, loss1))

                        # Retreive and print the labels and logits
                        print('Labels: %s' % np.squeeze(lbl1.astype(np.int8))[:20])
                        print('Logits: %s' % np.squeeze(np.argmax(logtz.astype(np.float), axis=1))[:20])

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries)

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, step)


                    if step % FLAGS.checkpoint_steps == 0:

                        print('-' * 70)
                        print ('Saving...')
                        Epoch = int((step*FLAGS.batch_size)/FLAGS.epoch_size)

                        # Define the filename
                        file = ('Epoch_%s' % Epoch)

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join('training/', file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)

                        # Sleep an amount of time to let testing catch up
                        time.sleep(7)

                    # Increment step
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done with Training - Epoch limit reached')

            finally:

                # Save the final checkpoint
                print(" ---------------- SAVING FINAL CHECKPOINT ------------------ ")
                saver.save(mon_sess, 'training/CheckpointFinal')

                # Stop threads when done
                coord.request_stop()

                # Wait for threads to finish before closing session
                coord.join(threads, stop_grace_period_secs=60)

                # Shut down the session
                mon_sess.close()


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists('training/'):
        tf.gfile.DeleteRecursively('training/')
    tf.gfile.MakeDirs('training/')
    train()

if __name__ == '__main__':
    tf.app.run()