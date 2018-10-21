""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time, os, glob

import BreastCalcMatrix as BreastMatrix
import numpy as np
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import tensorflow.contrib.slim as slim

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('epoch_size', 246, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 246, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")

# Test sizes: 30 = 13/16, 60 = 16/13 90 = 14/15 120 = 14/15
tf.app.flags.DEFINE_string('test_files', 'Old', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 256, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 128, """the dimensions fed into the network""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.998, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'New3/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

sdl = SDL.SODLoader(data_root='data/')


def eval():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Get a dictionary of our images, id's, and labels here
        _, validation = BreastMatrix.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss, _ = BreastMatrix.forward_pass(validation['data'], phase_train=phase_train)

        # To retreive labels
        labels = validation['label2']

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
        saver = tf.train.Saver(var_restore, max_to_keep=2)

        # Performance trackers
        best_MAE, best_epoch = 0, 0

        while True:

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
                    restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                    # Restore the graph
                    restorer.restore(mon_sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Define tester class instance
                sdt = SDT.SODTester(True, False)
                print ('Max Steps: ', max_steps)

                # Use slim to handle queues:
                with slim.queues.QueueRunners(mon_sess):

                    for step in range(max_steps):

                        # Load some metrics for testing
                        lbl, logtz, unique = mon_sess.run([labels, logits, validation['patient']], feed_dict={phase_train: False})

                        # Combine predictions
                        data, lbl, logtz = sdt.combine_predictions(lbl, logtz, unique, FLAGS.batch_size)

                        # Increment step
                        step += 1

                    # retreive metrics
                    sdt.calculate_metrics(logtz, lbl, 1, step)
                    sdt.retreive_metrics_classification(Epoch, True)
                    print ('------ Current Best AUC: %.4f (Epoch: %s) --------' %(best_MAE, best_epoch))

                    # Run a session to retrieve our summaries
                    summary = mon_sess.run(all_summaries, feed_dict={phase_train: False})

                    # Retreive step
                    try: step_retreived = int(int(Epoch) * 1.875)
                    except: step_retreived = 2625

                    # Add the summaries to the protobuf for Tensorboard
                    summary_writer.add_summary(summary, step_retreived)

                    # Lets save if they win accuracy
                    if sdt.AUC >= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filename
                        file = ('Epoch_%s_AUC_%0.3f' % (Epoch, sdt.AUC))

                        # Define the filenames
                        checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo, ('Epoch_%s_AUC_%0.3f' % (Epoch, sdt.AUC)))
                        csv_file = os.path.join('testing/' + FLAGS.RunInfo, ('%s_E_%s_AUC_%0.2f.csv' % (FLAGS.RunInfo[:-1], Epoch, sdt.AUC)))

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)
                        sdl.save_Dict_CSV(data, csv_file)

                        # Save a new best MAE
                        best_MAE = sdt.AUC
                        best_epoch = Epoch

            # Break if this is the final checkpoint
            if int(Epoch) > 900: break

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            filecheck = glob.glob(FLAGS.train_dir + FLAGS.RunInfo + '*')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                time.sleep(int(FLAGS.epoch_size * 0.05))

                # Recheck the folder for changes
                newfilec = glob.glob(FLAGS.train_dir + FLAGS.RunInfo + '*')


def main(argv=None):
    time.sleep(0)
    eval()


if __name__ == '__main__':
    tf.app.run()
