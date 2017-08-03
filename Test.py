""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time, os, glob

import BreastCalcMatrix as BreastMatrix
import numpy as np
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('epoch_size', 47, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 2, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', '4', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 128, """dimensions of the input pictures""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")

# Define a custom training class
def test():


    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Get a dictionary of our images, id's, and labels here
        images, validation = BreastMatrix.inputs(skip=True)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss = BreastMatrix.forward_pass(validation['image'], phase_train1=False)

        # To retreive labels
        labels = validation['label2']

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(0.998)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        best_MAE = 20

        while True:

            with tf.Session() as mon_sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state('training/')

                # Initialize the variables
                mon_sess.run(var_init)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the learned variables
                    restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                    # Restore the graph
                    restorer.restore(mon_sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('Epoch')[-1]

                # Initialize the thread coordinator
                coord = tf.train.Coordinator()

                # Start the queue runners
                threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

                # Initialize the step counter
                step = 0

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Running values for accuracy calculation
                right = 0
                total = 0

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        lbl1, logtz = mon_sess.run([labels, logits])

                        # Retreive and print the labels and logits
                        lbl = np.squeeze(lbl1.astype(np.int8))
                        logitz = np.squeeze(np.argmax(logtz.astype(np.float), axis=1))

                        # Print Summary
                        print('-' * 70)
                        print('Patient %s/%s Class: %s' %(step, max_steps, lbl))
                        print('Patient %s/%s Preds: %s' %(step, max_steps, logitz))

                        # retreive accuracy
                        for z in range(len(lbl)):
                            if lbl[z] == logitz[z]: right += 1
                        total += len(lbl)

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final MAE and ACC
                    accuracy = 100*right/total

                    # Print the final accuracies and MAE
                    print('-' * 70)
                    print(
                        '--- EPOCH: %s ACCURACY: %.2f %% (Old Best: %.1f) ---'
                        % (Epoch, accuracy, best_MAE))

                    # Lets save runs below 0.8
                    if accuracy >= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filename
                        file = ('Epoch_%s_MAE_%0.3f' % (Epoch, accuracy))

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join('testing/', file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)

                        # Save a new best MAE
                        best_MAE = accuracy

                    # Stop threads when done
                    coord.request_stop()

                    # Wait for threads to finish before closing session
                    coord.join(threads, stop_grace_period_secs=20)

                    # Shut down the session
                    mon_sess.close()

            # Break if this is the final checkpoint
            if 'Final' in Epoch: break

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            filecheck = glob.glob('training/' + '*')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                time.sleep(int(FLAGS.epoch_size * 0.05))

                # Recheck the folder for changes
                newfilec = glob.glob('training/' + '*')


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists('testing/'):
        tf.gfile.DeleteRecursively('testing/')
    tf.gfile.MakeDirs('testing/')
    test()


if __name__ == '__main__':
    tf.app.run()
