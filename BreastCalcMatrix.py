# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import tensorflow as tf
import Input
import SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")

# Retreive helper function object
sdn = SDN.SODMatrix()


def forward_pass(images, phase_train=True):

    """
    Performs the forward pass
    :param images: Input images
    :param phase_train: Training or testign phase
    :return:
    """

    # Images 0 is the scaled version, 1 is the regular
    img1, img2 = images[:, :, :, 1], images[:, :, :, 0]
    print (images, img1, img2)

    # First layer is conv
    conv = sdn.convolution('Conv1', tf.expand_dims(img2, -1), 3, 8, 1, phase_train=phase_train)
    print('Input Images: ', images)

    # Residual blocks
    conv = sdn.residual_layer('Residual1', conv, 3, 16, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual2', conv, 3, 16, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual3', conv, 3, 32, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual4', conv, 3, 32, 1, phase_train=phase_train)
    print('End Residual: ', conv)

    # Inception layers start 4x4
    conv = sdn.inception_layer('Inception5', conv, 64, S=2, phase_train=phase_train)
    conv = sdn.inception_layer('Inception6', conv, 64, S=1, phase_train=phase_train)
    conv = sdn.inception_layer('Inception7', conv, 64, S=1, phase_train=phase_train)
    conv = sdn.inception_layer('Inception8', conv, 64, S=1, phase_train=phase_train)
    print('End Inception', conv)

    # Linear layers
    fc = sdn.fc7_layer('FC', conv, 16, True, phase_train, FLAGS.dropout_factor, BN=True, override=3)
    fc = sdn.linear_layer('Linear', fc, 8, False, phase_train, BN=True)
    Logits = sdn.linear_layer('Output', fc, FLAGS.num_classes, False, phase_train, BN=False, relu=False, add_bias=False)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass_old(images, phase_train=True):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """


    # Images 0 is the scaled version, 1 is the regular
    img1, img2 = images[:, :, :, 1], images[:, :, :, 0]

    # The first layer.
    conv1 = sdn.convolution('Conv1', tf.expand_dims(img2, -1), 5, 16, 1, phase_train=phase_train, BN=False, relu=False)

    # The second  layer
    conv2a = sdn.residual_layer('Res0', conv1, 3, 16, 1, phase_train=phase_train, BN=False, relu=False)
    conv2 = sdn.residual_layer('Res1', conv2a, 3, 16, 1, phase_train=phase_train, BN=False, relu=False, DSC=False)

    # The third layer
    conv3 = sdn.residual_layer('Res2', conv2, 3, 16, 1, phase_train=phase_train, BN=True, relu=True, DSC=False)

    # Insert inception/residual layer here.
    conv4 = sdn.inception_layer('Inception1', conv3, 8, 2, phase_train=phase_train, BN=False, relu=False)

    # The 4th layer
    conv5 = sdn.residual_layer('Res3', conv4, 3, 32, 1, phase_train=phase_train, BN=False, relu=False, DSC=False)

    # 5th layer
    conv6 = sdn.residual_layer('Res4', conv5, 3, 32, 1, phase_train=phase_train, BN=True, relu=True, DSC=True)

    # The Fc7 layer
    fc7 = sdn.fc7_layer('FC7', conv6, 8, True, phase_train, FLAGS.dropout_factor, BN=False, override=3, pad='VALID')

    # the softmax layer
    Logits = sdn.linear_layer('Softmax', fc7, FLAGS.num_classes)

    # Retreive the weights collection
    weights = tf.get_collection('weights')

    # Sum the losses
    L2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in weights]), FLAGS.l2_gamma)

    # Add it to the collection
    tf.add_to_collection('losses', L2_loss)

    # Activation summary
    tf.summary.scalar('L2_Loss', L2_loss)
    print(conv4, conv6)

    return Logits, L2_loss


def total_loss(logits, labels):

    """
    Add loss to the trainable variables and a summary
        Args:
            logits: logits from the forward pass
            labels the true input labels, a 1-D tensor with 1 value for each image in the batch
        Returns:
            Your loss value as a Tensor (float)
    """

    # Calculate the AUC
    AUC = tf.contrib.metrics.streaming_auc(tf.argmax(logits, 1), labels)

    # Apply cost sensitive loss here
    if FLAGS.loss_factor != 1.0:

        # Make a nodule sensitive binary for values > 1 in this case
        lesion_mask = tf.cast(labels >= 1, tf.float32)

        # Now multiply this mask by scaling factor then add back to labels. Add 1 to prevent 0 loss
        lesion_mask = tf.add(tf.multiply(lesion_mask, FLAGS.loss_factor), 1)

    # Change labels to one hot
    labels = tf.one_hot(tf.cast(labels, tf.uint8), depth=FLAGS.num_classes, dtype=tf.uint8)

    # Calculate  loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(labels), logits=logits)

    # Apply cost sensitive loss here
    if FLAGS.loss_factor != 1.0: loss = tf.multiply(loss, tf.squeeze(lesion_mask))

    # Reduce to scalar
    loss = tf.reduce_mean(loss)

    # Output the losses
    tf.summary.scalar('Cross Entropy', loss)
    tf.summary.scalar('AUC', AUC[1])

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


def backward_pass(total_loss):

    """ This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training"""

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=1e-8)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([train_op, variable_averages_op]):  dummy_op = tf.no_op(name='train')  #

    return dummy_op


def inputs(skip=False):
    """ This function loads our raw inputs, processes them to a protobuffer that is then saved and
        loads the protobuffer into a batch of tensors """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not skip:

        # Part 1: Load the raw images and save to protobuf
        Input.pre_process(FLAGS.box_dims, FLAGS.cross_validations)

    else:
        print('-------------------------Previously saved records found! Loading...')

    # Part 2: Load the protobuff  -----------------------------
    print('----------------------------------------Loading Protobuff...')
    train = Input.load_protobuf()
    valid = Input.load_validation_set()

    # Part 3: Create randomized batches
    print('----------------------------------Creating and randomizing batches...')
    train = Input.sdl.randomize_batches(train, FLAGS.batch_size)
    valid = Input.sdl.val_batches(valid, FLAGS.batch_size)

    return train, valid