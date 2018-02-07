
import skimage
import skimage.io
import skimage.transform
import numpy as np

import tensorflow as tf

from skimage.transform import resize

import cv2
import scipy.misc
import SODLoader as SDL

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

sdl = SDL.SODLoader('/')

# returns the top1 string
def print_prob(prob):

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    if pred[0] == 0: top1 = 'ADH'
    else: top1 = 'Pure_DCIS'
    print("Top1: %s, Pred: %s, Prob: %s" %(top1, pred, prob))

    return top1


def visualize(image, conv_output, conv_grad, gb_viz, index, display):
    """

    :param image: The specific image in this batch. Must have 2+ channels...
    :param conv_output: The output of the last conv layer
    :param conv_grad: Partial derivatives of the last conv layer wrt class 1 ?
    :param gb_viz: gradient of the cost wrt the images
    :param index: which image in this batch we're working with
    :param display: whether to display images with matplotlib
    :return:
    """

    # Set output and grad
    output = conv_output  # [4,4,256]
    grads_val = conv_grad  # [4,4,256]
    print("grads_val shape: ", grads_val.shape, 'Output shape: ', output.shape)

    # Retreive mean weights of each filter?
    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [256]
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights): cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (FLAGS.network_dims, FLAGS.network_dims), preserve_range=True)

    # Generate image
    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    if display: sdl.display_single_image(img[:, :, 0], False, ('Input Image', img.shape))

    # Generate heatmap
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    if display: sdl.display_single_image(cam_heatmap, False, ('Grad-CAM', cam_heatmap.shape))

    # Generate guided backprop mask
    gb_viz = np.dstack((gb_viz[:, :, 0]))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
    if display: sdl.display_single_image(gb_viz[0], False, ('Guided Backprop', gb_viz.shape))

    # Generate guided grad cam
    gd_gb = np.dstack((gb_viz[0] * cam))
    if display: sdl.display_single_image(gd_gb[0], True, ('Guided Grad-CAM', gd_gb.shape))

    # Save Data
    scipy.misc.imsave((FLAGS.train_dir + FLAGS.RunInfo + ('Visualizations/%s_img.png' % index)), img[:,:,0])
    scipy.misc.imsave((FLAGS.train_dir + FLAGS.RunInfo + ('Visualizations/%s_Grad_Cam.png' % index)), cam_heatmap)
    scipy.misc.imsave((FLAGS.train_dir + FLAGS.RunInfo + ('Visualizations/%s_Guided_Backprop.png' % index)), gb_viz[0])
    scipy.misc.imsave((FLAGS.train_dir + FLAGS.RunInfo + ('Visualizations/%s_Guided_Grad_Cam.png' % index)), gd_gb[0])


def visualize_base(image, conv_output, conv_grad, gb_viz):
    output = conv_output           # [7,7,512]
    grads_val = conv_grad          # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)

    weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32)	# [7,7]


    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (FLAGS.network_dims ,FLAGS.network_dims), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255 *cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # print ('Image Shape ', img.shape, image.shape)
    # imgplot = plt.imshow(img[:,:,1])
    # ax.set_title('Input Image')
    sdl.display_single_image(img[:, :, 0], False, 'Input Image')
    sdl.display_single_image(img[:,:,1], False, 'Input Image 2')

    # fig = plt.figure(figsize=(12, 16))
    # ax = fig.add_subplot(131)
    # imgplot = plt.imshow(cam_heatmap)
    # ax.set_title('Grad-CAM')
    sdl.display_single_image(cam_heatmap, True, 'Grad-CAM')

    # gb_viz = np.dstack((gb_viz[:, :, 0], gb_viz[:, :, 1], gb_viz[:, :, 2]))
    print('GB 1', gb_viz.shape)
    gb_viz = np.dstack((gb_viz[:, :, 0]))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    # ax = fig.add_subplot(132)
    # imgplot = plt.imshow(gb_viz[0])
    # ax.set_title('guided backpropagation')
    sdl.display_single_image(gb_viz[0], False, 'Guided Backprop')


    #gd_gb = np.dstack((gb_viz[:, :, 0] * cam, gb_viz[:, :, 1] * cam, gb_viz[:, :, 2] * cam))
    gd_gb = np.dstack((gb_viz[0] * cam))
    print ('gd_gb shape: %s, cam shape: %s, gb_viz shape: %s' %(gd_gb.shape, cam.shape, gb_viz.shape))
    # ax = fig.add_subplot(133)
    # imgplot = plt.imshow(gd_gb[0])
    # ax.set_title('guided Grad-CAM')
    sdl.display_single_image(gd_gb[0], True, 'Guided Grad-CAM')

    #plt.show()
