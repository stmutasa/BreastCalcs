
import skimage
import skimage.io
import skimage.transform
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg

import tensorflow as tf

from skimage import io
from skimage.transform import resize

import cv2
import scipy.misc
import SODLoader as SDL

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

sdl = SDL.SODLoader('/')

# synset = [l.strip() for l in open('synset.txt').readlines()]

def resnet_preprocess(resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """
    channel_means = tf.constant([123.68, 116.779, 103.939],
                                dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    return resized_inputs - channel_means


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, normalize=True):
    """
    args:
        normalize: set True to get pixel value of 0~1
    """
    # load image
    img = skimage.io.imread(path)
    if normalize:
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()


    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224), preserve_range=True) # do not normalize at transform.
    return resized_img

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    if pred[0] == 0: top1 = 'ADH'
    else: top1 = 'Pure_DCIS'
    print("Top1: ", top1, pred)

    return top1


def visualize(image, conv_output, conv_grad, gb_viz, index, display):
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)
    display = []

    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights): cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (FLAGS.network_dims, FLAGS.network_dims), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    if display: sdl.display_single_image(img[:, :, 0], False, ('Input Image', img.shape))
    display.append(img[:, :, 0])

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    if display: sdl.display_single_image(cam_heatmap, True, ('Grad-CAM', cam_heatmap.shape))
    display.append(cam_heatmap)

    gb_viz = np.dstack((gb_viz[:, :, 0]))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
    if display: sdl.display_single_image(gb_viz[0], False, ('Guided Backprop', gb_viz.shape))
    display.append(gb_viz[0])

    gd_gb = np.dstack((gb_viz[0] * cam))
    if display: sdl.display_single_image(gd_gb[0], True, ('Guided Grad-CAM', gd_gb.shape))
    display.append(gd_gb)

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
