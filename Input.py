
import os, cv2, glob
import numpy as np
import tensorflow as tf
from random import shuffle
import matplotlib.pyplot as plt
from pathlib import Path

import SODLoader as SDL
import SOD_Display as SDD

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets'
data_dir = home_dir + '/BreastData/Mammo/Calcs'

sdl = SDL.SODLoader(data_root=data_dir)
sdd = SDD.SOD_Display()


def pre_process_DCISvsInv(box_dims):

    # Retreive filenames data/raw/pure/Patient 41/2/ser32770img00005.dcm
    filenames = []

    # Search each folder for the files
    pure_files = [x[0] for x in os.walk(data_dir + '/pure/')]
    inv_files = [x[0] for x in os.walk(data_dir + '/invasive/')]
    micro_files = [x[0] for x in os.walk(data_dir + '/microinvasion/')]

    # Append each file into filenames
    for z in range (len(pure_files)):
        if len(pure_files[z].split('/')) != 10: continue
        filenames.append(pure_files[z])

    for z in range(len(inv_files)):
        if len(inv_files[z].split('/')) != 10: continue
        filenames.append(inv_files[z])

    for z in range(len(micro_files)):
        if len(micro_files[z].split('/')) != 10: continue
        filenames.append(micro_files[z])

    # Shuffle filenames to create semi even protobufs
    shuffle(filenames)
    print(len(filenames), 'Files found: ', filenames)

    # Global variables
    lab1, lab2, index, filesave, pt = 0, 0, 0, 0, 0
    display, unique_ID, data = [], [], {}

    # Double the box size
    box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    # Single label files
    singles = ['pure/Patient 55', 'invasive/Patient 36', 'invasive/Patient 50']

    for file in filenames:

        #  Skip non patient folders
        if 'Patient' not in file: continue

        # Retreive the patient name
        patient = int(file.split('/')[-1].split(' ')[-1])

        # Retreive the type of invasion
        invasion = file.split('/')[-2]

        # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
        label_file = (file + '/%s-3.nii.gz' %patient)
        label_file2 = (file + '/%s-4.nii.gz' %patient)

        # Load the files
        image_file = sdl.retreive_filelist('dcm', False, path = (file + '/3'))
        image_file2 = sdl.retreive_filelist('dcm', False, path=(file + '/4'))

        # Load the 1st dicom
        image, acc, dims, window, _ = sdl.load_DICOM_2D(image_file2[0])

        # Load the label
        segments = np.squeeze(sdl.load_NIFTY(label_file2))

        # Flip x and Y
        segments = np.squeeze(sdl.reshape_NHWC(segments, False))

        # If this is one of the exceptions with only one series...
        if (invasion + '/' + file.split('/')[-1]) in singles:

            # Just copy it...
            image2, acc2, dims2, window2 = np.copy(image), acc, dims, window
            segments2 = np.copy(segments)

        else:

            # Load the 2nd dicom and labels
            image2, acc2, dims2, window2, _ = sdl.load_DICOM_2D(image_file[0])
            segments2 = np.squeeze(sdl.load_NIFTY(label_file))
            segments2 = np.squeeze(sdl.reshape_NHWC(segments2, False))

        # Assign labels
        if 'invasive' in invasion: label = 0
        elif 'micro' in invasion:  label = 1
        else: label = 2

        # Second labels
        if label < 2:
            label2 = 0
            lab1 += 2
        else:
            label2 = 1
            lab2 += 2

        # Retreive the center of the largest label
        blob, cn = sdl.largest_blob(segments)
        blob2, cn2 = sdl.largest_blob(segments2)

        # Calculate a factor to make our image size, call this "radius"
        radius, radius2 = np.sum(blob)**(1/3)*10, np.sum(blob2) ** (1 / 3) * 10

        # Normalize
        image = sdl.normalize_Mammo_histogram(image)
        image2 = sdl.normalize_Mammo_histogram(image2)

        # Make a 2dbox at the center of the label with size "radius" if scaled and 256 if not
        box_scaled, _ = sdl.generate_box(image, cn, int(radius)*2, dim3d=False)
        box_wide, _ = sdl.generate_box(image, cn, 512, dim3d=False)
        box_scaled2, _ = sdl.generate_box(image2, cn2, int(radius2) * 2, dim3d=False)
        box_wide2, _ = sdl.generate_box(image2, cn2, 512, dim3d=False)

        # Resize image
        box_scaled = cv2.resize(box_scaled, (box_dims, box_dims))
        box_wide = cv2.resize(box_wide, (box_dims, box_dims))
        box_scaled2 = cv2.resize(box_scaled2, (box_dims, box_dims))
        box_wide2 = cv2.resize(box_wide2, (box_dims, box_dims))

        # Save the boxes in one
        box = np.zeros(shape=(box_dims, box_dims, 2)).astype(np.float32)
        box[:, :, 0] = box_scaled
        box[:, :, 1] = box_wide
        box2 = np.zeros(shape=(box_dims, box_dims, 2)).astype(np.float32)
        box2[:, :, 0] = box_scaled2
        box2[:, :, 1] = box_wide2

        # Unique patient number
        accno = invasion + '_' + str(patient)

        # Set how many to iterate through
        # num_examples = int(2 - label2)
        num_examples = 1

        # Generate X examples
        for i in range (num_examples):

            # Generate the dictionary
            data[index] = {'data': box, 'label': label, 'label2': label2, 'patient': int(patient), 'accno': accno,
                    'series': file, 'invasion': invasion, 'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

            # Append counter
            index += 1

            # Generate the dictionary
            data[index] = {'data': box2, 'label': label, 'label2': label2, 'patient': int(patient), 'accno': accno,
                           'series': file, 'invasion': invasion, 'box_x': cn2[0], 'box_y': cn2[1], 'box_size': radius2}

            # Append counter
            index += 1

        # Finish this example of this patient
        pt +=1

        # Save the protobufs
        if pt % 25 == 0: print ('%s patients saved' %pt)

    # Save last protobuf for stragglers
    print('Creating a protocol buffer... %s examples from %s patients loaded, DCIS %s, Invasive: %s' % (len(data), pt, lab1, lab2))
    sdl.save_dict_filetypes(data[index - 1])
    sdl.save_tfrecords(data, 1, file_root=('data/DCIS_vs_Inv_Old' + str(filesave)))

    print('Complete... %s examples from %s patients loaded' % (index, pt))


def pre_process_INV_new(box_dims):

    # Retreive the files
    filenames = []
    new_files = sdl.retreive_filelist('nii.gz', True, (data_dir + '/New/'))

    # Append each file into filenames
    for z in new_files:
        if 'label' not in z: continue
        if 'ADH' in z: continue
        filenames.append(z)

    # Shuffle filenames to create semi even protobufs
    shuffle(filenames)

    # Global variables
    lab, index, laba, pt = [0, 0], 0, [0, 0], []
    display, unique_ID, data = [], [], {}

    # Double the box size
    box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    for file in filenames:

        # Retreive the patient name
        try: patient = int(file.split('/')[-1].split(' ')[0])
        except:
            print ('Cant retrieve patient ', file)
            continue

        # Retreive the type of invasion
        invasion = file.split('/')[-3]

        # Retreive filenames of the labels
        img_file = (file[:-13] + '.nii.gz')

        # retreive the projection
        projection = img_file.split('/')[-1].split(' ')[2].split('.')[0]

        # Load the image and label
        try: image = np.squeeze(sdl.load_NIFTY(img_file))
        except:
            print ('likely doubled segments ', file)
            continue
        segments = np.squeeze(sdl.load_NIFTY(file))

        # Assign labels
        if 'Invasive' in invasion: label = 0
        elif 'Micro' in invasion:  label = 1
        else: label = 2 # DCIS

        # Second labels
        if label < 2: label2 = 0
        else: label2 = 1
        lab[label2] += 1

        # Retreive the center of the largest label
        try: blob, cn = sdl.largest_blob(segments)
        except:
            print ('Retreive blob failed ', file)
            continue

        # Calculate a factor to make our image size, call this "radius"
        radius = np.sum(blob)**(1/3)*10

        # Normalize the mammo
        image = sdl.normalize_Mammo_histogram(image)

        # Make a 2dbox at the center of the label with size "radius" if scaled and 256 if not
        box_scaled, _ = sdl.generate_box(image, cn, int(radius)*2, dim3d=False)
        box_wide, _ = sdl.generate_box(image, cn, 512, dim3d=False)

        # Resize image
        box_scaled = cv2.resize(box_scaled, (box_dims, box_dims))
        box_wide = cv2.resize(box_wide, (box_dims, box_dims))

        # Save the boxes in one
        box = np.zeros(shape=(box_dims, box_dims, 2)).astype(np.float32)
        box[:, :, 0] = box_scaled
        box[:, :, 1] = box_wide

        # Generate the dictionary
        data[index] = {'data': box, 'label': label, 'label2': label2, 'patient': int(patient), 'accno': img_file.split('/')[-1],
                'series': projection, 'invasion': invasion, 'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

        # Append counters
        index += 1
        if patient not in pt:
            pt.append(patient)
            laba[label2]+=1

        # Display progress
        if index % 50 == 0: print('%s examples loaded in %s patients' %(index, len(pt)))

    # Save last protobuf for stragglers
    print('Creating a protocol buffer... %s examples loaded, %s patients, DCIS [%s] %s, Invasive: [%s] %s'
          % (len(data), len(pt), lab[0], laba[0], lab[1], laba[1]))
    sdl.save_dict_filetypes(data[index - 1])
    sdl.save_tfrecords(data, 1, file_root='data/DCIS_vs_Inv_New')

    print('Complete... %s examples from %s patients loaded' % (index, len(pt)))


def pre_process_adh_vs_pure(box_dims):

    # Retreive filenames data/raw/pure/Patient 41/2/ser32770img00005.dcm
    filenames = []

    # Search each folder for the files
    pure_files = [x[0] for x in os.walk(data_dir + '/pure/')]
    adh_files = [x[0] for x in os.walk(data_dir + '/ADH/')]

    # Append each file into filenames
    for z in range (len(pure_files)):
        if len(pure_files[z].split('/')) != 10: continue
        filenames.append(pure_files[z])

    for z in range(len(adh_files)):
        if len(adh_files[z].split('/')) != 10: continue
        filenames.append(adh_files[z])

    # Shuffle filenames to create semi even protobufs
    shuffle(filenames)
    print(len(filenames), 'Files found: ', filenames)

    # Global variables
    index, filesave, pt, lab1, lab2 = 0, 0, 0, 0, 0
    data, display, unique_ID = {}, [], []

    # Double the box size
    box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    # Single label files
    singles = ['pure/Patient 55', 'invasive/Patient 36', 'invasive/Patient 50']

    for file in filenames:

        #  Skip non patient folders
        if 'Patient' not in file: continue

        # Different code for loading ADH files: '/home/stmutasa/PycharmProjects/Datasets/BreastData/Mammo/Calcs/ADH/Patient 1 YES'
        if 'ADH' in file:

            # Retreive the patient name:
            patient = int(file.split('/')[-1].split(' ')[-2])

            # Retreive the type of invasion
            invasion = file.split('/')[-2]

            # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
            label_file = sdl.retreive_filelist('gz', False, path=(file + '/3'))[0]
            label_file2 = sdl.retreive_filelist('gz', False, path=(file + '/4'))[0]

            # Load the files
            image_file = sdl.retreive_filelist('dcm', False, path=(file + '/3'))[0]
            image_file2 = sdl.retreive_filelist('dcm', False, path=(file + '/4'))[0]

            # Load the 1st dicom
            image, acc, dims, window, _ = sdl.load_DICOM_2D(image_file2)

            # Load the label
            segments = np.squeeze(sdl.load_NIFTY(label_file2))

            # Flip x and Y
            segments = np.squeeze(sdl.reshape_NHWC(segments, False))

            # If this is one of the exceptions with only one series...
            try:

                # Load the 2nd dicom and labels
                image2, acc2, dims2, window2, _ = sdl.load_DICOM_2D(image_file)
                segments2 = np.squeeze(sdl.load_NIFTY(label_file))
                segments2 = np.squeeze(sdl.reshape_NHWC(segments2, False))

            except:

                # Likely no second, then just copy it...
                image2, acc2, dims2, window2 = np.copy(image), acc, dims, window
                segments2 = np.copy(segments)

        # Now load other files

        else:

            # Retreive the patient name old: data/raw/pure/Patient 41/2/ser32770img00005.dcm
            patient = int(file.split('/')[-1].split(' ')[-1])

            # Retreive the type of invasion
            invasion = file.split('/')[-2]

            # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
            label_file = (file + '/%s-3.nii.gz' % patient)
            label_file2 = (file + '/%s-4.nii.gz' % patient)

            # Load the files
            image_file = sdl.retreive_filelist('dcm', False, path=(file + '/3'))
            image_file2 = sdl.retreive_filelist('dcm', False, path=(file + '/4'))

            # Load the 1st dicom
            image, acc, dims, window, _ = sdl.load_DICOM_2D(image_file2[0])

            # Load the label
            segments = np.squeeze(sdl.load_NIFTY(label_file2))

            # Flip x and Y
            segments = np.squeeze(sdl.reshape_NHWC(segments, False))

            # If this is one of the exceptions with only one series...
            if (invasion + '/' + file.split('/')[-1]) in singles:

                # Just copy it...
                image2, acc2, dims2, window2 = np.copy(image), acc, dims, window
                segments2 = np.copy(segments)

            else:

                # Load the 2nd dicom and labels
                image2, acc2, dims2, window2, _ = sdl.load_DICOM_2D(image_file[0])
                segments2 = np.squeeze(sdl.load_NIFTY(label_file))
                segments2 = np.squeeze(sdl.reshape_NHWC(segments2, False))

        # All loaded, common pathway below

        # Fix weird segment shape where shape is yyy, 2, xxx
        if segments.shape[1]==2: segments = segments[:, 1, :]
        if segments2.shape[1] == 2: segments2 = segments2[:, 1, :]

        # Assign labels
        elif 'pure' in invasion:
            label = 1
            lab2 += 1
        else:
            label = 0
            lab1 += 2

        # Retreive the center of the largest label
        blob, cn = sdl.largest_blob(segments)
        blob2, cn2 = sdl.largest_blob(segments2)

        # Calculate a factor to make our image size, call this "radius"
        radius, radius2 = np.sum(blob) ** (1 / 3) * 10, np.sum(blob2) ** (1 / 3) * 10

        # Make a 2dbox at the center of the label
        box_scaled, _ = sdl.generate_box(image, cn, int(radius) * 2, dim3d=False)
        box_wide, _ = sdl.generate_box(image, cn, 512, dim3d=False)
        box_scaled2, _ = sdl.generate_box(image2, cn2, int(radius2) * 2, dim3d=False)
        box_wide2, _ = sdl.generate_box(image2, cn2, 512, dim3d=False)

        # Resize image
        box_scaled = cv2.resize(box_scaled, (box_dims, box_dims))
        box_wide = cv2.resize(box_wide, (box_dims, box_dims))
        box_scaled2 = cv2.resize(box_scaled2, (box_dims, box_dims))
        box_wide2 = cv2.resize(box_wide2, (box_dims, box_dims))

        # Save the boxes in one
        box = np.zeros(shape=(box_dims, box_dims, 2)).astype(np.float32)
        box[:, :, 0] = box_scaled
        box[:, :, 1] = box_wide
        box2 = np.zeros(shape=(box_dims, box_dims, 2)).astype(np.float32)
        box2[:, :, 0] = box_scaled2
        box2[:, :, 1] = box_wide2

        # Clip and normalize the boxes
        box [box > 3500] = 3500
        box2 [box2 >3500 ] = 3500
        mean = (np.mean(box) + np.mean(box2)) / 2
        std = (np.std(box) + np.std(box2)) / 2
        box = (box - mean) / std
        box2 = (box2 - mean) / std

        # Set how many to iterate through
        num_examples = 1

        # Generate X examples
        for i in range(num_examples):

            # Generate the dictionary
            data[index] = {'data': box.astype(np.float32), 'label': label, 'patient': int(patient), 'series': file, 'invasion': invasion,
                           'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

            # Append counter
            index += 1

            # Generate the dictionary
            data[index] = {'data': box2.astype(np.float32), 'label': label, 'patient': int(patient), 'series': file, 'invasion': invasion,
                           'box_x': cn2[0], 'box_y': cn2[1], 'box_size': radius2}

            # Append counter
            index += 1

        # Finish this patient
        pt += 1

        # Save the protobufs
        if pt % 30 == 0:

            # Now create a protocol buffer and save the data types
            print('Creating a protocol buffer... %s examples from 30 patients loaded, DCIS %s, ADH: %s' % (len(data), lab2, lab1))
            sdl.save_tfrecords(data, 1, file_root=('data/ADH_vs_Pure' + str(filesave)))

            # Trackers
            lab1, lab2 = 0, 0
            filesave += 1

            # Garbage
            del data
            data = {}

    # save last protobuf for stragglers
    print('Creating a protocol buffer... %s examples from 29 patients loaded, DCIS %s, ADH: %s' % (len(data), lab2, lab1))
    sdl.save_dict_filetypes(data[index - 1])
    sdl.save_tfrecords(data, 1, file_root=('data/ADH_vs_Pure' + str(filesave)))

    print('Complete... %s examples from %s patients loaded' % (index, pt))


def load_protobuf():
    """
    Loads the protocol buffer into a form to send to shuffle
    :param dimesion: String: Whether to load the 3dimensional version
    :return:
    """

    # Load all the filenames in glob
    filenames1 = glob.glob('data/*.tfrecords')
    filenames = []

    # Define the filenames to remove
    for i in range(0, len(filenames1)):
        if FLAGS.test_files not in filenames1[i]:
            filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # Load the dictionary. Remember we saved a doubled box
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims*2, tf.float32, channels=2)

    # Image augmentation

    # Random contrast and brightness
    #data['data'] = tf.image.random_brightness(data['data'], max_delta=2)
    data['data'] = tf.image.random_contrast(data['data'], lower=0.975, upper=1.05)

    # Random gaussian noise
    T_noise = tf.random_uniform([1], 0, 0.2)
    noise = tf.random_uniform(shape=[FLAGS.box_dims*2, FLAGS.box_dims*2, 2], minval=-T_noise, maxval=T_noise)
    data['data'] = tf.add(data['data'], tf.cast(noise, tf.float32))

    # Randomly rotate
    angle = tf.random_uniform([1], -0.52, 0.52)
    data['data'] = tf.contrib.image.rotate(data['data'], angle)

    # Crop center
    data['data'] = tf.image.central_crop(data['data'], 0.55)

    # Then randomly flip
    data['data'] = tf.image.random_flip_left_right(tf.image.random_flip_up_down(data['data']))

    # Random crop using a random resize
    data['data'] = tf.random_crop(data['data'], [FLAGS.box_dims, FLAGS.box_dims, 2])

    # Display the images
    tf.summary.image('Train Norm IMG', tf.reshape(data['data'][:, :, 0], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)
    tf.summary.image('Train Base IMG', tf.reshape(data['data'][:, :, 1], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)

    # Reshape image
    data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims])

    return sdl.randomize_batches(data, FLAGS.batch_size)


def load_validation_set():
    """
    Same as load protobuf() but loads the validation set
    :param dimension: whether to load the 2d or 3d version
    :return:
    """

    # Use Glob here
    filenames1 = glob.glob('data/*.tfrecords')

    # The real filenames
    filenames = []

    # Retreive only the right filename
    for i in range(0, len(filenames1)):
        if FLAGS.test_files in filenames1[i]:
            filenames.append(filenames1[i])

    print('Testing files: %s' % filenames)

    # Load the dictionary
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims*2, channels=2)

    # Crop center
    data['data'] = tf.image.central_crop(data['data'], 0.5)

    # Display the images
    tf.summary.image('Test Norm IMG', tf.reshape(data['data'][:, :, 0], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)
    tf.summary.image('Test Base IMG', tf.reshape(data['data'][:, :, 1], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)

    # Reshape image
    data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims])

    return sdl.val_batches(data, FLAGS.batch_size)


# pre_process_adh_vs_pure(256)
# pre_process_INV_new(256)
# pre_process_DCISvsInv(256)