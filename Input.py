
import os, cv2, glob
import numpy as np
import tensorflow as tf
from random import shuffle
import matplotlib.pyplot as plt

import SODLoader as SDL

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

sdl = SDL.SODLoader(os.getcwd())


def pre_process(box_dims, xvals):

    # Retreive filenames data/raw/pure/Patient 41/2/ser32770img00005.dcm
    filenames = []

    # Search each folder for the files
    pure_files = [x[0] for x in os.walk('data/raw/pure/')]
    inv_files = [x[0] for x in os.walk('data/raw/invasive/')]
    micro_files = [x[0] for x in os.walk('data/raw/microinvasion/')]

    # Append each file into filenames
    for z in range (len(pure_files)):
        if len(pure_files[z].split('/')) != 4: continue
        filenames.append(pure_files[z])

    for z in range(len(inv_files)):
        if len(inv_files[z].split('/')) != 4: continue
        filenames.append(inv_files[z])

    for z in range(len(micro_files)):
        if len(micro_files[z].split('/')) != 4: continue
        filenames.append(micro_files[z])

    # Shuffle filenames to create semi even protobufs
    shuffle(filenames)
    print(len(filenames), 'Files found: ', filenames)

    # Global variables
    index, filesave, pt = 0, 0, 0
    lab1, lab2 = 0, 0
    data = {}
    display, unique_ID = [], []

    # Double the box size
    #box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    # Single label files
    singles = ['pure/Patient 55', 'invasive/Patient 36', 'invasive/Patient 50']

    for file in filenames:

        #  Skip non patient folders
        if 'Patient' not in file: continue

        # Retreive the patient name
        patient = int(file.split('/')[3].split(' ')[-1])

        # Retreive the type of invasion
        invasion = file.split('/')[2]

        # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
        label_file = (file + '/%s-3.nii.gz' %patient)
        label_file2 = (file + '/%s-4.nii.gz' %patient)

        # Load the files
        image_file = sdl.retreive_filelist('dcm', False, path = (file + '/3'))
        image_file2 = sdl.retreive_filelist('dcm', False, path=(file + '/4'))

        # Load the 1st dicom
        image, acc, dims, window = sdl.load_DICOM_2D(image_file2[0])

        # Load the label
        segments = np.squeeze(sdl.load_NIFTY(label_file2))

        # Flip x and Y
        segments = np.squeeze(sdl.reshape_NHWC(segments, False))

        # If this is one of the exceptions with only one series...
        if (invasion + '/' + file.split('/')[3]) in singles:

            # Just copy it...
            image2, acc2, dims2, window2 = np.copy(image), acc, dims, window
            segments2 = np.copy(segments)

        else:

            # Load the 2nd dicom and labels
            image2, acc2, dims2, window2 = sdl.load_DICOM_2D(image_file[0])
            segments2 = np.squeeze(sdl.load_NIFTY(label_file))
            segments2 = np.squeeze(sdl.reshape_NHWC(segments2, False))

        # Assign labels
        if 'invasive' in invasion: label = 0
        elif 'micro' in invasion:  label = 1
        else: label = 2

        # Second labels
        if label < 2:
            label2 = 0
            lab2 += 4
        else:
            label2 = 1
            lab1 += 2

        # Retreive the center of the largest label
        blob, cn = sdl.largest_blob(segments, image)
        blob2, cn2 = sdl.largest_blob(segments2, image2)

        # Calculate a factor to make our image size, call this "radius"
        radius, radius2 = np.sum(blob)**(1/3)*10, np.sum(blob2) ** (1 / 3) * 10

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

        # Test: TODO
        if label != 2: continue
        sdl.display_single_image(box_scaled, False, label)
        display.append([box_scaled])

        # Save the boxes in one
        box = np.zeros(shape=(box_dims, box_dims, 2)).astype(np.float32)
        box[:, :, 0] = box_scaled
        box[:, :, 1] = box_wide
        box2 = np.zeros(shape=(box_dims, box_dims, 2)).astype(np.float32)
        box2[:, :, 0] = box_scaled2
        box2[:, :, 1] = box_wide2

        # Normalize the boxes
        box = (box - 2160.61) / 555.477
        box2 = (box2 - 2160.61) / 555.477

        # Set how many to iterate through
        num_examples = int(2 - label2)

        # Generate X examples
        for i in range (num_examples):

            # Generate the dictionary
            data[index] = {'data': box, 'label': label, 'label2': label2, 'patient': int(patient),
                    'series': file, 'invasion': invasion, 'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

            # Append counter
            index += 1

            # Generate the dictionary
            data[index] = {'data': box2, 'label': label, 'label2': label2, 'patient': int(patient),
                           'series': file, 'invasion': invasion, 'box_x': cn2[0], 'box_y': cn2[1], 'box_size': radius2}

            # Append counter
            index += 1

        # Finish this example of this patient
        pt +=1
        if pt > 16: break

        # Save if multiples of 49
        if pt % int(125/xvals) == 0:

            # Now create a protocol buffer
            print('Creating a protocol buffer... %s examples from %s patients loaded, Invasive %s, Pure DCIS: %s'
                  %(len(data), pt, lab2, lab1))

            # Open the file writer
            writer = tf.python_io.TFRecordWriter(os.path.join(('data/Data%s' % filesave) + '.tfrecords'))

            # Loop through each example and append the protobuf with the specified features
            for key, values in data.items():

                # Serialize to string
                example = tf.train.Example(features=tf.train.Features(feature=sdl.create_feature_dict(values, key)))

                # Save this index as a serialized string in the protobuf
                writer.write(example.SerializeToString())

            # Close the file after writing
            writer.close()

            # Trackers
            filesave += 1
            lab1, lab2 = 0, 0

            # Garbage
            del data
            data = {}


    # Finished all patients

    # Test TODO:
    #display.append(box_scaled)
    plt.show()
    #sdl.display_mosaic(display, True, title='Class1', cmap='gray')

    # Now create a protocol buffer
    print('Creating final protocol buffer... %s examples from %s patients loaded, Invasive %s, Pure DCIS: %s'
          % (index, pt, lab2, lab1))

    # Open the file writer
    writer = tf.python_io.TFRecordWriter(os.path.join(('data/DataFin') + '.tfrecords'))

    # Loop through each example and append the protobuf with the specified features
    for key, values in data.items():

        # Serialize to string
        example = tf.train.Example(features=tf.train.Features(feature=sdl.create_feature_dict(values, key)))

        # Save this index as a serialized string in the protobuf
        writer.write(example.SerializeToString())

    # Close the file after writing
    writer.close()


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
    for i in range (0, len(filenames1)):
        if FLAGS.test_files not in filenames1[i]:
            filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # now load the remaining files
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)

    reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = reader.read(filename_queue)  # Returns the next record (key:value) produced by the reader

    # Restore the feature dictionary to store the variables we will retrieve using the parse
    feature_dict = {'id': tf.FixedLenFeature([], tf.int64), 'data': tf.FixedLenFeature([], tf.string),
                    'patient': tf.FixedLenFeature([], tf.string), 'series': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string), 'label2': tf.FixedLenFeature([], tf.string),
                    'invasion': tf.FixedLenFeature([], tf.string), 'box_y': tf.FixedLenFeature([], tf.string),
                    'box_x': tf.FixedLenFeature([], tf.string), 'box_size': tf.FixedLenFeature([], tf.string)}

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    # Remember we saved a doubled box
    box_dims = int(FLAGS.box_dims*2)

    # Change the raw image data
    image = tf.decode_raw(features['data'], tf.float32)
    image = tf.reshape(image, shape=[box_dims, box_dims, 2])

    # Cast all our data to 32 bit floating point units. Cannot convert string to number unless you use that function
    id = tf.cast(features['id'], tf.float32)
    invasion = tf.cast(features['invasion'], tf.string)
    series = tf.cast(features['series'], tf.string)
    patient = tf.string_to_number(features['patient'], tf.float32)
    label = tf.string_to_number(features['label'], tf.float32)
    label2 = tf.string_to_number(features['label2'], tf.float32)

    # The box dimensions
    box_x = tf.string_to_number(features['box_x'], tf.float32)
    box_y = tf.string_to_number(features['box_y'], tf.float32)
    box_size = tf.string_to_number(features['box_size'], tf.float32)

    # Image augmentation
    angle = tf.random_uniform([1], -0.52, 0.52)

    # First randomly rotate
    image = tf.contrib.image.rotate(image, angle)

    # Crop center
    image = tf.image.central_crop(image, 0.55)

    # Then randomly flip
    image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))

    # Random crop using a random resize
    image = tf.random_crop(image, [FLAGS.box_dims, FLAGS.box_dims, 2])

    # Display the images
    tf.summary.image('Train Norm IMG', tf.reshape(image[:, :, 0], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)
    tf.summary.image('Train Base IMG', tf.reshape(image[:, :, 1], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)

    # Reshape image
    image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])

    # Return data as a dictionary
    final_data = {'image': image, 'label': label, 'patient':patient, 'box_size': box_size,
                  'label2': label2, 'invasion': invasion, 'series': series, 'box_loc': [box_y, box_x]}

    returned_dict = {}
    returned_dict['id'] = id
    for key, feature in final_data.items():
        returned_dict[key] = feature
    return returned_dict


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

    print ('Testing files: %s' %filenames)

    # Load the filename queue
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    val_reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = val_reader.read(
        filename_queue)  # Returns the next record (key:value) produced by the reader

    # Restore the feature dictionary to store the variables we will retrieve using the parse
    feature_dict = {'id': tf.FixedLenFeature([], tf.int64), 'data': tf.FixedLenFeature([], tf.string),
                    'patient': tf.FixedLenFeature([], tf.string), 'series': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string), 'label2': tf.FixedLenFeature([], tf.string),
                    'invasion': tf.FixedLenFeature([], tf.string), 'box_y': tf.FixedLenFeature([], tf.string),
                    'box_x': tf.FixedLenFeature([], tf.string), 'box_size': tf.FixedLenFeature([], tf.string)}

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    # Remember we saved a doubled box
    box_dims = int(FLAGS.box_dims * 2)

    # Change the raw image data
    image = tf.decode_raw(features['data'], tf.float32)
    image = tf.reshape(image, shape=[box_dims, box_dims, 2])

    # Cast all our data to 32 bit floating point units. Cannot convert string to number unless you use that function
    id = tf.cast(features['id'], tf.float32)
    invasion = tf.cast(features['invasion'], tf.string)
    series = tf.cast(features['series'], tf.string)
    patient = tf.string_to_number(features['patient'], tf.float32)
    label = tf.string_to_number(features['label'], tf.float32)
    label2 = tf.string_to_number(features['label2'], tf.float32)

    # The box dimensions
    box_x = tf.string_to_number(features['box_x'], tf.float32)
    box_y = tf.string_to_number(features['box_y'], tf.float32)
    box_size = tf.string_to_number(features['box_size'], tf.float32)

    # Crop center
    image = tf.image.central_crop(image, 0.5)

    # Display the images
    tf.summary.image('Test Norm IMG', tf.reshape(image[:, :, 0], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)
    tf.summary.image('Test Base IMG', tf.reshape(image[:, :, 1], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)

    # Resize image
    image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])

    # Return data as a dictionary
    final_data = {'image': image, 'label': label, 'patient': patient, 'box_size': box_size,
                  'label2': label2, 'invasion': invasion, 'series': series, 'box_loc': [box_y, box_x]}

    returned_dict = {}
    returned_dict['id'] = id
    for key, feature in final_data.items():
        returned_dict[key] = feature
    return returned_dict


def pre_process_adh(box_dims, xvals):

    # Retreive filenames data/raw/pure/Patient 41/2/ser32770img00005.dcm
    filenames = []

    # Search each folder for the files
    pure_files = [x[0] for x in os.walk('data/raw/pure/')]
    inv_files = [x[0] for x in os.walk('data/raw/invasive/')]
    micro_files = [x[0] for x in os.walk('data/raw/microinvasion/')]
    adh_files = [x[0] for x in os.walk('data/raw/ADH/')]

    # Append each file into filenames
    for z in range (len(pure_files)):
        if len(pure_files[z].split('/')) != 4: continue
        filenames.append(pure_files[z])

    for z in range(len(inv_files)):
        if len(inv_files[z].split('/')) != 4: continue
        filenames.append(inv_files[z])

    for z in range(len(micro_files)):
        if len(micro_files[z].split('/')) != 4: continue
        filenames.append(micro_files[z])

    for z in range(len(adh_files)):
        if len(adh_files[z].split('/')) != 4: continue
        filenames.append(adh_files[z])

    # Shuffle filenames to create semi even protobufs
    shuffle(filenames)
    print(len(filenames), 'Files found: ', filenames)

    # Global variables
    index, filesave, pt = 0, 0, 0
    lab1, lab2 = 0, 0
    data = {}
    display, unique_ID = [], []
    mean, std = 0,0

    # Double the box size
    box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    # Single label files
    singles = ['pure/Patient 55', 'invasive/Patient 36', 'invasive/Patient 50']

    for file in filenames:

        #  Skip non patient folders
        if 'Patient' not in file: continue

        # Different code for loading ADH files
        if 'ADH' in file:

            # Retreive the patient name
            patient = int(file.split('/')[3].split(' ')[-2])

            # Retreive the type of invasion
            invasion = file.split('/')[2]

            # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
            label_file = sdl.retreive_filelist('gz', False, path=(file + '/3'))[0]
            label_file2 = sdl.retreive_filelist('gz', False, path=(file + '/4'))[0]

            # Load the files
            image_file = sdl.retreive_filelist('dcm', False, path=(file + '/3'))[0]
            image_file2 = sdl.retreive_filelist('dcm', False, path=(file + '/4'))[0]

            # Load the 1st dicom
            image, acc, dims, window = sdl.load_DICOM_2D(image_file2)

            # Load the label
            segments = np.squeeze(sdl.load_NIFTY(label_file2))

            # Flip x and Y
            segments = np.squeeze(sdl.reshape_NHWC(segments, False))

            # If this is one of the exceptions with only one series...
            try:

                # Load the 2nd dicom and labels
                image2, acc2, dims2, window2 = sdl.load_DICOM_2D(image_file)
                segments2 = np.squeeze(sdl.load_NIFTY(label_file))
                segments2 = np.squeeze(sdl.reshape_NHWC(segments2, False))

            except:

                # Likely no second, then just copy it...
                image2, acc2, dims2, window2 = np.copy(image), acc, dims, window
                segments2 = np.copy(segments)

        # Now load other files

        else:

            # Retreive the patient name
            patient = int(file.split('/')[3].split(' ')[-1])

            # Retreive the type of invasion
            invasion = file.split('/')[2]

            # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
            label_file = (file + '/%s-3.nii.gz' % patient)
            label_file2 = (file + '/%s-4.nii.gz' % patient)

            # Load the files
            image_file = sdl.retreive_filelist('dcm', False, path=(file + '/3'))
            image_file2 = sdl.retreive_filelist('dcm', False, path=(file + '/4'))

            # Load the 1st dicom
            image, acc, dims, window = sdl.load_DICOM_2D(image_file2[0])

            # Load the label
            segments = np.squeeze(sdl.load_NIFTY(label_file2))

            # Flip x and Y
            segments = np.squeeze(sdl.reshape_NHWC(segments, False))

            # If this is one of the exceptions with only one series...
            if (invasion + '/' + file.split('/')[3]) in singles:

                # Just copy it...
                image2, acc2, dims2, window2 = np.copy(image), acc, dims, window
                segments2 = np.copy(segments)

            else:

                # Load the 2nd dicom and labels
                image2, acc2, dims2, window2 = sdl.load_DICOM_2D(image_file[0])
                segments2 = np.squeeze(sdl.load_NIFTY(label_file))
                segments2 = np.squeeze(sdl.reshape_NHWC(segments2, False))

        # All loaded, common pathway below

        # Assign labels
        if 'invasive' in invasion: label = 0
        elif 'micro' in invasion: label = 1
        elif 'pure' in invasion: label = 2
        else: label = 3

        # Second labels
        if label < 3:
            label2 = 0
            lab2 += 2
        else:
            label2 = 1
            lab1 += 4

        # Retreive the center of the largest label
        blob, cn = sdl.largest_blob(segments, image)
        blob2, cn2 = sdl.largest_blob(segments2, image2)

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

        # Normalize the boxes
        # scaled: Mean 2176.87447507, Std: 189.829622824
        # Wide: Mean 2176.05169742, Std: 191.406008206
        box = (box - 2176.8745) / 189.8296
        box2 = (box2 - 2176.8745) / 189.8296

        # For calculating Mean and STD
        mean += (np.mean(box_scaled) + np.mean(box_scaled2))/2
        std += (np.std(box_scaled) + np.std(box_scaled2))/2

        # Set how many to iterate through
        num_examples = int(1 + label2)

        # Generate X examples
        for i in range(num_examples):

            # Generate the dictionary
            data[index] = {'data': box, 'label': label, 'label2': label2, 'patient': int(patient),
                           'series': file, 'invasion': invasion, 'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

            # Append counter
            index += 1

            # Generate the dictionary
            data[index] = {'data': box2, 'label': label, 'label2': label2, 'patient': int(patient),
                           'series': file, 'invasion': invasion, 'box_x': cn2[0], 'box_y': cn2[1],
                           'box_size': radius2}

            # Append counter
            index += 1

        # Finish this patient
        pt += 1

        # Save the protobufs
        if pt % 38 == 0:

            # Now create a protocol buffer
            print('Creating a protocol buffer... %s examples from %s patients loaded, DCIS %s, ADH: %s'
                  % (len(data), pt, lab2, lab1))

            # Open the file writer
            writer = tf.python_io.TFRecordWriter(os.path.join(('data/Data%s' % filesave) + '.tfrecords'))

            # Loop through each example and append the protobuf with the specified features
            for key, values in data.items():
                # Serialize to string
                example = tf.train.Example(features=tf.train.Features(feature=sdl.create_feature_dict(values, key)))

                # Save this index as a serialized string in the protobuf
                writer.write(example.SerializeToString())

            # Close the file after writing
            writer.close()

            # Trackers
            filesave += 1
            lab1, lab2 = 0, 0

            # Garbage
            del data
            data = {}

    # Now create a protocol buffer
    print('Complete... %s examples from %s patients loaded' % (index, pt))

    # Print mean and STD
    print ('Mean %s, Std: %s' %((mean/pt), (std/pt)))
