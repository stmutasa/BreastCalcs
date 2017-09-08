
import os, cv2, glob
import numpy as np
import tensorflow as tf
from random import shuffle

import SODLoader as SDL

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

sdl = SDL.SODLoader(os.getcwd())


def pre_process(box_dims, xvals):

    # Retreive filenames data/raw/pure/Patient 41/2/ser32770img00005.dcm
    filenames = sdl.retreive_filelist('dcm', include_subfolders=True, path = 'data/raw/')

    # Shuffle filenames to create semi even protobufs
    shuffle(filenames)
    print ('Files: ', filenames)

    # Global variables
    index, filesave, pt = 0, 0, 0
    lab1, lab2 = 0, 0
    data = {}

    # Double the box size
    box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    for file in filenames:

        # Directory name
        dirname = os.path.dirname(file)

        # Series
        series = int(dirname.split('/')[-1])

        # Skip the non mag views (series 1 and 2)
        if int(series) < 3: continue

        # Retreive the patient name
        patient = int(dirname.split('/')[3].split(' ')[-1])

        # Retreive the type of invasion
        invasion = dirname.split('/')[2]

        # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
        dirname = dirname[:-2]
        label_file = (dirname + '/%s-%s.nii.gz' %(patient, series))

        # Load the dicom
        image, acc, dims, window = sdl.load_DICOM_2D(file)

        # Load the label
        segments = np.squeeze(sdl.load_NIFTY(label_file))

        # Flip x and Y
        segments = np.squeeze(sdl.reshape_NHWC(segments, False))

        # # Test TODO
        # print('Dir %s, ser %s, pt %s, inv %s, root %s, img %s, seg %s' %
        #       (label_file, series, patient, invasion, root, image.shape, segments.shape))

        # Assign labels
        if 'invasive' in invasion: label = 0
        elif 'micro' in invasion:  label = 1
        else: label = 2

        # Second labels
        if label < 2:
            label2 = 0
            lab2 += 2
        else:
            label2 = 1
            lab1 += 1

        # Retreive the center of the largest label
        blob, cn = sdl.largest_blob(segments, image)

        # Calculate a factor to make our image size, call this "radius"
        radius = np.sum(blob)**(1/3)*10

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

        # Normalize the boxes
        box = (box - 2160.61) / 555.477

        # Set how many to iterate through
        num_examples = int(2 - label2)

        # Generate X examples
        for i in range (num_examples):

            # Generate the dictionary
            data[index] = {'data': box, 'label': label, 'label2': label2, 'patient': int(patient),
                    'series': int(series), 'invasion': invasion, 'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

            # Append counter
            index += 1

            # Summary
            if index % 100 == 0: print (index, " Patient's loaded ", patient , series, label, label2, invasion, radius)

        # Finish this example of this patient
        pt +=1

        # Save if multiples of 49
        if pt % (243/xvals) == 0:

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

            # Garbage
            del data
            data = {}


    # Finished all patients

    # Now create a protocol buffer
    print (lab1, lab2, index)
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


def pre_process_fresh(box_dims, xvals):

    # Retreive filenames data/raw/pure/Patient 41/2/ser32770img00005.dcm
    filenames = sdl.retreive_filelist('dcm', include_subfolders=True, path = 'data/raw/')
    print ('Files: ', filenames)

    # Global variables
    index, lab1, lab2, filesave = 0, 0, 0, 0
    data = {}

    # Double the box size
    box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    for file in filenames:

        # Directory name
        dirname = os.path.dirname(file)

        # Series
        series = int(dirname.split('/')[-1])

        # Skip the non mag views (series 1 and 2)
        if int(series) < 3: continue

        # Retreive the patient name
        patient = int(dirname.split('/')[3].split(' ')[-1])

        # Retreive the type of invasion
        invasion = dirname.split('/')[2]

        # Retreive the root folder
        root = dirname.split('/')[-2]

        # Skip files: 57-3	9-3	2-4	7-4	7-3	3-4	8-3
        if (patient==7): continue
        elif (patient==57 and series==3) : continue
        elif (patient==9 and series==3) : continue
        elif (patient==3 and series==4) : continue
        elif (patient==8 and series==3)  : continue
        elif (patient == 2 and series == 4): continue

        # Load the dicom
        image, acc, dims, window = sdl.load_DICOM_2D(file)

        # now load the annotations
        for file2 in sdl.retreive_filelist('gz', include_subfolders=True, path = 'data/raw/'):

            # Skip if not the right patient or type (some patients have multiple types
            if (root not in file2) or (invasion not in file2): continue

            # Skip if not the right label
            if series != int(file2[-8]): continue

            # This is the one, load the label
            segments = np.squeeze(sdl.load_NIFTY(file2))

            # Flip x and Y
            segments = np.squeeze(sdl.reshape_NHWC(segments, False))

        # Should have everything loaded by now, implement error checking here

        # Assign labels
        if 'invasive' in invasion: label = 0
        elif 'micro' in invasion:  label = 1
        else: label = 2

        # Second labels
        if label < 2:
            label2 = 0
            lab2 += 3
        else:
            label2 = 1
            lab1 += 2

        # Retreive the center of the largest label
        blob, cn = sdl.largest_blob(segments, image)

        # Calculate a factor to make our image size, call this "radius"
        radius = np.sum(blob)**(1/3)*10

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

        # Set how many to iterate through
        num_examples = int(3 - label2)

        # Generate X examples
        for i in range (num_examples):

            # Generate the dictionary
            data[index] = {'data': box, 'label': label, 'label2': label2, 'patient': int(patient),
                    'series': int(series), 'invasion': invasion, 'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

            # Append counter
            index += 1

            # Summary
            if index % 10 == 0: print (index, " Patient's loaded ", patient , series, label, label2, invasion, radius)

        # Finish this example of this patient

    # Finished all patients

    # Now create a protocol buffer
    print('Creating final protocol buffer... %s patients loaded, Box Dimensions: %s' %(len(data), box_dims))
    print ('Invasive %s, Pure DCIS: %s' %(lab2, lab1))

    # Initialize normalization images array
    normz = np.zeros(shape=(len(data), box_dims, box_dims, 2), dtype=np.float32)

    # Normalize all the images. First retreive the images
    for key, dict in data.items(): normz[key, :, :, :] = dict['data']

    # Now normalize the whole batch
    print('Batch Norm: %s , Batch STD: %s' % (np.mean(normz), np.std(normz)))
    normz = sdl.normalize(normz, True, 0.01)

    # Return the normalized images to the dictionary
    for key, dict in data.items(): dict['data'] = normz[key]

    # generate x number of writers depending on the cross validations
    writer = []

    # Open the file writers
    for z in range(xvals): writer.append(tf.python_io.TFRecordWriter(os.path.join(('data/Data%s' %z) + '.tfrecords')))

    # Loop through each example and append the protobuf with the specified features
    z=0
    for key, values in data.items():

        # Serialize to string
        example = tf.train.Example(features=tf.train.Features(feature=sdl.create_feature_dict(values, key)))

        # Save this index as a serialized string in the protobuf
        writer[(z%xvals)].write(example.SerializeToString())
        z += 1

    # Close the file after writing
    for y in range (xvals): writer[y].close()


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
    patient = tf.string_to_number(features['patient'], tf.float32)
    series = tf.string_to_number(features['series'], tf.float32)
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
    patient = tf.string_to_number(features['patient'], tf.float32)
    series = tf.string_to_number(features['series'], tf.float32)
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
    adh_files = sdl.retreive_filelist('nii.gz', include_subfolders=True, path = 'data/raw/ADH/')
    filenames1 = sdl.retreive_filelist('dcm', include_subfolders=True, path='data/raw/')
    filenames = []

    # Remove ADH files from filenames
    for i in range(len(filenames1)):
        if 'ADH' not in filenames1[i]: filenames.append(filenames1[i])

    # Append and shuffle filenames to create semi even protobufs
    filenames.extend(adh_files)
    shuffle(filenames)
    print (len(filenames), 'Files: ', filenames)

    # Global variables
    index, filesave, pt = 0, 0, 0
    lab1, lab2 = 0, 0
    data = {}

    # Double the box size
    box_dims *= 2
    print ('Box Dimensions: ', box_dims, 'From: ', box_dims/2)

    for file in filenames:

        # Different code for loading ADH files
        if 'ADH' in file:

            # data/raw/ADH/Patient 60 YES/3/FINAL 20150708_092100ROUTINEs45994a000-label.nii.gz
            dirname = os.path.dirname(file)

            # Get series
            series = int(dirname.split('/')[-1])

            # Patient name
            patient = int(dirname.split('/')[-2].split(' ')[1])

            # Retreive the type of invasion
            invasion = dirname.split('/')[2]

            # file is the label file, now retreive image file
            image_file = sdl.retreive_filelist('nii', False, dirname)[0]

            # load the image
            image = np.squeeze(sdl.load_NIFTY(image_file))

            # Load segments
            segments = np.squeeze(sdl.load_NIFTY(file))

            # Some images have two channels for repeat mag views
            if segments.shape[0] == 2:

                # Make image equal the first channel
                image = image[0]
                segments = segments[0]

        # Now load other files
        else:

            # Directory name
            dirname = os.path.dirname(file)

            # Series
            series = int(dirname.split('/')[-1])

            # Skip the non mag views (series 1 and 2)
            if int(series) < 3: continue

            # Retreive the patient name
            patient = int(dirname.split('/')[3].split(' ')[-1])

            # Retreive the type of invasion
            invasion = dirname.split('/')[2]

            # Dir data/raw/invasive/Patient 53/3, ser 3, pt 53, inv invasive, root Patient 53
            label_file = (dirname[:-2] + '/%s-%s.nii.gz' %(patient, series))

            # Load the dicom
            image, acc, dims, window = sdl.load_DICOM_2D(file)

            # Load the label
            segments = np.squeeze(sdl.load_NIFTY(label_file))

            # Flip x and Y
            segments = np.squeeze(sdl.reshape_NHWC(segments, False))

        # All loaded, common pathway below

        # Assign labels
        if 'invasive' in invasion: label = 0
        elif 'micro' in invasion:  label = 1
        elif 'pure' in invasion: label = 2
        else: label = 3

        # Second labels. ADH = 1, other = 2
        if label < 3:
            label2 = 0
            lab2 += 1
        else:
            label2 = 1
            lab1 += 2

        # Retreive the center of the largest label
        try: blob, cn = sdl.largest_blob(segments, image)
        except: continue

        # Calculate a factor to make our image size, call this "radius"
        radius = np.sum(blob)**(1/3)*10

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

        # Normalize the boxes
        box = (box - 2160.61) / 555.477

        # Set how many to iterate through
        num_examples = int(1 + label2)

        # Generate X examples
        for i in range (num_examples):

            # Generate the dictionary
            data[index] = {'data': box, 'label': label, 'label2': label2, 'patient': int(patient),
                    'series': int(series), 'invasion': invasion, 'box_x': cn[0], 'box_y': cn[1], 'box_size': radius}

            # Append counter
            index += 1

        # Finish this example of this patient
        pt +=1

        # Save if multiples of x
        if pt % (380/xvals) == 0:

            # Now create a protocol buffer
            print('Creating a protocol buffer... %s examples from %s patients loaded, Ductal %s, ADH: %s'
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

            # Garbage
            del data
            data = {}


    # Finished all patients

    # Now create a protocol buffer
    print('Creating final protocol buffer (%s)... %s examples from %s patients loaded, Ductal %s, Pure ADH: %s'
          % (len(data), index, pt, lab2, lab1))

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
