'''This script contains the deep neural network and training of the deep neural network'''

from __future__ import print_function
import tensorflow as tf
import os
import glob
import numpy as np

#GPU training settings
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#Please enter a path containing training images here:
DATASET_PATH = ''

#Please enter a path containing test images here
Test_Dataset_path = ''

# Image Parameters
N_CLASSES = 4 # total number of classes
IMG_HEIGHT = 299 # the image height
IMG_WIDTH = 299 # the image width
CHANNELS = 3 # The 3 color channels


def read_images(dataset_path, batch_size):
    '''This function takes the dataset path. It convertes all the image paths to a tensor and decodes it from a jpeg format.
    After the decoding is done, it pushes the tensors into a queue. and returns batches of image and label tensors,
    depending upon the batch size'''

    directory = dataset_path
    #this variable will contain all the file names
    file_names = glob.glob(os.path.join(directory, '*.jpg'))
    #This variable will contain the number of files
    size_data = len(file_names)

    #This variable contains the image path as a list
    imagepaths = list()

    #This variable contains all the training lables in the form of a one-hot vector
    y_train = np.zeros((size_data, 4), dtype=np.int32)
    for i,j in range(size_data,4):
        y_train[i][j]=0

    #You can limit the number of training files using this variable
    sample_files = file_names[0:size_data]

    for i, label in enumerate(sample_files):
        for j in range(1):
            value = int(str(sample_files[i]).split('/')[-1].split('_')[j])
            y_train[i][value] = 1
            imagepaths.append(sample_files[i])

    #This variable contains the number of elemets in the list containing the file paths
    num_train_files= int(len(imagepaths))

    # Convert to images and labels to tensors
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(y_train, dtype=tf.int32)

    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)
    # Read images from disk
    image = tf.read_file(image)

    #Decodes the images from jpeg format
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalizes the images
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)
    return X,Y,num_train_files

def read_Test_images(dataset_path, batch_size):
    '''This function does the same work as the 'read_images(), but for the test files'''
    directory = dataset_path
    file_names = glob.glob(os.path.join(directory, '*.jpg'))
    size_data = len(file_names)
    np.random.seed(seed=2017)
    imagepaths = list()
    y_train = np.zeros((size_data, 4), dtype=np.int32)
    for i, j in range(size_data, 4):
        y_train[i][j] = 0
    sample_files = file_names[0:size_data]
    for i, label in enumerate(sample_files):
        for j in range(1):
            value = int(str(sample_files[i]).split('/')[-1].split('_')[j])
            # temp[value] = 1
            y_train[i][value] = 1
            imagepaths.append(sample_files[i])
    num_test_files=int(len(imagepaths))

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(y_train, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)
    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0 / 127.5 - 1.0

    # Create batches
    X_test, Y_test = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X_test, Y_test,num_test_files

# -----------------------------------------------
# This is the CNN model, it contains two conv layers, 2 pooling layers, one fully connected layer, a dropout and a softmax layer
# -----------------------------------------------


# Parameters
learning_rate = 0.00001
batch_size = 20
display_step = 20
num_epochs=10
dropout = 0.5

# read training images
X, Y,num_train_files = read_images(DATASET_PATH, batch_size)

#read testing images
X_test,Y_test,num_test_files = read_Test_images(Test_Dataset_path,batch_size)

#deciding the number of steps for tarining and testing
num_steps_train=int(num_train_files/batch_size)
num_steps_test= int(num_test_files/batch_size)

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out1 = tf.nn.softmax(out)

    return out1,out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training

probs,logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)

# Create another graph for testing that reuse the same weights
probs_test,logits_test = conv_net(X_test, N_CLASSES, dropout=1, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

    logits=logits_train, labels=Y))

#loss_op = tf.reduce_mean(logits_train)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_train, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Evaluate test accuracy
correct_pred_test = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y_test, 1))
accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    #Starting the queues
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for e in range(num_epochs):
        sum = 0
        print("-----------------------------------EPOCH:",e," ----------------------------------------")
        for step in range(1, num_steps_train+1):

            if step % display_step == 0:
                # Run optimization and calculate batch loss and accuracy
                _, loss, acc = sess.run([train_op, loss_op, accuracy])
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            else:
                # Only run the optimization op (backprop)
                sess.run(train_op)

        #Calculate the test accuracy at each epoch
        for i in range (num_steps_test):
            test_acc=sess.run(accuracy_test)
            sum+=test_acc
        print("test_accuracy=", sum/num_steps_test)

        #saving the model
        saver.save(sess, '/data/fayaz/Checkpoints/NY5/model.ckpt',global_step=e)
        print("Model saved.")
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)

