

from __future__ import print_function
import tensorflow as tf
import os
from Tkinter import Label,Tk
import tkFileDialog
from PIL import Image, ImageTk


os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Dataset Parameters - CHANGE HERE
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
Model_Path='' #example '/data/fayaz/Checkpoints/NY_final/model.ckpt-2'
# Image Parameters (same as in Train.py)
N_CLASSES = 4
IMG_HEIGHT = 299
IMG_WIDTH = 299
CHANNELS = 3


def read_Test_images(file,batch_size):
    ''' This script reads the image file, decodes it and returns the decoded image as a tensor'''
    file_names = file
    image = tf.read_file(file_names)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize
    image = image * 1.0 / 127.5 - 1.0
    # Create batches
    X_test = tf.train.batch([image], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)
    return X_test, 1


def conv_net(x, n_classes, dropout, reuse, is_training):
    '''This function has the convolutional neural network, same as the Train.py file'''


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
        out1 = tf.nn.softmax(out) #if not is_training else out

    return out1,out


def main():

    batch_size=1
    dropout=1

    #For UI
    root = Tk()
    root.title("image")
    root.geometry("600x600")
    root.configure(background='grey')

    #get the path tp the file
    path = tkFileDialog.askopenfilename(filetypes=[("Image File", '.jpg')])
    im = Image.open(path)
    tkimage = ImageTk.PhotoImage(im)
    myvar = Label(root, image=tkimage)
    myvar.image = tkimage
    myvar.pack()

    #Get the test image tensor
    X_test,num_files = read_Test_images(path,batch_size)

    #Get the class probabilities from the model
    prob_test,logits_test = conv_net(X_test, N_CLASSES, dropout, reuse=None, is_training=False)
    saver=tf.train.Saver()
    steps=int(num_files/batch_size)
    with tf.Session() as sess:
            #Initialise all variables
            sess.run(tf.global_variables_initializer())

            #restore the model file
            saver.restore(sess, Model_Path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range (steps):
                pred=sess.run(tf.arg_max(prob_test,1))
                if pred == 0:
                    text0="Backpack detected"
                    myvar2 = Label(root, text=text0)
                    myvar2.pack()
                    root.mainloop()
                elif pred == 1:
                    text1="Bag detected"
                    myvar2 = Label(root, text=text1)
                    myvar2.image = tkimage
                    root.mainloop()

                elif pred == 2:
                    text2="Luggage detected"
                    myvar2 = Label(root, text=text2)
                    myvar2.pack()

                    root.mainloop()
                elif pred == 3:
                    text3="Accessory detected"
                    myvar2 = Label(root, text=text3)
                    myvar2.pack()
                    root.mainloop()
                else:
                    print("There seems to have been some problem")

            coord.request_stop()
            coord.join(threads)

