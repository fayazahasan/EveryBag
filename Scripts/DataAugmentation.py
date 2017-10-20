'''This script is used for data augmentation using the various tensorflow functions to achieve the same.
It will take all the files in a particular class - Bags/BackPacks/Luggage/Accesories
At first all the files paths are loaded and converted into a tensor.
Next, these  path tensors are decoded into image tensors.
These image tensors are then augmented in 20 different ways until the number of images in each class becomes equal
to the number of images of the class with the maximum number of images'''

import tensorflow as tf
import glob
import os
import Files_Per_Class_Counter as t
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Three channels for an RGB image
CHANNELS = 3

#Path where the augmented files are to be stored here
TARGETPATH='/data/fayaz/Data/Nyris/Aug/'


def give_tensors(path):
    '''This function will return the decoded images as tensors, the paths where the original files are stored,
    and number of files it has read'''
    # This variable will hold the list of paths
    path_tensor = []

    #This variable will hold the read images but not decoded
    image_file_tensor = []

    #This variable will hold the decoded images
    image_decoded_tensor = []

    directory = path

    #Get all filenames ending with extension jpg
    file_names = glob.glob(os.path.join(directory, '*.jpg'))
    size_data = len(file_names)

    #list contaning all the image paths
    imagepaths = list()
    sample_files = file_names[0:size_data]
    for i, label in enumerate(sample_files):
        for j in range(1):
            imagepaths.append(sample_files[i])

    #Counting the number of files that are fed
    num_of_files=len(imagepaths)

    for i in range (num_of_files):
        # Convert to Tensor
        path_tensor.append(tf.convert_to_tensor(imagepaths[i], dtype=tf.string))

        #reading each element of the path tensor
        image_file_tensor.append(tf.read_file(path_tensor[i]))

        #decoding each of the files and appending it to a tensor
        image_decoded_tensor.append(tf.image.decode_jpeg(image_file_tensor[i], channels=CHANNELS))

    return image_decoded_tensor,path_tensor,num_of_files

def distort(image):
        '''This funciton will distort an image tensor in 20 different ways and store re-encode them into a JPG format.
        It will return all the distorted images as a list and the count of the number of distortions that are used'''

        num_distortions=20

        #This variable will hold the distorted images in jpg format
        distorted_images=[]

        # distortion 1
        flipped_image_tensor = tf.image.random_flip_left_right(image)
        flipped_image=tf.image.encode_jpeg(flipped_image_tensor, name="save_me")
        distorted_images.append(flipped_image)

        # distortion 2
        changed_brightness_tensor = tf.image.random_brightness(image, max_delta=63 / 255.0)
        changed_brightness_image = tf.image.encode_jpeg(changed_brightness_tensor, name="save_me")
        distorted_images.append(changed_brightness_image)

        # distortion 3
        changed_contrast_tensor = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        changed_contrast_image = tf.image.encode_jpeg(changed_contrast_tensor, name="save_me")
        distorted_images.append(changed_contrast_image)

        # distortion 4
        changed_hue_tensor = tf.image.random_hue(image, max_delta=0.25)
        changed_hue_image = tf.image.encode_jpeg(changed_hue_tensor, name="save_me")
        distorted_images.append(changed_hue_image)

        # distortion 5
        changed_saturation_tensor = tf.image.random_saturation(image, lower=0.2, upper=1.8, )
        changed_saturation_image = tf.image.encode_jpeg(changed_saturation_tensor, name="save_me")
        distorted_images.append(changed_saturation_image)

        # distortion 6
        changed_flipped_and_brightness_tensor = tf.image.random_brightness(flipped_image_tensor, max_delta=0.7)
        changed_flipped_and_brightness_image = tf.image.encode_jpeg(changed_flipped_and_brightness_tensor, name="save_me")
        distorted_images.append(changed_flipped_and_brightness_image)

        # distortion 7
        changed_flipped_and_contrast_tensor = tf.image.random_contrast(flipped_image_tensor, lower=0.2, upper=1.8)
        changed_flipped_and_contrast_image = tf.image.encode_jpeg(changed_flipped_and_contrast_tensor, name="save_me")
        distorted_images.append(changed_flipped_and_contrast_image)

        # distortion 8
        changed_flipped_and_hue_tensor = tf.image.random_hue(flipped_image_tensor, max_delta=0.25)
        changed_flipped_and_hue_image = tf.image.encode_jpeg(changed_flipped_and_hue_tensor, name="save_me")
        distorted_images.append(changed_flipped_and_hue_image)

        # distortion 9
        changed_flipped_and_saturation_tensor = tf.image.random_saturation(flipped_image_tensor, lower=0.3, upper=1.5, )
        changed_flipped_and_saturation_image = tf.image.encode_jpeg(changed_flipped_and_saturation_tensor, name="save_me")
        distorted_images.append(changed_flipped_and_saturation_image)

        # distortion 10
        changed_brightness_and_contrast_tensor = tf.image.random_contrast(changed_brightness_tensor, lower=0.2, upper=1.8)
        changed_brightness_and_contrast_image = tf.image.encode_jpeg(changed_brightness_and_contrast_tensor, name="save_me")
        distorted_images.append(changed_brightness_and_contrast_image)

        # distortion 11
        changed_brightness_and_hue_tensor = tf.image.random_hue(changed_brightness_tensor, max_delta=0.32)
        changed_brightness_and_hue_image = tf.image.encode_jpeg(changed_brightness_and_hue_tensor, name="save_me")
        distorted_images.append(changed_brightness_and_hue_image)

        # distortion 12
        changed_brightness_and_saturation_tensor = tf.image.random_saturation(changed_brightness_tensor, lower=0.1, upper=1.1, )
        changed_brightness_and_saturation_image = tf.image.encode_jpeg(changed_brightness_and_saturation_tensor,name="save_me")
        distorted_images.append(changed_brightness_and_saturation_image)

        # distortion 13
        changed_contrast_and_hue_tensor = tf.image.random_hue(changed_contrast_tensor, max_delta=0.10)
        changed_contrast_and_hue_image = tf.image.encode_jpeg(changed_contrast_and_hue_tensor, name="save_me")
        distorted_images.append(changed_contrast_and_hue_image)

        # distortion 14
        changed_contrast_and_saturation_tensor = tf.image.random_saturation(changed_contrast_tensor, lower=0.7, upper=1.9, )
        changed_contrast_and_saturation_image = tf.image.encode_jpeg(changed_contrast_and_saturation_tensor, name="save_me")
        distorted_images.append(changed_contrast_and_saturation_image)

        # distortion 15
        brt_cntrst_hue_tensor= tf.image.random_brightness(changed_contrast_and_hue_tensor, max_delta=63 / 255.0)
        brt_cntrst_hue_image = tf.image.encode_jpeg(brt_cntrst_hue_tensor, name="save_me")
        distorted_images.append(brt_cntrst_hue_image)

        # distortion 16
        brt_cntrst_sat_tensor = tf.image.random_brightness(changed_contrast_and_saturation_tensor, max_delta=63 / 255.0)
        brt_cntrst_sat_image = tf.image.encode_jpeg(brt_cntrst_sat_tensor, name="save_me")
        distorted_images.append(brt_cntrst_sat_image)

        # distortion 17
        flip_cntrst_sat_tensor = tf.image.random_flip_left_right(changed_contrast_and_saturation_tensor)
        flip_cntrst_sat_image=tf.image.encode_jpeg(flip_cntrst_sat_tensor, name="save_me")
        distorted_images.append(flip_cntrst_sat_image)

        # distortion 18
        sat_flip_cntrst_tensor= tf.image.random_saturation(changed_flipped_and_contrast_tensor, lower=0.2, upper=1.8, )
        sat_flip_cntrst_image = tf.image.encode_jpeg(sat_flip_cntrst_tensor, name="save_me")
        distorted_images.append(sat_flip_cntrst_image)

        # distortion 19
        hue_flip_cntrst_tensor = tf.image.random_hue(changed_flipped_and_contrast_tensor, max_delta=0.19)
        hue_flip_cntrst_image = tf.image.encode_jpeg(hue_flip_cntrst_tensor, name="save_me")
        distorted_images.append(hue_flip_cntrst_image)

        #distortion 20
        hue_brt_sat_tensor = tf.image.random_hue(changed_brightness_and_saturation_tensor, max_delta=0.19)
        hue_brt_sat_image = tf.image.encode_jpeg(hue_brt_sat_tensor, name="save_me")
        distorted_images.append(hue_brt_sat_image)


        return distorted_images,num_distortions


def main():
    '''In terminal type: python DataAugmentation.py -d <Path of the folder containing class-wise sorted data>'''
    parser = ArgumentParser()
    parser.add_argument("-d", dest="RootDir",
                            help="Directory of image files", metavar="FILE")
    args = parser.parse_args()
    dir = args.RootDir

    #This counter will see how many images are being distorted
    image_ctr=0

    #Will hold the path and the file name of the augmented image
    new_file=[]

    #Get the count of the maximum number of images in class
    max_files_in_class=t.files_per_class()

    #The image tensor, the path tensor and the number of files are obtained by calling function 'give_tensors'
    img_tensor, img_pth_tensor, num_files = give_tensors(dir)

    #Starting the tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #In this for loop every image tensor is read. Each of them are then distorted using the distort function
        #The distorted images are then saved as an auto-generated file name into the TARGET_PATH
        for i in range (num_files):
            original_file=img_pth_tensor[i].eval()
            distorted_images,num_dist=distort(img_tensor[i].eval())
            value = (str(original_file).split('/')[-1].split('.'))
            if image_ctr<=max_files_in_class:
                for j in range (num_dist):
                    new_file.append(str(str(TARGETPATH) + value[0] + "_" + str(j) + ".jpg"))
                    f=open(new_file[image_ctr],"wb+")
                    f.write(distorted_images[j].eval())
                    f.close()
                    image_ctr += 1
                del distorted_images
            else:
                break
        print("Augmentation Finished")

main()