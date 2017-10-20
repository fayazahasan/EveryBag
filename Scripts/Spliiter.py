'''This Script is used to split the data into training and test data. It takes one argument, the directory containing the images
categorised class wise. 10 percent of the images randomly of each class will be moved in order to create the test set.
To run the script, in terminal, run:

    python Splitter.py -d </path where the sorted images are present/backpacks>

Note: This script needs to run 4 times for the 4 different image directories containing backpacks,bags,luggage/accesories'''


import random
import os
import shutil
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-d", dest="RootDir",
                    help="Directory of image files", metavar="FILE")
args = parser.parse_args()
directory = args.RootDir

#Enter the path you wish for the test set here:
target_directory= r''
counter = 0
num_files=0
for root, dirs, files in os.walk((os.path.normpath(directory)), topdown=False):
    num_files=(len(files))

for i in range (int(num_files*0.1)):
    file = random.choice([x for x in os.listdir(directory)])
    if os.path.isfile(os.path.join(directory,file)):
        print("Moving random file no: ", i)
        shutil.move((os.path.join(directory,file)),(os.path.join(target_directory,file)))
    else:
        print("Already moved")
