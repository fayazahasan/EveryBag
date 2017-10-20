'''This simple script will count how many images are present in each category and return the maximum value out of the four'''


import os

def files_per_class():
    #The four directory variables will contain the paths of the individual directories containing images sorted class wise
    directory0 = '/home/fayazahasan/Downloads/luggage_case/LabelledDataClasswise/Backpacks'
    directory1 = '/home/fayazahasan/Downloads/luggage_case/LabelledDataClasswise/Bags'
    directory2 = '/home/fayazahasan/Downloads/luggage_case/LabelledDataClasswise/Luggage'
    directory3 = '/home/fayazahasan/Downloads/luggage_case/LabelledDataClasswise/Accesories'
    num_files0=0
    num_files1=0
    num_files2=0
    num_files3=0
    for root, dirs, files in os.walk((os.path.normpath(directory0)), topdown=False):
        num_files0=(len(files))
    for root, dirs, files in os.walk((os.path.normpath(directory1)), topdown=False):
        num_files1=(len(files))
    for root, dirs2, files in os.walk((os.path.normpath(directory2)), topdown=False):
        num_files2=(len(files))
    for root, dirs, files in os.walk((os.path.normpath(directory3)), topdown=False):
        num_files3=(len(files))
    return max(num_files0,num_files1,num_files2,num_files3)
