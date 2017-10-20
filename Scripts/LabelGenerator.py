'''This script is written with the focus to label the images. And storing them into 4 different directories of files containing
the images of each class. The script takes 3 inputs:
    *The root directory containing the images as the originally downloaded folders
    *Catetogry value (0 for Backpacks, 1 for Bags, 2 for Luggage an 3 for Accesories)
    *The target directory where the renamed files will be saved

In order to run the script the following command must be run on terminal

    python LabelGenerator.py -d </..../luggage_case/luggage_case/Backpacks -c <0|1|2|3> -t </.../Backpacks>
     
Note: This script needs to run four times parsing 4 different sets of arguments in order to label all the data'''


from argparse import ArgumentParser
import os
import shutil

parser = ArgumentParser()
parser.add_argument("-d", dest="RootDir",
                    help="Directory of image files", metavar="FILE")
parser.add_argument("-c", dest="category", type=int,help="Category of luggage")
parser.add_argument("-t", dest="TargetDir",
                    help="Directory of image files", metavar="FILE")
args = parser.parse_args()
RootDir1=args.RootDir
cat=args.category
TargetFolder = args.TargetDir
brandcounter=0
filecounter=0
for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
    for brand in dirs:
        print(brand)
        brandcounter+=1
        BrandFolder = os.path.join(root, brand)
        for dummy1, dummy2, files1 in os.walk((os.path.normpath(BrandFolder)), topdown=False):
            print("--------------------------------------------------------------------------------------------")
            for file in files1:
                filecounter+=1
                print (file)
                SourceFolder = os.path.join(BrandFolder,file )
                FormattedFileName = str(str(cat) + "_" + str(brandcounter) + "_" + str(filecounter) + ".jpg")
                print(FormattedFileName)
                new_name = os.path.join(TargetFolder, FormattedFileName)
                shutil.copy2(SourceFolder, new_name)