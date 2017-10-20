# EveryBag
A short task to classify images of luggage into 4 categories using computer vision and deep learning

Hi Markus
It has been an extremely interesting task for me to do. Within the short time I could make amidst work, I could only focus on categorising the images into the 4 major classes of backpacks, bags, luggages and accesories. I am sure I will spend some more time to solve this problem further. But for the sake of submission I am doing this now.

Let us start by talking about the dataset.
The images provided to me basically belonged to 4 classes. But the images from the class 'Luggage' has got roughly 80% of the images in the dataset. Having a dataset sructured like this will definitely cause problems to a network for a classification task. Since will easily classify one class. And since it is easy for the network to get the reward, it will always tend to classify other classes into this one class, hence making faulty prediction. To prove the same I have performed 4 seperate experiments
  
  * I have trained the network for the given data as it is (Experiment 1)
  * I have trained 400 images per class (Experiment 2)
  * I have augmented the data and tested it using a non-augmented dataset (Experiment 3)
  * I have augmented the data and tested it on an augmented dataset (Experiment 4)
  
The details on these experiments are mentioned in the experiments folder.
I also wrote several scripts which you can find on the 'Scripts' folder. 
Before running the scripts please ensure that all the dependencies are installed in your computer.
The dependancies are mentioned in the 'Dependencies' file
