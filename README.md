# EveryBag
A short task to classify images of luggage into 4 categories using computer vision and deep learning

Hi Markus
It has been an extremely interesting task for me to do. Within the short time I could make amidst work, I could only focus on categorising the images into the 4 major classes of backpacks, bags, luggages and accesories. I am sure I will spend some more time to solve this problem further. But for the sake of submission I am doing this now.

Let us start by talking about the dataset.
The images provided to me basically belonged to 4 classes. But the images from the class 'Luggage' has got roughly 80% of the images in the dataset. Having a dataset sructured like this will definitely cause problems to a network for a classification task. Since will easily classify one class. And since it is easy for the network to get the reward, it will always tend to classify other classes into this one class, hence making faulty prediction. To prove the same I have performed 4 seperate experiments
  
  * I have trained the network for the given data as it is
  * I have trained 400 images per class
  * I have augmented the data and tested it using a non-augmented dataset
  * I have augmented the data and tested it on an augmented dataset
