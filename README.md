# HOG-based-human-detection

### Project description
Write a program to compute the HOG (Histograms of Oriented Gradients)
feature from an input image and then classify the HOG feature vector into human or no-human by
using a 3-nearest neighbor (NN) classifier. In the 3-NN classifier, the distance between the input
image and a training image is computed by taking the histogram intersection of their HOG feature
vectors:
![image](https://b2-ac9137.s3.amazonaws.com/8c030-d1cd-d043-1dd-fe0c725f4ce_Untitled.png)
