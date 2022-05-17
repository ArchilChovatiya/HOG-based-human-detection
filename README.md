# HOG-based-human-detection

### Project description
Here is the python program to compute the HOG (Histograms of Oriented Gradients)
feature from an input image and then classify the HOG feature vector into human or no-human by
using a 3-nearest neighbor (NN) classifier. In the 3-NN classifier, the distance between the input
image and a training image is computed by taking the histogram intersection of their HOG feature
vectors:
\
![image](https://b2-ac9137.s3.amazonaws.com/8c030-d1cd-d043-1dd-fe0c725f4ce_Untitled.png)
\
where I is the HOG feature of the input image and M is the HOG feature of the training image;
the subscript j indicates the jth component of the feature vector and n is the dimension of the
HOG feature vector. The distance between the input image and each of the training images is
computed and the classification of the input image is taken to be the majority classification of the
three nearest neighbors. 

#### Conversion to grayscale.
The inputs to your program are color images cut out from a larger
image. First, convert the color images into grayscale using the formula 𝐼𝐼 = 𝑅𝑅𝑅𝑅𝑅𝑅𝑅𝑅𝑅𝑅(0.299𝑅𝑅 +
0.587𝐺𝐺 + 0.114𝐵𝐵) where R, G and B are the pixel values from the red, green and blue channels
of the color image, respectively, and Round is the round off operator.

#### Gradient operator
Here Prewitt’s operator is used for the computation of horizontal and vertical to compute gradient magnitudes.
Normalize and round off the gradient magnitude to integers within the range [0, 255]. Next, compute the gradient angle. For image
locations where the templates go outside of the borders of the image, assign a value of 0 to both
the gradient magnitude and gradient angle. Also, if both 𝐺𝐺𝑥𝑥 and 𝐺𝐺𝑦𝑦 are 0, assign a value of 0 to
both gradient magnitude and gradient angle.

#### HOG feature
Here we are using the unsigned representation and quantize the gradient angle into one of the 9 bins as shown in the
table below. If the gradient angle is within the range [180, 360), simply subtract the angle by 180
first. Use the following parameter values in your implementation: cell size = 8 x 8 pixels, block
size = 16 x 16 pixels (or 2 x 2 cells), block overlap or step size = 8 pixels (or 1 cell.) Use L2
norm for block normalization. Leave the histogram and final feature values as floating point
numbers
\
![image](https://b2-ac9137.s3.amazonaws.com/462251-df4e-b2c0-8825-fd286b77c4a_Untitled2.png)

#### Training and test images:
A set of 20 training images and a set of 10 test images in .bmp format
will be provided. The training set contains 10 positive (human) and 10 negative (no human)
samples and the test set contains 5 positive and 5 negative samples. All images are of size 160
(height) X 96 (width). With the given image size and the parameters given above for computing
the HOG feature, there are 20 X 12 cells and 19 X 11 blocks in the detection window. The
dimension of the HOG feature vector is 7,524.

[embed]http://example.com/file.pdf[/embed]
