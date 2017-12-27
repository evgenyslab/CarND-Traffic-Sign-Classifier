# **Traffic Sign Recognition**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

# Report

This project consisted of developing a CNN for recognizing German traffic signs. The objective of the project was to achieve over 93% classification rate on the validation portion, and then apply the net to the test dataset, and further, test the net on images from the web.

### Part 1: Data set Summary and Exploration

The dataset consisted of a training set, validation set, and test set. The training and validation sets were combined into a single training set with a size of 39209 samples, while the test set consisted of 12630 samples. A total of 42 classes were available. Each sample was composed of a 32x32 image with 3 color channels.

Combining the training and validation sets was an important step to ensure the net is not being validated on the same samples every time, thus ensuring robustness.

A preview of 10 samples from each class is provided in the notebook. The preview includes a bar chart indicating the total number of training samples for each class.

Two major points are noted:
1. Several classes have very little samples compared to others, and,
2. Some of the training images are poorly composed (i.e. low contrast)

These concerns can be dealt with individually through data augmentation, and pre-processing.

Data preprocessing was achieved through converting the images from color to grey (using openCV), then improving through contrast through historgram equalization, and finally, normalization. The normalization step was included to reduce the range of input data to the neural net.

### Part 2: Design of the Neural Net

The LeNet architecture was tested with the dataset, and provided as best 88% validation accuracy. To make the network building process easier, the convolution operations and fully-connected layer operations were abstracted into their own classes. Then, the LeNet architecture was tested with different convolution depths, and fully connected depths.

The final modified LeNet (re-named to LeCNN) follows:

| Layer         		|     Description	        					|
|:-----------------:|:---------------------------------:|
| Input         		| 32x32x1 Grey image   							|
| Convolution         		| 5x5 Kernel, output depth 18 layers, 32x32x1 &rarr; 28x28x18					|
| ReLU         		| 				   	|
| Max Pool  | 2x2 Stide, output 14x14x18 |
| Convolution         		| 5x5 Kernel, output depth 64 layers, 14x14x18 &rarr; 10x10x64					|
| ReLU         		| 				   	|
| Max Pool  | 2x2 Stide, output 5x5x64 |
| Flatten  |  |
| Fully-Connected  | 1600 &rarr; 1024  |
| Fully-Connected  | 1024 &rarr; 84  |
| Fully-Connected  | 84 &rarr; 42  |

### Part 3: Training

The net was trained with the following hyperparameters:
- Optimizer:      Adam
- Epochs:         60
- Batch Size:     256
- Learning Rate:  0.00009
- Device:         GTX1070

The final model results:

| Step         		|     Classification Rate	        					|
|:-----------------:|:---------------------------------:|
| Training  | 97.6%  |
| Validation  | 94.5%  |
| Testing  | 81.7%  |


### Part 4: Tuning process

The network was based off the LeNet architecture, with the same convolution layers and fully connected layers. The main difference is in the depth and size of the connected layers.

The original LeNet was a poor classifier on the traffic sign dataset, most likely due to the low number of kernel filters in the convolution layers. Thus, one of the first steps in testing the modified architecture, was to increase the number of kernel filters in each convolution. This improved the validation classification rate of the net. Increasing the number of kernel filters results in the net's ability to learn more complex shapes and patterns associated with the traffic signs.

Similarly, the sizes of the fully connected layers were increased to accommodate the larger convolution operations in contrast to the LeNet architecture.

It is noted, that data augmentation was not used for this project, thus, the training operation did exhibit an overfit phenomena after the peak training and validation rates were achieved. Similarly, implementing dropout and learning rate momentum should have counter the overfit.

The results of the network were favourable for training and validation, but did not fare as well with the testing dataset. This suggests that the chosen network did not fully encode the attributes of each traffic sign. This could have been improved with data augmentation (especially for classes with low samples), dropout (to improve resilience to data variance and decrease overfit), and with learning momentum to improve final weight optimization.

Even through the architecture only achieved 81.7% classification rate on the test data, it provided interesting results for traffic signs recovered from the web.

### Part 5: Test model on new images

To further test the model, 5 new images of German traffic signs were gathered from the web. Four of the signs were in the dataset, and one was not. The images were cropped and resized to 32x32 pixels, and fed through the pre-processing step, and then the net.

The net was correctly able to classify 4 of the 5 signs. Further, the correctly classified class has probabilities with over 99% certainty. This suggests that either the model learned those signs very well, or the web images were near ideal. The final sign that was not classified was because it was not trained by the model, namely a 'speed limit of 130km/h' sign was taken from the web, but did not exist as a class in the net. The response of the network was quite interesting, as the top 5 probabilities included the following signs:
- 20km/h (89.2%)
- 100km/h (7.5%)
- 30km/h (2.0%)
- 70km/h (0.78%)
- 50km/h (0.33%)

These probabilities indicate that the network was capable of picking out the more complex shapes associated with the sign, specifically the '100' portion and the '20' portion. Unfortunately, without a trained class, the net could not classify the sign correctly.

### Part 6: Further work
The proposed net approached used the LeNet architecture with tuned layers and hyperparameters. To further improve the network, the following could be tested:
- Data augmentation (improves robustness)
- dropout (improves robustness)
- learning rate momentum (improves optimization and classification rates)
- convolution layer forward pass. (modification of network structure - combination of improvements in robustness and classification rates)
