# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 shows the car navigating the track in autonomous mode.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following architecture<a name="table"></a>:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Lambda| pixel normalization   							|
| Cropping| Remove redundant image pixels (50 top 20 bottom, output 90x320x3  							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 90x320x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 45x160x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 45x160x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride
| Fully connected		| 120 out       									|
| RELU					|												|
| Fully connected		| 84 out       									|
| RELU					|												|
| Fully connected		| 43 out       									|
| RELU					|												|

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple ConvNet and process the training data.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate starting point because of it's simplicity which aided understanding the benefits of training data processing.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track; however I kept modifying the training data see v1 to v4.ipynb.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road with the original NN architecture I selected.

#### 2. Final Model Architecture

See [Network Architecture Definition](table).

#### 3. Creation of the Training Set & Training Process

These are the steps I took to process the training, validation data:

 1. To augment the data sat, I also flipped images and angles. This doubled the amount of training data. Also reduced the bias in the training data to left turns and initially kept the car in the center on straight runs.
 2. I then cropped the top 50 and bottom 20 pixels to remove unnecessary information needed for lane positioning. This allowed the network to train on more relevant features in the data. 
 3. I used side camera to give a perspective different to the center of the lane with a 0.2 offset on the steering angle toward the center. This allowed the NN to learn compensation to adjust back to the lane center.
 4. I finally randomly shuffled the data set and put 20% of the data into a validation set. 
 5. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5, in addition I used an adam optimizer so that manually training the learning rate wasn't necessary.
