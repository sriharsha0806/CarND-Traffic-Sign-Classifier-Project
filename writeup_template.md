#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: random_images_dataset.png "Images from Datset"
[image2]: random_image.png "random_image"
[image3]: histogram_train.png "Histogram of training Dataset "
[image4]: histogram_validation.png "Hitogram of validation Dataset"
[image5]: histogram_test.png "Histogram of Test Dataset"
[image6]: histogram_afterprocessing_train.png "histogram of training dataset after preprocessing step"
[image7]: test_images.png "Test images selected"
[image8]: modifiedLeNEt.jpeg "Architecture"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sriharsha0806/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pickle library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 

![alt text][image1]

The histogram of training Dataset

![alt text][image2]

The histogram of Validation Dataset

![alt text][image3]

The histogram of Test Dataset

![alt text][image4]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

The image Dataset has been preprocessed 
The following operations are included:
image rotation, Random Translation, Histogram Equalization to Diversify the Data and Generating equal number of Data images per class so that neural network architecture won't be biased one for any class.

The histogram of training class after preprocessing
![alt text][image5]

I have not normalized or grayscaled images in training dataset as i thought color features and prenormalized images can be helpful for the network. I wanted the neural network architecture to be as free from the basic preprocessing steps as possible.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1X1 stride, valid padding, outputs 10x10x16   |
| RELU     |            |
| Max pooling        | 2X2 stride, outputs 5X5X16       |
| Convolution 5X5      | 1X1 stride, valid padding, outputs 1x1x400   |
| RELU     |            |
| Flatten layers from conv2 and conv3|
| Concatenate flattened layers to a single size-800 layer|
| Dropout  |
| Fully connected	layer output 43       									|     

![alt text][image8]

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the twelth cell of the ipython notebook. 

To train the model, I used the Adam optimizer. The final settings are:
* batch size: 256
* epochs : 100
* learning rate: 0.001
* mu: 0
* sigma: 0.1
* Dropout: 0.5

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixteeth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of  94.5
* test set accuracy of 92

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* My architecture is based on Sermanet/LeCun model. I made few changes to the model. It was a hit and trial model. I tried to work on different optimizers and understand them from the following [website](http://sebastianruder.com/optimizing-gradient-descent/) 

| Type of Optimizer        		      |     validation_accuracy	| Test_accuracy|
|:--------------------------------:|:-----------------------:| :-----------------:|
| AdamOptimizer         		         | 92.4  							           | 91.4  |
| AdagradOptimizer                 | 75.3                    | 74.3   |
| FTRL                             | 83.5                    | 81     |
| ProximalGradientDescentOptimizer | 75.1                    | 73.1   |
| RMSOptimizer                     | 92.4                    | 91.2   |

The rest of the optimizers are not giving that high accuracy. I want to further study on and improve my knowledge on optimization techniques

* Why did you believe it would be relevant to the traffic sign  
* The architecture achieved one of highest accuracy on traffic sign Dataset in a competition. So i wanted to implement the following paper.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 * The difference between validation accuracy and test accuracy is very low. So the network is not overfitting.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the Eighteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop-Sign   									| 
| 60 Km/h     			| 120 Km/hr 										|
| Children Crossing					| Children Crossing											|
| Ahead Only | Turn left				 				|
| 3.5 ton 			| 3.5 ton     							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This model performed very poorly with new data.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were provided in ipython notebook. The model is a strongly biased when tested in more data. From Andrew ng Class, I can infer that i have to collect more data or more diversify the image preprocessing step. In the paper the model is implemented on YUV images. In future i will try to perform different image preprocessing steps. Initially my optimizer was RMS and There was no Dropout layer in my architecture. I tried Dropout as it is one of the method used for underfitting. But it is of no use. I changed the optimizer to Adam but still the network is strongly biased one. So i can conclude that data need to taken care. I have not implement preprocessing on test images.

