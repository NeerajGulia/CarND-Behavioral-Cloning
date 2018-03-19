# **Behavioral Cloning** 

## Writeup Report - Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/loss_graph_before_dropout.png "Loss Graph before adding Dropout"
[image2]: ./images/loss_graph_after_dropout.png "Loss Graph after adding Dropout"

## Rubric Points
** Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  **

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* neeraj_model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video-track1.mp4 autonomous video on track1
* video-track2.mp4 autonomous video on track2


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py neeraj_model.h5
```

#### 3. Submission code is usable and readable

The code in model.py uses a Python generator, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model which has been tried and tested for autonomous vehicle. For this I choose the nVidia model. 

#### 2. Tuning of model parameters

After running this model successfully, I realized that this model can be tweaked to lower down the parameters counts and overall footprint of the memory, so I included strides of 2x2 which helped in reduction of number of parameters without compromising the model performance.
The Total params are: 239,419.

At the end of the process, the vehicle is able to drive autonomously around both the tracks without leaving the road.

#### 3. Final Model Architecture

The final model architecture (model.py lines 95-108) consisted of a convolution neural network with the following layers and layer sizes ...     

| Layer (type)                    |  Output Shape        |  Param #   |  Connected to           |
|---------------------------------|----------------------|------------|-------------------------|
| lambda_1 (Lambda)               | (None, 160, 320, 3)  | 0          | lambda_input_1[0][0]    |
| cropping2d_1 (Cropping2D)       | (None, 78, 320, 3)   | 0          | lambda_1[0][0]          |
| convolution2d_1 (Convolution2D) | (None, 37, 158, 24)  | 1824       | cropping2d_1[0][0]      |
| convolution2d_2 (Convolution2D) | (None, 17, 77, 36)   | 21636      | convolution2d_1[0][0]   |
| dropout_1 (Dropout)             | (None, 17, 77, 36)   | 0          | convolution2d_2[0][0]   |
| convolution2d_3 (Convolution2D) | (None, 7, 37, 48)    | 43248      | dropout_1[0][0]         |
| convolution2d_4 (Convolution2D) | (None, 3, 18, 64)    | 27712      | convolution2d_3[0][0]   |
| convolution2d_5 (Convolution2D) | (None, 1, 16, 64)    | 36928      | convolution2d_4[0][0]   |
| flatten_1 (Flatten)             | (None, 1024)         | 0          | convolution2d_5[0][0]   |
| dense_1 (Dense)                 | (None, 100)          | 102500     | flatten_1[0][0]         |
| dropout_2 (Dropout)             | (None, 100)          | 0          | dense_1[0][0]           |
| dense_2 (Dense)                 | (None, 50)           | 5050       | dropout_2[0][0]         |
| dense_3 (Dense)                 | (None, 10)           | 510        | dense_2[0][0]           |
| dense_4 (Dense)                 | (None, 1)            | 11         | dense_3[0][0]           |


* Total params: 239,419
* Trainable params: 239,419
* Non-trainable params: 0

#### 4. Creation of the Training Set & Training Process

I knew that the average racing tracks always have some bias for one side of turn, like in Track1 a lot of left turns are there. To overcome this I flipped the images vertically which balanced out the left and right turns equally.

Track2 contains a lot of turns, while most of the Track1 has straight run. Hence when I combined the training data of Track1 and Track2 I knew that the straight drive will be balanced out with the turns.

This lead to successfull run of both Tracks by the **same model**. I kept running the car in autonomous mode for half an hour in Track2 and it kept running without hitting and stopping.

** Data sets ** (used in total 6):
* Track1:    
     a) Data set provided by Udacity    
     b) Normal traversal of Track1    
     c) Reverse traversal of Track1  
    
* Track2:    
     a) Normal traversal of Track2    
     b) Reverse traversal of Track2    
     c) There were some very steep turns, one dataset of those turns    
    
After the collection process, I had 124812 number of data points. I then preprocessed this data by normalizing it first and then by cropping the top 58 pixels and bottom 20 pixels, since the area of interest was inside those pixels only.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 5. Attempts to reduce overfitting in the model

I did not observe any overfitting and hence did not include any MaxPool or Dropouts in my original model. But the Udacity reviewer rejected my submission stating that as per Rubric some mechanism to reduce overfitting has to be done. 
So to satisfy the Project Rubric points, I have added two dropout layers in the model.

But as can be seen in the below graphs, overfitting was not an issue in my original model. With the same model I was able to successfully drive the car in autonomous mode for both the tracks.    

So, I was asked to tackle a problem which was not present at all in my original model.

** Training Loss Graph Before adding Dropout: **    

![Training Loss Graph][image1]

** Training Loss Graph after adding Dropout: **    

![Training Loss Graph][image2]

The model was trained and validated on different data sets. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 6. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

* Total data folders are: 6
* Train sample count:  124812
* Validation sample count:  31206

### Final Thoughts    
This is pretty evident that for any neural network to succeed we need to provide a lot of data. This has been proved here.
I believe this is the reason that **same model is able to run in autonomous mode in both Tracks** - Track1 as well as Track2.