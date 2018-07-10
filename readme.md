

Deep Learning aided Smartphone Pedestrian Dead Reckoning
======================================================
#### by Seo Yeon Stella Yang
Mechanical & Aerospace Engineering dep. M.S. Student in Seoul National University.






<img width="900" height="300" src = https://user-images.githubusercontent.com/10994112/42506381-b99237c8-847c-11e8-8ce0-07d87fbed14b.png></img>


- This project is deep learning Jeju camp sponsored by Google Tensorflow KR.

- My mentor Eunsol Kim in Kakao Brain also give me a lot of advice for the research.

   Thanks to them.

## 1. Introduction
My suggesting project is the deep learning application in smartphone sensor navigation research. Using smart phone sensor sequential data, we estimate the user's position with PDR(Pedestrian dead reckoning). But there is many error components. One of the critical error issue is related with "User's pose". The walking detection, Step counting, Step size algorithm is effected with this pose . If we know the pose context, we can choose the parameter more suitable. Also, when GPS outage occurs, the only INS navigation becomes fail because the gyro drifting error is bigger as time goes by. So that to Improve this error components, I want to apply the deep learning approach in this point.
If the model can classify the user pose mode and predict the INS error, the PDR navigation accuracy will be much higher. 

* input data : Smartphone INS(accel, gyro, mag) and GPS sequenctial data
* considering training model : CNN, RNN, LSTM
* output prediction : 
1) Human walking context : in-pocket, texting, calling, In-hand
2) Error estimation : In GPS outage situation, predicting the INS drifting error

## 2. Plan for Jeju Camp

* 1 week : 
	- Paper research 
    - Model design
      (Pose classification model, GPS/INS error regression model design)
* 2 week : 
     " TF CON in 07/13 "
    - Kalman fitting
    - Heading fitting
    - SC, WD, SL fitting (more data maybe need here) 
    - GPS code.
    - GPS experiment
    - GPS fitting
    - Code integration
    - Data acquisition
    - Find referenced code and Deal with cambridge data
* 3 week :
    - PDR algorithm editing
    - My model tensorflow code
      1)	Data loader
      2)	Preprocessing
      3)	Model (CNN)
      4)	Evaluation
      5)	Visualization
     - 	Model (RNN/LSTM)
      - Make 2 model which is similar but classification and regression
     - Training, Evaluating
      
* 4 week :	" Final Presentation in 07/26, 27 "
     - Discussion with mentor
     - Tensorflow code editing
     - Final presentation

## 3. Model configuration (TBD)

#### 1)	Plan 1 : Classify the User’s mode
###### Input : Cambridge smartphone dataset 
-	27 distinct users.
-	6 pose modes
-	Accelerometer, Gyro scope, Magnetometer 
-	Label : Pose mode label (one-hot)
###### Preprocessing : 
-	Combine the source code which is given from the open source
-	Change it as TFRecord form 
-	Data loader will load the data and shape it as same time interval (or spectroscopy) * (find example code : transform)
###### Model : Need to be specify what model will I use… -> discuss with mentor
-	Input side : (Dimension is needed to determined)
  1)	CNN
  2)	RNN
  3)	LSTM
 
 ** (Try first with open-sources and compare it’s accuracy and change the structure)
###### Output : User’s 6 pose modes (classification : softmax)

#### 2) Plan 2 : Prediction of INS error 
###### Input : 
- Smartphone INS sensor data : walking data in track 
- gps data : Smartphone GPS sensor data : : walking data in track
- gps true data : GPS instrument data : walking data in track

** (need to make by my self -> small dataset -> cross validation)
###### Preprocessing : same with plan 1
###### Model : same with plan 1 but the output needed to be care about because it is a regression.
###### Output : position change in epoch : regression (fully connected… in here need more info.)
+ more ideas : if it can compensate when gps is not in outage. Can compensate heading, step length etc.

 
