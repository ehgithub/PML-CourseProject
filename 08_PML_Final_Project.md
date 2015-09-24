# Practical Machine Learning - Course Project
Edgar Hamon  
September/2015  

## Executive Summary
The purpose of this project is to present a Machine Learning prediction based on accelerometers data and to present the results to the peer students of the John Hopkins data science specialization at Coursera. 

### Background 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


### Exploratory Data Analysis

First, lets set the R environment for the analysis


```r
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(rattle)

rm(list = ls())
```

#### Downloading & reading data
This portion of code assumes you will store the data files in the current working directory and does not validate if the files already exist. *Note:* code for downloading processing commented out.


```r
# trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download.file(trainUrl, destfile="pml-training.csv", method="auto")
# download.file(testUrl, destfile="pml-testing.csv", method="auto")

Dtrain <- read.csv("pml-training.csv")
Dtest  <- read.csv("pml-testing.csv")
```

#### Data Analysis and cleaning


```r
dim(Dtrain)
```

```
## [1] 19622   160
```

```r
dim(Dtest)
```

```
## [1]  20 160
```

```r
# summary(Dtrain)
# str(Dtrain)
```

We have 19622 observations and 160 variables in the training dataset, and 20 observations and 160 variables in the test data set.

Also, using "summary"" and "str"" functions we can see there are a lot of variables with NA's and many numeric variables load as factors due to "Div/0" text contained in the variable.
*note:* Summary and str functions are commented out in the code to save report space.

We are going to remove NA's columns and also to remove near zero variables. We apply the cleaning to both, the training and the test datesets


```r
Dtemp <- Dtrain[, colSums(is.na(Dtrain)) == 0] ## eliminates columns with NAs
trainnzv <- nearZeroVar(Dtemp, saveMetrics=FALSE)
Dtrain <- Dtemp[-trainnzv]   ## eliminates columns with near zero value
Dtrain <- Dtrain[-1]      ## Remove 1st column
### repeat same cleaning in test dataset
Dtemp <- Dtest[, colSums(is.na(Dtest)) == 0]
testnzv <- nearZeroVar(Dtemp, saveMetrics=FALSE)
Dtest <- Dtemp[-testnzv]   
Dtest <- Dtest[-1]      
dim(Dtrain)
```

```
## [1] 19622    58
```

```r
dim(Dtest)
```

```
## [1] 20 58
```

As a result, now we have a training dataset with 19622 observations and 58 variables. Train dataset has now 20 observations and 58 variables. (Instead of 160 variables we had originally)


### Prediction model

We can split the cleaned training set into a pure training data set (70%) and a verification data set (30%). 


```r
set.seed(1702) # For reproducibile purposes
inTrain <- createDataPartition(Dtrain$classe, p=0.70, list=F)
trainData <- Dtrain[inTrain, ]
veriData <- Dtrain[-inTrain, ]
```

Based on various tests with this dataset and also based on Course lectures, we choose to fit a predictive model for activity recognition and prediction using Random Forest algorithm because:

- It is capable to select automatically important variables
- It can to correlate covariates and outliers

We will use 8-fold cross validation for the algorithm and we will have 280 tree nodes.(These numbers were selected based on experimentations and trial & errors to find the optimal output)


```r
control <- trainControl(method="cv", 8)
model <- train(classe ~ ., data=trainData, method="rf", trControl=control, ntree=280)
model
```

```
## Random Forest 
## 
## 13737 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (8 fold) 
## Summary of sample sizes: 12018, 12018, 12020, 12022, 12021, 12021, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9887902  0.9858181  0.0019076105  0.0024135906
##   40    0.9994175  0.9992633  0.0006228961  0.0007878999
##   79    0.9991266  0.9988954  0.0006958319  0.0008800691
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 40.
```

Once we have the model, we estimate the performance of the model on the verification dataset.


```r
prediction <- predict(model, veriData)
confusionMatrix(veriData$classe, prediction)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    1 1138    0    0    0
##          C    0    2 1024    0    0
##          D    0    0    3  961    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9988          
##                  95% CI : (0.9976, 0.9995)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9982   0.9971   0.9990   1.0000
## Specificity            1.0000   0.9998   0.9996   0.9994   0.9998
## Pos Pred Value         1.0000   0.9991   0.9981   0.9969   0.9991
## Neg Pred Value         0.9998   0.9996   0.9994   0.9998   1.0000
## Prevalence             0.2846   0.1937   0.1745   0.1635   0.1837
## Detection Rate         0.2845   0.1934   0.1740   0.1633   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9990   0.9983   0.9992   0.9999
```

```r
accur <- postResample(prediction, veriData$classe)
accur
```

```
##  Accuracy     Kappa 
## 0.9988105 0.9984954
```

```r
model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 280, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 280
## No. of variables tried at each split: 40
## 
##         OOB estimate of  error rate: 0.06%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3906    0    0    0    0 0.0000000000
## B    1 2655    2    0    0 0.0011286682
## C    0    2 2394    0    0 0.0008347245
## D    0    0    2 2249    1 0.0013321492
## E    0    0    0    0 2525 0.0000000000
```

The estimated accuracy of the model is 99.8% and the estimated error is 0.06%.


### Predicting for Test Dataset

Now, we apply the model fitted above, to the original testing dataset (Taken from original data source). *Note:* problem_id column is removed first.

Not showing the final results here, since it will contain the answers for the second part of the course project


```r
final <- predict(model, Dtest[, -length(names(Dtest))])
varImp(model)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 79)
## 
##                                Overall
## raw_timestamp_part_1           100.000
## num_window                      47.216
## roll_belt                       41.803
## pitch_forearm                   26.145
## magnet_dumbbell_z               18.133
## yaw_belt                        12.988
## magnet_dumbbell_y               12.741
## pitch_belt                      11.998
## cvtd_timestamp30/11/2011 17:12  10.519
## roll_forearm                     9.244
## cvtd_timestamp02/12/2011 13:33   8.215
## cvtd_timestamp02/12/2011 14:58   8.058
## cvtd_timestamp28/11/2011 14:15   7.995
## magnet_dumbbell_x                7.845
## roll_dumbbell                    6.261
## accel_dumbbell_y                 5.717
## accel_belt_z                     5.551
## cvtd_timestamp05/12/2011 11:24   4.856
## accel_forearm_x                  4.682
## cvtd_timestamp02/12/2011 13:35   4.395
```
