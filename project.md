---
title: "Practical Machine Learning Course Project"
author: "Haojun Zhu"
date: "August 19, 2015"
output: html_document
---

## Introduction 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data Processing

### Data Input

First, load the necessary packages. 

```r
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(randomForest)
```

Download the training data and test data for this project. 


```r
setwd('~/Coursera/Practical Machine Learning/project')
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
file_train <- "~/pml-training.csv"
file_test <- "~/pml-testing.csv"
if (!file.exists(file_train)){
  download.file(url_train, destfile = file_train)
}
if (!file.exists(file_test)){
  download.file(url_test, destfile = file_test)
}
```

Read the raw data sets into R.

```r
train_raw <- read.csv(file_train)
test_raw <- read.csv(file_test)
```


### Data Cleaning 

First, columns that contain missing values are excluded. 


```r
train_raw <- train_raw[, colSums(is.na(train_raw)) == 0]
test_raw <- test_raw[, colSums(is.na(test_raw)) == 0]
```

Since there are several columns, such as timesteps, that do not contribute to the prediction, we remove them from both the training and test sets. Also, columns that consistute of non-numeric values are excluded as well. 


```r
# remove columns whose names contain X, timestamp, or window
trainRemove <- grepl("^X|timestamp|window", names(train_raw))
train_raw <- train_raw[, !trainRemove]

# only keep columns that have numeric values 
train_clean <- train_raw[, sapply(train_raw, is.numeric)]

train_clean$classe <- train_raw$classe


# remove columns whose names contain X, timestamp, or window
testRemove <- grepl("^X|timestamp|window", names(test_raw))
test_raw <- test_raw[, !testRemove]

# only keep columns that have numeric values
test_clean <- test_raw[, sapply(test_raw, is.numeric)]

test_clean$classe <- test_raw$classe
```

The remaining columns are therefore. 

```r
colnames(train_clean)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

### Data Slicing

For the cleaned training set, we split 60% into sub-training set, and 40% into sub-test set. 


```r
inTrain <- createDataPartition(train_clean$classe, p = 0.6, list = FALSE)
training <- train_clean[inTrain, ]
testing <- train_clean[-inTrain, ]
```

## Algorithm

### Random Forest 



```r
fitRF <- train(classe ~ ., data=training, method="rf", trControl = trainControl(method='cv', 3), ntree=200)
fitRF
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 7850, 7850, 7852 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9843751  0.9802334  0.0005259340  0.0006639278
##   27    0.9868376  0.9833497  0.0015570727  0.0019709066
##   52    0.9812329  0.9762597  0.0008992698  0.0011416747
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
predictRF <- predict(fitRF, testing)
confusionMatrix(testing$classe, predictRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228    4    0    0    0
##          B   12 1504    2    0    0
##          C    0   15 1346    7    0
##          D    0    1   19 1262    4
##          E    0    1    3    5 1433
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9907          
##                  95% CI : (0.9883, 0.9927)
##     No Information Rate : 0.2855          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9882          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.9862   0.9825   0.9906   0.9972
## Specificity            0.9993   0.9978   0.9966   0.9963   0.9986
## Pos Pred Value         0.9982   0.9908   0.9839   0.9813   0.9938
## Neg Pred Value         0.9979   0.9967   0.9963   0.9982   0.9994
## Prevalence             0.2855   0.1944   0.1746   0.1624   0.1832
## Detection Rate         0.2840   0.1917   0.1716   0.1608   0.1826
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9970   0.9920   0.9895   0.9935   0.9979
```
