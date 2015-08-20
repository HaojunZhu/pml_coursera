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

## Machine Learning Algorithm

We train the data with random forest algorithm. 

First, we apply 3-fold cross validation and preprocessing on training set. 


```r
fit1 <- train(classe ~ ., data=training, method="rf", trControl = trainControl(method="cv", 3), preProcess=c("center", "scale"))
fit1
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 7851, 7851, 7850 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9847147  0.9806616  0.0007622879  0.0009659367
##   27    0.9853092  0.9814154  0.0010285471  0.0013000979
##   52    0.9753738  0.9688452  0.0021345848  0.0026980906
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
predict1 <- predict(fit1, testing)
confusionMatrix(testing$classe, predict1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2225    5    1    0    1
##          B   13 1496    8    1    0
##          C    0   12 1350    6    0
##          D    0    3   17 1265    1
##          E    0    0    4    4 1434
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9903          
##                  95% CI : (0.9879, 0.9924)
##     No Information Rate : 0.2852          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9877          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9942   0.9868   0.9783   0.9914   0.9986
## Specificity            0.9988   0.9965   0.9972   0.9968   0.9988
## Pos Pred Value         0.9969   0.9855   0.9868   0.9837   0.9945
## Neg Pred Value         0.9977   0.9968   0.9954   0.9983   0.9997
## Prevalence             0.2852   0.1932   0.1759   0.1626   0.1830
## Detection Rate         0.2836   0.1907   0.1721   0.1612   0.1828
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9965   0.9917   0.9877   0.9941   0.9987
```

Then, we do not preprocess the training set but only use 3-fold cross validation. 


```r
fit2 <- train(classe ~ ., data=training, method="rf", trControl = trainControl(method="cv", 3))
fit2
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
## Summary of sample sizes: 7852, 7850, 7850 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD   
##    2    0.9855643  0.9817351  0.0023632490  0.002990823
##   27    0.9847997  0.9807696  0.0009626757  0.001219627
##   52    0.9763078  0.9700235  0.0031910652  0.004040216
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
predict2 <- predict(fit2, testing)
confusionMatrix(testing$classe, predict2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    0    1    0    0
##          B    7 1506    5    0    0
##          C    0   10 1357    1    0
##          D    0    0   21 1264    1
##          E    0    0    4    3 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9912, 0.9949)
##     No Information Rate : 0.2852          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9915          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9969   0.9934   0.9777   0.9968   0.9993
## Specificity            0.9998   0.9981   0.9983   0.9967   0.9989
## Pos Pred Value         0.9996   0.9921   0.9920   0.9829   0.9951
## Neg Pred Value         0.9988   0.9984   0.9952   0.9994   0.9998
## Prevalence             0.2852   0.1932   0.1769   0.1616   0.1830
## Detection Rate         0.2843   0.1919   0.1730   0.1611   0.1829
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9983   0.9958   0.9880   0.9968   0.9991
```

From the confusion matrices, we find that preprocessing actually decreases accuracy. We predict for test data without prepocessing. 


```r
# remove problem_id column
test_result <- predict(fit2, test_clean[, -53])
test_result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Submitting Results for Evaluation 

Use the function provided to generate files with prediction results.

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

if (!file.exists("./answer")) {
  dir.create("./answer")
}

setwd("./answer")
pml_write_files(test_result)
```
