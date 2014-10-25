Predicting if barell are lifted correctly
=========================================

This is my submission to the Pratical Machine Learning Course Project. I kept it very simple, so it should be fast to read :).

Required libraries
==================


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

Data loading and processing
===========================

First let's load the data (the files are included in the repository):


```r
# columns with incorrect values are set to nan
all_training <- read.csv('pml-training.csv', na.strings = c("", "NA", "#DIV/0!") )
all_testing <- read.csv('pml-testing.csv', na.strings = c("", "NA", "#DIV/0!") )
```

Next, we have to remove the first 7 columns as they are of little use.

```r
all_training = all_training[,-(1:7)]
all_testing = all_testing[,-(1:7)]
```

Removing all columns where there are at least one NA value (it might be a little extreme, but it is simpler like that).

```r
removeNaColumns     <- function(x) { x[, colSums(is.na(x)) == 0 ] }
all_training <- removeNaColumns(all_training)
all_testing <- removeNaColumns(all_testing)
```


Let's then separate our training data into two parts, for cross validation.

```r
partition  <- createDataPartition(all_training$classe, p=.6, list=FALSE)
raw_training <- all_training[partition,]
raw_validation  <- all_training[-partition,]
```

And we then apply PCA to reduce the number of dimensions:

```r
preProc = preProcess(raw_training[,-length(raw_training)], method="pca", thresh=0.9)
training <- predict(preProc, raw_training[,-length(raw_training)])
validation <- predict(preProc, raw_validation[,-length(raw_validation)])
```

Training
========
We trained the model on Random forest:

```r
modFit <- train(raw_training$classe~ .,data=training,method="rf",prox=FALSE)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

Results
=======
We can now predict our validation set:

```r
predicted = predict(modFit, validation)
```

The overall success rate is (that should be near our out of sample error):

```r
sum(predicted == raw_validation$classe) / length(predicted)
```

```
## [1] 0.9678
```

Here is the confusion matrix:

```r
confusionMatrix(predicted, raw_validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2202   49    8    2    0
##          B   13 1433   22    2   12
##          C   11   33 1322   57   11
##          D    6    1   13 1225    8
##          E    0    2    3    0 1411
## 
## Overall Statistics
##                                         
##                Accuracy : 0.968         
##                  95% CI : (0.964, 0.972)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.959         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.987    0.944    0.966    0.953    0.979
## Specificity             0.989    0.992    0.983    0.996    0.999
## Pos Pred Value          0.974    0.967    0.922    0.978    0.996
## Neg Pred Value          0.995    0.987    0.993    0.991    0.995
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.281    0.183    0.168    0.156    0.180
## Detection Prevalence    0.288    0.189    0.183    0.160    0.180
## Balanced Accuracy       0.988    0.968    0.975    0.974    0.989
```
