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
validation <- predict(preProc, raw_validation[,-length(raw_training)])
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
## [1] 0.9723
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
##          A 2206   37    4    6    2
##          B    6 1456   21    3    7
##          C    7   22 1323   49    6
##          D   12    2   18 1225    8
##          E    1    1    2    3 1419
## 
## Overall Statistics
##                                         
##                Accuracy : 0.972         
##                  95% CI : (0.968, 0.976)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.965         
##  Mcnemar's Test P-Value : 4.36e-07      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.988    0.959    0.967    0.953    0.984
## Specificity             0.991    0.994    0.987    0.994    0.999
## Pos Pred Value          0.978    0.975    0.940    0.968    0.995
## Neg Pred Value          0.995    0.990    0.993    0.991    0.996
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.281    0.186    0.169    0.156    0.181
## Detection Prevalence    0.287    0.190    0.179    0.161    0.182
## Balanced Accuracy       0.990    0.977    0.977    0.973    0.991
```
