#Creating a random forests model to flag whether or not credit card transactions are fraudlet
#Dataset Source: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
library(tidyverse)
library(dplyr)
library(caret)
library(lubridate)
library(parallel)
library(doParallel)
library(broom)
library(pROC)

df = read.csv('creditcard.csv', header = T)
#Reducing number of rows in dataset to make more computational processing more manageable by randomly selecting 100,000 rows
set.seed(1)
df = sample_n(df, 100000)

#Configuring Parallel Processing, for faster computational performance
cluster = makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl = trainControl(method = "cv",
                          number = 5,
                          allowParallel = TRUE)

#Casting variables as correct datatype
df$repeat_retailer = as.factor(df$repeat_retailer)
df$used_chip = as.factor(df$used_chip)
df$used_pin_number = as.factor(df$used_pin_number)
df$online_order = as.factor(df$online_order)
df$fraud = as.factor(df$fraud)

#Separating data into training and testing subsets
inTrain = createDataPartition(y=df$fraud, p=0.7, list=FALSE)
training = df[inTrain,]
testing = df[-inTrain,]

#Building random forests model
system.time(modfit <- train(fraud ~., method="rf", data=training, trControl = fitControl))

#Evaluating model vs testing data
pred = predict(modfit, testing)
testing$predCorrect = pred==testing$fraud
confusionMatrix(as.factor(pred), as.factor(testing$fraud))
qplot(distance_from_home, ratio_to_median_purchase_price, colour=predCorrect, data=testing, main="newdata Predictions" )

#######RESULTS########
#Confusion Matrix and Statistics

#          Reference
#Prediction     0     1
#         0 27398     4
#         1     0  2597
                                     
#               Accuracy : 0.9999     
#                 95% CI : (0.9997, 1)
#    P-Value [Acc > NIR] : <2e-16                        
#            Sensitivity : 1.0000     
#            Specificity : 0.9985  


