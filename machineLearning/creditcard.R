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
#Reducing number of rows in dataset to make more computational processing more manageable
df = df[-(100001:1000000),]
set.seed(1)

#Configuring Parallel Processing
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
#          Reference
#Prediction     0     1
#         0 27383     4
#         1     4  2609
                                          
#               Accuracy : 0.9997          
#                 95% CI : (0.9995, 0.9999)
#    P-Value [Acc > NIR] : <2e-16                                                    
                                          
#            Sensitivity : 0.9999          
#            Specificity : 0.9985


