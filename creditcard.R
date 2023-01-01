library(tidyverse)
library(dplyr)
library(caret)
library(lubridate)
library(parallel)
library(doParallel)
library(broom)
library(pROC)
df = read.csv('creditcard.csv', header = T)

set.seed(1)

#Configuring Parallel Processing
cluster = makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl = trainControl(method = "cv",
                          number = 5,
                          allowParallel = TRUE)

df$repeat_retailer = as.factor(df$repeat_retailer)
df$used_chip = as.factor(df$used_chip)
df$used_pin_number = as.factor(df$used_pin_number)
df$online_order = as.factor(df$online_order)
df$fraud = as.factor(df$fraud)

inTrain = createDataPartition(y=df$fraud, p=0.7, list=FALSE)
training = df[inTrain,]
testing = df[-inTrain,]


system.time(modfit <- train(fraud ~., method="rf", data=training, trControl = fitControl))


pred = predict(modfit, testing)
testing$predCorrect = pred==testing$fraud
table(pred, testing$fraud)

qplot(distance_from_home, ratio_to_median_purchase_price, colour=predCorrect, data=testing, main="newdata Predictions" )

