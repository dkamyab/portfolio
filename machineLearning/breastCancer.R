#Breast cancer pathology structure data from 569 patients
#Dataset Source: https://www.kaggle.com/datasets/sztuanakurun/breast-cancer
library(tidyverse)
library(dplyr)
library(ggplot2)
library(pROC)
library(caret)

#Configuring Parallel Processing, for faster computational performance
cluster = makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl = trainControl(method = "cv",
                          number = 5,
                          allowParallel = TRUE)

#Importing Data
df = read.csv('data.csv', header = T)
set.seed(1)

#Separating data into training and testing subsets
inTrain = createDataPartition(y=df$diagnosis, p=0.8, list=FALSE)
training = df[inTrain,]
testing = df[-inTrain,]

#Building random forests model
modFit = train(diagnosis ~., data=training, method="rf", prox=TRUE)
pred = predict(modFit, testing)

#Evaluating model vs testing data
testing$predCorrect = pred==testing$diagnosis
table(pred, testing$diagnosis)
confusionMatrix(as.factor(pred), as.factor(testing$diagnosis))
qplot(id, radius_mean, colour=predCorrect, data=testing, main= "newdata Predictions")


#########RESULTS#########

#Prediction  B  M
         B 70  4
         M  1 38
                                          
#               Accuracy : 0.9558          
#                95% CI : (0.8998, 0.9855)
#    P-Value [Acc > NIR] : <2e-16          
                                          
#            Sensitivity : 0.9859          
#            Specificity : 0.9048          
