#Analysis of Alzheimer's and Dementia Patient Disposition Following Hospital Admission in Maryland
library(tidyverse)
library(dplyr)
df = read.csv('raw.csv', header = T)

#This is done to relabel values within the dataset so they are more easily intepretable
df = df %>% mutate(SOURCE.HOME = factor(SOURCE.HOME,levels = c(0,1),labels = c("Out", "Home")))
df = df %>% mutate(DISPHOME = factor(DISPHOME,levels = c(0,1),labels = c("Out", "Home")))
df = df %>% mutate(MinOfALZDEM = factor(MinOfALZDEM, levels = c("R","A"),labels = c("R", "A")))
df = df %>% mutate(SEX = factor(SEX, levels = c(1,2),labels = c("Male", "Female")))
df = df %>% mutate(MARITALSTATUS = factor(MARITALSTATUS,levels = c(1,2,3,4,5,9),labels = c("Single","Married","Separated", "Divorced", "Widow/Widower","Unknown")))

#These items are being converted to the datatype of "Factor" because they are categorical variables
df = df %>% mutate(SOURCE.HOME = as.factor(SOURCE.HOME))
df = df %>% mutate(DISPHOME = as.factor(DISPHOME))
df = df %>% mutate(SEX = as.factor(SEX))
df = df %>% mutate(PatientRace = as.factor(PatientRace))
df = df %>% mutate(MARITALSTATUS = as.factor(MARITALSTATUS))

#In order for the logistic regression to make sense, the values need to be compared against a "reference" case
df$SOURCE.HOME = relevel(df$SOURCE.HOME, ref = "Home")
df$DISPHOME = relevel(df$DISPHOME, ref = "Home")
df$PatientRace = relevel(df$PatientRace, ref = "White")


lr =glm(DISPHOME ~ SOURCE.HOME + CountOfICD +MinOfALZDEM + SEX + AGE + PatientRace + MARITALSTATUS + TotalCharlson + CountOfICD*TotalCharlson + AGE*TotalCharlson ,data=df, family=binomial())
summary(lr)


lm=lm(TotalCharlson ~ AGE, data = df)
summary(lm)
plot(TotalCharlson ~ AGE, data = df)
plot(TotalCharlson ~ CountOfICD, data = df)



