rm = (list = ls())

setwd('C:/Users/user/Desktop/Loan_Predictor')

library(devtools)
library(caTools)
library(caret)
library(e1071)
library(randomForest)
library(ggplot2)
library(MASS, quietly = TRUE)

#let load the dataset
loandb <- read.csv(file.choose(), header = T)

#Want to know the columns names
colnames(loandb)



#Checking the missing Value
sapply(loandb , function(x) sum(is.na(x)))


# Y values are missing at random, then the incomplete cases contribute no information to the regression of Y on X1,...,Xp
# 
# In other words, when the X's are complete, there is no need for imputation, because 
# maximum-likelihood estimates can be obtained simply by deleting the cases with missing Y. 
# 
# Using imputed Y values in analysis would simply add noise to these estimates

#Let extract the default value which are not null.
#Keep the orginal dataset copy
orginal_db <- read.table(file.choose(), header = T, sep = ",")

#And lets Drop Nan Values
data_prep <- na.omit(loandb)

#Check Again for missing value
sapply(data_prep , function(x) sum(is.na(x)))


##check the accuracy
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}


#Splitting the Features

set.seed(123)
split = sample.split(data_prep$default, SplitRatio = 0.75)
training_set = subset(data_prep, split == TRUE)
test_set = subset(data_prep, split == FALSE)

length(data_prep)

#Scale the dataset
training_set[, 1:8] = scale(training_set[, 1:8])
test_set[, 1:8] = scale(test_set[, 1:8])

#training_set[, 1:8] = training_set[, 1:8]
#test_set[, 1:8] = test_set[, 1:8]


# # # Models to Evaluate

# We will compare five different machine learning Cassification models:

# 1 - Logistic Regression
# 2 - K-Nearest Neighbors Classification
# 3 - DecisionTreeClassifier
# 4 - RandomForestClassifier
# 5 - SVM


#************************************** Logistic Regression ***************************************
#Fitting Logistic Regression to training set
logisticRegression = glm(formula = default ~ ., family = binomial, data = training_set)

summary(logisticRegression)

#Predicting test set results
prob_pred = predict(logisticRegression, type = "response", newdata = test_set[-9] )

Y_prep = ifelse(prob_pred > 0.5, 1, 0)

#Making the Confusion Matrix
cm = table(test_set[, 9], Y_prep)
accuracy(cm)
#Accuracy is 73 percent

#FNR
#fnr = FN /FN + TP
#fnr_lr = 63.04


#    0   1
#0  111  18
#1  29  17

#**Confusion Matrix**
  
#True positives (TP): Predicted positive and are actually positive.

#False positives (FP): Predicted positive and are actually negative.

#True negatives (TN): Predicted negative and are actually negative.

#False negatives (FN): Predicted negative and are actually positive.

######################################################################

#Accuracy : Proportions of total number of correct result.

#(True Negative + True Positive / Total observation)

#Precision : Proportion of correct positive results out of all predicted positive results.

#(True Positive / True Positive + False Positive)

#Recall : Proportion of actual positive cases.

#(True Positive / True Positive + False Negative)

#F1 Score : Harmonic mean of Precision and Recall

#(2 * Precision * Recall / Precision + Recall)


caret::confusionMatrix(as.factor(Y_prep), as.factor(test_set[, 9]),
                       positive="1", mode="everything")

#Precision is 48 percent.
#From the confusion Matrix we can see that we make 10 wrong approval. So that could be risk.
#So we try to make less accuracy but the bad loan approvals can not be more than 5%
#So we will try to increase our precision

#Lets increase Threshold value
Y_prep_2 = ifelse(prob_pred > 0.80, 1, 0)

#Making the Confusion Matrix
cm_new = table(test_set[, 9], Y_prep_2)


##check the accuracy
accuracy(cm_new)

caret::confusionMatrix(as.factor(Y_prep_2), as.factor(test_set[, 9]),
                       positive="1", mode="everything")


lr_model <- train(default ~ ., data = data_prep, methods = 'lr',
                  trControl = caret::trainControl(method = 'cv',
                                               number = 10, verboseIter = TRUE))

lr_model$results
#Best tune mtry is 2
#Best RMSE Value is 0.3799328

#mtry: Number of variables randomly sampled as candidates at each split. ntree: Number of trees to grow

#mtry      RMSE     Rsquared   MAE     RMSESD     RsquaredSD   MAESD
#1    2 0.3799328 0.2572368 0.2966822 0.02662557 0.10814823 0.02011422
#2    6 0.3832223 0.2436553 0.2895288 0.02752700 0.10010814 0.02049789
#3   10 0.3840061 0.2407093 0.2875899 0.02779062 0.09860057 0.02156737


#Accuracy is 75 percent
#Now Precison is 71 percent


#So, After change the threshold value we came to an point the 80 percent id good for model
#   0   1
#0 123   6
#1  40   6

#Summary
#So After changing threshold values we came to an end. And 0.8 or 80 percent is good enough to predict.

#Good Loans
#Anything above 80 percent probability - Approve

#Bad Loans
#Anything below 60 percent probability - Reject

#Manual Check
#The remaining 61 - 79 percent we should manually check.


################################################################
#Lets work on AUC ROC Curve
library(ROCR)
#pred <- predict(logisticRegression, prob_pred, type = "prob")
pred <- prediction(prob_pred, test_set$default)
eval <- performance(pred, "acc")
plot(eval)

#Now to make best cutoff line in the chart
abline(h = 0.70, v = 0.20)


#**************************************** Gain And Lift Cart ************************************************
#Lets Create a Gain Chart.
gain <- performance(pred, "tpr", "rpp")
#rpp- Rate of positive predictions
plot(gain, main = "Gain Chart")

#Now lets work on Lift Cart
# Creating the cumulative density
data_prep$cumden <- cumsum(data_prep$default)/sum(data_prep$default)

# Creating the % of population
data_prep$perpop <- (seq(nrow(data_prep))/nrow(data_prep))*100

# Ploting
plot(data_prep$perpop,data_prep$cumden,type="l",xlab="% of Sample",ylab="% of Default's")

#*************************************** Random Forest ******************************************************
r_forest <- randomForest(x = training_set[-9], y = training_set$default, ntree=100)

Y_predict_rForest <- predict(r_forest, newdata = test_set[-9] )

Y_prep_rf = ifelse(Y_predict_rForest > 0.5, 1, 0)

#summary(r_forest)

rf_model <- train(default ~ ., data = data_prep, methods = 'rf',
                         trControl = caret::trainControl(method = 'cv',
                                                         number = 10, verboseIter = TRUE))


#Confusion MAtrix for random Forest
cm_rf = table(test_set[,9], Y_prep_rf)


#   0   1
#0 112  17
#1  28  18


##check the accuracy
accuracy(cm_rf)

caret::confusionMatrix(as.factor(Y_prep_rf), as.factor(test_set[, 9]),
                       positive="1", mode="everything")

#Accuracy is 73 percent
#Precision is 48 percent.

#FNR
#fnr = FN /FN + TP
#fnr_rf = 40.62

rf_model$results
#Best tune mtry is 2
#Best RMSE Value is 0.3765470

#mtry      RMSE  Rsquared       MAE     RMSESD    RsquaredSD      MAESD
#1    2 0.3765470 0.2830775 0.2950881 0.03393424  0.1110003 0.02648367
#2    6 0.3817009 0.2628440 0.2895317 0.03792641  0.1052058 0.02923205
#3   10 0.3811210 0.2657399 0.2860019 0.03941772  0.1071793 0.03050668


#*********************************** Desicision Tree *************************************************
#install.packages('rpart')
library('rpart')

training_set$default <- factor(training_set$default)
tree <- rpart(default ~ ., training_set)
predicted_classes <- predict(tree, test_set[-9], type = "class")

#Confusion Matrix for random Forest
cm_dt = table(test_set[,9], predicted_classes)

Y_prep_dt = as.numeric(as.character(predicted_classes))

##check the accuracy
accuracy(cm_dt)
#This Confusion Marix is for knowing all the Parameters
caret::confusionMatrix(as.factor(Y_prep_dt), as.factor(test_set[, 9]), 
                       positive="1", mode="everything")

#Accuracy is 70 percent
#Precision is 40 percent.

dt_model <- train(default ~ ., data = data_prep, methods = 'dt',
                  trControl = caret::trainControl(method = 'cv',
                                                  number = 10, verboseIter = TRUE))

dt_model$results
#Best tune mtry is 2
#Best RMSE Value is 0.3791385

#FNR
#fnr = FN /FN + TP
#fnr_rf = 50

#mtry      RMSE  Rsquared       MAE     RMSESD RsquaredSD      MAESD
#1    2 0.3791385 0.2638706 0.2981990 0.02486873 0.07260134 0.01988804
#2    6 0.3825914 0.2493257 0.2913170 0.02675885 0.07611397 0.02126562
#3   10 0.3846746 0.2432679 0.2902018 0.02764392 0.07706274 0.02191712

#*********************************** KNN *************************************************
library(class)
y_pred_knn = knn(train = training_set[, -9], test = test_set[, -9], 
                 cl = training_set[, 9], k = 5)

#Confusion Matrix for random Forest
cm_knn = table(test_set[,9], y_pred_knn)

Y_prep_knn = as.numeric(as.character(y_pred_knn))

##check the accuracy
accuracy(cm_knn)

caret::confusionMatrix(as.factor(Y_prep_knn), as.factor(test_set[, 9]),
                       positive="1", mode="everything")

#Accuracy is 70 percent
#Precision is 40 percent.



##*********************************** SVM *************************************************

classifier = svm(formula = default ~ ., 
                 data = training_set, 
                 type = 'C-classification', 
                 kernel = 'linear')

y_pred_svm = predict(classifier, newdata = test_set[-9])

#   0   1
#0 111  18
#1  28  18

#Confusion Matrix for random Forest
cm_svm = table(test_set[,9], y_pred_svm)



caret::confusionMatrix(as.factor(y_pred_svm), as.factor(test_set[, 9])
                       , positive="1", mode="everything")

#Accuracy is 73 percent
#Precision is 50 percent.

#To check other paarameters 
svm_model <- train(default ~ ., data = data_prep, methods = 'svm',
                   trControl = caret::trainControl(method = 'cv',
                                                   number = 10, verboseIter = TRUE))

svm_model$results
#mtry  Accuracy     Kappa    AccuracySD   KappaSD
#1    2 0.8096154 0.4398118 0.05978613 0.1984842
#2    5 0.8000726 0.4367423 0.06647885 0.2041928
#3    8 0.8153846 0.4876167 0.06213745 0.1734780

#And we can see the accuracy increased to 81 percent


#**************************************** Conclusion *************************************************
#Random Forest is intrinsically suited for multiclass problems,
#while SVM is intrinsically two-class

# So I prefered SVM over Random Forest for modeling

#So From the above calculation SVM 
#is a better fit because of Accuracy

## save this model
#save(rf_model, file = "rf_model.R")
saveRDS(svm_model, "svm_model.R")


## check if model exists? :
my_model <- readRDS("svm_model.R")

#Free of the memory
gc()
memory.size(max=F)

