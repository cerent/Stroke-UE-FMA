rm(list=ls())

# Load the library

install.packages("doMC")
install.packages("DMwR")
library(glmnet)
library(AUC)
library(caret)
library(CrossValidate)
library(cvTools)
library(devtools)
library(doMC)
library(doParallel)
library(DMwR)
library(e1071)
library(elasticnet)
library(foreach)
library(plotly)
library(pROC)
library(PRROC)
library(R.utils)
library(ROCR)
library(ROSE)
library(rminer)
library(ggplot2)
library(grid)
library(lattice)
library(MASS)
library(neuralnet)
library(nnet)
library(iterators)
library("foreach")
library("doParallel")
library(doMC)

#### Install Data ####

data_used # the data used for the project
output<-"Output_Regression" # the output that we want to predict (output is continuous)

# Create matrices to save the results during the CV process
result_cross_validation<-matrix(ncol=4,nrow=10) 
result_cross_validation_100iteration<-NULL
var_imp_matrix<-NULL

# Run outerlooop for 100 different partitions of dataset
results_final_regression<-for(iterouterloop in 1:100){
  
  # Create 10 different partitions
  folds_outerloop_combined<-createFolds(factor(data_used$Output_Regression),k=10,list = FALSE)
  
  # Outer loop of the nested CV
  for(outerloop in 1:10){
    
    # Split the data as train and test dataset (9 partitions over 10 for train dataset and 1 partition over 10 for the test dataset)
    trainData <- data_used[folds_outerloop_combined != outerloop, ] 
    testData <- data_used[folds_outerloop_combined == outerloop, ]
                                                                         
# tune the hyperparameters using 10 fold CV and repeat it 10 times     
cv_10 = trainControl(method = "repeatedcv", number = 10,repeats=10)
hit_elnet = train(Output_Regression ~ ., data = trainData,method = "glmnet",trControl = cv_10,tuneLength=10,center = TRUE, scale = TRUE,
                  tuneGrid = expand.grid(.alpha = seq(0,1,0.1),.lambda =seq(0,3,0.01)))

# Best hyperparameters
alpha1_best<-hit_elnet$bestTune$alpha
lambda1_best<-hit_elnet$bestTune$lambda
    
# Prepare the train data to fit the model
x<-as.matrix(trainData[,-which(names(trainData) %in% "Output_Regression")])
y<-as.matrix(trainData[,which(names(trainData) %in% "Output_Regression")])

# Fit the model with training data
fit_train_combined<-glmnet(x,y,lambda=lambda1_best,alpha=alpha1_best,family = "gaussian",center = TRUE, scale = TRUE) # fit the model with variable selection
    
# Calculate the variable importance using the parameter coefficients
coef_table_combined=coef(fit_train_combined,s=lambda1_best)

# Predict for the Test Data
testData <- data_used[folds_outerloop== outerloop, ]
testData$Output_Regression<-NULL
new_testdata_combined<-as.matrix(testData)
predict_test_elasticnet_combined<-predict(fit_train_combined,newx = new_testdata_combined,type="response")
    
    
# calculate RMSE and R2 after the prediction of the post-intervention UE-FMA
testData <- data_used[folds_outerloop == outerloop, ]
rmse_final<-rmse(actual=testData$Output_Regression, predicted=predict_test_elasticnet_combined)

preds <- predict_test_elasticnet_combined
actual <-testData$Output_Regression
rss <- sum((preds - actual) ^ 2)  ## residual sum of squares
tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
rsq_final <- 1 - rss/tss

# Save the best hyperparameters and the results in a matrix
result_cross_validation[outerloop,1]<-alpha1_best
result_cross_validation[outerloop,2]<-lambda1_best
result_cross_validation[outerloop,3]<-rsq_final
result_cross_validation[outerloop,4]<-rmse_final

}
  
# Save the results for 100 iterations
result_cross_validation_100iteration<-cbind(result_cross_validation_100iteration,result_cross_validation)
var_imp_matrix<-cbind(var_imp_matrix,as.matrix(coef_table_combined))
}
