rm(list=ls())

# Load the library

install.packages("doMC")
install.packages("DMwR")
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
                                                                        
## Cross-Validation to choose the best hyperparameters (max depth and minbucket)

# create a matrix to save results in the CV
cp_error_combined<-matrix(ncol=3,nrow=36)
k=1
for (maxdepth1 in seq(2,10,1)) {
  for (minbucket1 in seq(5,20,5)) {
    
    
    err_cv<-matrix(ncol=2,nrow=10) 
    err_cv1<-NULL
    # Repeat CV 10 times for different partitions
    for (iterinner in 1:10) {
      # Create partitions from train dataset
      folds_outerloop_inner<-createFolds(factor(trainData$Output_Regression),k=10,list = FALSE)
      
      for(innerloop in 1:10){
        
        trainingData <- trainData[folds_outerloop_inner != outerloop, ] 
        validationData <- trainData[folds_outerloop_inner == outerloop, ]
        
        # Fit the model using a couple of the hyperparameters
        cart_fit_training<-rpart(Output_Regression~.,data = trainingData,
                                 control = rpart.control(maxdepth = maxdepth1,minbucket=minbucket1))
        
        #Predict the validation dataset
        cart_predict_validation<-predict(cart_fit_training,validationData)
        err_cv[innerloop,1]<-innerloop
        err_cv[innerloop,2]<-rmse(actual=validationData$Output_Regression, predicted=cart_predict_validation)
      }
      err_cv1<-rbind(err_cv1,err_cv)
    }}
  
  # Decide the best hyperparameters based on RMSE
  mean_error_10cv<-mean(err_cv1[,2])
  cp_error_combined[k,1]<-maxdepth1
  cp_error_combined[k,2]<-minbucket1
  cp_error_combined[k,3]<-mean_error_10cv
  
  k=k+1
}

# select the best hyperparameters that minimize the mean of RMSE (over 10 iteration)
best_maxdepth<-cp_error_combined[which.min(cp_error_combined[,3]),1]
best_minbucket<-cp_error_combined[which.min(cp_error_combined[,3]),2]

# Fit the model using the best hyperparameters and train dataset
cart_fit_test<-rpart(Output_Regression~.,data = trainData,control = rpart.control(maxdepth = best_maxdepth,minbucket=best_minbucket))

# Calculate the variable importance
VarImpValue<-as.matrix(cart_fit_test$variable.importance)
VarImpNames<-as.matrix(names(cart_fit_test$variable.importance))
VarImpValueNames<-cbind(VarImpValue,VarImpNames,as.numeric(seq(1,length(VarImpValue),1)))
var_imp_cart_reg_postFMA<-cbind(var_imp_cart_reg_postFMA,VarImpValueNames)

# Predict using the test dataset
cart_predict_test<-predict(cart_fit_test,testData)

# save the predicted and observed post-intervention UE-FMA
predicted_combined1<-cbind(cart_predict_test,testData$Output_Regression)
predicted_output<-rbind(predicted_output,predicted_combined1)
              
# Calculate R-squared                                                          
preds <- cart_predict_test
actual <-testData$Output_Regression
rss <- sum((preds - actual) ^ 2)  ## residual sum of squares
tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
rsq_final <- 1 - rss/tss

# Calculate RMSE
rmse_final<-rmse(actual=testData$Output_Regression, predicted=cart_predict_test)


# Save the best hyperparameters and the results in a matrix
result_cross_validation[outerloop,1]<-best_maxdepth
result_cross_validation[outerloop,2]<-best_minbucket
result_cross_validation[outerloop,3]<-rsq_final
result_cross_validation[outerloop,4]<-rmse_final


  }
  # Save the results for 100 iterations
  result_cross_validation_100iteration<-cbind(result_cross_validation_100iteration,result_cross_validation)
  var_imp_matrix<-cbind(var_imp_matrix,as.matrix(svm.imp$value))
  
}
