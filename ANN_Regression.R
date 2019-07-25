
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
folds_outerloop<-createFolds(factor(data_used$Output_Regression),k=10,list = FALSE)
                                                    
# Outer loop of the nested CV
for(outerloop in 1:10){
                                                      
# Split the data as train and test dataset (9 partitions over 10 for train dataset and 1 partition over 10 for the test dataset)
trainData <- data_used[folds_outerloop != outerloop, ] 
testData <- data_used[folds_outerloop == outerloop, ]
                                                      
# tune the hyperparameters using the train dataset using 10 fold CV and by repeating 10 times     
trainData$Output_Regression<-as.numeric(trainData$Output_Regression)
tuneResult <- tune(nnet, Output_Regression ~ .,  data = trainData, nrepeat = 10,MaxNWts=1000,
ranges = list(decay=c(0.0001, 0.001, 0.01, 0.1, 0.5),size=seq(5,10,1), maxit=300,center = TRUE, scale = TRUE,
trace=FALSE,sampling = "cross",cross = 10,fix = 1/10))
                                                      
# Predict the test data using the best hyperparameters
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, testData,type="raw")

# save the predicted and observed post-intervention UE-FMA
predicted_combined1<-cbind(tunedModelY,testData$Output_Regression)
predicted_output<-rbind(predicted_output,predicted_combined1)
                                                      
# calculate RMSE and R2 after the prediction of the post-intervention UE-FMA
rmse <- rmse(actual=testData$Output_Regression, predicted=tunedModelY)

preds <- tunedModelY
actual <-testData$Output_Regression
rss <- sum((preds - actual) ^ 2)  ## residual sum of squares
tss <- sum((actual - mean(actual)) ^ 2)  ## total sum of squares
rsq_final <- 1 - rss/tss                                                   
                                                   
# Calculate the variable importance
var_imp<-varImp(tunedModel)
var_imp<-as.data.frame(var_imp$Overall)
names_data<-as.data.frame(names(data_used))
names_data1<-as.matrix(names_data[-which(names(trainData) %in% "Output_Regression"),])
var_imp1<-cbind(names_data1,var_imp)
var_imp2<-as.data.frame(var_imp1)
                                                      
# Save the best hyperparameters and the results in a matrix
result_cross_validation[outerloop,1]<-tuneResult$best.parameters$decay
result_cross_validation[outerloop,2]<-tuneResult$best.parameters$size
result_cross_validation[outerloop,3]<-rsq_final
result_cross_validation[outerloop,4]<-rmse


}
# Save the results for 100 iterations
result_cross_validation_100iteration<-cbind(result_cross_validation_100iteration,result_cross_validation)
var_imp_matrix<-cbind(var_imp_matrix,as.matrix(var_imp2[,2]))

}
  