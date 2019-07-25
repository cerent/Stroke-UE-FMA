


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

data_used # name of the data used for the study

output<-"Output_class" # the name of the output

# Create a function to write the classification results and variable importance measurements in the same list
multiResultClass <- function(result1=NULL,result2=NULL)
{me <- list(
  result1 = result1,
  result2 = result2)
#Set the name for the class
class(me) <- append(class(me),"multiResultClass")
return(me)}

# Create matrices to save the results during the cross-validation process
result_cross_validation<-matrix(ncol=13,nrow=10) 
result_cross_validation_100iteration<-NULL
var_imp_matrix<-NULL

# Prepare the codes for the parallelisation
myCluster <- makeCluster(10)
registerDoMC(10)


# Turn 10 parallel program 10 times (different partition for each iteration)

result_parallel<-foreach(it=rep(1:10,10), .combine = rbind,.multicombine=TRUE,
                                             .packages=c("nnet","rminer","caret","AUC","e1071","randomForest","rpart")) %dopar% {cat(it)
                                               
                                               # Split the data in 10 partitions
                                               folds_outerloop<-createFolds(factor(data_used$Output_class),k=10,list = FALSE)                                                                   
                                               
                                               # Start the outer loop of the nested CV
                                               for(outerloop in 1:10){
                                                 
                                                 # Create a train dataset using 9 partitions over 10 
                                                 trainData <- data_used[folds_outerloop != outerloop, ]
                                                 
                                                 # Create a test dataset using 1 partition over 10  
                                                 testData <- data_used[folds_outerloop == outerloop, ]
                                                 
                                                 # loop to chose best hyperparameters
                                                 error_innerloop<-matrix(ncol=3,nrow=3311)
                                                 k=1
                                                 for (alpha1 in seq(0,1,0.1)) {
                                                   for (lambda1 in seq(0,3,0.01)) {
                                                     
                                                     
                                                     err_cv<-matrix(ncol=2,nrow=10) 
                                                     err_cv1<-NULL
                                                     
                                                     # Repeat CV 10 times for different partitions
                                                     for (iterinner in 1:10) {
                                                       
                                                       # Create partitions from train dataset
                                                       folds_outerloop_inner<-createFolds(factor(trainData$Output_class),k=10,list = FALSE)
                                                       
                                                       for(innerloop in 1:10){
                                                         
                                                         # Create a training dataset using 9 partitions over 10 
                                                         
                                                         trainingData <- trainData[folds_outerloop_inner != innerloop, ] 
                                                         
                                                         # Create a validation dataset using 1 partitions over 10 
                                                         
                                                         validationData <- trainData[folds_outerloop_inner == innerloop, ]
                                                         
                                                         
                                                         # Normalize training dataset
                                                         performScaling <- TRUE  # Turn it on/off for experimentation.
                                                         
                                                         if (performScaling) {
                                                           
                                                           # Loop over each column.
                                                           for (colName in names(trainingData)) {
                                                             
                                                             # Check if the column contains numeric data.
                                                             if(class(trainingData[,colName]) == 'integer' | class(trainingData[,colName]) == 'numeric') {
                                                               
                                                               # Scale this column (scale() function applies z-scaling).
                                                               trainingData[,colName] <- as.numeric(scale(trainingData[,colName]))
                                                             }
                                                           }
                                                         }
                                                         
                                                         
                                                         # Balance the normalized training dataset
                                                         library(DMwR)
                                                         trainingData_smote <- SMOTE(Output_class ~ ., trainingData, perc.over = 100)
                                                         
                                                         # Fit the EN method
                                                         trainingData_smote$Output_Class<-as.factor(trainingData_smote$Output_Class)
                                                         x<-data.matrix(trainingData_smote[,-which(names(trainingData_smote) %in% "Output_class")])
                                                         y<-data.matrix(trainingData_smote[,which(names(trainingData_smote) %in% "Output_class")])
                                                         innermodel_enet_prob <- glmnet(x,y,lambda=lambda1,alpha=alpha1,family = "binomial")      
                                                         
                                                         # Normalize the validation dataset using the information of training dataset
                                                         normParam <- preProcess(trainingData)
                                                         norm.validationData <- predict(normParam, validationData)
                                                         
                                                         #Predict the validation dataset
                                                         norm.validationData$Output_class<-NULL
                                                         # Predict the validation dataset
                                                         predict_validation_enet_prob<-predict(innermodel_enet_prob,newx = data.matrix(norm.validationData),s=lambda1,type="response") 
                                                         
                                                         #AUC
                                                         y.test<-as.numeric(norm.validationData[,which(names(norm.validationData) %in% output)])
                                                         y.test<-factor(y.test)
                                                         predict_validation_enet_prob<-as.numeric(predict_validation_enet_prob)
                                                         auc_test<-auc(roc(predict_validation_enet_prob,labels = y.test))
                                                         
                                                         err_cv[innerloop,1]<-innerloop
                                                         err_cv[innerloop,2]<-auc_test
                                                         
                                                       }
                                                       # Decide the best hyperparameters based on AUC
                                                       
                                                       mean_error_10cv<-mean(err_cv[,2])
                                                       error_innerloop[k,1]<-alpha1
                                                       error_innerloop[k,2]<-lambda1
                                                       error_innerloop[k,3]<-mean_AUC_inner_10cv
                                                       k=k+1
                                                     }
                                                   }
                                                   
                                                   # best (decay,size) couple 
                                                   best_alpha<-error_innerloop[which.max(error_innerloop[,3]),1]
                                                   best_lambda<-error_innerloop[which.max(error_innerloop[,3]),2]
                                                   
                                                   # Normalize the train dataset
                                                   library(DMwR)
                                                   # Normalize training dataset
                                                   performScaling <- TRUE  # Turn it on/off for experimentation.
                                                   
                                                   if (performScaling) {
                                                     
                                                     # Loop over each column.
                                                     for (colName in names(trainData)) {
                                                       
                                                       # Check if the column contains numeric data.
                                                       if(class(trainData[,colName]) == 'integer' | class(trainData[,colName]) == 'numeric') {
                                                         
                                                         # Scale this column (scale() function applies z-scaling).
                                                         trainData[,colName] <- as.numeric(scale(trainData[,colName]))
                                                       }
                                                     }
                                                   }
                                                   
                                                   # Balance the normalized train dataset
                                                   trainData_SMOTE <- SMOTE(Output_class ~ ., trainData, perc.over = 100)
                                                   
                                                   x<-data.matrix(trainData_SMOTE[,-which(names(trainData_SMOTE) %in% "Output_class")])
                                                   y<-data.matrix(trainData_SMOTE[,which(names(trainData_SMOTE) %in% "Output_class")])
                                                   fit_train <- glmnet(x,y,lambda=best_lambda,alpha=best_alpha,family = "binomial") 
                                                   
                                                   # Variable Importance
                                                   coef_table=coef(fit_train,s=best_lambda)
                                                   var_imp_matrix<-rbind(var_imp_matrix,as.matrix(coef_table))
                                                   # Normalize the test dataset using the information of train dataset
                                                   normParam <- preProcess(trainData)
                                                   norm.testData <- predict(normParam, testData)
                                                   
                                                   # Predict test dataset (PROBA)
                                                   norm.testData$Output_class<-NULL
                                                   enet_predict_proba<-predict(fit_train,newx = data.matrix(norm.testData),s=best_lambda,type="response") 
                                                   
                                                   y.test<-as.numeric(norm.testData[,which(names(norm.testData) %in% output)])
                                                   y.test<-factor(y.test)
                                                   enet_predict_proba<-as.numeric(enet_predict_proba)
                                                   auc_test<-auc(roc(enet_predict_proba,labels = y.test))
                                                   
                                                   # Predict test dataset (CLASS)
                                                   norm.testData$Output_class<-NULL
                                                   predict_test_elasticnet_class<-predict(fit_train,newx = data.matrix(norm.testData),s=best_lambda,type="class") 
                                                   
                                                   
                                                   #### Confusion Matrix ###
                                                   library(caret)
                                                   testData <- data_used[folds_outerloop == outerloop, ]
                                                   normParam <- preProcess(trainData)
                                                   norm.testData <- predict(normParam, testData) 
                                                   truth<-as.factor(norm.testData$Output_class)
                                                   levels(predict_test_elasticnet_class)=levels(truth)
                                                   
                                                   result <- confusionMatrix(predict_test_elasticnet_class,truth,positive="1")
                                                   
                                                   #Save the results in a matrix                  
                                                   result_cross_validation[outerloop,1]<-best_alpha
                                                   result_cross_validation[outerloop,2]<-best_lambda
                                                   result_cross_validation[outerloop,3]<-outerloop
                                                   result_cross_validation[outerloop,4]<-auc_test 
                                                   
                                                   result_cross_validation[outerloop,5]<-result$byClass['Balanced Accuracy']
                                                   result_cross_validation[outerloop,6]<-result$byClass['Sensitivity']
                                                   result_cross_validation[outerloop,7]<-result$byClass['Specificity']
                                                   result_cross_validation[outerloop,8]<-result$byClass['Precision']
                                                   result_cross_validation[outerloop,9]<-result$byClass['Recall']
                                                   result_cross_validation[outerloop,10]<-result$byClass['F1']
                                                   result_cross_validation[outerloop,11]<-result$byClass['Balanced Accuracy']
                                                   result_cross_validation[outerloop,12]<-result$byClass['Pos Pred Value']
                                                   result_cross_validation[outerloop,13]<-result$byClass['Neg Pred Value']
                                                   
                                                   # Variable Importance
                                                   coef_table=coef(fit_train,s=best_lambda)
                                                   var_imp_matrix<-rbind(var_imp_matrix,as.matrix(coef_table))                       
                                                 }
                                                 # Save the results for each iteration
                                                 result_cross_validation_100iteration<-rbind(result_cross_validation_100iteration,result_cross_validation)
                                                 
                                                 # Save the classification results and variable importance in the same list
                                                 result <- multiResultClass()
                                                 result$result1 <- result_cross_validation_100iteration
                                                 result$result2 <- var_imp_matrix
                                                 return(result)
                                               }
                                             }




