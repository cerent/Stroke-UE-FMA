

# Load the library
library(PRROC)
library(CrossValidate)
library(cvTools)
library(ROCR)
library(R.utils)
library(plotly)
library(pROC)
library(elasticnet)
library(glmnet)
library(MASS)
library(devtools)
library(cvTools)
library(grid)
library(caret)
library(neuralnet)
library(cart)
library(lattice)
library(ggplot2)
library(AUC)
library(e1071)
library(rminer)
library("foreach")
library("doParallel")
library(doMC)
library(rpart)
library(randomForest)
library(PRROC)
library(CrossValidate)
library(cvTools)
library(ROCR)
library(R.utils)
library(plotly)
library(pROC)
library(elasticnet)
library(glmnet)
library(MASS)
library(devtools)
library(cvTools)
library(grid)
library(caret)
library(neuralnet)
library(nnet)
library(lattice)
library(ggplot2)
library(AUC)
library(e1071)
library(rminer)
library("foreach")
install.packages("doMC")
library(doMC)
library(R.matlab)
install.packages("DMwR")

#### Install Data ####
data_used # name of the data used for the study

output<-"Output_class" # the name of the output


# Create a function to write the classification results and variable importance measurements in the same list
multiResultClass <- function(result1=NULL,result2=NULL,result3=NULL)
{
  me <- list(
    result1 = result1,
    result2 = result2,
    result3 = result3
  )
  
  ## Set the name for the class
  class(me) <- append(class(me),"multiResultClass")
  return(me)
}

# Create matrices to save the results during the cross-validation process
result_cart_MCID_class_clinical_100iter<-matrix(ncol=8,nrow=10) 
result_cart_MCID_class_clinical_100iter1<-NULL
var_imp_cart_class_MCID<-NULL

# Prepare the codes for the parallelisation
getDoParWorkers(1:10)
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
                                                 error_innerloop<-matrix(ncol=3,nrow=36)
                                                 k=1
                                                 for (maxdepth1 in seq(2,10,1)) {
                                                   for (minbucket1 in seq(5,20,5)) {
                                                     
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
                                                         
                                                         # Fit the CART method
                                                         trainingData_smote$Output_Class<-as.factor(trainingData_smote$Output_Class)
                                                         cart_fit_test<-rpart(Output_Class~., method = "class",data = trainData,control = rpart.control(maxdepth = maxdepth1,minbucket= minbucket1))
                                                         
                                                         # Normalize the validation dataset using the information of training dataset
                                                         normParam <- preProcess(trainingData_smote)
                                                         norm.validationData <- predict(normParam, validationData)
                                                         
                                                         #Predict the validation dataset
                                                         cart_predict<-as.matrix(predict(cart_fit_test,norm.validationData,type="prob")[,2])
                                                         
                                                         #AUC
                                                         y.test<-as.numeric(norm.validationData[,which(names(norm.validationData) %in% output)])
                                                         y.test<-factor(y.test)
                                                         cart_predict<-as.numeric(cart_predict)
                                                         auc_test<-auc(roc(cart_predict,labels = y.test))
                                                         
                                                         err_cv[innerloop,1]<-innerloop
                                                         err_cv[innerloop,2]<-auc_test
                                                         
                                                       }
                                                       # Decide the best hyperparameters based on AUC
                                                       
                                                       mean_error_10cv<-mean(err_cv[,2])
                                                       error_innerloop[k,3]<-mean_error_10cv
                                                       error_innerloop[k,2]<-minbucket1
                                                       error_innerloop[k,1]<-maxdepth1
                                                       k=k+1
                                                     }
                                                   }
                                                   
                                                   # best (decay,size) couple 
                                                   maxdepthbest<-error_innerloop[which.min(error_innerloop[,3]),1]
                                                   minbucketbest<-error_innerloop[which.min(error_innerloop[,3]),2]
                                                   
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
                                                   
                                                   trainData_SMOTE$Output_Class<-as.factor(trainData_SMOTE$Output_Class)
                                                   cart_fit_test<-rpart(Output_Class~., method = "class",data = trainData,control = rpart.control(maxdepth = maxdepthbest,minbucket= minbucketbest))
                                                   
                                                   # Normalize the test dataset using the information of train dataset
                                                   normParam <- preProcess(trainData_SMOTE)
                                                   norm.testData <- predict(normParam, testData)
                                                   
                                                   # Predict test dataset (PROBA)
                                                   cart_predict_proba2<-predict(cart_fit_test,norm.testData,type="prob")[,2]
                                                   cart_predict_proba<-as.matrix(cart_predict_proba2)
                                                   
                                                   y.test<-as.numeric(norm.testData[,which(names(norm.testData) %in% output)])
                                                   y.test<-factor(y.test)
                                                   cart_predict_proba<-as.numeric(cart_predict_proba)
                                                   auc_test<-auc(roc(cart_predict_proba,labels = y.test))
                                                   
                                                   # Predict test dataset (CLASS)
                                                   cart_predict_class<-predict(cart_fit_test,norm.testData,type="class")[,2]
                                                   cart_predict_class<-as.matrix(cart_predict_class)
                                                   
                                                   #### Confusion Matrix ###
                                                   library(caret)
                                                   cart_predict_class<-as.factor(cart_predict_class[,2])
                                                   truth<-as.factor(testData$Output_Class)
                                                   result <- confusionMatrix(cart_predict_class,truth)
                                                 
                                                 #Save the results in a matrix                  
                                                 result_cross_validation[outerloop,1]<-maxdepthbest
                                                 result_cross_validation[outerloop,2]<-minbucketbest
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
                                                 
                                                 # Variable Importance #                        
                                                 var_imp_matrix<-cbind(var_imp_matrix,cart_fit_test$variable.importance)
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




