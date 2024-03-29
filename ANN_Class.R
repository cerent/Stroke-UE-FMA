
  
  
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

data_used # name of the data used for the study

output<-"Output_class" # the name of the output

# Create a function to write the classification results and variable importance measurements in the same list
multiResultClass <- function(result1=NULL,result2=NULL,result3=NULL)
{me <- list(
  result1 = result1,
  result2 = result2,
  result3 = result3)
#Set the name for the class
class(me) <- append(class(me),"multiResultClass")
return(me)}

# Create matrices to save the results during the cross-validation process
result_cross_validation<-matrix(ncol=13,nrow=10) 
result_cross_validation_100iteration<-NULL
var_imp_matrix<-NULL
predicted_output<-NULL

# Prepare the codes for the parallelisation
myCluster <- makeCluster(10)
registerDoMC(10)

# Turn 10 parallel program 10 times (different partition for each iteration)

result_parallel<-foreach(it=rep(1:10,10), .combine = rbind,.multicombine=TRUE,
                                                     .packages=c("nnet","rminer","caret","AUC","e1071")) %dopar% {cat(it)
                                                       
                                                       # Split the data in 10 partitions
                                                       folds_outerloop<-createFolds(factor(data_used$Output_class),k=10,list = FALSE)                                                                   
                                                       
                                                       # Start the outer loop of the nested CV
                                                       for(outerloop in 1:10){
                                                         
                                                         # Create a train dataset using 9 partitions over 10 
                                                         trainData <- data_used[folds_outerloop != outerloop, ]
                                                         
                                                         # Create a test dataset using 1 partition over 10  
                                                         testData <- data_used[folds_outerloop == outerloop, ]
                                                         
                                                         # loop to chose best hyperparameters
                                                         error_innerloop<-matrix(ncol=3,nrow=30)
                                                         k=1
                                                         for (decay1 in c(0.0001, 0.001, 0.01, 0.1, 0.5)) {
                                                           for (size1 in seq(5,10,1)) {
                                                             
                                                          
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
                                                                 
                                                                 # Fit the model using a couple of the hyperparameters
                                                                 innermodel_ann_prob<- nnet(Output_class ~ .,  data = trainingData,size=size1,decay=decay1,probability=TRUE,
                                                                                            maxit = 300,  trace = F, linout = F,MaxNWts=50000)
                                                                 
                                                               
                                                                 # Predict the validation dataset
                                                                 predict_validation_ann_prob<-predict(innermodel_ann_prob,validationData)
                                                                 err_cv[innerloop,1]<-innerloop
                                                                 err_cv[innerloop,2]<- auc(roc(validationData$Output_class,predict_validation_ann_prob,plot=FALSE,direction="auto",quiet = TRUE))
                                                               }
                                                               err_cv1<-rbind(err_cv1,err_cv)
                                                             }
                                                           }
                                                           
                                                           # Decide the best hyperparameters based on AUC
                                                           mean_AUC_inner_10cv<-mean(err_cv1[,2])
                                                           error_innerloop[k,1]<-size1
                                                           error_innerloop[k,2]<-decay1
                                                           error_innerloop[k,3]<-mean_AUC_inner_10cv
                                                           
                                                           k=k+1
                                                         }
                                                         
                                                         # select the best hyperparameters that minimize the mean of RMSE (over 10 iteration)
                                                         best_size<-error_innerloop[which.max(error_innerloop[,3]),1]
                                                         best_decay<-error_innerloop[which.max(error_innerloop[,3]),2]
                                                         
                      
                                                         # Fit the model with best hyperparameters using train data (PROBA)
                                                         tunedModel_prob<- nnet(Output_class ~ .,  data = trainData,size=best_size,decay=best_decay,probability=TRUE,
                                                                                maxit = 300,  trace = F, linout = F,MaxNWts=50000)
                                                         
                                                       
                                                         # Predict the model using test data
                                                         ann_predict_proba <- predict(tunedModel_prob, testData,probability = TRUE) 
                                                         
                                                         # Calculate AUC
                                                         require(PRROC)
                                                         auc_test<- auc(roc(testData$Output_class,ann_predict_proba,plot=FALSE,direction="auto",quiet = TRUE))
                                                         
                                                         # Predict the class for the test dataset
                                                         ann_predict_class<-predict(tunedModel_prob, norm.testData, type="class")
                                                         
                                                         #### Confusion Matrix ###  
                                                         library(caret)
                                                         ann_predict_class2<-as.factor(as.numeric(as.factor(ann_predict_class))-1)
                                                         truth<-as.factor(testData$Output_class)
                                                         levels(ann_predict_class2)=levels(truth)
                                                         
                                                         result <- confusionMatrix(ann_predict_class2,truth,positive="1")
                                                         
                                                         result_cross_validation[outerloop,1]<-best_size
                                                         result_cross_validation[outerloop,2]<-best_decay
                                                         result_cross_validation[outerloop,3]<-outerloop
                                                         result_cross_validation[outerloop,4]<-auc_test 
                                                         
                                                         result_cross_validation[outerloop,5]<-result$byClass['Balanced Accuracy']
                                                         result_cross_validation[outerloop,6]<-result$byClass['Sensitivity']
                                                         result_cross_validation[outerloop,7]<-result$byClass['Specificity']
                                                         result_cross_validation[outerloop,8]<-result$byClass['Precision']
                                                         result_cross_validation[outerloop,9]<- result$byClass['Recall']
                                                         result_cross_validation[outerloop,10]<-result$byClass['F1']
                                                         result_cross_validation[outerloop,11]<-result$byClass['Balanced Accuracy']
                                                         result_cross_validation[outerloop,12]<-result$byClass['Pos Pred Value']
                                                         result_cross_validation[outerloop,13]<-result$byClass['Neg Pred Value']
                                                         
                                                         # Variable Importance #  
                                                         var_imp<-varImp(tunedModel_prob_varimp)
                                                         var_imp<-as.data.frame(var_imp$Overall)
                                                         names_data<-as.data.frame(names(data_used))
                                                         names_data1<-as.matrix(names_data[-which(names(trainData) %in% "Output_class"),])
                                                         var_imp1<-cbind(names_data1,var_imp)
                                                         var_imp2<-as.data.frame(var_imp1)
                                                         var_imp_matrix<-cbind(var_imp_matrix,as.matrix(var_imp2[,2]))
                                                         
                                                         
                                                         # create a matrix from predicted and output classes
                                                         predicted_proba<-ann_predict_proba
                                                         predicted_class<-as.matrix(ann_predict_class2)
                                                         output<-as.matrix(testData$Output_class)
                                                         predicted_output<-rbind(predicted_output,cbind(predicted_proba,predicted_class,output,seq(1,length(predicted_class),1)))
                                                       }
                                                       
                                                       # Save the results for 100 iterations
                                                       result_cross_validation_100iteration<-rbind(result_cross_validation_100iteration,result_cross_validation)
                                                       
                                                       result <- multiResultClass()
                                                       result$result1 <- result_ann_qsm_MCID_class_100iter1
                                                       result$result2 <- var_imp_ann_class_MCID
                                                       result$result3 <- predicted_output
                                                       
                                                       return(result)
                                                     }



