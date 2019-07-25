# Codes-Stroke-UE-FMA

Five machine learning methods (Support Vector Machines [SVM], Artificial Neural Networks [ANN], Elastic-Net [NET], Classification and Regression Trees), and Random Forest [RF]) were used to perform regression (prediction of continous variables) and classification (prediction of binary variables) analyses.

Each method was trained with two loops (outer and inner) of k-fold cross validation (k = 10) to optimize the hyperparameters and test the performance of the methods. The outer loop provided a training set (9/10 folds) for model building and test set (1/10 folds) for model assessment. Thus, the training dataset included approximately 90% of patients while the testing dataset included 10% patients. The inner loop (repeated over 10 different partitions of the training dataset only) performed grid-search to find the set of model hyperparameters that minimized the average hold-out RMSE for regression or maximized AUC for classification. A model was fitted using those optimal hyperparameters on the entire training dataset and assessed on the hold-out test set from the outer loop. The outer loop was repeated for 100 different random partitions of the data. The mean test R^2 and RMSE in the regression analysis, and the mean test AUC,F1 score, sensitivity, specificity, PPV, and NPV in the classification analysis (over all 10 folds x 100 iterations = 1000 test sets) were calculated to assess the performance of the method.

---

The codes provided in this repository were used to perform machine learning methods to 
1. predict a chronic stroke individual’s motor function (Upper-extremity Fugl-Meyer Assessment (UE-FMA) after 6 weeks of intervention using pre-intervention demographic, clinical, neurophysiological and imaging data,
2. identify which data elements were most important in predicting chronic stroke patients’ impairment after 6 weeks of intervention,
3. to classify stroke patients based on their significant motor outcome (UE-FMA) change after therapy. 

Please find the manuscript via [link](https://www.biorxiv.org/content/10.1101/457416v1 ) 

---

Please contact Ceren Tozlu (cet2005@med.cornell.edu) for any questions about the repository.



