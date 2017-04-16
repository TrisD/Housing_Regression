# Housing_Regression
Comparison of Linear, Ridge and lasso regression with xgboost on the Ames data set on Kaggle

This analysis was developed to compare various Linear models results to a kaggle competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Data is imported, analysed to find the key relationships between variables and the target. Variables are log transformed to remove skewness
Further transforms are performed to convert categorical variables to integers that can be interpreted by SKlearn.
Null values are replaced with the mean value of their feature.

We define a function that calculates the RMSE from a cross validation of 5 folds. This gives us a real idea of how the model is performing.

A set of the most important uncorrelated features are used to perform a linear regression with a RMSE of: 0.16524087909

We then look at Ridge regression attempting to find the best value of alpha to optimise the model. 
The Ridge model returns a RMSE of: 0.12733734668670774
Resource on Ridge: https://www.youtube.com/watch?v=5asL5Eq2x0A

We next look at a LASSO regression again optimising the alphas, this returns the best RMSE yet of: 0.12314421090977432
Looking at the important co-efficients we see that some of these are probably spurious and shouldnt be used in a more precise answer.
Resource on LASSO: https://www.youtube.com/watch?v=jbwSCwoT51M

Finally we run an XGBoost and combine this with the lasso results as suggested by: https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
This scores an RMSE of: 0.12087 on the test set.
