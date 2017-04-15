################################################################
#        
################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
# Import data and check out what we've got ourselves into
df_train = pd.read_csv('~/Desktop/Git/pricing_regression/train.csv')
df_test = pd.read_csv('~/Desktop/Git/pricing_regression/test.csv')

All_Data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))

#What do we have to work with here?
print(df_train.columns)

#Look at SalePrice
print(df_train['SalePrice'].describe())

# Historgram
sns.distplot(df_train['SalePrice'])

#Let's quantify that non-normality
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#What's what? 
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
           
#Some variables seem too related
#Three groups of related, YearBuilt+YearRemodAdd, TotalBsmtSF+1stFlrSF and GarageYrBlt+GarageCarss+GarageArea
#The following seem to drive SalePrice: OverallQual, TotalBsmtSF, GrvLivArea and we probably need some others

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Keep ones with high correlation, pick single ones of related bunches
#'SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'
#scatterplot

#Lets look at these closely to see which may need transformation
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show(); 
        
#We can use cols later to get our regression going
#First lets do some transfomration
        
#Lets find all the skew variables we need
num_feats = All_Data.dtypes[All_Data.dtypes != "object"].index
#print(num_feats)
skew_feats = df_train[num_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#print(skew_feats)
skew_feats = skew_feats[skew_feats > 0.75] # Normally distributed data has a skew of 0
#print(skew_feats)
skew_feats = skew_feats.index
print(skew_feats)

#Lets normalise those right out
All_Data[skew_feats] = np.log1p(All_Data[skew_feats])
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

### Convert categorical data to numerical
All_Data = pd.get_dummies(All_Data)

#Now what are we missing? 
#missing data
total = All_Data.isnull().sum().sort_values(ascending=False)
percent = (All_Data.isnull().sum()/All_Data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

### Now we have unskewed datawe can look at replacing NULLS this is a simple way to do it
All_Data = All_Data.fillna(All_Data.mean())

### Convert categorical data to numerical
All_Data = pd.get_dummies(All_Data)
### Now we have unskewed datawe can look at replacing NULLS this is a simple way to do it
All_Data = All_Data.fillna(All_Data.mean())
### split into train and testing sets again
X_train = All_Data[:df_train.shape[0]]
#print(X_train)
X_test = All_Data[df_train.shape[0]:]
#print(X_test)
Y = df_train.SalePrice
#print(Y)

#transformed histogram and normal probability plot
sns.distplot(Y, fit=norm);
fig = plt.figure()
res = stats.probplot(Y, plot=plt)

#Sweet! That looks way more linear

#Import a bunch of regression models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.model_selection import cross_val_score

### Define a function that calculates the Root Mean Square of Errors for a given model using a cross_validation 
def RMSE_CV(model):
    RMSE = np.sqrt(-cross_val_score(model, X_train, Y, scoring="neg_mean_squared_error", cv = 5))
    return(RMSE)
cols.remove('SalePrice')
model_reg = LinearRegression().fit(X_train[cols],Y)

#That's pretty great for a linear regression on only 6 variables!
print("Linear model error",RMSE_CV(model_reg).mean()) # RMSE 0.16524087909

#Get the coefficents
coef_reg = pd.Series(model_reg.coef_,index = X_train[cols].columns)

print("Coefs reg = ",coef_reg)

#Now why dont we just use all the variables when doing Lin regression?
#Must be orthonormal variables, but what if we use a regression method that can get around this?
#OLS can't deal with highly correlated co-effs the variance on the OLS co-effs blow out 
#varying from sample to sample we can curtail this using essentailly a lagrandge multiplier 
#This constrains the co-efficients size
# We look at ridge regression first
# https://www.youtube.com/watch?v=5asL5Eq2x0A great video on ridge
# Min B ||y-AB||^2_2 + lambda||B||^2_1 where _2-> cartesian distance _1 = |A| + |B|
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [RMSE_CV(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

#Graphically look for alpha that minimises RMSE
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

#We get to use a LOT more variables here so lets see how we do
cv_ridge.min() # RMSE of 0.12733734668670774

#What about another linear model like LASSO
# https://www.youtube.com/watch?v=jbwSCwoT51M
# Min B ||y-AB||^2_2 + lambda||B||_1 
# Constraint is essentially a diamond now, becomes v likely that level curves hit at diamond corners
# Diamond corners are more likely to be hit than anywhere else
# This means some variables are reduced to having nearly no impact.

## Fin