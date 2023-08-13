#!/usr/bin/env python
# coding: utf-8

# # import the library

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# # import dataset

# In[2]:


USAHousing = pd.read_csv('USA_Housing.csv')
USAHousing.head()


# In[3]:


# To find the information about the dataset
USAHousing.info()


# In[4]:


USAHousing.describe()


# # Data Preprocessing 

# ### 1) Missing value treatement

# In[5]:


USAHousing.isnull().sum()


# In[6]:


USAHousing.isnull().sum()/len(USAHousing)*100


# In[7]:


# Check outlier and then will decide whether we have to use mean or median approach


# In[8]:


sns.boxplot(y = 'Avg. Area Income', data=USAHousing)
plt.show()


# In[9]:


USAHousing['Avg. Area Income'] = USAHousing['Avg. Area Income'].fillna(USAHousing['Avg. Area Income'].median())


# In[10]:


sns.boxplot(y = 'Avg. Area Number of Rooms', data=USAHousing)
plt.show()


# In[11]:


USAHousing['Avg. Area Number of Rooms'] = USAHousing['Avg. Area Number of Rooms'].fillna(USAHousing['Avg. Area Number of Rooms'].median())


# In[12]:


sns.boxplot(y = 'Avg. Area Number of Bedrooms', data=USAHousing)
plt.show()


# In[13]:


USAHousing['Avg. Area Number of Bedrooms'].describe()


# In[14]:


USAHousing['Avg. Area Number of Bedrooms'] = USAHousing['Avg. Area Number of Bedrooms'].fillna(USAHousing['Avg. Area Number of Bedrooms'].mean())


# In[15]:


USAHousing.isnull().sum()


# # Part 2 - Encoding concept

# In[16]:


USAHousing.head(2)


# In[17]:


USAHousing['Address'][0]


# In[18]:


USAHousing['Address'][1]


# In[19]:


# Address is non-significant variable to predict USA Housing price. hence, we have to drop this variable
USAHousing = USAHousing.iloc[:,0:-1]


# In[20]:


USAHousing.head()


# # Part 3 - Handling outlier 
# # it's mandatory part whenever you solve regression problem
# # outlier is sentive to regression problem

# In[21]:


def distplots(col):
    sns.distplot(USAHousing[col])
    plt.show()
    
for i in list(USAHousing.columns)[0:]:
    distplots(i)


# In[22]:


def boxplots(col):
    sns.boxplot(USAHousing[col])
    plt.show()
    
for i in list(USAHousing.select_dtypes(exclude=['object']).columns)[0:]:
    boxplots(i)


# In[23]:


USAHousing.columns


# In[24]:


# Please use capping method - one by one to handle the dataset - home work
# capping method required

Q1 = USAHousing.quantile(0.25)
Q3 = USAHousing.quantile(0.75)
IQR = Q3 - Q1

pos_outlier = Q3 + 1.5 * IQR

neg_outlier = Q1 - 1.5 * IQR


# In[25]:


print(Q1)
print("*************"*5)
print(Q3)
print("*************"*5)
print(IQR)
print("*************"*5)
print(pos_outlier)
print("*************"*5)
print(neg_outlier)
print("*************"*5)


# In[26]:


new_df = USAHousing.copy()


# In[27]:


# 'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Area Population'


# In[28]:


income_q1 = new_df['Avg. Area Income'].quantile(0.25)
income_q3 = new_df['Avg. Area Income'].quantile(0.75)
income_iqr = income_q3 -  income_q1
income_upper = income_q3 + 1.5 * income_iqr
income_lower = income_q1 - 1.5 * income_iqr


# In[29]:


new_df['Avg. Area Income'] = np.where(new_df['Avg. Area Income'] > income_upper, income_upper,
                                     np.where(new_df['Avg. Area Income'] < income_lower, income_lower,
                                            new_df['Avg. Area Income']) )


# In[30]:


age_q1 = new_df['Avg. Area House Age'].quantile(0.25)
age_q3 = new_df['Avg. Area House Age'].quantile(0.75)
age_iqr = age_q3 - age_q1
age_upper = age_q3 + 1.5 * age_iqr
age_lower = age_q1 - 1.5 * age_iqr


# In[31]:


new_df['Avg. Area House Age'] = np.where(new_df['Avg. Area House Age'] > age_upper,age_upper,
                                     np.where(new_df['Avg. Area House Age'] < age_lower, age_lower,
                                            new_df['Avg. Area House Age']) )


# In[32]:


room_q1 = new_df['Avg. Area Number of Rooms'].quantile(0.25)
room_q3 = new_df['Avg. Area Number of Rooms'].quantile(0.75)
room_iqr = room_q3 - room_q1
room_upper = room_q3 + 1.5 * room_iqr
room_lower = room_q1 - 1.5 * room_iqr


# In[33]:


new_df['Avg. Area Number of Rooms'] = np.where(new_df['Avg. Area Number of Rooms'] > room_upper,room_upper,
                                     np.where(new_df['Avg. Area Number of Rooms'] < room_lower, room_lower,
                                            new_df['Avg. Area Number of Rooms']) )


# In[34]:


pop_q1 = new_df['Area Population'].quantile(0.25)
pop_q3 = new_df['Area Population'].quantile(0.75)
pop_iqr = pop_q3 - pop_q1
pop_upper = pop_q3 + 1.5 * pop_iqr
pop_lower = pop_q1 - 1.5 * pop_iqr


# In[35]:


new_df['Area Population'] = np.where(new_df['Area Population'] > pop_upper,pop_upper,
                                     np.where(new_df['Area Population'] < pop_lower, pop_lower,
                                            new_df['Area Population']) )


# In[36]:


def boxplots(col):
    sns.boxplot(new_df[col])
    plt.show()
    
for i in list(new_df.select_dtypes(exclude=['object']).columns)[0:]:
    boxplots(i)


# In[37]:


new_df.head(2)


# In[38]:


# Part 4 - Feature Scaling 
# we can only do with independent variable

# split the data into independent variable and dependent variable
x = new_df.iloc[:,0:-1]
y = new_df['Price']


# In[39]:


x.head()


# In[40]:


y.head()


# In[41]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc_x = sc.fit_transform(x)
pd.DataFrame(sc_x)


# In[42]:


# Finding correlation
plt.figure(figsize=(20,15))
corr = new_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# # VIF - Variance Inflation Factor - to check multicollinearity

# In[43]:


variable = sc_x
variable.shape


# In[44]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variable = sc_x

vif = pd.DataFrame()

vif['Variance Inflation Factor'] = [variance_inflation_factor(variable, i ) for i in range(variable.shape[1])]

vif['Features'] = x.columns

A variance inflation factor (VIF) is a measure of the amount of multicollinearity in regression analysis. Multicollinearity exists when there is a correlation between multiple independent variables in a multiple regression model.
# ![image.png](attachment:image.png)

# In[45]:


vif


# # Split the data into training and test for building the model and for prediction

# In[46]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# # Building Linear Regression Model

# ## Approach no - 1

# In[47]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)


# In[48]:


print(lm.intercept_)
print()
print(lm.coef_)


# In[49]:


x.columns

price = intercept + slope1*Avg. Area Income + slope2*Avg. Area House Age+slope3*Avg. Area Number of Rooms+
            slope4*Avg. Area Number of Bedrooms+slope5*Area Population
    
price = -2657920.671950032 + 2.17329521e+01*Avg. Area Income + 1.65689998e+05*Avg. Area House Age+1.21585113e+05*Avg. Area Number of Rooms+
            1.72972862e+03*Avg. Area Number of Bedrooms+1.53020383e+01*Area Population
    

# In[50]:


price = -2657920.671950032 + 2.17329521e+01*85000+ 1.65689998e+05*2+1.21585113e+05*3+ 1.72972862e+03*3+1.53020383e+01*38000
price


# In[51]:


# Predict house price by using lm model with test dataset

y_pred_price = lm.predict(x_test)
y_pred_price_train = lm.predict(x_train)


# In[52]:


y_pred_price


# In[53]:


y_test


# In[54]:


# Validate the actual price of the test data and predicted price

from sklearn.metrics import r2_score
r2_score(y_test, y_pred_price)


# In[55]:


r2_score(y_train, y_pred_price_train)


# # Approach no 2 - OLS Method

# In[56]:


from statsmodels.regression.linear_model import OLS
import statsmodels.regression.linear_model as smf


# In[57]:


reg_model = smf.OLS(endog = y_train, exog=x_train).fit()


# In[58]:


reg_model.summary()


# In[59]:


# There is no autocorrelation in Linear Regression model
# what to do when the assumption (autocorrelation) fails ?
## 1) Lagged variable - 
## 2) Difference - 
## 3) Generalized least squares
## Time Series Models - AR, MA, ARMA, ARIMA,STL
## Robust Standard Error - HAC


# ![image.png](attachment:image.png)

# In[60]:


# Check linearity

plt.scatter(y_test, y_pred_price)


# In[61]:


# Normality of Residual

sns.distplot((y_test - y_pred_price), bins=50)
plt.show()


# In[62]:


# Conclude this model
# Data Preprocessing - 
# EDA
# Slip the data into train and test


## Adj. R-squared (uncentered):	0.964
## All variable is statically significant (p <= 0.05) 
# task - please drop "Avg. Area Number of Bedrooms" and then follow the same approach
# check underfitting or overfitting problem - no bias and variance found
## Assumptions

# 1) Linearity - Satisfied
# 2) Normality of Residuals- Satisfied
# 3) Homoscedasticity - Satisfied (there is no outlier and residual is normaly distributed)
# 4) No autocorrelation - Satisfied
# 5) No or little Multicollinearity - satisfied
# 6) No endogenity problem - satisfied 


# In[63]:


# Regularisazation
# Gradient Descent 


# In[64]:


# By using sklearn linear model
# training accuracy : 91.6%
# test accuracy = 91.3%


# # Lasso regularization

# In[65]:


from sklearn.linear_model import Lasso 
lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
print("Lasso Model :", (lasso.coef_))


# In[66]:


y_pred_train_lasso = lasso.predict(x_train)
y_pred_test_lasso = lasso.predict(x_test)


# In[67]:


print("Training Accuracy :", r2_score(y_train, y_pred_train_lasso))
print()
print("Test Accuracy :", r2_score(y_test, y_pred_test_lasso))


# In[ ]:





# In[68]:


# Part 2 : Ridge Regression (L2- Regularization)
# closure to zero but not exact zero
# penalty - 0.3
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.3)
ridge.fit(x_train, y_train)
print("Ridge Model :", (ridge.coef_))


# In[69]:


y_pred_train_ridge = ridge.predict(x_train)
y_pred_test_ridge = ridge.predict(x_test)


# In[70]:


print("Training Accuracy :", r2_score(y_train, y_pred_train_ridge))
print()
print("Test Accuracy :", r2_score(y_test, y_pred_test_ridge))


# In[ ]:





# # ElasticNet

# In[71]:


from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=0.3, l1_ratio=0.1)
elastic.fit(x_train, y_train)


# In[72]:


y_pred_train_elastic = elastic.predict(x_train)
y_pred_test_elastic = elastic.predict(x_test)


# In[73]:


print("Training Accuracy :", r2_score(y_train, y_pred_train_elastic))
print()
print("Test Accuracy :", r2_score(y_test, y_pred_test_elastic))


# In[ ]:





# # Performance matrix

# ## Mean Absolute Error (MAE)

# In[74]:


from sklearn import metrics


# In[76]:


print("MAE :", metrics.mean_absolute_error(y_test, y_pred_price))


# In[ ]:





# ## Mean Absolute Percent Error (MAPE)

# In[78]:


print("MAPE :", metrics.mean_absolute_error(y_test, y_pred_price)/100)


# In[ ]:





# ## Mean Squared Error (MSE)

# In[79]:


print("MSE :", metrics.mean_squared_error(y_test, y_pred_price))


# In[ ]:





# ## Root Mean Squared Error (RMSE)

# In[80]:


print("RMSE :", np.sqrt(metrics.mean_squared_error(y_test, y_pred_price)))


# In[ ]:





# # Gradient Descent 

# In[ ]:





# In[87]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sc_x, y, test_size=0.25, random_state=101)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[ ]:





# In[88]:


from sklearn.linear_model import SGDRegressor


# In[89]:


gd_model = SGDRegressor()
gd_model.fit(x_train, y_train)


# In[90]:


y_pred_gd_train = gd_model.predict(x_train)

y_pred_gd_test = gd_model.predict(x_test)


# In[91]:


print("GD Trainging Accuracy :", r2_score(y_train, y_pred_gd_train))

print()

print("GD Test Accuracy :", r2_score(y_test, y_pred_gd_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




