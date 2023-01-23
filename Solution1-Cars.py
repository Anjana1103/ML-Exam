#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Data Understanding and Exploration

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Importing the Dataset
df_cars=pd.read_csv(r'C:\Users\Anjana Anilkumar\Desktop\Exam_ML\data_1.csv')


# In[3]:


# Checking whether the dataset has been loaded properly or not
# Checking the First 5 records
df_cars.head()


# In[4]:


# Checking the last 5 records
df_cars.tail()


# In[5]:


# Checking the number of records(ie. rows) and features(ie. columns) in the dataset
df_cars.shape


# In[6]:


# We Observe that there are 301 records and 9 columns in the dataset


# In[7]:


# Checking the information about the Dataset, with column names, Non-null count and Datatype.
df_cars.info()


# In[8]:


# Checking the statistical information about the Numerical data in the dataset
df_cars.describe()


# # Data Cleaning

# In[9]:


# Checking for missing values, if any, in the dataset
df_cars.isnull().sum()


# In[10]:


# Observation:
# There are no missing values in the Dataset


# In[11]:


# Checking if there are Duplicate values in the Dataset
df_cars.duplicated().sum()


# In[12]:


# Dropping the duplicate values
df_cars=df_cars.drop_duplicates()


# In[13]:


df_cars.shape


# In[14]:


# Observation: 
# The duplicated values have been dropped as previously the row count was 301, Now it is 299.


# ##### Business Objectives:
# You as a Data scientist are required to apply some data science techniques  for the price of cars with the available independent variables.  That should help the management to understand how exactly the prices vary with the independent variables. 

# In[15]:


df_cars.head()


# # Visualization to understand the data

# In[16]:


sns.countplot(df_cars['Selling_Price'])


# In[17]:


# Checking the variation in Selling_price over the Years
fig=plt.figure(figsize=(10,10))
sns.relplot('Year','Selling_Price', data=df_cars, kind='line')


# In[18]:


# Plotting graph between Present Price and Selling Price to Undertand the relation between them.
plt.scatter(df_cars.Present_Price, df_cars.Selling_Price, c='b')
plt.xlabel('Present Price')
plt.ylabel('Selling Price')
plt.title('Relation Between Present Price Price and Selling Price')


# In[19]:


# Plotting graph between Fuel type and selling price
plt.bar(df_cars.Fuel_Type, df_cars.Selling_Price)
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.title('Relation Between Fuel Type and Selling Price')


# In[20]:


# Finding the Correlation between the features
corr=df_cars.corr()
corr.style.background_gradient(cmap='coolwarm')


# # Data Preparation
# 

# In[21]:


df_cars.head()


# In[22]:


# selecting columns :
df_new=df_cars[['Year','Selling_Price', 'Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]


# In[23]:


sns.pairplot(df_new)


# In[24]:


df_new.head()


# In[25]:


df_new=pd.get_dummies(df_new)


# In[26]:


df_new.head()


# In[27]:


df_new.shape


# In[28]:


x=df_new.iloc[:,[0,2,3,4,5,6,7,8,9,10,11]]


# In[29]:


x


# In[30]:


y=df_new.iloc[:,1]
y


# In[31]:


# Splitting the Data into Training Set and Testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Model Building and Evaluation: 

# In[32]:


# Model Building: Linear Regression model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model


# In[33]:


model1=model.fit(x_train,y_train)
model1


# In[34]:


# Predicted values
y_pred=model1.predict(x_test)


# In[35]:


y_pred


# In[36]:


# Calculating value of coefficients
c=model1.coef_
c


# In[37]:


# Calculating values of intercept
m=model1.intercept_
m


# In[38]:


# Mean Squared Error calculation
from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)


# In[39]:


# Printing Summary
import statsmodels.api as sm
x_stats=sm.add_constant(x_train)
summ=sm.OLS(y_train,x_stats).fit()
summ.summary()


# ##### Observation: 
# We see that Mean Squared Error value is very less and R-Squared value is 0.891.
# Hence the Model built is high on accuracy

# In[40]:


# Splitting the Data into Training Set and Testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[41]:


# Model Building: Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
model1=regressor.fit(X_train,Y_train)
model1


# In[42]:


# Predicted Values
y_pred1=model1.predict(X_test)


# In[43]:


y_pred1


# In[44]:


#Mean Squared Error
metrics.mean_squared_error(Y_test,y_pred1)


# In[45]:


# Printing Summary
import statsmodels.api as sm
X_stats=sm.add_constant(X_train)
summ=sm.OLS(Y_train,X_stats).fit()
summ.summary()


# ##### Observation: 
# We see that Mean Squared Error value is very less and R-Squared value is 0.891.
# Hence the Model built is high on accuracy
