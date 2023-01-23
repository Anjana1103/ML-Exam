#!/usr/bin/env python
# coding: utf-8

# # Data Understanding and Exploration

# In[163]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[164]:


# Importing the Dataset
df=pd.read_excel(r'C:\Users\Anjana Anilkumar\Desktop\Exam_ML\data_final.xlsx')


# In[165]:


# Checking whether the dataset has been loaded properly or not
# Checking the First 5 records
df.head()


# In[166]:


# Checking the last 5 records
df.tail()


# In[167]:


# Checking the number of records(ie. rows) and features(ie. columns) in the dataset
df.shape


# In[56]:


# We Observe that there are 100 records and 3 columns in the dataset


# In[168]:


# Checking and understanding the Dataset, with column names, Non-null count and Datatype.
df.info()


# In[169]:


# Checking the statistical information about the Numerical data in the dataset
df.describe()


# # Data Cleaning

# In[170]:


# Checking for missing values
df.isnull().sum()


# In[ ]:


# Observation: There are no missing values in the dataset


# In[171]:


# Checking for duplicated values
df.duplicated().sum()


# In[ ]:


# Observation: There are no duplicate values in the dataset


# We can conclude that our dataset is clean.

# # Data Preperation

# In[172]:


df.head()


# In[175]:


# Selecting the Independent column 'Observation' into a dataframe
x=df.iloc[:,:-1].values
x


# In[176]:


# Selecting the dependent column 'Price' into dataframe
y=df.iloc[:,-1].values
y


# In[177]:


# Splitting the data into Training and Testing Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[178]:


len(x_train)


# In[179]:


len(x_test)


# In[180]:


# Checking Training set
x_train


# In[ ]:


# Observation: Random values from 'Observation' have been selected into the Training Set


# # Model Building: 
# 

# In[181]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[182]:


from sklearn.metrics import r2_score


# In[183]:


y_pred=model.predict(x_test)
print(r2_score(y_test,y_pred))


# In[187]:


y_pred


# In[188]:


c=model.coef_
c


# In[189]:


m=model.intercept_
m


# In[190]:


# MSE
from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)


# In[191]:


# Printing Summary
import statsmodels.api as sm
x_stats=sm.add_constant(x_train)
summ=sm.OLS(y_train,x_stats).fit()
summ.summary()


# In[192]:


from sklearn.preprocessing import PolynomialFeatures
reg=PolynomialFeatures(degree=3)
poly=reg.fit_transform(x)
model1=LinearRegression()
model2=model1.fit(poly,y)
model2

