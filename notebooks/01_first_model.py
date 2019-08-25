#!/usr/bin/env python
# coding: utf-8

# ## 01 First Model
# 
# Little data exploratin and a VERY simple OLS model

# In[1]:


import pandas as pd
import pandas_profiling as pp


# In[2]:


df = pd.read_csv("../data/raw/train.csv")


# In[3]:


df.head()


# In[4]:


pr = pp.ProfileReport(df)


# In[5]:


pr


# In[6]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## First model
# 
# Very first model, lets try something realy simple:
# 
# * pick top 10 most highly correlated features with the target.
# * fill any missing values with the mode
# * Fit an OLS regression

# In[14]:


feats = pr.description_set['correlations']['spearman']["SalePrice"].sort_values(ascending=False).index[1:11]
target = "SalePrice"


# In[15]:


df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])


# In[16]:


pp.ProfileReport(df[feats])


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(df[feats], df[target], test_size=0.3)


# In[18]:


rgr = linear_model.LinearRegression().fit(X_train, y_train)


# In[19]:


mean_squared_error(y_test, rgr.predict(X_test))**0.5

