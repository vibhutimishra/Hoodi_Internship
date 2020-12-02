#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from pandas import DataFrame


# In[2]:


dataframe=pd.read_csv("final.csv")


# In[6]:


X=dataframe.iloc[:,2:5].values
y=dataframe.iloc[:,5]


# In[30]:


Labelencoder_X=LabelEncoder()
X[:,1]=Labelencoder_X.fit_transform(X[:,1])


# In[31]:


ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

#avoiding the dummy variable trap
X=X[:,1:]


# In[32]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)


# In[33]:


forest_model = RandomForestRegressor(n_estimators=1000,random_state=1)
forest_model.fit(Xtrain, ytrain)
preds = forest_model.predict(Xtest)
print(mean_absolute_error(ytest, preds))


# In[34]:


plt.scatter(Xtest[:,3],ytest)
plt.title("actual")
plt.xlabel("time in weeks")
plt.ylabel("yield per hectre in tonnes")


# In[35]:


plt.scatter(Xtest[:,3],preds)
plt.title("predicted")
plt.xlabel("time in weeks")
plt.ylabel("yield per hectre in tonnes")


# In[36]:


ytest=np.array(ytest)


# In[37]:


preds


# In[38]:


plt.hist(ytrain,bins=50,ec='black')
plt.xlabel('yeild/area')
plt.ylabel('no of farmers')


# In[39]:


sns.distplot(ytrain)
plt.title("actual")


# In[40]:


sns.distplot(ytest)
plt.title("predicted")
dataframe['yield/area'].corr(dataframe['Crop age in Weeks'])


# In[41]:


dataframe.head()


# In[42]:


dataframe.corr()


# In[43]:


regr = LinearRegression()
regr.fit(Xtrain,ytrain)


# In[44]:


pred2=regr.predict(Xtest)
print(mean_absolute_error(ytest, pred2))


# In[45]:


plt.scatter(Xtest[:,3],pred2)
plt.title("predicted")
plt.xlabel("time in weeks")
plt.ylabel("yield per hectre in tonnes")


# In[46]:


plt.scatter(Xtest[:,3],ytest)
plt.title("actual")
plt.xlabel("time in weeks")
plt.ylabel("yield per hectre in tonnes")


# In[47]:


Xtrain=pd.DataFrame(data=Xtrain)
pd.DataFrame(data=regr.coef_,index=Xtrain.columns)


# ### data transformation

# In[48]:


dataframe['yield/area'].skew()

