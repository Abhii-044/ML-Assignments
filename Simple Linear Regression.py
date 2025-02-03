#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[4]:


data1.info()


# In[5]:


data1.describe()


# In[6]:


fig,axes=plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["daily"],ax=axes[0],color='skyblue',width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("daily")
sns.histplot(data1["daily"],kde=True,ax=axes[1],color='Purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Daily")
axes[1].set_ylabel("Y")
plt.tight_layout()
plt.show()


# In[7]:


plt.scatter(data1["daily"],data1["sunday"])


# In[8]:


data1["daily"].corr(data1["sunday"])


# In[9]:


model = smf.ols("sunday~daily",data = data1).fit()
model.summary()


# In[10]:


x=data1['daily'].values
y=data1['sunday'].values
plt.scatter(x,y,color='m',marker='o',s=30)
b0=13.84
b1=1.33
y_hat=b0+b1*x
plt.plot(x,y_hat,color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[11]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0,max(x)+100)
plt.ylim(0,max(y)+100)
plt.show()


# In[12]:


data1.corr(numeric_only=True)


# In[13]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[14]:


model1.summary()


# In[ ]:





# In[19]:


model1.params


# In[24]:


print(f'model t-values:\n{model.tvalues}\n----------\nmodel p-values: \n{model.pvalues}')


# In[25]:


(model.rsquared,model.rsquared_adj)


# In[27]:


newdata=pd.Series([200,300,1500])


# In[28]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[29]:


model.predict(data_pred)


# In[30]:


pred = model1.predict(data1["daily"])
pred


# In[31]:


data1["Y_hat"] = pred
data1


# In[33]:


data1["residuals"]=data1["sunday"]-data1["Y_hat"]
data1


# In[35]:


mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[36]:


plt.scatter(data1["Y_hat"], data1["residuals"])


# In[37]:


import statsmodels.api as sm
sm.qqplot(data1["residuals"],line='45',fit=True)
plt.show()


# In[38]:


sns.histplot(data1["residuals"],kde=True)


# In[ ]:




