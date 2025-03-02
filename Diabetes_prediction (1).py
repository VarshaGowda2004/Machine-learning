#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[7]:


dataset=pd.read_csv("diabetes.csv")


# In[8]:


dataset.head()


# In[9]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[10]:


dataset.tail(10)


# In[11]:


print(dataset.shape)


# In[12]:


#getting statistical measures
dataset.describe()


# In[15]:


dataset['Outcome'].value_counts()


# In[20]:


dataset.groupby('Outcome').mean()


# In[22]:


#separating the data and labels
x=dataset.drop(columns='Outcome',axis=1)
y=dataset['Outcome']


# In[24]:


print(x)


# In[25]:


print(y)


# In[49]:


#data standardization
scaler=StandardScaler()


# In[51]:


scaler.fit(x)


# In[53]:


standardized_data=scaler.transform(x)


# In[54]:


print(standardized_data)


# In[55]:


x=standardized_data
y=dataset['Outcome']


# In[56]:


print(x)
print(y)


# In[57]:


#train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[58]:


print(x.shape ,x_train.shape,x_test.shape)


# In[59]:


#lets train the model
classifier=svm.SVC(kernel='linear')


# In[60]:


classifier.fit(x_train,y_train)


# In[61]:


#evalution of our model
#accuracy score on training data
x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[62]:


print('accuracy score of the training data :',training_data_accuracy)


# In[63]:


#accuracy score on test data
x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[64]:


print('accuracy score of the training data :',test_data_accuracy)


# In[69]:


import numpy as np

# input data
input_data = (5, 116, 74, 0, 0, 25.6, 0.201, 30)
# Convert to numpy array
input_data_array = np.asarray(input_data)
# Reshape for a single sample (our model is trained on 760 data so reshaping is imp)
input_data_reshaped = input_data_array.reshape(1, -1)
# Apply the same scaler used for training
std_data = scaler.transform(input_data_reshaped)
print(std_data)
# Make prediction
prediction = classifier.predict(std_data)
if prediction[0] == 0:
    print("wohoooo congratsss you're NOT diabetic.")
else:
    print("you're diabetic.")


# In[ ]:




