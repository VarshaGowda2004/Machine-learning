#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as  np


# In[89]:




class Linear_regression():
    # Initiating parameters (hyperparameters)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations  # Fixed the typo

    def fit(self, X, Y):
        # No of training examples & features
        self.m, self.n = X.shape

        # Initializing weights and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Implementing gradient descent
        for i in range(self.no_of_iterations):  # Now correctly references self.no_of_iterations
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # Calculate gradients (Fixed self_Y typo)
        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = -2 * np.sum(self.Y - Y_prediction) / self.m

        # Updating the weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b  # y = wx + c


# here we are using salary dataset and with increase in experience the salary also increases hence we are using linear regression model for prediction

# In[90]:


#import dependencies,
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[91]:


salary =pd.read_csv('salary.csv')


# In[92]:


salary.head()


# In[93]:


salary.describe()


# In[94]:


salary.tail()


# In[95]:


salary.shape


# In[96]:


salary.isnull().sum()


# # splitting the features & target

# In[97]:


X =salary.iloc[:,:-1].values 
Y= salary.iloc[:,1].values


# In[98]:


print(X)


# In[99]:


print(Y)


# # splitting dataset into train and test data

# In[100]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=2)


# # training linear regression model

# In[101]:


model=Linear_regression(learning_rate=0.02, no_of_iterations=1000)


# In[102]:


model.fit(X_train,Y_train)


# In[103]:


#printing the parameter values (weight & bias)
print('weight=',model.w[0])
print('bias=',model.b)


# In[104]:


# salary=9514.400999035135(experience)+9514.400999035135
#our trained linear regression model equation


# # predicting the salary value for test data

# In[106]:


test_data_prediction=model.predict(X_test)


# In[107]:


print(test_data_prediction)


# # visualizing the predicted and actual values

# In[110]:


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,test_data_prediction,color='blue')
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()


# In[ ]:




