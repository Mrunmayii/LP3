#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Implement Gradient Descent Algorithm to find the local minima of a function.
#For example, find the local minima of the function y=(x+3)² starting from the point x=2.


# In[21]:


import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as pyplot
from matplotlib import pyplot


# In[22]:


def function(x):
    return (x + 3) ** 2


# In[23]:


def gradient(x):
    return 2 * (x + 3)


# In[24]:


# Hyperparameters for the gradient descent algorithm
learning_rate = 0.1  # Adjust this as needed
max_iterations = 10000
tolerance = 1e-6


# In[ ]:





# In[25]:


x = 2


# In[29]:


for iteration in range(max_iterations):
    gradient_x = gradient(x)
    x -= learning_rate * gradient_x  
    
    delta_x = abs(learning_rate * gradient_x)
    if delta_x < tolerance:
        break

print(f"Local minimum found at x = {x}, y = {function(x)}")


# In[ ]:





# In[30]:


start = 2
alpha = 0.1
y=calculate_gradient(alpha,start,max_iterations)
#alpha = learning_Rate
#start= start value of x(i.e 2)
x_cor=np.linspace(-5,5,100)
pyplot.plot(x_cor,function(x_cor))

x_arr=np.array(x)
pyplot.plot(x_arr,function(x_arr),'.-',color='red')
pyplot.show()


# In[ ]:




