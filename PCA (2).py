#!/usr/bin/env python
# coding: utf-8

# In[34]:


from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from mpl_toolkits import mplot3d
from csv import reader
import os
import csv
import numpy as np
import math
import operator
from math import exp
import matplotlib.pyplot as plt
train_sample = r'\training_validation'


# In[35]:


def path_f(file_name):
    d_file = open(file_name, "r")
    lines = reader(d_file)
    data_load = list(lines)
    array = ''
    vect_array = []
    for r in range(len(data_load)):                  #each sample file are load and stored in list
        array += data_load[r][0]
    for i in range(len(array)):
        vect_array.append(int(array[i]))
    return vect_array


# In[36]:


def classdata(file_name, path):
    f_path = os.listdir(path)
    class_data = []
    for f in f_path:                                  #path to training dataset is provided with samples
        if f.startswith(file_name):
            class_data.append(path_f(path + '\\' + f))
    return class_data


# In[37]:


class_0 = classdata('class_0', train_sample)
class_1 = classdata('class_1', train_sample)
class_2 = classdata('class_2', train_sample)
class_3 = classdata('class_3', train_sample)
class_4 = classdata('class_4', train_sample)
class_5 = classdata('class_5', train_sample)
class_6 = classdata('class_6', train_sample)
class_7 = classdata('class_7', train_sample)
class_8 = classdata('class_8', train_sample)
class_9 = classdata('class_9', train_sample)


# In[38]:


train = class_0 + class_1 + class_2 + class_3 + class_4 +class_5 + class_6 + class_7 + class_8 + class_9
total_array = np.array(train)
print(total_array.shape) 
print(len(train))


# In[39]:


shapee = total_array.T
print(shapee.shape)


# In[72]:


for i in range(len(shapee)):
    Meann = np.mean(shapee,axis=1)            #mean of the dataset
print(Meann.shape)


# In[74]:


center_data = shapee.T - Meann
#print("center data",center_data)
print("shape of center data", center_data.shape)
#print(len(p))
#print(p.shape)                                     # calculating the center_data 
#print(p)
cov_matrixx = np.cov(center_data)                  #The covariance matrix of the center data
#print("covariance ",cov_matrixx)
print("shape of covariance matrix",cov_matrixx.shape)


# In[75]:


cov_matrixx


# In[78]:


U,S,V = np.linalg.svd(center_data.T)


# In[79]:


U.shape


# In[80]:


S.shape


# In[81]:


V.shape


# In[82]:


sum =0
for i in range(len(S)):
    sum = sum + S[i]
print("sum of eigenvalues: ",sum)


# In[83]:


s = int(sum) * 0.95
print("considering 95% of original data=",s)


# In[87]:


sum =0
for i in range(len(S)):
    sum = sum + S[i]
    #print("sum",round(sum))
    #print("total components:",count)
    if (sum>np.sum(S)*0.95):
        print("Total number of reduced features: ",i)
        break


# In[53]:


new_matrix = shapee[0:833,0:1934]           #slicing the as per the reduced features


# In[54]:


class_0=new_matrix[:,0:189]
mat0 = class_0.T.tolist()
mat1_with_labels = []                 #adding labels to classes
for i in range(len(mat0)):
    mat0[i].append(0)


# In[55]:


class_1=new_matrix[:,189:387]
mat1 = class_1.T.tolist()
mat1_with_labels = []
for i in range(len(mat1)):
    mat1[i].append(1)


# In[56]:


class_2=new_matrix[:,387:581]
mat2 = class_2.T.tolist()
mat2_with_labels = []
for i in range(len(mat2)):
    mat2[i].append(2)


# In[57]:


class_3=new_matrix[:,581:780]
mat3 = class_3.T.tolist()
mat3_with_labels = []
for i in range(len(mat3)):
    mat3[i].append(3)


# In[58]:


class_4=new_matrix[:,780:966]
mat4 = class_4.T.tolist()
mat4_with_labels = []
for i in range(len(mat4)):
    mat4[i].append(4)
    
class_5=new_matrix[:,966:1153]
mat5 = class_5.T.tolist()
for i in range(len(mat5)):
    mat5[i].append(5)

    
class_6=new_matrix[:,1153:1349]
mat6 = class_6.T.tolist()
for i in range(len(mat6)):
    mat6[i].append(6)
    
class_7=new_matrix[:,1349:1550]
mat7 = class_7.T.tolist()
for i in range(len(mat7)):
    mat7[i].append(7)

class_8=new_matrix[:,1550:1730]
mat8 = class_8.T.tolist()
for i in range(len(mat8)):
    mat8[i].append(8)    
    
class_9=new_matrix[:,1730:1934]
mat9 = class_9.T.tolist()
for i in range(len(mat9)):
    mat9[i].append(9)    


# In[59]:


T_data = mat0+ mat1 + mat2 + mat3 + mat4 + mat5 + mat6 + mat7 + mat8 + mat9
T_data                                           #adding all dataset
len(T_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




