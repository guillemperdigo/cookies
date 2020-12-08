#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[23]:


cookies = pd.read_csv("data/cookies.csv")


# In[25]:


cookies_original = cookies


# In[26]:


cookies.drop(columns=["butter type", "mixins"], inplace=True)


# In[27]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(cookies,
                                       test_size=0.2,
                                       random_state=123)

train_X = train_set.drop(columns="quality")
train_y = train_set["quality"]

test_X = test_set.drop(columns="quality")
test_y = test_set["quality"]


# In[43]:


param_distribs={"n_neighbors": randint(low=3, high=30),
                "weights":["uniform", "distance"],
                "p":[1,2]}

neigh3_search = RandomizedSearchCV(KNeighborsRegressor(),
                                   param_distribs,
                                   scoring="r2",
                                   n_iter=10,
                                   cv=5,
                                   n_jobs=4,
                                   random_state=123)

model_pipeline = make_pipeline(SimpleImputer(),
                               StandardScaler(),
                               PCA(n_components=0.9),
                               neigh3_search)


# In[44]:


model_pipeline.fit(train_X, train_y)


# In[45]:


# save the model to disk
filename = 'Models/model_pipeline1.sav'
pickle.dump(model_pipeline, open(filename, 'wb'))

