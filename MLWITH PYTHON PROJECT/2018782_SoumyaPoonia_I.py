#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for mathematical operations
import numpy as np
# for dataframe operations
import pandas as pd
# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


# reading the datasets
input_train_data = pd.read_csv("train_LZdllcl (1).csv")
input_test_data = pd.read_csv("test_2umaH9m (1).csv")
submission = pd.read_csv("sample_submission_M0L0uXE.csv")
input_train_data.head()


# In[17]:


label = input_train_data["is_promoted"]
input_train_data.drop(columns=["is_promoted", "employee_id"], axis=0, inplace=True)
input_test_data.drop(columns=["employee_id"], axis=0, inplace=True)
a=label.value_counts()
a


# In[18]:


a.plot(kind='pie',title="is promoted", autopct='%1.1f%%')


# In[73]:


input_train_data.describe()


# filling null value in datasheet
# 

# In[21]:


totaldata = input_train_data.append(input_test_data, ignore_index=True)
print(totaldata["education"].value_counts())
input_train_data["education"] = input_train_data["education"].fillna("Bachelor's")
input_test_data["education"] = input_test_data["education"].fillna("Bachelor's")


print(totaldata["previous_year_rating"].value_counts())
input_train_data["previous_year_rating"] = input_train_data["previous_year_rating"].fillna(3.0)
input_test_data["previous_year_rating"] = input_test_data["previous_year_rating"].fillna(3.0)


# 

# In[22]:


totaldata.head()


# Plotting all the attributes 

# In[25]:


## Count Function
def count_func(input_train_data,col):
    categories = input_train_data[col].dropna().unique()
    data = {}
    for cat in categories:
        data[cat] = len(input_train_data[input_train_data[col]==cat])
    data = pd.DataFrame(data,index=[0]).T
    data.columns = [col]
    print(data)
    return data


# In[26]:


#no_of_trainings count
count_df = count_func(input_train_data,'no_of_trainings')
count_df.plot(kind='bar',title='no_of_trainings count') #using pandas
plt.xlabel('no_of_trainings')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["no_of_trainings"], label)
plt.title('Relation between no_of_trainings and label')


# In[78]:


#department count
count_df = count_func(input_train_data,'department')
count_df.plot(kind='bar',title='department count') #using pandas
plt.xlabel('department')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["department"], label)


# In[14]:


#region count
count_df = count_func(input_train_data,'region')
count_df.plot(kind='bar',title='region count') #using pandas
plt.xlabel('region')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["region"], label)


# In[80]:


#education count
count_df = count_func(input_train_data,'education')
count_df.plot(kind='bar',title='education count') #using pandas
plt.xlabel('education')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["education"], label)


# In[15]:


#gender count
count_df = count_func(input_train_data,'gender')
count_df.plot(kind='bar',title='gender count') #using pandas
plt.xlabel('gender')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["gender"], label)


# In[82]:


#age count
count_df = count_func(input_train_data,'age')
count_df.plot(kind='bar',title='age count') #using pandas
plt.xlabel('age')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["age"], label)


# In[83]:


#previous_year_rating	 count
count_df = count_func(input_train_data,'previous_year_rating')
count_df.plot(kind='bar',title='previous_year_rating	 count') #using pandas
plt.xlabel('previous_year_rating	')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["previous_year_rating"], label)


# In[84]:


#previous_year_rating	 count
count_df = count_func(input_train_data,'previous_year_rating')
count_df.plot(kind='bar',title='previous_year_rating	 count') #using pandas
plt.xlabel('previous_year_rating	')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["previous_year_rating"], label)


# In[85]:


#length_of_service count
count_df = count_func(input_train_data,'length_of_service')
count_df.plot(kind='bar',title='length_of_service count') #using pandas
plt.xlabel('length_of_service')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["length_of_service"], label)


# In[86]:


#KPIs_met >80%	 count
count_df = count_func(input_train_data,'KPIs_met >80%')
count_df.plot(kind='bar',title='KPIs_met >80%	 count') #using pandas
plt.xlabel('KPIs_met >80%	')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["KPIs_met >80%"], label)


# In[87]:


#awards_won?	 count
count_df = count_func(input_train_data,'awards_won?')
count_df.plot(kind='bar',title='awards_won? count') #using pandas
plt.xlabel('awards_won?	')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["awards_won?"], label)


# In[88]:


#avg_training_score	 count
count_df = count_func(input_train_data,'avg_training_score')
count_df.plot(kind='bar',title='avg_training_score count') #using pandas
plt.xlabel('avg_training_score	')
plt.ylabel('count')
plt.show()
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
sns.barplot(input_train_data["avg_training_score"], label)


# Feature Engineering
# 

# In[89]:


# lets create some extra features from existing features to improve our Model

# creating a Metric of Sum
input_train_data['sum_metric'] = input_train_data['awards_won?']+input_train_data['KPIs_met >80%'] + input_train_data['previous_year_rating']
input_test_data['sum_metric'] = input_test_data['awards_won?']+input_test_data['KPIs_met >80%'] + input_test_data['previous_year_rating']

# creating a total score column
input_train_data['total_score'] = input_train_data['avg_training_score'] * input_train_data['no_of_trainings']
input_test_data['total_score'] = input_test_data['avg_training_score'] * input_test_data['no_of_trainings']


# In[90]:


#  removing some of the columns which are not very useful for predicting the promotion.
input_train_data = input_train_data.drop(['recruitment_channel', 'region'], axis = 1)
input_test_data = input_test_data.drop(['recruitment_channel', 'region'], axis = 1)
input_train_data.columns


# Dealing with Categorical Columns

# In[91]:


# Lets check the categorical columns present in the data
input_train_data.select_dtypes('object').head()


# In[92]:


# lets check the value counts for the education column
input_train_data['education'].value_counts()


# In[93]:


# lets start encoding these categorical columns to convert them into numerical columns

# lets encode the education in their degree of importance 
input_train_data['education'] = input_train_data['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))
input_test_data['education'] = input_test_data['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))
input_train_data['education'].head(20)


# In[94]:


# lets use Label Encoding for Gender and Department to convert them into Numerical
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
input_train_data['department'] = le.fit_transform(input_train_data['department'])
input_test_data['department'] = le.fit_transform(input_test_data['department'])
input_train_data['gender'] = le.fit_transform(input_train_data['gender'])
input_test_data['gender'] = le.fit_transform(input_test_data['gender'])

input_train_data.head()


# In[95]:


input_test_data.head()


# Splitting the Data

# In[96]:


# lets split the target data from the train data

y = label
x = input_train_data
x_test = input_test_data

# lets print the shapes of these newly formed data sets
print("Shape of the x :", x.shape)
print("Shape of the y :", y.shape)
print("Shape of the x Test :", x_test.shape)


# In[97]:


from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[98]:


print("Shape of the x Train :", x_train.shape)
print("Shape of the y Train :", y_train.shape)
print("Shape of the x Valid :", x_valid.shape)
print("Shape of the y Valid :", y_valid.shape)
print("Shape of the x Test :", x_test.shape)


# Feature Scaling

# In[100]:


# lets import the standard scaler library from sklearn to do that
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)


# Model Development

# In[113]:


# Lets use Decision Trees to classify the data
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)
y_pred


# In[112]:


# lets take a look at the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_valid, y_pred)
print(cr)


# In[108]:


#predtiction for testing data
y_pred2=model.predict(x_test)
y_pred2.shape


# In[109]:


y_test=submission['is_promoted']
y_test.shape


# In[111]:


df=pd.DataFrame({'actual':y_test,'predicted':y_pred2})
df.head(10)


# In[114]:


cr2=classification_report(y_test, y_pred2)
print(cr2)

