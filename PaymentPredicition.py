#!/usr/bin/env python
# coding: utf-8

# # Importing modules and dataset

# In[1]:


# importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import sklearn.metrics as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[2]:


df = pd.read_csv('1804476.csv')
df


# # PRE-PROCESSING

# In[3]:


df.info()


# We can see that there are NULL values in clear_date --> which we will use to split in test and train later.     
# area_business is totally NULL. 
# There are 3 NULL values in invoice_id, we'll look into it later.

# In[4]:


df.isnull().sum()


# ### Date-Time Conversion

# In[5]:


df['document_create_date.1']=pd.to_datetime(df['document_create_date.1'],format='%Y%m%d')
df['document_create_date.1'].head()


# In[6]:


df['clear_date']=pd.to_datetime(df['clear_date'],format='%Y-%m-%d %H:%M:%S')
df['clear_date'].head()


# In[7]:


df['posting_date']=pd.to_datetime(df['posting_date'],format='%Y-%m-%d')
df['posting_date'].head()


# In[8]:


df['due_in_date']=pd.to_datetime(df['due_in_date'],format='%Y%m%d')
df['due_in_date'].head()


# In[9]:


df['baseline_create_date']=pd.to_datetime(df['baseline_create_date'],format='%Y%m%d')
df['baseline_create_date'].head()


# In[10]:


# looking at the converted data-types
df.info()


# ### Dropping columns with ALL NULL values.

# In[11]:


df.dropna(axis=1,how='all',inplace=True)
df.shape


# We dropped area_business

# ### Dropping null values in invoice id
# invoice id has only 49997 unique values of 50000. So let's analyse it a bit.

# In[12]:


df["document type"].value_counts()


# We see that where the invoice id is missing the document type is X2 and posting date is different from document create date. Since their are only 3 values and invoice_id needs to be unique we can drop these rows

# In[13]:


df.dropna(axis=0,subset=['invoice_id'],inplace=True)
df.reset_index(drop=True,inplace=True)
df.shape
df


# ### Checking for unique values in each column

# In[14]:


for cols in df.columns:   
   print(f"{cols} - {df[f'{cols}'].nunique()}")


# There is only 1 unique value in document type and posting_id. We can drop these.

# In[15]:


unique_cols =  [x for x in df.columns if df[x].nunique()==1] 
print(unique_cols)
df.drop(unique_cols,axis=1,inplace=True)
df.columns


# Now let's check for NULL values again.

# In[16]:


df.isnull().any()


# No null(except clear-date) values so moving on to removing duplicate columns.

# ### Duplicate column removal

# In[17]:


# function to find duplicate columns
def findDuplicateColumns(df):
    duplicatecolumns=set()
    for x in range(df.shape[1]):
        col1=df.iloc[:,x]
        for y in range(x+1,df.shape[1]):
            col2=df.iloc[:,y]
            if col1.equals(col2):
                duplicatecolumns.add(df.columns.values[x])
    return list(duplicatecolumns)


# Using user-defined function as the in-built function was computationally expensive.

# In[18]:


dr=findDuplicateColumns(df)
dr


# In[19]:


df.drop(columns=dr,inplace=True) 


# We dropped posting_date and doc_id.
# 
# Now, to avoid discrepancy between document_create_date and document_create_date.1 we drop document_create_date and instead use document_create_date.1

# In[20]:


df.drop(columns=['document_create_date'],inplace=True)


# Also, invoice id is unique for each transaction and don't affect the sales, so we drop it.

# In[21]:


df.drop(columns=['invoice_id'],inplace=True) 
df


# # TARGET VARIABLE AND SORT
# Our model cant take in dates for prediction so our target column will not be clear_date. Instead it will be the delay column as delay will be in int format which our model can predict.
# 
# clear_date - due_date will tell us how much delay was there in paying. 
#  
#  "-" indicates it has been paid off before due date. 
#  "+" indicates it has been paid off after the due date

# In[4]:


df['delay']=(df['clear_date']-df['due_in_date']).dt.days


# Sorting in ascending order by document_create_date.1

# In[23]:


df.sort_values(by='document_create_date.1',inplace=True) # sorting in ascending order by document_create_date.1
df


# # SPLITTING
# Creating test set on null clear_date

# In[24]:


test_data = df[df.clear_date.isnull()].reset_index()
test_data.drop(columns=['index'],inplace=True)
test_data


# In[25]:


ts = test_data.copy()


# Splitting test_set into x_test and y_test

# In[26]:


y_test = test_data["delay"]
y_test


# In[27]:


x_test = test_data.iloc[:,:-1].copy()
x_test


# ### Creating a seprate data frame out of which we will extract val1, val2 and final train set.
# WHOLE DATA ---> [TRAIN_DATA + VAL1 (1.5 MONTHS) + VAL2 (1.5 MONTHS)] + TEST_DATA

# In[28]:


train_data2 = df[df.clear_date.notnull()].reset_index() #train set on non-null clear date
train_data2.drop(columns=['index'],inplace=True)
train_data2


# In[29]:


# Making use of offset to find date 1.5 months prior to 2020-02-27
x = train_data2['document_create_date.1'].iloc[-1] - pd.DateOffset(months=1, days=15)
y = train_data2['document_create_date.1'].loc[train_data2['document_create_date.1'] <= x].iloc[-1]
print(y) 
train_data2['document_create_date.1'].loc[train_data2['document_create_date.1'] == y].last_valid_index()


# This gives the date which is 1.5 months prior to 2020-02-27.
# 
# val2 will start from 2020-01-13 to 2020-02-27

# In[30]:


# dataframe for val2
val2 = train_data2.iloc[40519: , :].copy()
val2.reset_index(drop=True,inplace=True)
val2


# Removing val2 from train_data2 and storing into new dataframe to extract val1.

# In[31]:


train_data1 = pd.concat([train_data2, val2]).drop_duplicates(keep=False)
train_data1


# In[32]:


x2 = train_data1['document_create_date.1'].iloc[-1] - pd.DateOffset(months=1, days=15)
y2 = train_data1['document_create_date.1'].loc[train_data1['document_create_date.1'] <= x2].iloc[-1]
print(y2)
train_data1['document_create_date.1'].loc[train_data1['document_create_date.1'] == y2].last_valid_index() 


# This gives the date which is 1.5 months prior to 2020-01-13 (start of val2).
# 
# val1 will start from 2019-11-28 to 2020-01-12.

# In[33]:


val1 = train_data1.iloc[37251: , :].copy()
val1.reset_index(drop=True,inplace=True)
val1


# Removing val1 from the dataframe and making the final train set.

# In[34]:


train_data = pd.concat([train_data1, val1]).drop_duplicates(keep=False)
train_data


# We successfully splitted the data into train, val1, val2, and test set.

# # EDA on train set

# ## UNIVARIATE ANALYSIS

# In[35]:


train_data.info()


# We can see there are no null values in our dataset. There are only 3 continuous variable column i.e. buisness_year, total_open_amount and delay,

# ### business_code
# company code of the account

# In[36]:


train_data.business_code.value_counts()


# In[37]:


sns.countplot(x = train_data['business_code'])


# As there are only 6 values we can apply encoding on it later.

# ### cust_number
# customer number given to all the customers of the Account.

# In[38]:


train_data.cust_number.value_counts()


# In[39]:


len(dict(train_data.cust_number.value_counts()))


# There are 1359 unique customers or comapny that we have transactions with.
# 
# cust_number represent account no of different companies. For eg WAL-MAR has many varities that is WAL-MAR corporation, WAL-MAR systems etc but they have the same cust_number.

# ### name_customer
# name of the customer.

# In[40]:


train_data.name_customer.value_counts()


# Companies Like WAL-MAR, WAL-MAR systems, etc belong to company WAL-MAR and have a unique customer_num to their name hence this column will be dropped.

# ### clear_date
# The date on which the customer clears an invoice, or in simple terms, they make the full payment.

# In[41]:


train_data.clear_date.min()


# In[42]:


train_data.clear_date.max()


# Our clear_date ranges for 362 days.

# In[43]:


clear_month = train_data.clear_date.dt.month
clear_month.value_counts()


# In[44]:


sns.countplot(x=clear_month, palette="hls")


# May is the month where the companies completed their maximum transations.

# ### buisness_year
# indicates the year of clear date

# In[45]:


train_data.buisness_year.value_counts()


# As there is only 1 buisness_year we will drop this column.

# ### document_create_date.1
# The date on which the invoice document was created

# In[46]:


train_data["document_create_date.1"].min()


# In[47]:


train_data["document_create_date.1"].max()


# It ranges for 334 days, almost less than a year

# ### invoice_currency
# The currency of the invoice amount in the document for the invoice

# In[48]:


train_data["invoice_currency"].value_counts()


# As there are two currencies involved, we can convert either one them. However, as there is not much difference between USD and CAD, there is not much use of converting them.

# ### total_open_amount
# The amount that is yet to be paid for that invoice

# In[49]:


train_data['total_open_amount'].describe().apply(lambda x: format(x,'f'))


# Automatically binning the amount using pd.qcut.

# In[50]:


open_amount_bins = pd.qcut(train_data['total_open_amount'],q=10)
open_amount_bins.value_counts()


# In[51]:


amount_bins = [0,5000,10000,50000,100000,1100000]
new_open_amount_bin = pd.cut(train_data['total_open_amount'], bins = amount_bins)


# In[52]:


plt.xticks(fontsize=10, rotation=90)
sns.countplot(x=open_amount_bins, palette="hls")


# The number of companies in each distribution is same.

# ### baseline_create_date
# The date on which the Invoice was created.

# In[53]:


train_data.loc[train_data["document_create_date.1"]!=train_data["baseline_create_date"]]


# Baseline create date and document create date are same except if the company is canadian there is a delay in filing the document and creating an invoice. 
# 
# It's not that important to our model so we'll drop it.

# ### cust_payment_terms
# Business terms and agreements between customers and accounts on discounts and days of payment.

# In[54]:


train_data.cust_payment_terms.value_counts()


# In[55]:


len(dict(train_data.cust_number.value_counts()))


# It is same as cust_number.

# ### isOpen
# - Tells whether a transaction is open or closed

# In[56]:


train_data.isOpen.value_counts()


# isOpen is 0 for all the train set as all the invoices has been closed. So we can drop this column.

# ### Delay
# The negative delay specifies that the amount was cleared before the due date and thus there was no delay. Therefore we have capped the negative values to 0.

# In[57]:


train_data.delay[train_data.delay<0].count()


# In[58]:


train_data['delay']=train_data.delay.apply(lambda x: 0 if x<0 else x)
train_data.delay[train_data.delay<0].count()


# In[59]:


train_data.delay.describe()


# ## MULTIVARIATE ANALYSIS

# ### Relation b/w business_code with invoice_currency

# In[60]:


train_data['invoice_currency'].value_counts()


# In[61]:


train_data.groupby("business_code").invoice_currency.value_counts()


# We see that business code first letter describes in which currency the transaction took place except for 2 entries. Let's explore them further

# In[62]:


train_data.groupby("invoice_currency").business_code.value_counts()


# business_code U013 is an US based company but paying in CAD

# In[63]:


temp = train_data.loc[train_data['business_code'] == 'U013']
temp.loc[temp['invoice_currency']=='CAD']


# We see that only PRATT company is the only US based company paying in CAD.
# 
# Morever they are only 2 transactions so it's not as significant. They don't even have any delay so this won't affect our model.

# ### Relation between total_open_amount and delay

# In[64]:


delay_bins = [0,15,30,45,60,300]
delay_bucket = pd.cut(train_data['delay'], bins = delay_bins)
pd.crosstab(index = new_open_amount_bin, columns =delay_bucket)


# In[65]:


plt.figure(figsize=(8,5))
plt.xlabel("Amount bin")
sns.countplot(x = new_open_amount_bin, hue=delay_bucket)


# As we can see from the graph for each bucket the amount was returned in 0-15 delay days.

# In[66]:


train_data.corr()


# In[67]:


sns.pairplot(train_data, height=4)


# We can infer: The total open amount decreses as the delay increases. There is negative co-relation.

# # FEATURE ENGINEERING

# In[68]:


train_data.drop(columns=['business_code', 'name_customer','buisness_year','invoice_currency','baseline_create_date', 'cust_payment_terms', 'isOpen'], inplace=True)
train_data


# Extracting day and month from document_create_date.1 and due_in_date.

# In[69]:


train_data["doc_create_day"] = train_data["document_create_date.1"].dt.day
train_data["doc_create_month"] = train_data["document_create_date.1"].dt.month

train_data["due_day"] = train_data["due_in_date"].dt.day
train_data["due_month"] = train_data["due_in_date"].dt.month


# In[70]:


train_data


# In[71]:


train_data.drop(columns=['clear_date', 'document_create_date.1', 'due_in_date'], inplace=True)
train_data


# Extracting the integer part from cust_number

# In[72]:


train_data['cn'] = train_data['cust_number'].str.extract('(\d+)')

# then will drop that column
train_data.drop('cust_number',axis=1,inplace=True)

#lets see the dataframe
train_data


# In[73]:


train_data.info()


# Converting the data-type of cn as float

# In[74]:


train_data['cn']=train_data['cn'].astype(str).astype(float)
train_data.info()


# Splitting into x_train and y_train

# In[75]:


y_train = train_data['delay']
y_train


# In[76]:


x_train = train_data.drop(["delay"], axis=1).copy()
x_train


# ##### Now we need to repeat the same process on val1, val2 and x_test.

# In[77]:


val1.info()


# In[78]:


val1['delay']=val1.delay.apply(lambda x: 0 if x<0 else x)
val1.drop(['business_code', 'name_customer', 'clear_date', 'buisness_year', 'invoice_currency', 'baseline_create_date', 'cust_payment_terms', 'isOpen'], axis=1, inplace=True)

val1["doc_create_day"] = val1["document_create_date.1"].dt.day
val1["doc_create_month"] = val1["document_create_date.1"].dt.month

val1["due_day"] = val1["due_in_date"].dt.day
val1["due_month"] = val1["due_in_date"].dt.month

val1['cn'] = val1['cust_number'].str.extract('(\d+)')
val1.drop('cust_number',axis=1,inplace=True)

val1.drop(columns=['document_create_date.1', 'due_in_date'], inplace=True)
val1


# In[79]:


val1['cn']=val1['cn'].astype(str).astype(float)
val1.info()


# Splitting into x_val and y_val

# In[80]:


y_val1 = val1['delay']
y_val1


# In[81]:


x_val1 = val1.drop(["delay"], axis=1).copy()
x_val1


# In[82]:


val2['delay']=val2.delay.apply(lambda x: 0 if x<0 else x)
val2.drop(['business_code', 'name_customer', 'clear_date', 'buisness_year', 'invoice_currency', 'baseline_create_date', 'cust_payment_terms', 'isOpen'], axis=1, inplace=True)

val2["doc_create_day"] = val2["document_create_date.1"].dt.day
val2["doc_create_month"] = val2["document_create_date.1"].dt.month

val2["due_day"] = val2["due_in_date"].dt.day
val2["due_month"] = val2["due_in_date"].dt.month

val2['cn'] = val2['cust_number'].str.extract('(\d+)')
val2.drop('cust_number',axis=1,inplace=True)

val2.drop(columns=['document_create_date.1', 'due_in_date'], inplace=True)
val2


# In[83]:


val2['cn']=val2['cn'].astype(str).astype(float)
val2.info()


# In[84]:


y_val2 = val2['delay']
y_val2


# In[85]:


x_val2 = val2.drop(["delay"], axis=1).copy()
x_val2


# In[86]:


x_test.drop(['business_code', 'name_customer', 'clear_date', 'buisness_year', 'invoice_currency', 'baseline_create_date', 'cust_payment_terms', 'isOpen'], axis=1, inplace=True)

x_test["doc_create_day"] = x_test["document_create_date.1"].dt.day
x_test["doc_create_month"] = x_test["document_create_date.1"].dt.month

x_test["due_day"] = x_test["due_in_date"].dt.day
x_test["due_month"] = x_test["due_in_date"].dt.month

x_test['cn'] = x_test['cust_number'].str.extract('(\d+)')
x_test.drop('cust_number',axis=1,inplace=True)

x_test.drop(columns=['document_create_date.1', 'due_in_date'], inplace=True)
x_test


# In[113]:


x_test['cn']=x_test['cn'].astype(str).astype(float)
x_test.info()


# # FEATURE SELECTION

# In[87]:


plt.figure(figsize=(14,10))
cor = train_data.corr()
sns.heatmap(cor,cmap = 'viridis',annot=True)


# As we can see from the above graph due_month and doc_create_month are highly correlated and thus they will have a negative impact on our model. So drop remove due_month. 

# In[88]:


x_train = x_train.drop(["due_month"], axis=1).copy()
x_val1 = x_val1.drop(["due_month"], axis=1).copy()
x_val2 = x_val2.drop(["due_month"], axis=1).copy()
x_test = x_test.drop(["due_month"], axis=1).copy()


# # MODELING

# As this is a regression model we will use the following models:-
# - Linear Regressor
# - SVR
# - Decision Tree
# - Random Forest
# - XgBoost

# ### XGBoost

# In[89]:


#fitting the model 

clf = xgb.XGBRegressor()
clf.fit(x_train, y_train)

# Predicting the Validation Set Results
predicted = clf.predict(x_val1)


# In[90]:


# predicting all types of error and accuracy

print("Mean absolute error =", round(sm.mean_absolute_error(y_val1, predicted), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_val1, predicted), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_val1, predicted), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_val1, predicted), 2)) 
print("R2 score =", round(sm.r2_score(y_val1, predicted), 2))
print("Accuracy= ", clf.score(x_val1,y_val1))


# ### Random Forest

# In[91]:


clf2 = RandomForestRegressor()
clf2.fit(x_train, y_train)

# Predicting the Validation Set Results
predicted2 = clf2.predict(x_val1)


# In[92]:


# predicting all types of error and accuracy

print("Mean absolute error =", round(sm.mean_absolute_error(y_val1, predicted2), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_val1, predicted2), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_val1, predicted2), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_val1, predicted2), 2)) 
print("R2 score =", round(sm.r2_score(y_val1, predicted2), 2))
print("Accuracy= ", clf2.score(x_val1,y_val1))


# ### DecisionTreeRegressor

# In[93]:


# Fitting Decision Tree Regressor to the Training Set
clf3 = DecisionTreeRegressor()
clf3.fit(x_train, y_train)

# Predicting the Validation Set Results
predicted3 = clf3.predict(x_val1)


# In[94]:


# predicting all types of error and accuracy

print("Mean absolute error =", round(sm.mean_absolute_error(y_val1, predicted3), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_val1, predicted3), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_val1, predicted3), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_val1, predicted3), 2)) 
print("R2 score =", round(sm.r2_score(y_val1, predicted3), 2))
print("Accuracy= ", clf3.score(x_val1,y_val1))


# ### LinearRegression

# In[95]:


# Fitting Simple Linear Regression to the Training Set
clf5 = LinearRegression()
clf5.fit(x_train, y_train)

# Predicting the Test Set Results
predicted5 = clf.predict(x_val1)


# In[96]:


# predicting all types of error and accuracy

print("Mean absolute error =", round(sm.mean_absolute_error(y_val1, predicted5), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_val1, predicted5), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_val1, predicted5), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_val1, predicted5), 2)) 
print("R2 score =", round(sm.r2_score(y_val1, predicted5), 2))
print("Accuracy= ", clf5.score(x_val1,y_val1))


# #### We'll continue with XgBoost as it gives us the best result from all the other models.

# ### HYPER-PARAMETER TUNING

# In[97]:


# now we will hypertune our parameters for better and accurate results and avoiding overfitting

# we will set some parameter
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In this we will fit and check for best parameters for our model using RandomizedSearchCV

# In[98]:


reg=xgb.XGBRegressor()
random_search=RandomizedSearchCV(reg,param_distributions=params,n_iter=5,n_jobs=-1,cv=5,verbose=3)


# In[99]:


random_search.fit(x_train, y_train)


# Checking for best parameters.

# In[100]:


random_search.best_estimator_


# Using these parameters and checking against val2

# In[101]:


cl = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0.2, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.2, max_delta_step=0, max_depth=8,
             min_child_weight=3, monotone_constraints='()',
             n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
cl.fit(x_train, y_train)
# Predicting the Test Set Results
predicted = cl.predict(x_val2)


# In[102]:


# predicting all types of error and accuracy

print("Mean absolute error =", round(sm.mean_absolute_error(y_val2, predicted), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_val2, predicted), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_val2, predicted), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_val2, predicted), 2)) 
print("R2 score =", round(sm.r2_score(y_val2, predicted), 2))
print("Accuracy= ", cl.score(x_val2,y_val2))


# # TEST

# In[114]:


predicted1 = cl.predict(x_test)


# In[115]:


predicted1=np.around(predicted1)
predicted1.astype(int)


# In[116]:


len(predicted1)


# We got the delay on test set.
Creating a new dataframe
# In[117]:


Col=pd.DataFrame()


# Appending the delay predicted column in it

# In[118]:


Delay=[]
for x in predicted1:
    Delay.append(pd.Timedelta(days=x))
Col['Delay'] = Delay
Col


# Predicting the clear date

# In[119]:


Col['clear_date'] = ts['due_in_date']+Col['Delay']
Col


# # Creating the Aging Bucket

# In[120]:


aging_bucket = []
for x in predicted1:
    if x<=15:
        aging_bucket.append("0-15days")
    elif x<=30:
        aging_bucket.append("16-30days")
    elif x<=45:
        aging_bucket.append("31-45days")
    elif x<=60:
        aging_bucket.append("46-60days")
    else:
        aging_bucket.append("Greater than 60 days")
Col['Aging Bucket']= aging_bucket
Col.drop(['Delay'],axis=1,inplace=True)
Col

