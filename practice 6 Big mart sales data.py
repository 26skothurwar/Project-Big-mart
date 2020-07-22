#!/usr/bin/env python
# coding: utf-8

# In[686]:


import pandas as pd
dftrain=pd.read_csv(r'/users/sowmya/downloads/bigmart-sales-data/train.csv')
dftest=pd.read_csv(r'/users/sowmya/downloads/bigmart-sales-data/test.csv')


# In[687]:


print(dftrain.shape)
print(dftest.shape)
df.columns


# In[688]:


dftrain['source']='dftrain'
dftest['source']='dftest'
df=pd.concat([dftrain,dftest],ignore_index=True)


# In[689]:


print(dftrain.shape,dftest.shape,df.shape)


# In[690]:


df.apply(lambda x:sum(x.isnull()))


# In[691]:


df.describe()


# In[692]:


df.apply(lambda x:len(x.unique()))


# In[693]:


#filter categorical variables
categorical_columns=[x for x in df.dtypes.index if df.dtypes[x]=='object']

#exclude Id and source
categorical_columns=[x for x in categorical_columns if x not in['Item_Identifier','Outlet_Identifier','source']]

#print frequency of categories
for col in categorical_columns:
    print(df[col].value_counts())
    


# In[694]:


replacevalues={'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}
df=df.replace({"Item_Fat_Content":replacevalues})
df['Item_Fat_Content'].value_counts()


# In[695]:


df.head()


# In[ ]:



    


# In[696]:


#modify item_visibility as it has 0 values which makes no sense
avg_item_visibility=df.pivot_table(values='Item_Visibility',index='Item_Identifier')
print(avg_item_visibility)

#assign boolean values to 0 values in item_visibility
mis_bool=(df['Item_Visibility']==0)
print(sum(mis_bool))

#impute 0 values to avg value
df.loc[mis_bool,'Item_Visibility']=df.loc[mis_bool,'Item_Identifier'].apply(lambda x:avg_item_visibility.loc[x])
print(sum(df['Item_Visibility']==0))


# In[697]:


#Determine the average weight per item:
item_avg_weight = df.pivot_table(values='Item_Weight', index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
miss_bool = df['Item_Weight'].isnull() 
print(sum(miss_bool))

#Impute data and check #missing values before and after imputation to confirm
print ('Orignal #missing: %d'% sum(miss_bool))
df.loc[miss_bool,'Item_Weight'] = df.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
print ('Final #missing: %d'% sum(df['Item_Weight'].isnull()))


# In[698]:


from scipy.stats import mode
outlet_size_mode=df.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]))
print(outlet_size_mode)

#get a boolean varaible for missing values in Outlet_size
miss_bool=df['Outlet_Size'].isnull()
print(sum(miss_bool))

#impute the values
df.loc[miss_bool,'Outlet_Size']=df.loc[miss_bool,'Outlet_Type'].apply(lambda x:outlet_size_mode[x])
print(sum(df['Outlet_Size'].isnull()))


# In[699]:


df['Item_visibility_meanratio']=df.apply(lambda x:x['Item_Visibility']/avg_item_visibility.loc[x['Item_Identifier']],axis=1)


# In[700]:


df['Item_visibility_meanratio'].head()


# In[701]:


#get the first two character of ID
df['Item_Type_combined']=df['Item_Identifier'].apply(lambda x:x[0:2])

#rename the values to some intuitive categories
df['Item_Type_combined']=df['Item_Type_combined'].map({'FD':'Food','NC':'Non consumable','DR':'Drinks'})

df['Item_Type_combined'].value_counts()


# In[702]:


#determine the years of operation of the store

df['Years_of_operation']=df['Outlet_Establishment_Year'].apply(lambda x:2013-x)
df['Years_of_operation'].head()
df['Years_of_operation'].describe()


# In[703]:


#mark non consumable as seperate category as they should not be having fat content
df.loc[df['Item_Type_combined']=="Non consumable",'Item_Fat_Content']="Non edible"
df['Item_Fat_Content'].value_counts()


# In[704]:


from sklearn.preprocessing import LabelEncoder
my_encoder=LabelEncoder()
df['Outlet']=my_encoder.fit_transform(df['Outlet_Identifier'])
col_encoding=['Item_Fat_Content','Item_Type_combined','Outlet_Location_Type','Outlet_Size','Outlet_Type']
for col in col_encoding:
    df[col]=my_encoder.fit_transform(df[col])


# In[705]:


df.head()


# In[706]:


print(df.columns)


# In[707]:


import numpy as np
df=pd.get_dummies(data=df,columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_combined','Outlet'],dtype=float)


# In[708]:


df.dtypes


# In[709]:


df.dtypes


# In[710]:


df[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# In[711]:


#drop the columns which have been changed to different type
df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)


# In[712]:


#divide into test and train
traindf=df.loc[df['source']=="dftrain"]
testdf=df.loc[df['source']=="dftest"]


# In[713]:


#drop unecessary columns
traindf.drop(['source'],axis=1,inplace=True)
testdf.drop(['source','Item_Outlet_Sales'],axis=1,inplace=True)


# In[714]:


#export files as modified versions
traindf.to_csv("train_modified.csv",index=False)
testdf.to_csv("test_modified.csv",index=False)


# In[786]:


#Define target and ID columns:
from math import sqrt
import numpy as np
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.metrics import mean_squared_error
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validate(alg, dtrain[predictors], dtrain[target], cv=20,scoring='neg_mean_squared_error')
    
   
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
   
    
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


# In[787]:


#Linear Regression Model
import pandas as pd
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression,Ridge,Lasso
predictors=[x for x in traindf.columns if x not in [target]+IDcol]


alg1=LinearRegression(normalize=True)
modelfit(alg1,traindf,testdf,predictors,target,IDcol,"alg1.csv")
coef1=pd.Series(alg1.coef_,predictors).sort_values
coef1().plot(kind='bar',title='Model Coefficients')


# In[788]:


predictors = [x for x in traindf.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, traindf, testdf, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
 


# In[781]:


from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in traindf.columns if x not in [target]+IDcol]
alg3=DecisionTreeRegressor(max_depth=15,min_samples_leaf=100)
modelfit(alg3,traindf,testdf,predictors,target,IDcol,'alg3.csv')
coef3=pd.Series(alg3.feature_importances_,predictors).sort_values(ascending=False)
coef3.plot(kind='bar',title='Feature Importances')




# In[791]:


predictors=['Item_MRP','Outlet_Type_0','Outlet_5','Years_of_operation']
alg4=DecisionTreeRegressor(max_depth=8,min_samples_leaf=150)
modelfit(alg4,traindf,testdf,predictors,target,IDcol,'alg4.csv')
coef4=pd.Series(alg4.feature_importances_,predictors).sort_values(ascending=False)
coef4.plot(kind='bar',title='Feature importances')


# In[797]:


from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in traindf.columns if x not in [target]+IDcol]
alg5=RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=100,n_jobs=4)
modelfit(alg5, traindf, testdf, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')
 


# In[799]:


predictors = [x for x in traindf.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, traindf, testdf, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')


# In[ ]:




