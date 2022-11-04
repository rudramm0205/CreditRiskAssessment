#!/usr/bin/env python
# coding: utf-8

# # <h1><center> Credit Risk Prediction </center><h1>

# ## <h1><center> Introduction </center></h1>

# Credit Risk is a risk of default on a debt that may arise from a non payment of borrower. It is a common problem in financial institutions all over the world and if the financial institutions dont calculate the risk factor it will lead to a closure of that institution. With the help of this dataset and Machine learning methods, we can help the financial institutions identify what are the risk factors that makes people default credit. Our Predictor variable loan status is bivariate in nature. We have applied Logistic Regression, Random Forest Classifier and Decision Tree Classifier to make the Credit risk prediction.

# In[122]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[123]:


##To read the hear data.
cr_data = pd.read_csv("credit_risk_dataset (1).csv")


# In[124]:


## To see the first 10 observations of the Credit Risk Prediction dataframe.
cr_data.head(10)


# In[125]:


## Total number of rows and columns.
cr_data.shape


# In[126]:


cr_data.info()


# ## <h1><center> Data Cleaning </center></h1>

# In[127]:


## To check the null values present in our data.
cr_data.isnull().sum()


# In[128]:


cr_data.dtypes


# In[129]:


## Imputation of null values that is present in 'person_emp_length' with median.
cr_data['person_emp_length']=cr_data['person_emp_length'].fillna(cr_data['person_emp_length'].median())


# In[130]:


## Imputation of null values that is present in 'loan_int_rate' with median.
cr_data['loan_int_rate']=cr_data['loan_int_rate'].fillna(cr_data['loan_int_rate'].median())


# In[131]:


cr_data.isnull().sum()


# In[132]:


## To present the statistical summary of the dataframe.
cr_data.describe()


# ## <h1><center> Exploratory Data Analysis </center></h1>

# In[133]:


## Plot to show the distribution of observations in person_age variable
sns.distplot(cr_data['person_age'])


# In[134]:


## Plot to show the distribution of observations in person_income variable
sns.distplot(cr_data['person_income'])


# In[135]:


## Plot to show the distribution of observations in person_emp_length variable
sns.distplot(cr_data['person_emp_length'])


# In[136]:


## To identify Outliers in person_age column
cr_data['person_age'].plot.box()


# In[137]:


## To identify Outliers in person_income column
cr_data['person_income'].plot.box()


# In[138]:


## To identify Outliers in person_emp_length column
cr_data['person_emp_length'].plot.box()


# In[139]:


sns.boxplot(data=cr_data,y=cr_data['person_emp_length'],x=cr_data['loan_status'])


# In[140]:


sns.boxplot(data=cr_data,y=cr_data['loan_amnt'],x=cr_data['loan_status'])


# In[141]:


cr_data['loan_grade'].value_counts().plot.bar()


# In[142]:


# numerical variebles
num_cols = pd.DataFrame(cr_data[cr_data.select_dtypes(include=['float', 'int']).columns])
# print the numerical variebles
num_cols.columns


# In[143]:


num_cols_hist = num_cols.drop(['loan_status'], axis=1)
# visualize the distribution for each varieble
plot.figure(figsize=(12,16))

for i, col in enumerate(num_cols_hist.columns):
    idx = int('42'+ str(i+1))
    plot.subplot(idx)
    sns.distplot(num_cols_hist[col], color='blue', 
                 kde_kws={'color': 'indianred', 'lw': 2, 'label': 'KDE'})
    plot.title(col+' distribution', fontsize=14)
    plot.ylabel('Probablity', fontsize=12)
    plot.xlabel(col, fontsize=12)
    plot.xticks(fontsize=12)
    plot.yticks(fontsize=12)
    plot.legend(['KDE'], prop={"size":12})

plot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                    wspace=0.35)
plot.show()


# In[144]:


## Label Encoder for converting the categorical variables into integer type in the data frame.
new_column1 = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file',]
new_enc = preprocessing.LabelEncoder()
for col in new_column1:
    cr_data[col]=  new_enc.fit_transform(cr_data[col])


# In[145]:


cr_data.dtypes


# In[146]:


## This correlation heatmap to show relationship of variables with our predictor variable loan_status
c1, axes1 = plot.subplots(figsize = (14,12))
sns.heatmap(cr_data.corr(), annot=True, ax=axes1)


# # <h1><center> Modelling </center></h1>

# In[147]:


## To drop the predictor variable 'loan_status' from the independent variables for model training. 
x=cr_data.drop(['loan_status'],axis=1)
y=cr_data['loan_status']


# In[148]:


x.shape


# In[149]:


y.shape


# In[150]:


from sklearn import model_selection,linear_model, metrics


# In[151]:


## To split the data into train and test for model building. We seperated into 70:30 ratio.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3,random_state=0)


# In[152]:


## The number of observations after splitting.
x_train.shape,x_test.shape


# In[153]:


## Standard scaler is used for normalization of our training and testing data to Remove Outliers
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# In[154]:


## For importing Logistic regression and its following metrices...
from sklearn.linear_model import LogisticRegression as LgRg
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,precision_score,recall_score,classification_report


# In[155]:


## To fit the Logistic Regression model into the training set and then predict on the testing set. 
lg2= LgRg()
lg2.fit(x_train,y_train)
y_lg_pred = lg2.predict(x_test)
score_lg=accuracy_score(y_test,y_lg_pred)*100
print("training accuracy score: ",accuracy_score(y_train,lg2.predict(x_train))*100)
print("testing accuracy score: ",score_lg)


# In[156]:


## Feature Importance for Logistic Regression Model
imp2 = lg2.coef_[0]


# In[157]:


for i,v in enumerate(imp2):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[158]:


# Plotting Feature Importance of Each Variable
plot.bar([x for x in range(len(imp2))], imp2)
plot.show()


# In[159]:


from sklearn.neighbors import KNeighborsClassifier


# In[160]:


def model_assess(model, name='Default'):
    '''
    This function is used to test model performance 
    
    Input: model, defined classifer
    Output: print the confusion matrix
    
    '''
    
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    preds_proba = model.predict_proba(x_test)
    print(name, '\n',classification_report(y_test, model.predict(x_test)))


# In[161]:


knn = KNeighborsClassifier(n_neighbors=21)
model_assess(knn, name='KNN')


# In[162]:


scoreListknn = []
for i in range(1,21):
    KNclassifier = KNeighborsClassifier(n_neighbors = i)
    KNclassifier.fit(x_train, y_train)
    scoreListknn.append(KNclassifier.score(x_test, y_test))
    
plot.plot(range(1,21), scoreListknn)
plot.xticks(np.arange(1,21,1))
plot.xlabel("K value")
plot.ylabel("Score")
plot.show()
KNAcc = max(scoreListknn)
print("KNN best accuracy: {:.2f}%".format(KNAcc*100))


# In[52]:


# Performing Random Forest Classifier on Training and Test Data Set
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(random_state=200)
rfc = rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
ac = accuracy_score(y_test, y_pred_rfc)
print('Testing Accuracy score is:', ac)
print('Training Accuracy score is:',accuracy_score(y_train,rfc.predict(x_train)))
cm = confusion_matrix(y_test, y_pred_rfc)
sns.heatmap(cm, annot = True, fmt = "d")


# In[53]:


## To Predict the Feature Importance of Random Forest Classifier
imp3 = rfc.feature_importances_


# In[54]:


for i,v in enumerate(imp3):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[55]:


## Plotting Feature importance of Random Forest Classifier
plot.bar([x for x in range(len(imp3))], imp3)
plot.show()


# In[56]:


## Performing Decision Tree Classifier with Random State = 20
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(random_state=20)


# In[57]:


## Fitting the Decision Tree Classifier into our training dataframe
dt_model.fit(x_train,y_train)


# In[58]:


## Predicting the score of our Decision Tree Classifier on Training DataFrame
dt_model.score(x_train,y_train)


# In[59]:


## Predicting the score of our Decision tree Classifier on Test DataFrame
dt_model.score(x_test,y_test)


# In[60]:


dt_model.predict(x_train)


# In[61]:


dt_model.predict_proba(x_test)


# In[62]:


y_pred=dt_model.predict_proba(x_test)[:,1]


# In[63]:


y_new=[]
for i in range(len(y_pred)):
    if y_pred[i]<=0.7:
        y_new.append(0)
    else:
        y_new.append(1)


# In[64]:


accuracy_score(y_test,y_new)


# In[65]:


train_accuracy=[]
test_accuracy=[]
for depth in range(1,10):
    dt_model=DecisionTreeClassifier(max_depth=depth,random_state=20)
    dt_model.fit(x_train,y_train)
    train_accuracy.append(dt_model.score(x_train,y_train))
    test_accuracy.append(dt_model.score(x_test,y_test))


# In[66]:


frame=pd.DataFrame({'max_depth':range(1,10),'train_acc':train_accuracy,'test_cc':test_accuracy})
frame.head()


# In[67]:


## Finding the Depth Of Tree to optimize the Decision Tree Classifier 
plot.figure(figsize=(12,6))
plot.plot(frame['max_depth'],frame['train_acc'],marker='o')
plot.plot(frame['max_depth'],frame['test_cc'],marker='o')
plot.xlabel('Depth of a tree')
plot.ylabel('performance')
plot.legend()


# In[68]:


## Predicting Feature Importance of Decision Tree Classifier
imp4 = dt_model.feature_importances_


# In[69]:


for i,v in enumerate(imp4):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[70]:


## Plotting Feature Importance of Decision Tree Classifier
plot.bar([x for x in range(len(imp3))], imp3)
plot.show()


# ## <h1><center> Observations </center></h1>

# ## 1. After Fitting 3 Machine Learning Algorithms i.e. Logistic Regression, Random Forest Classifier, Decision Tree Classifier the testing accuracy score of Random Forest Classifier gives 89% Accuracy score.
# 
# ## 2. Out of all we got only 4 important features namely Person's Income, Person Home Ownership, Loan Grade & Loan Interest Rate.

# ## <h1><center> References </center></h1>

# https://www.kaggle.com/code/zhaoyunma/credit-risk-prediction/notebook<br>
# https://en.wikipedia.org/w/index.php?title=Credit_risk&oldid=1095601631
