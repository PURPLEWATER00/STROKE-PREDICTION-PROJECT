import numpy as np
import pandas as pd
import pickle

# visiualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib import rcParams

# hypothesis testing 
import scipy.stats as stats
from random import sample

# imbalanced data undersampler
from imblearn.under_sampling import RandomUnderSampler
#from imblearn.combine import SMOTEENN
#from imblearn.combine import SMOTETomek
#from imblearn.over_sampling import SMOTE

# model constructing libraries
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score


train1_df=pd.read_csv('train-1.csv')
train2_df=pd.read_csv('train-2.csv')
train_df=pd.concat((train1_df,train2_df))
test_df=pd.read_csv('test.csv')

train_df['smoking_status'].replace('Unknown', train_df['smoking_status'].mode().values[0], inplace = True)
test_df['smoking_status'].replace('Unknown', test_df['smoking_status'].mode().values[0], inplace = True)

train_df = train_df[train_df['gender']!='Other']
train_df['bmi'].fillna(train_df['bmi'].median(),inplace=True)

x = train_df.drop(['id','stroke','hypertension', 'heart_disease'], axis = 1)
cat_val = x.select_dtypes('object')
x = pd.get_dummies(x, columns= cat_val.columns)
x=x.drop(['gender_Female', 'ever_married_No', 'smoking_status_never smoked', 'Residence_type_Rural'], axis=1)

y = train_df['stroke']

us = RandomUnderSampler()
x_res, y_res = us.fit_resample(x, y)

x_train, x_test,y_train, y_test =  train_test_split(x_res,y_res,test_size=0.2)

param_grid =  { 'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf','linear']  }
svr = SVC(probability=True)
grid = GridSearchCV(svr, param_grid, refit = True, verbose = 3, scoring='f1')

grid.fit(x_train,y_train)

