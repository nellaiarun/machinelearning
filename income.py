#!usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:27:40 2016
@author: Arunachalam Thirunavukkarasu
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectPercentile
from sklearn.grid_search import GridSearchCV

#Main function
def main():
  data = pd.read_csv("train_data.csv")
  testdata = pd.read_csv("test_data.csv")
  cat_vars = data.dtypes[data.dtypes=='object'].index #Categorical Variables
  con_vars = data.dtypes[data.dtypes=='int64'].index #Continuous Variables
  print "Basic Statistics of Continuous Variables: "
  print data.describe()
  print "Unique values of Categorical Variables: "
  print data[cat_vars].apply(lambda x: len(x.unique()))
  mv_vars = data.apply(lambda x: sum(x.isnull()) > 0) #Missing values Series
  mv_vars = mv_vars[mv_vars].index #Extract Variables
  print "Categorical Variables with Null/ Missing Values: "
  print mv_vars
  #Treat missing values with mode.
  for var in mv_vars:
    data[var].fillna(data[var].mode()[0], inplace=True)
    testdata[var].fillna(testdata[var].mode()[0], inplace=True)
  #Transform/ Normalize Categorical Variables.  
  for vt_var in cat_vars:
    vt_values = data[vt_var].value_counts()/data.shape[0]*100
    vt_values = vt_values[vt_values < 5.0]
    for vt_value in vt_values:
      data[vt_var].replace(vt_value,'Others',inplace=True)
    if vt_var.strip().lower() == 'income.group': continue
    vt_values = testdata[vt_var].value_counts()/testdata.shape[0]*100
    vt_values = vt_values[vt_values < 5.0]
    for vt_value in vt_values:
      testdata[vt_var].replace(vt_value,'Others',inplace=True)
    
  target_var = 'Income.Group'
  predictor_vars = [x for x in data.dtypes.index if x not in ['ID', target_var]]
  X = data[predictor_vars]
  y = data[target_var]
  
  le = LabelEncoder()
  for var in cat_vars:
    if var == target_var:     
      y = le.fit_transform(y)
      continue
    X[var] = le.fit_transform(X[var]) #Label Encode only Categorical variables
    testdata[var] = le.fit_transform(testdata[var])
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #80/20 split
  
  logreg_model = LogisticRegression()
  logreg_model.fit(X_train,y_train) #Train the model
  y_test_pred = logreg_model.predict(X_test)
  y_test_accuracy = accuracy_score(y_test, y_test_pred)
  #Quick check of evaluation metric (accuracy) with 20% split obtained from train_test_split
  print("y_test accuracy is:" + str(y_test_accuracy))
  
  #Comprehensive evaluation using cross validation with 10 folds.  
  scores = cross_val_score(logreg_model, X, y, scoring='accuracy', cv=10) #K-Folds=10
  print "Mean Accuracy score of Logistic Regression model for K folds, i.e., 10 folds, is: " + str(scores.mean())
  print "Null score for training data is: " + str(max(y.mean(), 1-y.mean()))
  #Closer look at confusion matrix
  print confusion_matrix(y_test, y_test_pred)
  logreg_model.predict_proba(X_test)[0:10,:]
  y_test_pred_prob = logreg_model.predict_proba(X_test)[:,1] #1st Col is prob for 1's predictions.
  %matplotlib inline
  plt.hist(y_test_pred_prob,bins=10) #Observation: Whopping majority falls below 0.5 threshold
  y_test_pred_new = binarize(y_test_pred_prob,0.3)[0] #Adjust prediction threshold
  print confusion_matrix(y_test, y_test_pred_new) #Check confusion matrix post threshold adjustment
  
  #Moving onto Ensembles with Bagging - Random Forest
  k_scores = []
  best_k = 10
  max_k_score = 0
  #Determine optimal # of trees.
  for k in range(10,101,10):
    rfc = RandomForestClassifier(n_estimators=k)
    rfc_scores = cross_val_score(rfc, X, y, scoring='accuracy', cv=10)
    k_scores.append(rfc_scores.mean())
    if rfc_scores.mean() > max_k_score:
      best_k = k
      max_k_score = rfc_scores.mean()
  
  #Grid search other hyperparameters
  criterion_options = ['gini','entropy']
  min_samples_leaf_options = range(1,11,1)
  max_depth_options = range(20,101,20)
  param_grid = dict(criterion=criterion_options, min_samples_leaf=min_samples_leaf_options, max_depth=max_depth_options, n_estimators=[best_k], n_jobs=[-1])
  rfc = RandomForestClassifier()
  grid = GridSearchCV(rfc, param_grid, cv=10, scoring='accuracy')
  grid.fit(X,y)
  print grid.best_estimator_
  print grid.best_params_
  
  #Create RFC with findings from Grid Search
  rfc = RandomForestClassifier(n_estimators=80, n_jobs=-1, criterion='entropy', max_depth=60, min_samples_leaf=10)
  rfc.fit(X_train, y_train)
  y_test_pred_rfc = rfc.predict(X_test)
  print accuracy_score(y_test, y_test_pred_rfc)
  print confusion_matrix(y_test, y_test_pred_rfc)
  print f1_score(y_test, y_test_pred_rfc)
  y_test_pred_prob_rfc = rfc.predict_proba(X_test)[:,1]
  plt.hist(y_test_pred_prob_rfc,bins=10)
  y_test_pred_rfc_new = binarize(y_test_pred_prob_rfc,0.3)[0]
  print accuracy_score(y_test, y_test_pred_rfc_new)  
  print confusion_matrix(y_test, y_test_pred_rfc_new)
  print f1_score(y_test, y_test_pred_rfc_new)
"""  
  fs = SelectPercentile(feature_selection.chi2, percentile=60)
  X_fs = fs.fit_transform(X, y)
  testdata_fs = fs.transform(testdata[predictor_vars])
  rfc = RandomForestClassifier(n_estimators=best_k)
  X_train_fs, X_test_fs, y_train, y_test = train_test_split(X_fs, y, test_size=0.2)
  rfc.fit(X_train_fs, y_train)
  y_test_pred_rfc_fs = rfc.predict(X_test_fs)
  print accuracy_score(y_test, y_test_pred_rfc_fs)
  print confusion_matrix(y_test, y_test_pred_rfc_fs)
  print f1_score(y_test, y_test_pred_rfc_fs)
  y_test_pred_prob_rfc_fs = rfc.predict_proba(X_test_fs)[:,1]
  plt.hist(y_test_pred_prob_rfc_fs,bins=10)
  y_test_pred_rfc_fs_new = binarize(y_test_pred_prob_rfc_fs,0.4)[0]
  print accuracy_score(y_test, y_test_pred_rfc_fs_new)
  print confusion_matrix(y_test, y_test_pred_rfc_fs_new)
  print f1_score(y_test, y_test_pred_rfc_fs_new)
"""  
  testdata_pred_rfc = rfc.predict(testdata[predictor_vars])
  output = pd.DataFrame(np.vstack((testdata['ID'].values, testdata_pred_rfc)).T)
  #testdata_pred_rfc_prob = rfc.predict_proba(testdata[predictor_vars])[:,1]
  #testdata_pred_rfc_new = binarize(testdata_pred_rfc_prob, 0.45)[0]
  #output = pd.DataFrame(np.vstack((testdata['ID'].values, testdata_pred_rfc_new)).T)
  output['Income.Group'] = output[1].map(lambda x: '<=50K' if x==0 else '>50K')
  output.drop(1,axis=1)
  output.to_csv('submit.csv',sep=',',header=True,index=False)

#Boiler Plate
if __name__ == '__main__': main()
