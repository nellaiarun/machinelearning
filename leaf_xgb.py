#!usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 16:27:19 2016
@author: Arunachalam Thirunavukkarasu
©Copyrights Reserved by Author.
"""
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
import xgboost as xgb

def getTrainTest(args):
  (train, test) = (r'', r'')
  if not args:
    print "usage: --traindata filename --testdata filename"
    sys.exit(1)
  if args[0].strip().lower() == "--traindata":
    train = args[1].strip().lower()
    del args[0:2]
  if args[0].strip().lower() == "--testdata":
    test = args[1].strip().lower()
    del args[0:2]
  return train, test
  
def getCatVars(data):
  return data.dtypes[data.dtypes == 'object'].index

def getConVars(data):
  return data.dtypes[data.dtypes != 'object'].index

def understandData(data):
  print "----------------------Exploratory Data Analyses----------------------"
  print "{1} Columns and {0} Rows".format(data.shape[0], data.shape[1])
  print "\nCategorical Variables: {0}".format(getCatVars(data))
  print "\nContinuous Variables: {0}".format(getConVars(data))
  print "\nBasic Stats: {0}".format(data.describe())
  raw_input("Press RETURN key to continue analyses...")
  print "\nSkew of Data: {0}".format(data.skew())
  print "\nVariable Correlation: {0}".format(data.corr(method='pearson'))
  print "---------------------------------------------------------------------"

def showUniqueValues(data, cat_vars):
  print data[cat_vars].apply(lambda x: len(x.unique()))
  
def showUniqueValuesPercent(data, cat_vars):
  recordCount = data.shape[0]
  for cat_var in cat_vars:
    print "\n----------{0}-------------\n".format(cat_var)    
    print (data[cat_var].value_counts()/recordCount).round(decimals=4) #Series has .round function.

def getMissingValueVars(data):
  mv_vars = data.apply(lambda x: sum(x.isnull()))
  mv_vars = mv_vars[mv_vars > 0].index
  if len(mv_vars) == 0:
    print "\nThere are no variables with missing values in the given dataset!"
    return mv_vars
  print "\nList of Missing Variables:"
  for mv_var in mv_vars:
    print "\n{0}".format(mv_var)
  return mv_vars

def replaceMVwithMode(data, mv_vars):
  for var in mv_vars:
    data[var].fillna(data[var].mode()[0],inplace=True)

def encodeCatVariables(data, cat_vars):
  if type(data) == pd.core.frame.DataFrame: data.is_copy = False #Offset Warning
  le = LabelEncoder()
  if len(cat_vars) == 1:
    data = le.fit_transform(data)
    return data, list(le.classes_)
  if len(cat_vars) > 1:
    for cat_var in cat_vars:  
      data.loc[:,cat_var] = le.fit_transform(data[cat_var])
    return data, [0]

def main(args):
  trainfile, testfile = getTrainTest(args)
  traindata = pd.read_csv(trainfile)
  understandData(traindata)
  train_cat_vars = getCatVars(traindata)
  train_con_vars = getConVars(traindata)
  print "\nCount of Unique values for categorical variables:"
  showUniqueValues(traindata, train_cat_vars)
  print "\n% of Unique values for each variable:"
  showUniqueValuesPercent(traindata, train_cat_vars)
  
  train_mv_vars = getMissingValueVars(traindata)
  if len(train_mv_vars) > 0: replaceMVwithMode(traindata, train_mv_vars)
  target_var = ['species']
  unwanted_vars = ['id']
  X = traindata.values[:,2:]
  y = traindata.values[:,1]
  scaler = StandardScaler().fit(X)
  X = scaler.transform(X)
  y, y_classes = encodeCatVariables(y, target_var)
  sss = StratifiedShuffleSplit(y, n_iter=10, test_size=0.2, random_state=23)
  for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

  testdata = pd.read_csv(testfile)
  test_cat_vars = getCatVars(testdata)
  test_con_vars = getConVars(testdata)
  test_mv_vars = getMissingValueVars(testdata)
  if len(test_mv_vars) > 0: replaceMVwithMode(testdata, test_mv_vars)
  X_submit = testdata.values[:,1:]
  X_submit = scaler.transform(X_submit)
  submission_ids = testdata['id']  
  
  xg_train = xgb.DMatrix(X_train, label=y_train)
  xg_test = xgb.DMatrix(X_test, label=y_test)
  xg_submit = xgb.DMatrix(X_submit)
  param = {}
  param['objective'] = 'multi:softprob'
  param['eta'] = 0.3 #Learning Rate
  param['max_depth'] = 3 #Tree Depth
  param['silent'] = 1 #Activate silent mode
  param['nthread']= 4 #Parallelism
  param['num_class'] = 99 #Classes
  param['min_child_weight'] = 0.1
  param['colsample_bytree'] = 0.4
  param['gamma'] = 0.0
  param['sub_sample'] = 1.0
  param['eval_metric'] = 'mlogloss'
  param['reg_alpha'] = 0.01
  num_round = 100 #Trees
  watchlist = [(xg_train, 'train'), (xg_test, 'test')]
  bst = xgb.train(param, xg_train, num_round, watchlist)
  yprob = bst.predict(xg_submit).reshape(testdata.shape[0],len(y_classes))
  df = pd.DataFrame(yprob, index=submission_ids, columns=y_classes)
  df.to_csv('xgb_sub.csv')
  
if __name__ == '__main__': main(sys.argv[1:])