#!usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:38:20 2016
@author: Arunachalam T
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def getMissingVars(data):
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

def getCatVars(data):
  return data.dtypes[data.dtypes == 'object'].index

def getConVars(data):
  return data.dtypes[data.dtypes != 'object'].index

def encodeCatVariables(data, cat_vars):
  le = LabelEncoder() #Instantiate Label Encoder, without which methods like fit_transform can't be executed.
  for cat_var in cat_vars:  
    data.loc[:,cat_var] = le.fit_transform(data[cat_var])

def encodeY(y_data):
  le = LabelEncoder()
  y_data = le.fit_transform(y_data)
  return y_data, list(le.classes_)

def standardNormalization(data):
  scaler = StandardScaler().fit(data) #Compute or Learn Mean/ Median for scaling. No need to scale y - it's a categorical variable and gets encoded up next.
  data = scaler.transform(data) #Apply the learnt Mean/ Median and perform scaling.
  return data

def create_nn_model(neurons=384, init_mode='normal', optimizer='Adam', nb_epoch=150, batch_size=100):
  np.random.seed(seed)
  nn_model = Sequential()
  #Dense is a fully connected layer
  #Dense args: # of neurons, input dimension (for 1st hidden layer only), activation function for all neurons in that layer.
  nn_model.add(Dense(neurons, input_dim=192, init=init_mode, W_regularizer=l2(0.01), activation='relu'))#RELU (Rectified Linear Unit) a.k.a Rectifier activation function used for this 1st hidden layer. Historically only SIGMOID function was used. Another practice is to use hyperbolic tangent (tanh) function for hidden layers.
  nn_model.add(Dropout(0.2))
  nn_model.add(Dense(neurons/2, init=init_mode, W_regularizer=l2(0.01), activation='relu'))
  nn_model.add(Dropout(0.1))
  nn_model.add(Dense(99, init=init_mode, activation='softmax'))#softmax by default provides 'probability' as its outpute. No need to use predict_proba for softmax. To predict the classes using softmax use predict_classes method.
  #learning_rate = 0.01
  #decay_rate = learning_rate / nb_epoch  
  ###rmsprop_optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8, decay=decay_rate)
  #Compile method configures the learning process. It's a must prior to model training.
  nn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])#For multi-class classification, 'cross entropy' cost function and 'softmax' activation function for output layer is the best choice.
  return nn_model

def main():
  np.random.seed(seed)
  trainfile = r"/Arun/ML/Practice/Leaves/train.csv"
  testfile = r"/Arun/ML/Practice/Leaves/test.csv" #File for submission
  traindata = pd.read_csv(trainfile)
  testdata = pd.read_csv(testfile) #Data for prediction and submission
  print "\nDataset has {1} columns and {0} Rows".format(traindata.shape[0], traindata.shape[1])
  print "\nBasic Stats: \n{0}".format(traindata.describe())
  #----Missing Value Treatment----
  tr_mv_vars = getMissingVars(traindata)
  if len(tr_mv_vars) > 0: replaceMVwithMode(traindata, tr_mv_vars)
  te_mv_vars = getMissingVars(testdata)
  if len(te_mv_vars) > 0: replaceMVwithMode(testdata, te_mv_vars)
  #-----X, y assignment-----
  tr_cat_vars = list(getCatVars(traindata))
  tr_con_vars = list(getConVars(traindata))
  te_cat_vars = list(getCatVars(testdata))
  te_con_vars = list(getConVars(testdata))
  target_var = 'species'
  X = traindata.values[:,2:]
  y = traindata.values[:,1] #target_var's all data values
  X_submit = testdata.values[:,1:]#There's no target_var or y value in this dataset
  submission_ids = testdata['id']
  #-------Encoding----------
  if target_var in tr_cat_vars:
    tr_cat_vars.remove(target_var)
    y, y_classes = encodeY(y)
  if len(tr_cat_vars) > 0: encodeCatVariables(X, tr_cat_vars)
  if len(te_cat_vars) > 0: encodeCatVariables(X_submit, te_cat_vars)
  #--------Scaling--------
  ###X = standardNormalization(X) #Disabled to provide improvised approach via estimators[]. Enable it while using it w/ GridSearchCV.
  #--------Modeling-------
  model = KerasClassifier(build_fn=create_nn_model, verbose=0)
  estimators = []
  estimators.append(('normalization', StandardScaler()))
  estimators.append(('MLP', model)) #Multi Layer Perceptron (a.k.a Artificial Neural Network)
  leaf_pipeline = Pipeline(estimators)
  kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
  result = cross_val_score(leaf_pipeline, X, y, cv=kfold, n_jobs=-1)
  print "\nResult: Acc{0}% std({1}%)".format(result.mean()*100, result.std()*100)
  model.fit(X, y)
  y_pred = model.predict(X_submit)
  y_prob = model.predict_proba(X_submit)
  submission = pd.DataFrame(y_prob, index=submission_ids, columns=y_classes)
  submission.to_csv('submit.csv')
  ###model = KerasClassifier(build_fn=create_nn_model, verbose=0)
  ###kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
  ###batches = [20, 100]
  ###init_mode = ['uniform', 'normal', 'zero']
  ###optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
  ###epochs = [100, 150, 200]
  ###param_grid = dict(nb_epoch=epochs)#These args init_mode, optimizers etc corresponds to build_fn i.e., create_nn_model()'s arguments. Inside that function actual Keras function argument is 'init' for init_mode for instance.
  ###grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, n_jobs=-1)
  ###grid_result = grid.fit(X, y)
  ###print "\nBest score: {0} obtained w/ parameters: {1}".format(grid_result.best_score_, grid_result.best_params_)
  ###----------Plotting------
  ###%matplotlib inline
  ###plt.plot(grid_result.cv_results_.get('mean_train_score'))
  ###plt.plot(grid_result.cv_results_.get('mean_test_score'))
  ###plt.xlabel('Folds/ Iterations')
  ###plt.ylabel('Scoring=Accuracy')
  ###plt.legend(['train','test'], loc='lower right')
  ###plt.ylim(ymin=0.88, ymax=1.02)
  ###plt.show()

seed = 7
#Boiler Plate
if __name__ == '__main__': main()