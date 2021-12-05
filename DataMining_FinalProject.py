#!/usr/bin/env python
# coding: utf-8

# In[147]:


import numpy as np 
import pandas as pd 
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import collections
import math


# In[148]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score as acs
from sklearn.svm import SVC


# In[172]:


from prettytable import PrettyTable
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense,Activation,Dropout
import warnings
warnings.filterwarnings("ignore")


# # Pre-Processing of the data

# ## Dataset Used (Breast Cancer Diagnostic)

# In[157]:


data=pd.read_csv("data.csv")
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
display(data.head(20))


# In[158]:


data.shape


# In[216]:


X=data.iloc[:,1:32]
y=data.iloc[:,0]


# In[217]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# # Function To Calcuate Metrices 

# # ML Models

# In[218]:


def data_met(tn, fp, fn, tp):
    result = []
    result.append(tn)
    result.append(fp)
    result.append(fn)
    result.append(tp)
    tpr = tp/(tp+fn)
    result.append(tpr)
    tnr = tn/(tn+fp)
    result.append(tnr)
    fpr = fp/(tn+fp)
    result.append(fpr)
    fnr = fn/(tp+fn)
    result.append(fnr)
    recall = tp/(tp+fn)
    result.append(recall)
    precision = tp/(tp+fp)
    result.append(precision)
    f1 = (2*tp)/(2*tp+fp+fn)
    result.append(f1)
    acc = (tp+tn)/(tp+fp+fn+tn)
    result.append(acc)
    err = (fp+fn)/(tp+fp+fn+tn)
    result.append(err)
    bacc = (tpr+tnr)/2
    result.append(bacc)
    tss = tp/(tp+fn) - fp/(fp+tn)
    result.append(tss)
    hss = 2*(tp*tn - fp*fn)/((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))
    result.append(hss)
    return np.array(result)


# In[233]:


cross_validation_folds = KFold(n_splits=10,shuffle=True, random_state=3030)
fold = 0
for train_index, test_index in cross_validation_folds.split(X, y):
    fold += 1
    print("Fold", str(fold))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #Random Forrest
    print("\tRandom Forest model result:")
    rf = RandomForestClassifier(max_depth=5, random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    rf_result = data_met(tn, fp, fn, tp)
    print("\t\tTrue negative:", rf_result[0])
    print("\t\tFalse positive:", rf_result[1])
    print("\t\tFalse negative:", rf_result[2])
    print("\t\tTrue positive:", rf_result[3])
    print("\t\tTrue positive rate:", rf_result[4])
    print("\t\tTrue negative rate:", rf_result[5])
    print("\t\tFalse positive rate:", rf_result[6])
    print("\t\tFalse negative rate:", rf_result[7])
    print("\t\tRecall:", rf_result[8])
    print("\t\tPrecision:", rf_result[9])
    print("\t\tF1:", rf_result[10])
    print("\t\tAccuracy:", rf_result[11])
    print("\t\tError Rate:", rf_result[12]) 
    print("\t\tBalance Accuracy:", rf_result[13])
    print("\t\tTrue skill statistics:", rf_result[14])
    print("\t\tHeidke skill score:", rf_result[15])
     #SVM
    print("\tSVM model result:")
    svc = SVC(gamma='auto')
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svc).ravel()
    svc_result = data_met(tn, fp, fn, tp)
    print("\t\tTrue negative:", svc_result[0])
    print("\t\tFalse positive:", svc_result[1])
    print("\t\tFalse negative:", svc_result[2])
    print("\t\tTrue positive:", svc_result[3])
    print("\t\tTrue positive rate:", svc_result[4])
    print("\t\tTrue negative rate:", svc_result[5])
    print("\t\tFalse positive rate:", svc_result[6])
    print("\t\tFalse negative rate:", svc_result[7])
    print("\t\tRecall:", svc_result[8])
    print("\t\tPrecision:", svc_result[9])
    print("\t\tF1:", svc_result[10])
    print("\t\tAccuracy:", svc_result[11])
    print("\t\tError Rate:", svc_result[12]) 
    print("\t\tBalance Accuracy:", svc_result[13])
    print("\t\tTrue skill statistics:", svc_result[14])
    print("\t\tHeidke skill score:", svc_result[15])
    print("\n\n")
    ##lstm
    #print (X[:5])
    #print (y[:5],set(y))
    #print (y[:5],set(y))
    
    
    X_bar = X.copy()
    y_bar = np.array([1 if k == 'B' else 0 for k in list(y)])
    
    X_train_fold = np.array(X_bar[train_index])
    y_train_fold = np.array(y_bar[train_index]).reshape(-1,1).astype(int)
    X_val_fold = np.array(X_bar[test_index])
    y_val_fold = np.array(y_bar[test_index]).reshape(-1,1).astype(int)
    
    X_train_fold = np.reshape(X_train_fold,(X_train_fold.shape[0], 1, X_train_fold.shape[1])).astype(int)
    X_val_fold = np.reshape(X_val_fold, (X_val_fold.shape[0], 1, X_val_fold.shape[1])).astype(int)
    
    model = Sequential()

    model.add(LSTM(60, return_sequences=True, input_shape=(1,30)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('Softmax'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    history = model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=20, validation_data=(X_val_fold, y_val_fold),
                        verbose=1, shuffle=False)
    
    ## Report LSTM Result
    y_pred_lstm = model.predict(X_val_fold)
    y_pred_lstm = np.array([t[0][0] for t in y_pred_lstm]).ravel()
    y_val_fold = np.array(y_bar[test_index]).ravel()
    tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred_lstm).ravel()
    lstm_result = data_met(tn, fp, fn, tp)
    print("LSTM:\n\n")
    print("\t\tTrue negative:", lstm_result[0])
    print("\t\tFalse positive:", lstm_result[1])
    print("\t\tFalse negative:", lstm_result[2])
    print("\t\tTrue positive:", lstm_result[3])
    print("\t\tTrue positive rate:", lstm_result[4])
    print("\t\tTrue negative rate:", lstm_result[5])
    print("\t\tFalse positive rate:", lstm_result[6])
    print("\t\tFalse negative rate:", lstm_result[7])
    print("\t\tRecall:", lstm_result[8])
    print("\t\tPrecision:", lstm_result[9])
    print("\t\tF1:", lstm_result[10])
    print("\t\tAccuracy:", lstm_result[11])
    print("\t\tError Rate:", lstm_result[12]) 
    print("\t\tBalance Accuracy:", lstm_result[13])
    print("\t\tTrue skill statistics:", lstm_result[14])
    print("\t\tHeidke skill score:", lstm_result[15])
    
    
    table=PrettyTable()
    table.field_names = ['Model','TN','FP','FN','TP','TPR','TNR','FPR','FNR','recall','precision','F1','accuracy','ER','BA','TSS','HSS']
    table.add_row(['RandomForest',rf_result[0],rf_result[1],rf_result[2],rf_result[3],rf_result[4],rf_result[5],rf_result[6],rf_result[7],rf_result[8],rf_result[9],rf_result[10],rf_result[11],rf_result[12],rf_result[13],rf_result[14],rf_result[15]])
    table.add_row(['SVM',svc_result[0],svc_result[1],svc_result[2],svc_result[3],svc_result[4],svc_result[5],svc_result[6],svc_result[7],svc_result[8],svc_result[9],svc_result[10],svc_result[11],svc_result[12],svc_result[13],svc_result[14],svc_result[15]])
    table.add_row(['lstm',lstm_result[0],lstm_result[1],lstm_result[2],lstm_result[3],lstm_result[4],lstm_result[5],lstm_result[6],lstm_result[7],lstm_result[8],lstm_result[9],svc_result[10],lstm_result[11],lstm_result[12],lstm_result[13],lstm_result[14],lstm_result[15]])
    print(table)
    print("\n\n")


# In[ ]:





# In[ ]:




