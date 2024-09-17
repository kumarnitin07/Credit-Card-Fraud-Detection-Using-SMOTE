#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
# import all the necessary library in sub-branch

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam

from scipy.stats import boxcox
from scipy.stats import norm

warnings.filterwarnings("ignore")


df = pd.read_csv('creditcard.csv')
df.head()



#visualization
sns.countplot('Class', data=df, color='r')

sns.distplot(df['Amount'], color='r')

sns.distplot(df['Time'], color='r')


#checking NEGATIVE values
print('No. of Negative Values in Amount col =>', np.sum((df['Amount'] < 0).values.ravel()))

print('No. of Negative Values in Time col =>', np.sum((df['Time'] < 0).values.ravel()))

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')


X = df.drop('Class', axis=1)
y = df['Class']


#DROP col with LOW VARIANCE
describe=X.describe()
describe=describe.transpose()
std_dev=describe['std']


#correlation
corr_matrix = X.corr()
corr_matrix.style.background_gradient(cmap='coolwarm')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
dropped_cols = [column for column in upper.columns if any(upper[column] > 0.9)]


# Drop features which are highly correlated 
X.drop(dropped_cols, axis=1, inplace=True)

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()
full_data = X

full_data['scaled_amount'] = rob_scaler.fit_transform(full_data['Amount'].values.reshape(-1,1))
full_data['scaled_time'] = rob_scaler.fit_transform(full_data['Time'].values.reshape(-1,1))
full_data.drop(['Amount', 'Time'], axis=1, inplace=True)


X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(full_data.values)

# PCA scatter plot
f, ax1 = plt.subplots(1, figsize=(15,6))
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
ax1.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', 
            label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', 
            label='Fraud', linewidths=2)
ax1.set_title('PCA', fontsize=14)
ax1.grid(True)
ax1.legend(handles=[blue_patch, red_patch])



#OUTLIERS
f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x="Class", y="V1", data=df,  ax=axes[0])
axes[0].set_title('V1 vs Class')

sns.boxplot(x="Class", y="V2", data=df, ax=axes[1])
axes[1].set_title('V2 vs Class')

sns.boxplot(x="Class", y="V3", data=df, ax=axes[2])
axes[2].set_title('V3 vs Class')

sns.boxplot(x="Class", y="V4", data=df, ax=axes[3])
axes[3].set_title('V4 vs Class')

plt.show()



col_name='V1'
f, ax1 = plt.subplots(1, figsize=(10, 5))
fraud_dist = df[col_name].loc[df['Class'] == 1].values
not_fraud_dist = df[col_name].loc[df['Class'] == 0].values
sns.distplot(fraud_dist,ax=ax1, fit=norm, color='#FB8861', label="Fraud")
sns.distplot(not_fraud_dist,ax=ax1, fit=norm, color='#56F9BB', label="Normal")
ax1.set_title('Distribution of V14 Fraud Transactions vs  Normal Transactions', fontsize=14)
plt.legend(loc="upper right")
plt.show()


#Sampling and Splitting
train=df.loc[df['Class'] == 0]
test=df.loc[df['Class'] == 1]
train_sub = train.sample(n=len(test)*10)
sample_indexes = train_sub.index.tolist()


#dropping sampled rows from original data
train1=train.drop(sample_indexes)

train1['scaled_amount'] = rob_scaler.fit_transform(train1['Amount'].values.reshape(-1,1))
train1['scaled_time'] = rob_scaler.fit_transform(train1['Time'].values.reshape(-1,1))

test['scaled_amount'] = rob_scaler.fit_transform(test['Amount'].values.reshape(-1,1))
test['scaled_time'] = rob_scaler.fit_transform(test['Time'].values.reshape(-1,1))   

train_sub['scaled_amount'] = rob_scaler.fit_transform(train_sub['Amount'].values.reshape(-1,1))
train_sub['scaled_time'] = rob_scaler.fit_transform(train_sub['Time'].values.reshape(-1,1))


train1.drop(['Amount', 'Time'], axis=1, inplace=True)
test.drop(['Amount', 'Time'], axis=1, inplace=True)
train_sub.drop(['Amount', 'Time'], axis=1, inplace=True)


X_train = train1.drop('Class', axis=1)
X_test_1 = test.drop('Class', axis=1)
X_test_0 = train_sub.drop('Class', axis=1)
    

def training(X_train, X_test):
    input_dim=len(X_train.columns)
    autoencoder = Sequential()
    autoencoder.add(Dense(30,  activation='elu', input_shape=(input_dim,)))
    autoencoder.add(Dense(50,  activation='elu'))
    autoencoder.add(Dense(100, activation='linear', name='bottleneck'))
    autoencoder.add(Dense(50,  activation='elu'))
    autoencoder.add(Dense(30,  activation='elu'))
    autoencoder.add(Dense(input_dim,  activation='elu'))
    
    autoencoder.compile(loss='mean_squared_error', optimizer = Adam())
    
    trained_model = autoencoder.fit(X_train, X_train, batch_size=1024, epochs=100, 
                                    verbose=1, validation_data=(X_test, X_test))
    return (trained_model,autoencoder)

model=training(X_train, X_test_1)


#PLOTTING TRAINING AND VALIDATION LOSS
plt.plot(model[0].history['loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()

plt.plot(model[0].history['val_loss'])
plt.title('model_val_loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['val'], loc='upper right')
plt.show()


# bottleneck representation
#encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
#encoded_data = encoder.predict(X_train)  
# reconstruction

def decoding(model,X_test):
    decoded_output = model.predict(X_test)
    errors = decoded_output - X_test
    squared_error = np.square(errors)
    mse_error = squared_error.mean(axis=1)
    output = boxcox(mse_error)
    return (output[0])

mse_train_0 = decoding(model[1],X_train)
mse_test_0 = decoding(model[1],X_test_0)
mse_test_1 = decoding(model[1],X_test_1)


(mu1, sigma1) = norm.fit(mse_train_0)
(mu2, sigma2) = norm.fit(mse_test_1)

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

roots=solve(mu1, mu2, sigma1, sigma2)

threshold = max(roots)
print(threshold)  


f, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
sns.distplot(mse_train_0,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('Normal Transactions \n Distribution', fontsize=14)
sns.distplot(mse_test_1,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('Fraud Transactions \n Distribution', fontsize=14)
plt.show()



f, ax1 = plt.subplots(1, figsize=(15, 5))
sns.distplot(mse_train_0,ax=ax1, fit=norm, color='#FB8861', label="Normal")
sns.distplot(mse_test_1,ax=ax1, fit=norm, color='#56F9BB', label="Fraud")
ax1.set_title('Distribution of Reconstruction Error \n Normal Transactions vs Fraud Transactions', 
              fontsize=18)
plt.axvline(x=threshold, color='red')
plt.text((threshold-.4),-0.012,round(threshold,2),rotation=0, color='red')
plt.legend(loc="upper right")
ax1.annotate('threshold', xy=(threshold, 0.21), xytext=(threshold+2, 0.212),
            arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

#prediction


def cmm(normal,anomaly,threshold):
    filter1 = ((normal <= threshold))
    filter1 = 1*filter1
    
    filter2 = ((anomaly >= threshold))
    filter2 = 1*filter2

    tab= [[      '', 'Normal', 'Anomaly'],
          ['Normal', sum(filter1), len(filter1)-sum(filter1)],
          ['Anomaly',len(filter2)-sum(filter2), sum(filter2)]]
    
    cm=pd.DataFrame(tab[1:], columns=tab[0])
    print(cm)
    print('precision :', cm.iloc[1,2]/(cm.iloc[1,2]+cm.iloc[0,2]))
    print('recall    :', cm.iloc[1,2]/(cm.iloc[1,2]+cm.iloc[1,1]))
    return cm


cm1 = cmm(mse_test_0,  mse_test_1, threshold)









