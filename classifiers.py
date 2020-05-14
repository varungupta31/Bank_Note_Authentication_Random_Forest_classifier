# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 02:52:58 2020

@author: varun
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot




dataset=pd.read_csv("BNA.csv")

#Data set Cleaning and Preprocessing
#----------------------------------------

#Search for nan or missing values.

print(dataset[dataset['variance']==np.nan]) 
print(dataset[dataset['skewness']==np.nan]) 
print(dataset[dataset['curtosis']==np.nan]) 
print(dataset[dataset['entropy']==np.nan]) 
print(dataset[dataset['class']==np.nan]) 

#No nan or missing values in the data set...

#Feature Extraction Process

X=dataset.iloc[:,[0,1,2,3]].values
y=dataset.iloc[:,4].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


sc=StandardScaler()
X_sc=sc.fit_transform(X)
y_sc=sc.fit_transform(y)
X_trainSC=sc.fit_transform(X_train)
X_testSC=sc.fit_transform(X_test)


#Exploratory data analysis
#-------------------------------------------
details=dataset.info()

# RangeIndex: 1372 entries, 0 to 1371
# Data columns (total 5 columns):
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   variance  1372 non-null   float64
#  1   skewness  1372 non-null   float64
#  2   curtosis  1372 non-null   float64
#  3   entropy   1372 non-null   float64
#  4   class     1372 non-null   int64  
# dtypes: float64(4), int64(1)
# memory usage: 53.7 KB

labels=['class 0','class 1']

sizes=[dataset['class'].value_counts()[0],dataset['class'].value_counts()[1]]

plot1 = plt.pie(sizes,labels=labels,shadow=True,autopct='%1.2f%%')
plt.title("Binary Classified values")

plot2=sns.pairplot(dataset,hue='class',palette = "muted")
plot2.fig.suptitle("Pairplot")

plot3=plt.boxplot(X_sc[:,[0]],showmeans=True)
plt.title("Variance values box plot")

plot4=plt.boxplot(X_sc[:,[1]],showmeans=True)
plt.title("skewness values box plot")

plot5=plt.boxplot(X_sc[:,[2]],showmeans=True)
plt.title("curtosis values box plot")

plot6=plt.boxplot(X_sc[:,[3]],showmeans=True)
plt.title("entropy values box plot")




# Building the random forest classifier.



classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
#classifier=RandomForestClassifier(n_estimators=100,criterion='entropy')

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

print(classifier.score(X_test,y_test))
#0.9963636363636363



cm=confusion_matrix(y_test,y_pred)


plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,cmap='YlGnBu',fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.title("Confusion matrix")

print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred)) 


# Plotting Decision Trees


 

features = list(dataset.columns[1:])


dot_data = StringIO()  
export_graphviz(classifier.estimators_[0], out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


export_graphviz(classifier.estimators_[1], out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[1].create_png())

export_graphviz(classifier.estimators_[5], out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[1].create_png())







