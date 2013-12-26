#!/usr/bin/env python
'''
PREDICT_SURVIVORS.PY

Author: Mikhail Klassen
Email: mikhail.klassen@gmail.com
Created: Dec 25, 2013

Description:
    Predicts the survivors of the RMS Titanic from the passenger
    manifest information. One of the 'Getting Started' competitions
    on Kaggle. See http://www.kaggle.com/c/titanic-gettingStarted

    Uses Random Forests method to predict survivors.
'''

import numpy as np
import csv
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm

####
#
# (1) READ IN DATA. CLEAN DATA
#
####

# Load in the training csv file
train_file_object = csv.reader(open('train.csv', 'rb')) 
test_file_object = csv.reader(open('test.csv', 'rb')) 

# Get the header 
train_header = train_file_object.next() 
test_header = test_file_object.next()

# Create variables to load with CSV data
train_data=[] 
test_data=[]
ids = []
for row in train_file_object: 
    train_data.append(row[1:]) 
for row in test_file_object:
    ids.append(row[0])
    test_data.append(row[1:])

# Then convert from a list to an array
train_data = np.array(train_data)
test_data = np.array(test_data)

# Convert string classifiers to integers
# Male = 1, female = 0:
train_data[train_data[0::,3]=='male',3] = 1
train_data[train_data[0::,3]=='female',3] = 0
test_data[test_data[0::,2]=='male',2] = 1
test_data[test_data[0::,2]=='female',2] = 0
# Embark C = 0, S = 1, Q = 2
train_data[train_data[0::,10] =='C',10] = 0
train_data[train_data[0::,10] =='S',10] = 1
train_data[train_data[0::,10] =='Q',10] = 2
test_data[test_data[0::,9] =='C',9] = 0 
test_data[test_data[0::,9] =='S',9] = 1
test_data[test_data[0::,9] =='Q',9] = 2

# Separate into classes
class1tr = train_data[train_data[:,1] == '1',:]
class2tr = train_data[train_data[:,1] == '2',:]
class3tr = train_data[train_data[:,1] == '3',:]

class1te = test_data[test_data[:,0] == '1',:]
class2te = test_data[test_data[:,0] == '2',:]
class3te = test_data[test_data[:,0] == '3',:]

class1_median_age = np.median(class1tr[class1tr[:,4] != '',4].astype(float))
class2_median_age = np.median(class2tr[class2tr[:,4] != '',4].astype(float))
class3_median_age = np.median(class3tr[class3tr[:,4] != '',4].astype(float))

# For all the ages with no data, assume age is median for each pclass
for i in xrange(np.size(train_data[0::,0])):
    if train_data[i,4] == '':
        train_data[i,4] = np.median(train_data[(train_data[0::,4] != '') & (train_data[0::,0] == train_data[i,0]),4].astype(np.float))
for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,3] == '':
        test_data[i,3] = np.median(test_data[(test_data[0::,3] != '') & (test_data[0::,0] == test_data[i,0]),3].astype(np.float))

# Exclude training rows with missing age data
#train_data = np.delete(train_data, np.where(train_data[:,4] == '')[0],0)

# For all missing ebmbarks, make embark most common place
train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10]\
                                                   != '',10].astype(np.float)))
test_data[test_data[0::,9] == '',9] = np.round(np.mean(test_data[test_data[0::,9]\
                                                   != '',9].astype(np.float)))

# For all the missing fares, assume median of respectice class
for i in xrange(np.size(train_data[0::,0])):
    if train_data[i,8] == '':
        train_data[i,8] = np.median(train_data[(train_data[0::,8] != '') & (train_data[0::,0] == train_data[i,0]),8].astype(np.float))
for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,7] == '':
        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') & (test_data[0::,0] == test_data[i,0]),7].astype(np.float))

# Delete non-informative columns: Name, Ticket Number, Cabin Number
train_data = np.delete(train_data,[2,7,9],1) 
test_data = np.delete(test_data,[1,6,8],1)
features = np.delete(train_header,[0,1,3,8,10])

####
#
# (2) TRAIN CLASSIFIER. RUN ON TEST DATA
#
####

print 'Training...'
classifier = 'forest'

if classifier == 'forest':
    n_ests = 200
    print 'Training Random Forest Classifier with {0} estimators'.format(n_ests)
    clf = RandomForestClassifier(n_estimators=n_ests, criterion="entropy", max_features=None)
    #clf = ExtraTreesClassifier(n_estimators=n_ests)
    #clf = AdaBoostClassifier(n_estimators=n_ests)
    #clf = GradientBoostingClassifier(n_estimators=n_ests)
elif classifier == 'SVM':
    print 'SVM not really working very well. Get\'s caught computing forever...'
    clf = svm.SVC(kernel='poly',gamma=3)
else:
    sys.exit('Bad classifier.')

X = train_data[0::,1::] # Features
y = train_data[0::,0]   # Qualifier

clf = clf.fit(X,y)

print 'Predicting...'
output = clf.predict(test_data)

####
#
# (3) CROSS-VALIDATE AND FEATURE RANKING
#
####

scores = cross_val_score(clf, X, y)
print 'Accuracy: {0:5.2f} (+/-{1:5.2f})'.format(scores.mean(), scores.std()*2)

if classifier == 'forest':
    importances = clf.feature_importances_
    n_feats = len(features)
    feat_std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(n_feats):
        print '{0:2} - {1:12}: {2:5.4f} (std: {3:5.4f})'.format(f+1,features[indices[f]],importances[indices[f]],feat_std[indices[f]])

####
#
# (4) OUTPUT RESULTS 
#
####

open_file_object = csv.writer(open("titanic_predictions.csv", "wb"))
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
