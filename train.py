import numpy as np
import  urllib.request
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score



f1 = open("train.csv", "r")
train_data = f1.readlines()
f1.close()


data_train = np.loadtxt(train_data, delimiter = ',')

x_train = data_train[:, 0:784]
y_train = data_train[:, 784]


f2 = open("test.csv", "r")
test_data = f2.readlines()
f2.close()


data_test = np.loadtxt(test_data, delimiter = ',')


x_test = data_test[:, 0:784]
y_test = data_test[:, 784]



#SVM======================================
from sklearn import svm
print("\n\n")    

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

predicted = model.predict(x_test)
accuracy = accuracy_score(y_test, predicted)
print("ACCURACY oF SVM")
print(str(accuracy*100)+"%")
	

#MLP-------------------------------------
from sklearn.metrics import accuracy_score
print("\n\n")
mlp = MLPClassifier(solver='lbfgs',   hidden_layer_sizes= (10, ))
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("ACCURACY oF MLP")
print (str(accuracy*100)+"%")

#Decision Tree-----------------------------------
print("\n\n")

mode=DecisionTreeClassifier()
mode.fit(x_train,y_train)
y_pred = mode.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("DecisionTreeClassifier : ", str(accuracy_score(y_test, y_pred)*100.0)+"%")
    

# Naive Bayes ------------------------------------
print("\n\n")
Multi = MultinomialNB()
Multi.fit(x_train, y_train)
y_pred = Multi.predict(x_test)

print("Naive Bayes : ", str(accuracy_score(y_test, y_pred)*100.0)+"%")



# Random Forest -------------------------------------
from sklearn.ensemble import RandomForestClassifier
print("\n\n")


trees = 0

while trees<36:
    trees+=3

    RF = RandomForestClassifier(n_estimators=trees)
    RF.fit(x_train, y_train)
    y_pred = RF.predict(x_test)
    accc = accuracy_score(y_test, y_pred)
    print("Random Forset (num of trees = " + str(trees)+" ) : ", str(accc*100.0)+"%")

