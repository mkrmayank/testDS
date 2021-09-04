import numpy as np # linear algebra
import pandas as pd # data processing
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
train=pd.read_csv("../input/titanic/train.csv")
train.head()

test=pd.read_csv("../input/titanic/test.csv")
test.head()

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
gender_submission.head()

train.shape

import seaborn as sns

sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train,palette='winter')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='PuBu')

train['Age'].plot.hist()
train['Fare'].plot.hist(bins=20,figsize=(10,5))

sns.countplot(x='SibSp',data=train)

train['Parch'].plot.hist()

sns.countplot(x='Parch',data=train)

train.isnull().sum()

train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)

train.head()
test.head()

train.dropna(inplace=True)
#train.dropna(inplace=True)

test.shape
train.isnull().sum()
test.isnull().sum()

train['Sex'] = train['Sex'].map({'male':1,'female':0})
test['Sex'] = test['Sex'].map({'male':1,'female':0})

train.head()

train.drop(['PassengerId','Ticket','Embarked','Name','Fare'],axis=1,inplace=True)
test.drop(['PassengerId','Ticket','Embarked','Name','Fare'],axis=1,inplace=True)

test['Age']=test.fillna(test['Age'].mean())

train.head()
test.head()

#model section
X=train.drop('Survived',axis=1)
y=train['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

#logic regression
from sklearn.linear_model import LogisticRegression
Logistic=LogisticRegression()
Logistic.fit(X_train,y_train)

predictions=Logistic.predict(X_test)
Logistic.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier
Random_forest=RandomForestClassifier()
Random_forest.fit(X_train,y_train)

predictions_2=Random_forest.predict(X_test)
Random_forest.score(X_test,y_test)

#decision tree classifier
from sklearn import tree
Decision_tree=tree.DecisionTreeClassifier()
Decision_tree.fit(X_train,y_train)
predictions_3=Decision_tree.predict(X_test)
Decision_tree.score(X_test,y_test)

#k neighbour classifier
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
KNN.fit(X_train,y_train)

predictions_4=KNN.predict(X_test)
KNN.score(X_test,y_test)

Logistic.fit(X,y)
pred=Logistic.predict(test)
submission=pd.DataFrame({"PassengerId": gender_submission["PassengerId"],"Survived":pred})
submission.to_csv('submission.csv',index=False)