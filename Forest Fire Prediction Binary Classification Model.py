#Import Prerequisites
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Import Data
data=pd.read_csv('ForestFireDataset.csv')

#Converting data into Train and Test data
X=data.drop(['Area', 'Fire Occurrence'], axis=1)
y = data['Fire Occurrence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a model
model=LogisticRegression()

#Train the model
model.fit(X_train,y_train)

#Run the test on the model
y_pred = model.predict(X_test)

#Test the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
