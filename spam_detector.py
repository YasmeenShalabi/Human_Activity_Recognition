#Spam email detector using scikit-learn

import pandas as pd #data manipulation library to load and manipulate data in a structured format like a dataframe
from sklearn.model_selection import train_test_split #splits data into train and test sets
from sklearn.linear_model import LogisticRegression #used for binary classification problems (Ex:spam vs not spam)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score #confusion shows performance of the classification; accuracy calculates how often models predictions are correct; precision measures proportion of positive predictions that are correct; recall measures proportion of actual positives that are correctly identified; f1_score is the harmonic mean of precision and recall
import seaborn as sns #library for data viz, built on top of matplotlib and makes it easy to create graphics
import matplotlib.pyplot as plt #to plot graphs and charts

#load the dataset and split into training and testing sets

data= pd.read_csv('spambase.csv')
X= data.drop('spam', axis=1) #features
y= data['spam']  #target labels (1 = spam, 0 = not spam)

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42) #returns 4 arrays

#train the logistic regression model to classify emails as spam or not spam

model= LogisticRegression()
model.fit(X_train, y_train) #training the model
y_pred= model.predict(X_test) #uses model to predict the labels(spam or not spam) on the test set X_test. Predictions stored in y_pred

print(X_test) #print data

#evaluate model using accuracy, confusion matrix, precision, recall, and F1 score
accuracy= accuracy_score(y_test, y_pred) # percent of correct predictions
precision= precision_score(y_test, y_pred) #calcualtes proportion of emails predicted as spam that are actually spam
recall= recall_score(y_test, y_pred) #proportion of actual spam emails that are correctly predicted
f1= f1_score (y_test, y_pred) #harmonic mean of precision and recall, provides a single metric that balances both precision and recall

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

#Visualize the confusion matrix using Seaborns heatmap
cm= confusion_matrix(y_test, y_pred) #table with 4 sections (true positive-emails correctly predicted as spam; true neg-emails correctly predicted as not spam; false pos -emails incorrectly pred as spam but were not spma; false neg-emails incorrectly pred as not spam but were actually spam)
sns.heatmap(cm, annot=True, fmt='d') #uses seaborn to visualize the confusion matrix as heatmap; annot=True annotates heatmap with actual values from confusion matrix; fmt=d ensures values are shown as integers and not scientific notation
plt.title('Confusion Matrix')
plt.show()