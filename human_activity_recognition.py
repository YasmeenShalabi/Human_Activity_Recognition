#Human activity recognition with smartphones

import pandas as pd #load and manipulate dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #used for human activity recognition
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns #to create heatmap from confusion matrix visualization
import matplotlib.pyplot as plt #to plot confusion matrix

# load dataset
# https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
df= pd.read_csv('har_data.csv')

#preprocess the dataset
X= df.drop('Activity', axis=1) #removes activity and leaves only the features (sensory readings)
y= df['Activity']

#split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random state ensures reproducability of the split

#train the Randon Forest Classifier
model= RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators=> trees
model.fit(X_train, y_train)

#make predictions on test set
y_pred= model.predict(X_test)

#evaluate model using accuracy, precision, recall, f1-score
accuracy= accuracy_score(y_test, y_pred)
precision= precision_score(y_test, y_pred, average='macro')
recall= recall_score(y_test, y_pred, average='macro')
f1= f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy * 100:.2f}")
print(f"Precision: {precision * 100:.2f}")
print(f"Recall: {recall * 100:.2f}")
print(f"F1_score: {f1 * 100:.2f}")

#visualize the confusion matrix using Seaborns heatmap
cm= confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()