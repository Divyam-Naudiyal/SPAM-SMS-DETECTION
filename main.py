import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

with open(r"C:\Users\lenovo\PycharmProjects\CreditCardFraud\spam.csv",  encoding='ISO-8859-1') as file:
    first_row = file.readline().strip()
    columns_with_data = [column.strip() for column in first_row.split(',') if column.strip() != '']

# Read the CSV file with only the selected columns
data = pd.read_csv(r"C:\Users\lenovo\PycharmProjects\CreditCardFraud\spam.csv", usecols=columns_with_data, encoding='ISO-8859-1')

# Display the data
print(data.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['v1'])

# Text preprocessing
data['v2'] = data['v2'].str.lower()
data['v2'] = data['v2'].str.replace('[^a-zA-Z]', ' ', regex=True)
data['v2'] = data['v2'].str.split()

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['v2'].apply(lambda x: ' '.join(x)))

X_train, X_test, y_train, y_test = train_test_split(X, data['v1'], test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{confusion}')
print(f'Classification Report: \n{report}')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

