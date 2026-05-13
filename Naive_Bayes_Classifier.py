# import pandas as pd

# data = pd.read_csv(r"Classification\naive_bayes_play_tennis.csv")

# print(data)

# X = data[['Outlook', 'Temperature', 'Humidity', 'Wind']]
# y = data['PlayTennis']

# X = pd.get_dummies(X)

# from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()

# y = encoder.fit_transform(y)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# from sklearn.naive_bayes import GaussianNB

# model = GaussianNB()

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score

# print("Confusion Matrix")
# print(confusion_matrix(y_test, y_pred))

# print("Accuracy:", accuracy_score(y_test, y_pred))

# print("Precision:", precision_score(y_test, y_pred))

# print("Recall:", recall_score(y_test, y_pred))

# print("F1 Score:", f1_score(y_test, y_pred))

# sample = pd.DataFrame({
#     'Outlook': ['Sunny'],
#     'Temperature': ['Cool'],
#     'Humidity': ['High'],
#     'Wind': ['Strong']
# })

# sample = pd.get_dummies(sample)

# sample = sample.reindex(columns=X.columns, fill_value=0)

# prediction = model.predict(sample)

# print("Prediction:", encoder.inverse_transform(prediction))
import pandas as pd

data = pd.read_csv(r"Classification\naive_bayes_play_tennis.csv")

print(data)

X = data[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = data['PlayTennis']

X = pd.get_dummies(X)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y)

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()

model.fit(X, y)

y_pred = model.predict(X)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Confusion Matrix")
print(confusion_matrix(y, y_pred))

print("Accuracy:", accuracy_score(y, y_pred))

print("Precision:", precision_score(y, y_pred))

print("Recall:", recall_score(y, y_pred))

print("F1 Score:", f1_score(y, y_pred))

sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
})

sample = pd.get_dummies(sample)

sample = sample.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(sample)

print("Prediction:", encoder.inverse_transform(prediction))