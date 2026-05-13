import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"Classification\svm_classifier_dataset.csv")

X = data[['feature1', 'feature2']]
y = data['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

model = SVC(
    kernel='rbf',
    probability=True
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:,1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("Train Score:", model.score(X_train, y_train))
print("Test Score:", model.score(X_test, y_test))

plt.scatter(data['feature1'], data['feature2'], c=data['target'])
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.title("SVM Classification")
plt.show()