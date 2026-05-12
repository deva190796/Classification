#Regression metrics

from sklearn.metrics import confusion_matrix

y_true = [1,0,1,0,1,0,1,0]
y_pred = [1,0,1,1,0,0,1,0]

Confusion_matrix = confusion_matrix(y_true, y_pred)
print("Confusion_Matrix",Confusion_matrix)

from sklearn.metrics import accuracy_score
Accuracy_score = accuracy_score(y_true, y_pred)
print("Accuracy: ",Accuracy_score)

from sklearn.metrics import precision_score
Precison_Score = precision_score(y_true, y_pred)
print("Precison_Score",Precison_Score)

from sklearn.metrics import recall_score

Recall_Score = recall_score(y_true, y_pred)
print("Recall Score",Recall_Score)

from sklearn.metrics import roc_auc_score

y_true = [1,1,0,0]
y_prob = [0.9,0.8,0.7,0.2]

print(roc_auc_score(y_true, y_prob))