import numpy as np
import matplotlib.pyplot as plt

import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from data_preprocessing import load_training_data, load_test_data, get_pipeline


# ----------------------------------------------------
# Load data

X_train, X_holdout, y_train, y_holdout = load_training_data()
X_test = load_test_data()

# K-fold cross validation for different k

cv = skl_ms.StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
K = np.arange(1, 50)
misclassification = np.zeros(len(K))

for j, k in enumerate(K):
    pipeline = get_pipeline(skl_nb.KNeighborsClassifier(n_neighbors=k))
    scores = skl_ms.cross_val_score(pipeline, X_train, y_train, cv=cv)
    misclassification[j] = 1 - np.mean(scores)

plt.plot(K, misclassification, marker='o', markersize=4)
plt.title('Cross validation error for kNN')
plt.xlabel('k')
plt.ylabel('Validation error')
plt.show()

#recall value for different K

K = np.arange(1, 50)
recall_low = []
recall_high = []


for k in K:
    pipeline = get_pipeline(skl_nb.KNeighborsClassifier(n_neighbors=k))
    y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv)
    report = classification_report(y_train, y_pred, output_dict=True)
    
    recall_low.append(report.get("low_bike_demand", {}).get("recall", 0))
    recall_high.append(report.get("high_bike_demand", {}).get("recall", 0))


plt.figure(figsize=(10, 6))
plt.plot(K, recall_low, label="low_bike_demand", marker='o', markersize=4)
plt.plot(K, recall_high, label="high_bike_demand", marker='s', markersize=4)
plt.title("Recall as function of k")
plt.xlabel("k")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

kNN = get_pipeline(skl_nb.KNeighborsClassifier(n_neighbors=2))

y_pred = cross_val_predict(kNN, X_train, y_train, cv=cv)

print(classification_report(y_train, y_pred))

ConfusionMatrixDisplay.from_predictions(y_train, y_pred)
plt.show()
