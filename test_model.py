import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("test_dataset.csv")

X_test = df.drop(columns=["label"])
y_test = df["label"]

model = joblib.load("gesture_model.pkl")

y_pred = model.predict(X_test)

print("classification report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("predicted label")
plt.ylabel("true label")
plt.tight_layout()
plt.show()