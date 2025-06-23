import numpy as np
from keras.models import load_model
from tcn import TCN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = np.load('test_dataset.npz')
X_test = data['X']
y_test = data['y']

class_names = ['double', 'flick', 'infinity', 'junk', 'kiss'] 

model = load_model('gesture_model.h5', custom_objects={'TCN': TCN})

y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)

print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

num_classes = len(class_names)
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

plt.ylabel('True label')
plt.xlabel('Predicted label')

thresh = cm.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()