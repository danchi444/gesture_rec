import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from micromlgen import port

df = pd.read_csv("train_dataset.csv")

X = df.drop(columns=['label'])
y = df['label']

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=175, random_state=42)
)

model.fit(X, y)
print(model.named_steps['randomforestclassifier'].classes_)

joblib.dump(model, "gesture_model.pkl")

c_code = port(model.named_steps['randomforestclassifier'])
with open('final/gesture_model.h', 'w') as f:
    f.write(c_code)