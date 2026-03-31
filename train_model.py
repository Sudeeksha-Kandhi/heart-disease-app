import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_excel('processed.cleveland (2).xlsx')

# Clean data
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.median(numeric_only=True))

# Convert target (IMPORTANT)
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Features & target
X = df.drop('num', axis=1)
y = df['num']

# ✅ BALANCED MODEL (THIS IS THE FIX 🔥)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight='balanced'
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Accuracy check
accuracy = pipeline.score(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(pipeline, "heart_disease_pipeline.joblib")

print("✅ New model saved successfully!")