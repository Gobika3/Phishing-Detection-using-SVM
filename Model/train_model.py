import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Model/kdd_train.csv")

# Define selected features based on form input
selected_features = [
    'duration', 'protocol_type', 'src_bytes', 'dst_bytes',
    'is_host_login', 'is_guest_login', 'diff_srv_rate',
    'srv_diff_host_rate', 'flag'
]

# Copy data and keep only selected features + target
data = df[selected_features + ['labels']].copy()

# Encode categorical columns
encoders = {}
for column in selected_features:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders[column] = le

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['labels'])
X = data[selected_features]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
with open("Model/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("Model/feature_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("Model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model trained using 9 features and saved successfully.")
