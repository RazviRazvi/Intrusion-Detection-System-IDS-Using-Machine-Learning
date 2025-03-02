import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Load the Dataset 
train_path = "kdd_train.csv"
test_path = "kdd_test.csv"

df_train = pd.read_csv(train_path, header=None, low_memory=False)
df_test = pd.read_csv(test_path, header=None, low_memory=False)

#  Assign Column Names 
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels"
]

df_train.columns = column_names
df_test.columns = column_names

#  Encode Categorical Features 
categorical_cols = ["protocol_type", "service", "flag"]
encoders = {}

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df_train[col] = encoders[col].fit_transform(df_train[col])
    df_test[col] = encoders[col].transform(df_test[col])

#  Encode Labels: Handle Unknown Classes 
label_encoder = LabelEncoder()
df_train["labels"] = df_train["labels"].astype(str)
df_test["labels"] = df_test["labels"].astype(str)

df_train.loc[df_train["labels"].isna(), "labels"] = "unknown"
df_test.loc[df_test["labels"].isna(), "labels"] = "unknown"

unique_labels = list(df_train["labels"].unique()) + ["unknown"]
label_encoder.fit(unique_labels)

df_train["labels"] = label_encoder.transform(df_train["labels"])
df_test["labels"] = df_test["labels"].apply(lambda x: x if x in label_encoder.classes_ else "unknown")
df_test["labels"] = label_encoder.transform(df_test["labels"])

#  Convert Numeric Columns 
for col in df_train.columns:
    if col not in categorical_cols + ["labels"]:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

#  Separate Features and Target 
X_train = df_train.drop(columns=["labels"])
y_train = df_train["labels"]
X_test = df_test.drop(columns=["labels"])
y_test = df_test["labels"]

#  Undersampling: Balance Normal and Attack Traffic 
majority_class = y_train.mode()[0]
df_majority = df_train[df_train["labels"] == majority_class]
df_minority = df_train[df_train["labels"] != majority_class]

df_majority_sampled = df_majority.sample(n=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_sampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

#  Prepare Balanced Training Data 
X_train_balanced = df_balanced.drop(columns=["labels"])
y_train_balanced = df_balanced["labels"]

#  Normalize the Data 
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

#  Train the Cybersecurity Model (Random Forest Classifier) 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced_scaled, y_train_balanced)

#  Make Predictions with Anomaly Detection 
y_pred_prob = model.predict_proba(X_test_scaled)
threshold = 0.6  
y_pred = np.argmax(y_pred_prob, axis=1)

# Check confidence and assign "unknown" if below threshold
for i, probs in enumerate(y_pred_prob):
    if max(probs) < threshold:
        y_pred[i] = label_encoder.transform(["unknown"])[0]  

#  Save Attacks and Anomalies Separately 
df_test["Predicted_Label"] = label_encoder.inverse_transform(y_pred)

#  Save ONLY "Unknown" attacks (anomalies)
anomalies = df_test[df_test["Predicted_Label"] == "unknown"]
anomalies[["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "count", "srv_count"]].to_csv("detected_anomalies.csv", index=False)

#  Save ALL Attacks (Known + Unknown)
attacks = df_test[df_test["Predicted_Label"] != "normal"]  # Exclude normal traffic
attacks[["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "count", "srv_count", "Predicted_Label"]].to_csv("detected_attacks.csv", index=False)

#  Save Label Encoders for Decoding 
import pickle
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print(f" Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("- Unknown attacks → `detected_anomalies.csv`")
print("- All detected attacks → `detected_attacks.csv`")
