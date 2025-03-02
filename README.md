# Intrusion-Detection-System-IDS-Using-Machine-Learning
A Machine Learning-powered Intrusion Detection System (IDS) designed to detect cyber threats in network traffic.

This IDS identifies both known and unknown attacks using a trained Random Forest model and anomaly detection techniques.

Features

Preprocessing: Encodes categorical features, normalizes numeric values, and handles imbalanced data.
Machine Learning Model: Trains a Random Forest classifier to classify network attacks with approximately 92 percent accuracy.
Anomaly Detection: Identifies unknown attack types dynamically based on confidence thresholds.
Attack Logging: Saves detected threats in CSV format for analysis.
Human-Readable Reports: Converts encoded attack data into interpretable text using a decoder.
Scalability: Supports additional attack types through continuous learning.

How It Works

Data Preprocessing
Encodes categorical features such as protocol type, service, and flag into numerical values.
Handles missing values and assigns "unknown" to unseen attack types.
Normalizes numerical values such as source bytes and destination bytes using StandardScaler.
Handling Imbalanced Data
Most datasets contain significantly more "normal" traffic than attack data.
Undersampling is applied to balance attack and normal samples.
The dataset is shuffled to ensure the model learns attack patterns effectively.
Machine Learning Model
A Random Forest Classifier is trained on the balanced dataset.
The model predicts attack types in the test dataset.
Anomaly detection is applied: If the confidence score is below a threshold, the sample is marked as "unknown."
Attack Logging and Decoding
Detected attacks are saved in detected_attacks.csv.
Unknown threats are logged separately in detected_anomalies.csv.
The decoder.py script converts numeric attack logs into human-readable format, restoring protocol type, service, and flag values to their original text representation.


Installation and Setup

Prerequisites
Ensure you have the following installed on your system:

Python 3.6 or later
Pandas
NumPy
Scikit-learn
Install Required Dependencies
Run the following command in your terminal:

pip install pandas numpy scikit-learn
How to Run the IDS

Train and Detect Attacks
Run the main IDS script to analyze network traffic and detect threats:

python IDS.py
This generates two files:

detected_attacks.csv: Contains all attacks detected.
detected_anomalies.csv: Contains unknown threats detected.
Convert Encoded Data to Human-Readable Format
Since the IDS encodes categorical features, use the decoder to convert them back into human-readable format:

python decoder.py
This generates two human-readable files:

detected_attacks_text.csv: Contains the decoded attack logs.
detected_anomalies_text.csv: Contains the decoded unknown threats.
