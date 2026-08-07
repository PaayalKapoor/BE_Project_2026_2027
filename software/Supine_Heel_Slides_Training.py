import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('docs\supine_heel_slides_data.csv')
labels = ['label_correct', 'label_heel_lift', 'label_speed_error']
X = df.drop[labels].values
y = df[labels].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



