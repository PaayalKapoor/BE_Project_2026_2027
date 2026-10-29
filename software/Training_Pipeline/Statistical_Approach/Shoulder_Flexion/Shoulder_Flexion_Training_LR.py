import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier

df = pd.read_csv(r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Dataset for Exercises - Shoulder_flexion_1.csv")
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'peak_frame']
drop = ['S1_shoulder_mean', 'S1_elbow_mean', 'S1_trunk_mean', 'S1_sho_height_mean', 'S1_wrist_gap_mean', 'S1_velocity_mean', 'S1_acceleration_mean',
        'S2_shoulder_mean', 'S2_elbow_mean', 'S2_trunk_mean', 'S2_sho_height_mean', 'S2_wrist_gap_mean', 'S2_velocity_mean', 'S2_acceleration_mean',
        'S3_shoulder_mean', 'S3_elbow_mean', 'S3_trunk_mean', 'S3_sho_height_mean', 'S3_wrist_gap_mean', 'S3_velocity_mean', 'S3_acceleration_mean',
        'S4_shoulder_mean', 'S4_elbow_mean', 'S4_trunk_mean', 'S4_sho_height_mean', 'S4_wrist_gap_mean', 'S4_velocity_mean', 'S4_acceleration_mean',
        'trunk_compensation_s2', 'trunk_compensation_s3', 'S1_shoulder_vis_frames', 'S1_elbow_vis_frames', 'S1_trunk_vis_frames', 'S1_sho_height_vis_frames',
        'S1_wrist_gap_vis_frames', 'S1_velocity_vis_frames', 'S1_acceleration_vis_frames', 'S2_shoulder_vis_frames', 'S2_elbow_vis_frames', 'S2_trunk_vis_frames', 'S2_sho_height_vis_frames',
        'S2_wrist_gap_vis_frames', 'S2_velocity_vis_frames', 'S2_acceleration_vis_frames', 'S3_shoulder_vis_frames', 'S3_elbow_vis_frames', 'S3_trunk_vis_frames', 'S3_sho_height_vis_frames',
        'S3_wrist_gap_vis_frames', 'S3_velocity_vis_frames', 'S3_acceleration_vis_frames', 'S4_shoulder_vis_frames', 'S4_elbow_vis_frames', 'S4_trunk_vis_frames', 'S4_sho_height_vis_frames',
        'S4_wrist_gap_vis_frames', 'S4_velocity_vis_frames', 'S4_acceleration_vis_frames', 'Correct']
labels = ['Elbow Bent', 'Momentum', 'Shoulder Shrug', 'all']
feature_cols = [c for c in df.columns if c not in drop + metadata + labels]

X = df[feature_cols].values
y = df[labels].values
groups = df['patient_id'].values

sgkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]  #The rows in each fold are divided into train and test sets 
        y_train, y_test = y[train_idx], y[test_idx]

        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), 'Data Leak'  #assert acts as a command in python to stop execution in case of data leak

        scaler = StandardScaler() #Scaling to ensure all features belong to the same range and the model does not provide more weights to the features with higher range. 
        X_train = scaler.fit_transform(X_train) #We standardize the feature values
        X_test = scaler.transform(X_test)


        lr_model = OneVsRestClassifier(LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)) #Load the adaboost model into a variable
        lr_model.fit(X_train, y_train) #Train the model
        y_pred = lr_model.predict(X_test) #Make predictions
        #Print the metrics for each fold
        micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred,  average="macro", zero_division=0)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)


        print(f"Fold {fold}: F1={macro_f1:.3f}, Recall={macro_recall:.3f}, "
              f"Precision={macro_precision:.3f}, test_positives={y_test.sum()}")

        fold_metrics.append({"fold": fold, "accuracy": accuracy, "precision": macro_precision,
                              "recall": macro_recall, "f1": macro_f1})

metrics_df = pd.DataFrame(fold_metrics)
print(f"\nLogistic Regression - mean/std across folds:")
print(metrics_df.drop(columns="fold").agg(["mean", "std"]))
#Train on the entire dataset
final_scaler = StandardScaler()
X_final = final_scaler.fit_transform(X)

final_model = OneVsRestClassifier(LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000))
final_model.fit(X_final, y)

lr_models = final_model.estimators_
for label, lr_model in zip(labels, lr_models):
        coefficients = pd.Series(lr_model.coef_[0], index = feature_cols)
        top_10 = coefficients.abs().sort_values(ascending=False).head(10)
        top_10_coefficients = coefficients.loc[top_10.index]

        print(f"\nTop 10 Features for {label}:")
        print(top_10_coefficients)

#OneVsRestClassifier is used here with logistic regression since LR does not accept multi-labels. OneVsRestClassifier basically trains 3 separate logistic regression models. 
#These individual models can be accessed using .estimators_
#OneVsRestClassifier does not take the average metrics of all 3 models, instead it provides the individual inferences which are combined to get the final output

