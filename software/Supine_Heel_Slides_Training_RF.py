import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold


df = pd.read_csv(r'C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\supine_heel_slides_ML.csv')
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'valley_frame']
drop = ['Diff-S2range', 'S1min-S3max', 'Diff-S4range', 'S1_knee_mean', 'S1_knee_std', 'S1_hip_mean', 'S1_hip_std', 'S1_ankle_mean', 'S1_ankle_std', 'S1_pelvic_mean', 'S1_pelvic_std', 'S1_heel_mean', 'S1_heel_std',
        'S2_knee_mean', 'S2_knee_std', 'S2_hip_mean', 'S2_hip_std', 'S2_ankle_mean', 'S2_ankle_std', 'S2_pelvic_mean', 'S2_pelvic_std', 'S2_heel_mean', 'S2_heel_std',
        'S3_knee_mean', 'S3_knee_std', 'S3_hip_mean', 'S3_hip_std', 'S3_ankle_mean', 'S3_ankle_std', 'S3_pelvic_mean', 'S3_pelvic_std', 'S3_heel_mean', 'S3_heel_std',
        'S4_knee_mean', 'S4_knee_std', 'S4_hip_mean', 'S4_hip_std', 'S4_ankle_mean', 'S4_ankle_std', 'S4_pelvic_mean', 'S4_pelvic_std', 'S4_heel_mean', 'S4_heel_std', 'pelvic_lift_max', 'hip_compensation', 'S2_S4_speed_ratio', 'label_correct',
        'heel_lift_max', 'heel_lift_mean', 'S1_pelvic_min', 'S1_pelvic_max', 'S1_pelvic_range', 'S1_pelvic_vis_frames', 'S2_pelvic_vis_frames', 'S3_pelvic_vis_frames', 'S4_pelvic_vis_frames',
        'S2_pelvic_min', 'S2_pelvic_max', 'S2_pelvic_range', 'S3_pelvic_min', 'S3_pelvic_max', 'S3_pelvic_range', 'S4_pelvic_min', 'S4_pelvic_max', 'S4_pelvic_range']
labels = ['label_heel_lift', 'label_speed_error']
feature_cols = [c for c in df.columns if c not in labels + drop + metadata]
X = df.drop(columns = labels + drop + metadata).values
y = df[labels].values
groups = df['patient_id'].values

gkf = GroupKFold(n_splits=5)
fold_metrics = []
fold_reports = []

for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    assert set(groups[train_index]).isdisjoint(set(groups[test_index])), 'Data Leak'
    
# #Here we use GroupShuffleSplit function in python to avoid having a particular patient's reps in the training and testing set both. A particular patient's reps should either only be in the training or testing set
# gss = GroupShuffleSplit(n_splits = 1, test_size=0.2, random_state = 42)
# train_index, test_index = next(gss.split(X, y, groups=groups)) #Used instead of a for loop

# X_train, X_test = X[train_index], X[test_index]
# y_train, y_test = y[train_index], y[test_index]

# #Check to make sure that no patient's reps are in the training and testing set both
# train_patients = set(groups[train_index])
# test_patients = set(groups[test_index])
# assert train_patients.isdisjoint(test_patients), "Data Leak" #assert acts as a command in python to stop execution in case of data leak

    print("Train label counts: ", pd.DataFrame(y_train, columns = labels).sum())
    print("Test label counts: ", pd.DataFrame(y_test, columns=labels).sum())

#Learned that standard scaler is not required for random forest or tree based models since they analyze one feature at a time so they are invariant to scaling. 
    scaler = StandardScaler()

#The standard scaler function in python is used to standardize (bring all the feature values in the range of 0-1) the feature set in order to avoid the model from giving more weightage to the features with a higher range.
    X_train = scaler.fit_transform(X_train)
#Standardize the testing data on the fitted scaler 
    X_test = scaler.transform(X_test)

    rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)

    correct_true = (y_test.sum(axis=1)==0).astype(int)
    correct_pred = (y_pred_rf.sum(axis=1)==0).astype(int)
    correct_f1 = f1_score(correct_true, correct_pred, zero_division=0)

    print(f"Fold: {fold}")
    print(f"Training Reps: {len(train_index)}, Testing Reps: {test_index}")
    print(f"Test patients: {len(set(groups[test_index]))}")
    print(f"Test label counts:" , pd.DataFrame(y_test, columns = labels).sum())
    print(f"Accuracy: {accuracy}, Precision: {precision}, F1 Score: {f1}, Recall: {recall}, Correct F1: {correct_f1}")

    fold_metrics.append({"fold": fold, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "correct_f1": correct_f1})
    fold_reports.append(classification_report(y_test, y_pred_rf, target_names=labels, zero_division=0, output_dict=True))


metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df.drop(columns = "fold").agg(["mean", "std"]))

for label in labels:
    label_f1s = [rep[label]["f1-score"] for rep in fold_reports]
    label_recall = [rep[label]["recall"] for rep in fold_reports]
    print(f"{label}: mean F1 = {np.mean(label_f1s)} (+/- {np.std(label_f1s)})\n"
          f"mean Recall = {np.mean(label_recall)} (+/- {np.std(label_recall)})")

#We finally train on the entire dataset. We perform cross validation only to confirm that one particular partition of data is not giving us skewed results
final_scaler = StandardScaler()
X_final = final_scaler.fit_transform(X)

rf = RandomForestClassifier(random_state = 42, class_weight="balanced")
rf.fit(X_final, y)

importances = pd.Series(rf.feature_importances_, index =feature_cols)
print("Top 15 feature importances: ")
print(importances.sort_values(ascending=False).head(15))

