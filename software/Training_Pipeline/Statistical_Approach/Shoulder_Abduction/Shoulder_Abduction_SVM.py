from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
import joblib

df = pd.read_csv(r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Datasets\Shoulder_Abduction\Dataset for Exercises - shoulder_abduction.csv")
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'peak_frame']
drop = ['S1_arm_trunk_vis_frames', 'S1_arm_vertical_vis_frames', 'S1_shoulder_tilt_vis_frames', 'S1_elbow_vis_frames', 'S1_trunk_vis_frames', 'S1_sho_height_vis_frames', 'S1_velocity_vis_frames', 'S1_acceleration_vis_frames',
        'S2_arm_trunk_vis_frames', 'S2_arm_vertical_vis_frames', 'S2_shoulder_tilt_vis_frames', 'S2_elbow_vis_frames', 'S2_trunk_vis_frames', 'S2_sho_height_vis_frames', 'S2_velocity_vis_frames', 'S2_acceleration_vis_frames',
        'S3_arm_trunk_vis_frames', 'S3_arm_vertical_vis_frames', 'S3_shoulder_tilt_vis_frames', 'S3_elbow_vis_frames', 'S3_trunk_vis_frames', 'S3_sho_height_vis_frames', 'S3_velocity_vis_frames', 'S3_acceleration_vis_frames',
        'S4_arm_trunk_vis_frames', 'S4_arm_vertical_vis_frames', 'S4_shoulder_tilt_vis_frames', 'S4_elbow_vis_frames', 'S4_trunk_vis_frames', 'S4_sho_height_vis_frames', 'S4_velocity_vis_frames', 'S4_acceleration_vis_frames',
        'shoulder_shrug_s2', 'shoulder_shrug_s3', 'correct']

def train_binary_model_svm(df, target_label, other_label):
    drop_cols = drop + [other_label]
    feature_cols = [c for c in df.columns if c not in [target_label] + drop_cols + metadata]

    X = df[feature_cols].values
    y = df[target_label].values
    groups = df['patient_id'].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), 'Data Leak'

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Predicted: ", y_pred)
        print("Actual: ", y_test)
        probabilities = model.predict_proba(X_test)
        print(probabilities)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Fold {fold}: Accuracy={accuracy:.3f}, F1={f1:.3f}, Recall={recall:.3f}, "f"Precision={precision:.3f}, test_positives={y_test.sum()}")

        fold_metrics.append({"fold": fold, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n{target_label} (SVM) — mean/std across folds:")
    print(metrics_df.drop(columns="fold").agg(["mean", "std"]))

    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X)
    final_model = SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=42)
    final_model.fit(X_final, y)

    return final_model, metrics_df


print("Elbow Bent Model")
elbowbent_model_svm, elbowbent_metrics_svm = train_binary_model_svm(df, 'elbow_bent', 'velocity')
joblib.dump(elbowbent_model_svm, "elbowbent_model_svm.joblib")

# print("Momentum Model")
# momentum_model_lr, momentum_metrics_lr = train_binary_model_lr(df, 'Momentum', 'Elbow Bent', 'Shoulder Shrug', 'all') 
# joblib.dump(momentum_model_lr, "momentum_model_lr.joblib")

print("Velocity Model")
vel_model_svm, vel_metrics_svm = train_binary_model_svm(df, 'velocity', 'elbow_bent') 
joblib.dump(vel_model_svm, "vel_model_svm.joblib")
