from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
import joblib

df = pd.read_csv(r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Datasets\Shoulder_Flexion\Dataset for Exercises - Copy of Shoulder_flexion_1.csv")
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'peak_frame']
drop = ['S1_shoulder_mean', 'S1_elbow_mean', 'S1_trunk_mean', 'S1_sho_height_mean', 'S1_wrist_gap_mean', 'S1_velocity_mean', 'S1_acceleration_mean',
        'S2_shoulder_mean', 'S2_elbow_mean', 'S2_trunk_mean', 'S2_sho_height_mean', 'S2_wrist_gap_mean', 'S2_velocity_mean', 'S2_acceleration_mean',
        'S3_shoulder_mean', 'S3_elbow_mean', 'S3_trunk_mean', 'S3_sho_height_mean', 'S3_wrist_gap_mean', 'S3_velocity_mean', 'S3_acceleration_mean',
        'S4_shoulder_mean', 'S4_elbow_mean', 'S4_trunk_mean', 'S4_sho_height_mean', 'S4_wrist_gap_mean', 'S4_velocity_mean', 'S4_acceleration_mean',
        'S1_shoulder_std', 'S1_elbow_std', 'S1_trunk_std', 'S1_sho_height_std', 'S1_wrist_gap_std', 'S1_velocity_std', 'S1_acceleration_std',
        'S2_shoulder_std', 'S2_elbow_std', 'S2_trunk_std', 'S2_sho_height_std', 'S2_wrist_gap_std', 'S2_velocity_std', 'S2_acceleration_std',
        'S3_shoulder_std', 'S3_elbow_std', 'S3_trunk_std', 'S3_sho_height_std', 'S3_wrist_gap_std', 'S3_velocity_std', 'S3_acceleration_std', 
        'S4_shoulder_std', 'S4_elbow_std', 'S4_trunk_std', 'S4_sho_height_std', 'S4_wrist_gap_std', 'S4_velocity_std', 'S4_acceleration_std',
        'S1_wrist_gap_min', 'S1_wrist_gap_max', 'S1_wrist_gap_range', 'S2_wrist_gap_min', 'S2_wrist_gap_max', 'S2_wrist_gap_range', 'S3_wrist_gap_min', 'S3_wrist_gap_max', 'S3_wrist_gap_range',
        'S4_wrist_gap_min', 'S4_wrist_gap_max', 'S4_wrist_gap_range', 'trunk_compensation_s2', 'trunk_compensation_s3', 'S1_shoulder_vis_frames', 'S1_elbow_vis_frames', 'S1_trunk_vis_frames', 'S1_sho_height_vis_frames',
        'S1_trunk_min', 'S1_trunk_max', 'S1_trunk_range', 'S2_trunk_min', 'S2_trunk_max', 'S2_trunk_range', 'S3_trunk_min', 'S3_trunk_max', 'S3_trunk_range',
        'S4_trunk_min', 'S4_trunk_max', 'S4_trunk_range',
        'S1_wrist_gap_vis_frames', 'S1_velocity_vis_frames', 'S1_acceleration_vis_frames', 'S2_shoulder_vis_frames', 'S2_elbow_vis_frames', 'S2_trunk_vis_frames', 'S2_sho_height_vis_frames',
        'S2_wrist_gap_vis_frames', 'S2_velocity_vis_frames', 'S2_acceleration_vis_frames', 'S3_shoulder_vis_frames', 'S3_elbow_vis_frames', 'S3_trunk_vis_frames', 'S3_sho_height_vis_frames',
        'S3_wrist_gap_vis_frames', 'S3_velocity_vis_frames', 'S3_acceleration_vis_frames', 'S4_shoulder_vis_frames', 'S4_elbow_vis_frames', 'S4_trunk_vis_frames', 'S4_sho_height_vis_frames',
        'S4_wrist_gap_vis_frames', 'S4_velocity_vis_frames', 'S4_acceleration_vis_frames', 'Correct']
FEATURES = {
  "Elbow Bent": ["elbow_deviation_s2", "elbow_deviation_s3", "S2_elbow_min", "S3_elbow_min", "S2_elbow_range", "S3_elbow_range", "S2_elbow_std", "S3_elbow_std"],
  "Shoulder Shrug": ["shoulder_shrug_s2", "shoulder_shrug_s3","S2_sho_height_max", "S3_sho_height_max", "S2_sho_height_std", "S3_sho_height_std"],
  "Momentum": ["s2_momentum_ratio", "s2_peak_to_mean_vel", "s2_time_to_peak_vel", "S2_velocity_max", "S2_acceleration_max", "s2_s4_speed_ratio", "peak_overshoot", "extension_at_bottom"],
   "all": ["elbow_deviation_s2", "elbow_deviation_s3", "S2_elbow_min", "S3_elbow_min", "S2_elbow_range", "S3_elbow_range", "S2_elbow_std", "S3_elbow_std", "shoulder_shrug_s2", "shoulder_shrug_s3", "S2_sho_height_max", "S3_sho_height_max",
            "S2_sho_height_std", "S3_sho_height_std", "s2_momentum_ratio", "s2_peak_to_mean_vel", "s2_time_to_peak_vel", "S2_velocity_max", "S2_acceleration_max", "s2_s4_speed_ratio", "peak_overshoot", "extension_at_bottom"]
}

def train_binary_model_svm(df, target_label, other_label_1, other_label_2, other_label_3):
    drop_cols = drop + [other_label_1] + [other_label_2] + [other_label_3] 
    feature_cols = [c for c in df.columns if c not in [target_label] + drop_cols + metadata]

    X = df[FEATURES[target_label]].values
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
elbowbent_model_svm, elbowbent_metrics_svm = train_binary_model_svm(df, 'Elbow Bent', 'Momentum', 'Shoulder Shrug', 'all')
joblib.dump(elbowbent_model_svm, "elbowbent_model_svm.joblib")

# print("Momentum Model")
# momentum_model_lr, momentum_metrics_lr = train_binary_model_lr(df, 'Momentum', 'Elbow Bent', 'Shoulder Shrug', 'all') 
# joblib.dump(momentum_model_lr, "momentum_model_lr.joblib")

print("Shoulder Shrug Model")
shouldershrug_model_svm, shouldershrug_metrics_svm = train_binary_model_svm(df, 'Shoulder Shrug', 'Elbow Bent', 'Momentum', 'all') 
joblib.dump(shouldershrug_model_svm, "shouldershrug_model_svm.joblib")

print("All Errors Model")
all_model_svm, all_metrics_svm = train_binary_model_svm(df, 'all', 'Elbow Bent', 'Momentum', 'Shoulder Shrug') 
joblib.dump(all_model_svm, "all_model_svm.joblib")