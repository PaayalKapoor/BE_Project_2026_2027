import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.model_selection import train_test_split
import joblib

recall_weighted_scorer = make_scorer(fbeta_score, beta=2, zero_division = 0) #fbeta is used to find a harmonic mean between recall and precision. The weight depends on the beta value. 
#if beta = 1, both, precision and recall are given the same weightage. beta > 1 provides more weightage to recall (since missed positive cases are a greater concern). beta < 1 provides more weightage to precision.
df = pd.read_csv(r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Dataset for Exercises - Shoulder_flexion_1.csv")
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
    
def train_binary_model_lr(df, target_label, other_label_1, other_label_2, other_label_3):
    drop_cols = drop + [other_label_1] + [other_label_2] + [other_label_3] #Drop certain columns that do not contribute as much to the model's prediction along with the other label (already present in the dataset)
    feature_cols = [c for c in df.columns if c not in [target_label] + drop_cols + metadata] #Feature set

    X = df[FEATURES[target_label]].values #Define the features
    y = df[target_label].values #Define the target variable
    groups = df['patient_id'].values #Group patients with the same patient_id to avoid data leakage 

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) #Stratified groupkfold is used to group similar datapoints. This specific split of data can be used for binary or 1d datasets only
    fold_metrics = [] #Save the fold metrics for each fold

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]  #The rows in each fold are divided into train and test sets 
        y_train, y_test = y[train_idx], y[test_idx]

        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), 'Data Leak'  #assert acts as a command in python to stop execution in case of data leak

        scaler = StandardScaler() #Scaling to ensure all features belong to the same range and the model does not provide more weights to the features with higher range. 
        X_train = scaler.fit_transform(X_train) #We standardize the feature values
        X_test = scaler.transform(X_test)


        lr_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000) #Load the adaboost model into a variable
        lr_model.fit(X_train, y_train) #Train the model
        y_pred = lr_model.predict(X_test) #Make predictions
        print(y_pred)
        print(y_test)
        probabilities = lr_model.predict_proba(X_test)
        print(probabilities)
        
        #Print the metrics for each fold
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)


        print(f"Fold {fold}: F1={f1:.3f}, Recall={recall:.3f}, "
              f"Precision={precision:.3f}, test_positives={y_test.sum()}")

        fold_metrics.append({"fold": fold, "accuracy": accuracy, "precision": precision,
                              "recall": recall, "f1": f1})

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n{target_label} (Logistic Regression) - mean/std across folds:")
    print(metrics_df.drop(columns="fold").agg(["mean", "std"]))
    #Train on the entire dataset
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X)

    final_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    final_model.fit(X_final, y)

    # coefficients = pd.Series(final_model.coef_[0],index=feature_cols)
    # print(f"\nTop 10 features for {target_label} (Logistic Regression):")
    # print(coefficients.abs().sort_values(ascending=False).head(10).index) #Interpretation: If it is a positive coefficient - higher value indicates a higher probability of label = 1
    # #Negative coefficient - higher value is associated with a lower probability of label = 1

    return final_model, metrics_df 


print("Elbow Bent Model")
elbowbent_model_lr, elbowbent_metrics_lr = train_binary_model_lr(df, 'Elbow Bent', 'Momentum', 'Shoulder Shrug', 'all')
joblib.dump(elbowbent_model_lr, "elbowbent_model_lr.joblib")

# print("Momentum Model")
# momentum_model_lr, momentum_metrics_lr = train_binary_model_lr(df, 'Momentum', 'Elbow Bent', 'Shoulder Shrug', 'all') 
# joblib.dump(momentum_model_lr, "momentum_model_lr.joblib")

print("Shoulder Shrug Model")
shouldershrug_model_lr, shouldershrug_metrics_lr = train_binary_model_lr(df, 'Shoulder Shrug', 'Elbow Bent', 'Momentum', 'all') 
joblib.dump(shouldershrug_model_lr, "shouldershrug_model_lr.joblib")

print("All Errors Model")
all_model_lr, all_metrics_lr = train_binary_model_lr(df, 'all', 'Elbow Bent', 'Momentum', 'Shoulder Shrug') 
joblib.dump(all_model_lr, "all_model_lr.joblib")
