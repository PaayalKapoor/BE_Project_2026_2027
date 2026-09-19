import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

#Since the dataset is very imbalanced. 153 heel lift reps, 19 velocity reps and 220 correct reps - the model will not be able to learn the velocity error 
#and generalise on new data. Therefore, we split the multiclass model into two binary classification models

df = pd.read_csv(r'C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Datasets\Supine_Heel_Slides\supine_heel_slides_velocity.csv')
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'valley_frame']

drop = ['S1_knee_mean', 'S1_knee_std', 'S1_hip_mean', 'S1_hip_std', 'S1_ankle_mean', 'S1_ankle_std', 'S1_pelvic_mean', 'S1_pelvic_std', 'S1_heel_mean', 'S1_heel_std', 'S1_velocity_mean', 'S1_velocity_std',
        'S2_knee_mean', 'S2_knee_std', 'S2_hip_mean', 'S2_hip_std', 'S2_ankle_mean', 'S2_ankle_std', 'S2_pelvic_mean', 'S2_pelvic_std', 'S2_heel_mean', 'S2_heel_std', 'S2_velocity_mean', 'S2_velocity_std',
        'S3_knee_mean', 'S3_knee_std', 'S3_hip_mean', 'S3_hip_std', 'S3_ankle_mean', 'S3_ankle_std', 'S3_pelvic_mean', 'S3_pelvic_std', 'S3_heel_mean', 'S3_heel_std', 'S3_velocity_mean', 'S3_velocity_std',
        'S4_knee_mean', 'S4_knee_std', 'S4_hip_mean', 'S4_hip_std', 'S4_ankle_mean', 'S4_ankle_std', 'S4_pelvic_mean', 'S4_pelvic_std', 'S4_heel_mean', 'S4_heel_std', 'S4_velocity_mean', 'S4_velocity_std',
        'pelvic_lift_max', 'hip_compensation', 'S2_S4_speed_ratio', 'S1_pelvic_min', 'S1_pelvic_max', 'S1_pelvic_range', 'S1_pelvic_vis_frames', 'S2_pelvic_vis_frames', 
        'S3_pelvic_vis_frames', 'S4_pelvic_vis_frames', 'S2_pelvic_min', 'S2_pelvic_max', 'S2_pelvic_range', 'S3_pelvic_min', 'S3_pelvic_max', 'S3_pelvic_range', 'S4_pelvic_min', 
        'S4_pelvic_max', 'S4_pelvic_range', 'S1_ankle_vis_frames', 'S1_ankle_mean', 'S1_ankle_min', 'S1_ankle_max', 'S1_ankle_range', 'S1_ankle_std', 'S2_ankle_vis_frames', 
        'S2_ankle_mean', 'S2_ankle_min', 'S2_ankle_max', 'S2_ankle_range', 'S2_ankle_std', 'S3_ankle_vis_frames', 'S3_ankle_mean', 'S3_ankle_min', 'S3_ankle_max', 'S3_ankle_range', 
        'S3_ankle_std', 'S4_ankle_vis_frames', 'S4_ankle_mean', 'S4_ankle_min', 'S4_ankle_max', 'S4_ankle_range', 'S4_ankle_std', 'S1_hip_vis_frames', 'S1_hip_mean', 'S1_hip_min', 
        'S1_hip_max', 'S1_hip_range', 'S1_hip_std', 'S2_hip_vis_frames', 'S2_hip_mean', 'S2_hip_min', 'S2_hip_max', 'S2_hip_range', 'S2_hip_std', 'S3_hip_vis_frames', 'S3_hip_mean', 
        'S3_hip_min', 'S3_hip_max', 'S3_hip_range', 'S3_hip_std', 'S4_hip_vis_frames', 'S4_hip_mean', 'S4_hip_min', 'S4_hip_max', 'S4_hip_range', 'S4_hip_std', 'S1_knee_range', 
        'S2_knee_range', 'S3_knee_range', 'S4_knee_range', 'S1_velocity_vis_frames', 'S2_velocity_vis_frames', 'S3_velocity_vis_frames', 'S4_velocity_vis_frames', 'label_correct']

def train_binary_model(df, target_label, other_label):
    drop_cols = drop + [other_label]  #We drop certain columns that do not contribute to the model as much and the other label that is in the dataset 
    feature_cols = [c for c in df.columns if c not in [target_label] + drop_cols + metadata] #The target variable along with metadata and a few other columns are dropped from the dataset. This becomes the feature set.

    X = df[feature_cols].values #feature set 
    y = df[target_label].values #Target variable
    groups = df['patient_id'].values #We group patients by patient_id to avoid data leakage

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)  #Stratified group k fold works only on a binary classification
    fold_metrics = [] #The metrics of each fold are saved in this array

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx] #The rows in each fold are divided into train and test sets 
        y_train, y_test = y[train_idx], y[test_idx]

        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), 'Data Leak' #assert acts as a command in python to stop execution in case of data leak
 
        scaler = StandardScaler() #Scaling to ensure all features belong to the same range and the model does not provide more weights to the features with higher range. 
        X_train = scaler.fit_transform(X_train) #We standardize the feature values
        X_test = scaler.transform(X_test)
        #The weights are calculated separately since XGBoost has no argument like RF's 'class_weight' 
        pos = y_train.sum()
        neg = len(y_train) - pos
        weight = neg / max(pos, 1)

        model = xgb.XGBClassifier(random_state=42, scale_pos_weight=weight) #Here we load the XGBoost model into a variable 
        model.fit(X_train, y_train) #Train the model on the given dataset 
        y_pred = model.predict(X_test) #Make predictions
        #Print the metrics for each fold 
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Fold {fold}: pos_weight={weight:.2f}, F1={f1:.3f}, Recall={recall:.3f}, "
              f"Precision={precision:.3f}, test_positives={y_test.sum()}")

        fold_metrics.append({"fold": fold, "accuracy": accuracy, "precision": precision,
                              "recall": recall, "f1": f1})

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n{target_label} — mean/std across folds:")
    print(metrics_df.drop(columns="fold").agg(["mean", "std"]))

    #Final model trained on everything, for feature importance / deployment
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X)
    pos = y.sum(); neg = len(y) - pos
    final_model = xgb.XGBClassifier(random_state=42, scale_pos_weight=neg / max(pos, 1))
    final_model.fit(X_final, y)

    importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    print(f"\nTop 10 features for {target_label}:")
    print(importances.sort_values(ascending=False).head(10))

    return final_model, metrics_df


print("Heel Lift Model")
heel_model, heel_metrics = train_binary_model(df, 'label_heel_lift', 'label_speed_error')

print("Velocity Error Model")
velocity_model, velocity_metrics = train_binary_model(
    df, 'label_speed_error', 'label_heel_lift',
)