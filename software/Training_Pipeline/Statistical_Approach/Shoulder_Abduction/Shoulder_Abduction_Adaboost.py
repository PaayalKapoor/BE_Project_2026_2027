
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import joblib

#Since the dataset is very imbalanced. 153 heel lift reps, 19 velocity reps and 220 correct reps - the model will not be able to learn the velocity error 
#and generalise on new data. Therefore, we split the multiclass model into two binary classification models
df = pd.read_csv(r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Datasets\Shoulder_Abduction\Dataset for Exercises - shoulder_abduction.csv")
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'peak_frame']
drop = ['S1_arm_trunk_vis_frames', 'S1_arm_vertical_vis_frames', 'S1_shoulder_tilt_vis_frames', 'S1_elbow_vis_frames', 'S1_trunk_vis_frames', 'S1_sho_height_vis_frames', 'S1_velocity_vis_frames', 'S1_acceleration_vis_frames',
        'S2_arm_trunk_vis_frames', 'S2_arm_vertical_vis_frames', 'S2_shoulder_tilt_vis_frames', 'S2_elbow_vis_frames', 'S2_trunk_vis_frames', 'S2_sho_height_vis_frames', 'S2_velocity_vis_frames', 'S2_acceleration_vis_frames',
        'S3_arm_trunk_vis_frames', 'S3_arm_vertical_vis_frames', 'S3_shoulder_tilt_vis_frames', 'S3_elbow_vis_frames', 'S3_trunk_vis_frames', 'S3_sho_height_vis_frames', 'S3_velocity_vis_frames', 'S3_acceleration_vis_frames',
        'S4_arm_trunk_vis_frames', 'S4_arm_vertical_vis_frames', 'S4_shoulder_tilt_vis_frames', 'S4_elbow_vis_frames', 'S4_trunk_vis_frames', 'S4_sho_height_vis_frames', 'S4_velocity_vis_frames', 'S4_acceleration_vis_frames',
        'shoulder_shrug_s2', 'shoulder_shrug_s3', 'correct']

def train_binary_model_adaboost(df, target_label, other_label):
    drop_cols = drop + [other_label] #Drop certain columns that do not contribute as much to the model's prediction along with the other label (already present in the dataset)
    feature_cols = [c for c in df.columns if c not in [target_label] + drop_cols + metadata] #Feature set

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
        sample_weight = np.where(y_train == 1, weight, 1.0)

        model = AdaBoostClassifier(random_state=42, n_estimators=200) #Here we load the XGBoost model into a variable 
        model.fit(X_train, y_train, sample_weight=sample_weight) #Train the model on the given dataset 
        y_pred = model.predict(X_test) #Make predictions
        print("Predicted: ", y_pred)
        print("Actual: ", y_test)
        probabilities = model.predict_proba(X_test)
        print(probabilities)
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
    final_weight = neg / max(pos, 1)
    final_sample_weight = np.where(y == 1, final_weight, 1.0)
    final_model = AdaBoostClassifier(random_state=42, n_estimators=200)
    final_model.fit(X_final, y, sample_weight=final_sample_weight)

    # importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    # print(f"\nTop 10 features for {target_label}:")
    # print(importances.sort_values(ascending=False).head(10))

    return final_model, metrics_df

print("Elbow Bent Model")
elbowbent_model_adaboost, elbowbent_metrics_adaboost = train_binary_model_adaboost(df, 'elbow_bent', 'velocity')
joblib.dump(elbowbent_model_adaboost, "elbowbent_model_adaboost.joblib")

# print("Momentum Model")
# momentum_model_lr, momentum_metrics_lr = train_binary_model_lr(df, 'Momentum', 'Elbow Bent', 'Shoulder Shrug', 'all') 
# joblib.dump(momentum_model_lr, "momentum_model_lr.joblib")

print("Velocity Model")
vel_model_adaboost, shouldershrug_metrics_adaboost = train_binary_model_adaboost(df, 'velocity', 'elbow_bent') 
joblib.dump(vel_model_adaboost, "vel_model_adaboost.joblib")

