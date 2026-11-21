import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, fbeta_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.model_selection import train_test_split
import joblib

recall_weighted_scorer = make_scorer(fbeta_score, beta=2, zero_division = 0) #fbeta is used to find a harmonic mean between recall and precision. The weight depends on the beta value. 
#if beta = 1, both, precision and recall are given the same weightage. beta > 1 provides more weightage to recall (since missed positive cases are a greater concern). beta < 1 provides more weightage to precision.
df = pd.read_csv(r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Datasets\Shoulder_Abduction\Dataset for Exercises - shoulder_abduction.csv")
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'peak_frame']
drop = ['S1_arm_trunk_vis_frames', 'S1_arm_vertical_vis_frames', 'S1_shoulder_tilt_vis_frames', 'S1_elbow_vis_frames', 'S1_trunk_vis_frames', 'S1_sho_height_vis_frames', 'S1_velocity_vis_frames', 'S1_acceleration_vis_frames',
        'S2_arm_trunk_vis_frames', 'S2_arm_vertical_vis_frames', 'S2_shoulder_tilt_vis_frames', 'S2_elbow_vis_frames', 'S2_trunk_vis_frames', 'S2_sho_height_vis_frames', 'S2_velocity_vis_frames', 'S2_acceleration_vis_frames',
        'S3_arm_trunk_vis_frames', 'S3_arm_vertical_vis_frames', 'S3_shoulder_tilt_vis_frames', 'S3_elbow_vis_frames', 'S3_trunk_vis_frames', 'S3_sho_height_vis_frames', 'S3_velocity_vis_frames', 'S3_acceleration_vis_frames',
        'S4_arm_trunk_vis_frames', 'S4_arm_vertical_vis_frames', 'S4_shoulder_tilt_vis_frames', 'S4_elbow_vis_frames', 'S4_trunk_vis_frames', 'S4_sho_height_vis_frames', 'S4_velocity_vis_frames', 'S4_acceleration_vis_frames',
        'shoulder_shrug_s2', 'shoulder_shrug_s3', 'correct']

    
def train_binary_model_rf(df, target_label, other_label):
    drop_cols = drop + [other_label] #Drop certain columns that do not contribute as much to the model's prediction along with the other label (already present in the dataset)
    feature_cols = [c for c in df.columns if c not in [target_label] + drop_cols + metadata] #Feature set

    X = df[feature_cols].values #Define the features
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


        rf_model = RandomForestClassifier(random_state=42, class_weight="balanced") #Load the adaboost model into a variable
        rf_model.fit(X_train, y_train) #Train the model
        y_pred = rf_model.predict(X_test) #Make predictions
        print("Predicted: ", y_pred)
        print("Actual: ", y_test)
        probabilities = rf_model.predict_proba(X_test)
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
    print(f"\n{target_label} (Random Forest) - mean/std across folds:")
    print(metrics_df.drop(columns="fold").agg(["mean", "std"]))
    #Train on the entire dataset
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X)

    final_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    final_model.fit(X_final, y)

    # coefficients = pd.Series(final_model.coef_[0],index=feature_cols)
    # print(f"\nTop 10 features for {target_label} (Logistic Regression):")
    # print(coefficients.abs().sort_values(ascending=False).head(10).index) #Interpretation: If it is a positive coefficient - higher value indicates a higher probability of label = 1
    # #Negative coefficient - higher value is associated with a lower probability of label = 1

    return final_model, metrics_df 


print("Elbow Bent Model")
elbowbent_model_rf, elbowbent_metrics_rf = train_binary_model_rf(df, 'elbow_bent', 'velocity')
joblib.dump(elbowbent_model_rf, "elbowbent_model_rf.joblib")

# print("Momentum Model")
# momentum_model_lr, momentum_metrics_lr = train_binary_model_lr(df, 'Momentum', 'Elbow Bent', 'Shoulder Shrug', 'all') 
# joblib.dump(momentum_model_lr, "momentum_model_lr.joblib")

print("Velocity Model")
vel_model_rf, vel_metrics_rf = train_binary_model_rf(df, 'velocity', 'elbow_bent') 
joblib.dump(vel_model_rf, "vel_model_rf.joblib")

