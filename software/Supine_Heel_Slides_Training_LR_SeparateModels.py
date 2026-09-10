import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\docs\Dataset for Exercises - HeightNormalization.csv')
metadata = ['patient_id', 'rep_id', 'start_frame', 'end_frame', 'valley_frame']
drop = ['S1_knee_mean', 'S1_knee_std', 'S1_hip_mean', 'S1_hip_std', 'S1_ankle_mean', 'S1_ankle_std', 'S1_pelvic_mean', 'S1_pelvic_std', 'S1_heel_mean', 'S1_heel_std',
        'S2_knee_mean', 'S2_knee_std', 'S2_hip_mean', 'S2_hip_std', 'S2_ankle_mean', 'S2_ankle_std', 'S2_pelvic_mean', 'S2_pelvic_std', 'S2_heel_mean', 'S2_heel_std',
        'S3_knee_mean', 'S3_knee_std', 'S3_hip_mean', 'S3_hip_std', 'S3_ankle_mean', 'S3_ankle_std', 'S3_pelvic_mean', 'S3_pelvic_std', 'S3_heel_mean', 'S3_heel_std',
        'S4_knee_mean', 'S4_knee_std', 'S4_hip_mean', 'S4_hip_std', 'S4_ankle_mean', 'S4_ankle_std', 'S4_pelvic_mean', 'S4_pelvic_std', 'S4_heel_mean', 'S4_heel_std', 
        'pelvic_lift_max', 'hip_compensation', 'S1_pelvic_min', 'S1_pelvic_max', 'S1_pelvic_range', 'S1_pelvic_vis_frames', 'S2_pelvic_vis_frames', 
        'S3_pelvic_vis_frames', 'S4_pelvic_vis_frames', 'S2_pelvic_min', 'S2_pelvic_max', 'S2_pelvic_range', 'S3_pelvic_min', 'S3_pelvic_max', 'S3_pelvic_range', 'S4_pelvic_min', 
        'S4_pelvic_max', 'S4_pelvic_range', 'S1_ankle_vis_frames', 'S1_ankle_mean', 'S1_ankle_min', 'S1_ankle_max', 'S1_ankle_range', 'S1_ankle_std', 'S2_ankle_vis_frames', 
        'S2_ankle_mean', 'S2_ankle_min', 'S2_ankle_max', 'S2_ankle_range', 'S2_ankle_std', 'S3_ankle_vis_frames', 'S3_ankle_mean', 'S3_ankle_min', 'S3_ankle_max', 'S3_ankle_range', 
        'S3_ankle_std', 'S4_ankle_vis_frames', 'S4_ankle_mean', 'S4_ankle_min', 'S4_ankle_max', 'S4_ankle_range', 'S4_ankle_std', 'S1_hip_vis_frames', 'S1_hip_mean', 'S1_hip_min', 
        'S1_hip_max', 'S1_hip_range', 'S1_hip_std', 'S2_hip_vis_frames', 'S2_hip_mean', 'S2_hip_min', 'S2_hip_max', 'S2_hip_range', 'S2_hip_std', 'S3_hip_vis_frames', 'S3_hip_mean', 
        'S3_hip_min', 'S3_hip_max', 'S3_hip_range', 'S3_hip_std', 'S4_hip_vis_frames', 'S4_hip_mean', 'S4_hip_min', 'S4_hip_max', 'S4_hip_range', 'S4_hip_std', 'S1_knee_range', 
        'S2_knee_range', 'S3_knee_range', 'S4_knee_range']

    
def train_binary_model_ada(df, target_label, other_label):
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


        lr_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000) #Load the adaboost model into a variable
        lr_model.fit(X_train, y_train) #Train the model
        y_pred = lr_model.predict(X_test) #Make predictions
        #Print the metrics for each fold
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Fold {fold}: F1={f1:.3f}, Recall={recall:.3f}, "
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

    coefficients = pd.Series(final_model.coef_[0],index=feature_cols)
    print(f"\nTop 10 features for {target_label} (Logistic Regression):")
    print(coefficients.abs().sort_values(ascending=False).head(10).index) #Interpretation: If it is a positive coefficient - higher value indicates a higher probability of label = 1
    #Negative coefficient - higher value is associated with a lower probability of label = 1

    return final_model, metrics_df


print("Heel Lift Model")
heel_model_ada, heel_metrics_ada = train_binary_model_ada(df, 'label_heel_lift', 'label_speed_error')

print("Velocity Error Model")
velocity_model_ada, velocity_metrics_ada = train_binary_model_ada(df, 'label_speed_error', 'label_heel_lift',) 
    





