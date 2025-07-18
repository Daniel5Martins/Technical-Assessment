# --- Imports ---
import os
import pandas as pd
import numpy as np
import shap

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

import aux_functions as aux

# --- Load Dataset ---
def load_data():
    # If running on jupyter, changing directory is not needed.
    running_on_jupyter = False
    if running_on_jupyter:
        pass
    else:
        # This script assumes that the data is found at "./ml_dataset/ml_dataset_test.csv" relative to the script location.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)

    df = pd.read_csv("./ml_dataset/ml_dataset_test.csv")
    return df

def preprocessing_data(df, remove_ids = True, prs_imputation = "KNN"):
    if remove_ids:
        # Dropping subject IDs, not necessary for this project
        df = df.drop(columns=["subject_id"])

    ### At this point df copies for eventual comparisons may be helpful in other contexts, but they are not needed for this project
    #Handling Missing Values - Imputation
    df['age'].fillna(df['age'].median(), inplace=True) # Continuous value and minor missingnes (1.3%), median is acceptable
    df['gender'].fillna(df['gender'].mode()[0], inplace=True) # Only one missing value, minor impact

    snp_columns = [col for col in df.columns if str(col)[:3] == "var"]
    for col in snp_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encoding variables
    df['gender'] = df['gender'].map({'male': 0, 'female': 1, 'Male': 0, 'Female': 1})
    df['is_smoker'] = df['is_smoker'].astype(int)

    # Encode target variable
    label_encoder = LabelEncoder()
    df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])

    df = aux.snp_encoding(df, snp_columns) # To find more details on SNP encoding, refer to aux_functions.py
    
    if prs_imputation == "KNN":
        imputer = KNNImputer(n_neighbors=5)
        df[snp_columns + ['prs']] = imputer.fit_transform(df[snp_columns + ['prs']])

    # Final SNP encodings
        # Keeping genotypes as categories may be helpful in some cases and one-hot encoding numerical values in others.
        # Using both strategies for downstream flexibility.
    ## One-hot Encoding
    df_onehot = df.copy()
    df_onehot = pd.get_dummies(df_onehot, columns=snp_columns, drop_first=True)
    # *Note* drop_first to avoid "dummy variable trap", one of the columns can be entirely predicted by the values in the other

    ## Casting SNP columns to categorical values
    df[snp_columns] = df[snp_columns].round().astype(int).astype('category')

    return (df, df_onehot)

def LogReg(df, seed = 2718, print_res = False):
    ### Simple logistic Regression for baseline results
    X_train, X_test, y_train, y_test = aux.data_preparation(df)

    # Scaling columns with numerical values
    numeric_cols = ['age', 'prs', 'feat1']

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Fit Logistic Regression
    logreg = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs', random_state=seed, max_iter=1000)
    logreg.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = logreg.predict(X_test)
    if print_res:
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("Accuracy:", balanced_accuracy_score(y_test, y_pred))

def RandomForests(df, seed = 42, print_res = False):
    X_train, X_test, y_train, y_test = aux.data_preparation(df)

    #scaler = StandardScaler()
    ### ... Tested for concordance with LogReg. Scaling didn't affect results as expected for Random Forests

    # Initialize RF with balanced class weights to address imbalance
    rf = RandomForestClassifier(random_state=seed,
                                class_weight='balanced',
                                n_estimators=1000,
                                min_samples_split=5,
                                min_samples_leaf=2,
                                max_features='log2',
                                max_depth=5,
                                bootstrap=True
                                )
    
    ### Randomized search for initial hyperparameter approximation, commented for redundancy swith grid search
    #aux.rf_rand_search(rf, X_train, y_train, seed)
    ### Grid search for final tuning, "best parameters" now used in original model call
    ###   Exception to bootstrap parameter (T instead of F)
    #grid_search = aux.rf_grid_search(rf, X_train, y_train, seed)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    if print_res:
        print("Classification report:\n", classification_report(y_test, y_pred, target_names=['AD', 'CN', 'MCI']))
        print("Accuracy:", balanced_accuracy_score(y_test, y_pred))

    return rf, X_train, X_test, y_train, y_test

### Baseline
    # ~44% balanced accuracy.
    # 55/32/46 % recall for AD/MCI/CN respectively.
    # 47/50/33 % precision for AD/MCI/CN respectively.

### First RandomForests model
# Balanced accuracy = 44.8%

# With permutation importances:
### 'feat1' consistently affected model performance
### Due to its unknown origin and meaning, new tests were performed after removing its column

# Balanced accuracy = 47.1%
# 68/30/43 % recall for AD/MCI/CN respectively.
# 54/55/31 % precision for AD/MCI/CN respectively.

def main():
    # Load and preprocess data
    df = load_data()
    df, df_onehot = preprocessing_data(df)

    # LogReg(df_onehot, print_res=True)

    # Run Random Forest on full dataframe (with 'feat1')
    rf_initial, X_train, X_test, y_train, y_test = RandomForests(df, print_res=True)
    aux.permut_imp(rf_initial, X_test, y_test)
    # Run Random Forest after removing 'feat1' feature
    df_no_feat1 = df.drop(columns=['feat1'])
    rf_no_feat1, X_train, X_test, y_train, y_test = RandomForests(df_no_feat1, print_res=True)
    aux.permut_imp(rf_no_feat1, X_test, y_test, model_name = "rf_no_feat1")
    # Generate SHAP values for interpretation
    shap_values = aux.shap_explain(rf_no_feat1, X_test)

    # Optionally plot SHAP summary (uncomment if needed)
    aux.shap_plots(shap_values, X_test, classes=['AD', 'CN', 'MCI'], plot_list=['all'])

if __name__ == "__main__":
    main()

