import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.inspection import permutation_importance

import shap

def snp_encoding(df, snp_cols, reference = "major"):
    # Reference option would enable the possibility to provide a list or dictionary of ref alleles.
    # This approach is not implemented as the SNPs are anonymized.
    # For the current project, the major (most frequent) allele will be considered the reference.
    for col in snp_cols:
        col_encoded = genotype_mapping(df[col]) 
        df[col] = col_encoded
    return df

def genotype_mapping(col):
    # Mapping genotypes to 0/1/2
    major, minor = find_major(col)

    # Define genotype map
    genotype_map = {
        f'{major}/{major}': 0,
        f'{major}/{minor}': 1,
        f'{minor}/{major}': 1,  # preventing instances where the major allele is the "alternative" 
        f'{minor}/{minor}': 2
    }

    return col.map(genotype_map)
    
def find_major(col):
    # Flatten genotype column
    # "C/T" results in ['C', 'T'], "C/C" in ['C', 'C']
    # .sum() extend the growing list of alleles in the column
    alleles = col.apply(lambda x: x.split('/')).sum()

    # Count alleles to find major/minor
    allele_counts = Counter(alleles)
    if len(allele_counts) > 2:
        print(f"Warning: SNP {col} has more than 2 alleles: {allele_counts}")
        # Tested for detection of non-biallelic SNPs. For the present project, all SNPs are biallelic.
        # Next line still selects for the 2 most common alleles, additional mapping entries should be designed in the case of multi-allelic SNPs.
    major, minor = allele_counts.most_common(2) # returns [("allele", count), ("allele", count)]
    major_allele = major[0]
    minor_allele = minor[0]

    return major_allele, minor_allele

def data_preparation(df, test_prop = 0.2):
    # X, y Split
    X = df.drop(columns=['diagnosis', 'diagnosis_encoded'])
    y = df['diagnosis_encoded']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_prop, random_state=672, stratify=y
    )
    return X_train, X_test, y_train, y_test

def rf_rand_search(rf_model, X_train, y_train, seed, top_n = 10):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search_params = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False]
    }

    search = RandomizedSearchCV(
        rf_model,
        param_distributions=search_params,
        n_iter=50,
        scoring='balanced_accuracy',
        cv=cv,
        random_state=seed,
        verbose=2,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    # Create DataFrame from cv_results_
    results_df = pd.DataFrame(search.cv_results_)

    # Sort by mean_test_score descending and select top_n
    top_results = results_df.sort_values(by='mean_test_score', ascending=False).head(top_n)

    # Print the top n results nicely
    for i, row in top_results.iterrows():
        print(f"Rank {i+1}:")
        print(f"Params: {row['params']}")
        print(f"Mean Balanced Accuracy: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})\n")
    # Best performances were achieved with:
    #   - n_estimators in [500, 1000]
    #   - min_samples_split in [2, 5]
    #   - min_samples_leaf = 2
    #   - max_features in [log2, sqrt]
    #   - max_depth = 5
    #   - bootstrap = False
    ### With this, a more well-defined GridSearch was designed

    return search

def rf_grid_search(rf_model, X_train, y_train, seed, scoring='balanced_accuracy'):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    param_grid = {
        'n_estimators': [500, 750, 1000],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [5],
        'bootstrap': [False]
    }

    search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    # Print the best results
    print("\nBest parameters:")
    print(search.best_params_)
    print(f"Best {scoring} (CV): {search.best_score_:.4f}")

    return search

def permut_imp(model, X_test, y_test, scoring_metrics = ['balanced_accuracy', 'roc_auc_ovr_weighted', 'f1_weighted', 'recall_weighted'], seed = 777, model_name = 'rf'):
    # Permutation importance
    print("\nCalculating permutation importance...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=seed, scoring=scoring_metrics)
    
    for s in scoring_metrics:
        # Sort features by importance
        sorted_idx = perm_importance[s].importances_mean.argsort()[::-1]
        print("Features by permutation importance:")
        for i in sorted_idx:
            print(f"{X_test.columns[i]}: Mean = {perm_importance[s].importances_mean[i]:.4f}; std = {perm_importance[s].importances_std[i]:.4f}")
        
        # Plot importance
        plt.figure(figsize=(10,6))
        plt.barh(X_test.columns[sorted_idx], perm_importance[s].importances_mean[sorted_idx])
        plt.xlabel(f"Mean decrease in {s}")
        plt.title("Permutation Feature Importance")
        plt.gca().invert_yaxis()
        plt.savefig(f'images/perm_imp/{model_name}_perm_imp_{s}.png', dpi=300)
        plt.close()

def shap_explain(model, X_test):
    # SHAP explainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test).values
    
    return shap_values

def shap_plots(shap_values, X_test, classes, plot_list):
    for class_idx, class_name in enumerate(classes):
        if 'bar' in plot_list or 'all' in plot_list:
            shap.summary_plot(shap_values[:, :, class_idx], X_test, show=False, plot_type="bar", class_names=[class_name])
            plt.savefig(f"images/shap/shap_bar_summary_{class_name}.png", dpi=300)
            plt.close()

        if 'bee' in plot_list or 'all' in plot_list:
            shap.summary_plot(shap_values[:, :, class_idx], X_test, show=False, class_names=[class_name])
            plt.savefig(f"images/shap/shap_beeswarm_summary_{class_name}.png", dpi=300)
            plt.close()

    if 'dependence' in plot_list or 'all' in plot_list:
        # If no features specified, default to first feature
        features_to_plot = ['var4', 'var8', 'prs', 'age', 'gender']
        for feature in features_to_plot:
            shap.dependence_plot(
                feature, 
                shap_values[:, :, 0], 
                X_test, 
                show=False
            )
            plt.savefig(f"images/shap/shap_AD_feat_dependence_{feature}.png", dpi=300)
            plt.close()