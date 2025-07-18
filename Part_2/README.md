# Machine Learning Assessment

This repository implements a machine learning pipeline for classifying Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects based on clinical and genetic data, including SNP genotypes and Polygenic Risk Scores (PRS). The workflow includes data loading, preprocessing, model training, evaluation, and interpretability analysis via SHAP values.

---

## Project Structure

- `ML_script.py`: Main pipeline script for preprocessing, modeling, evaluation, and SHAP-based interpretation.
- `aux_functions.py`: Auxiliary module with functions for SNP encoding, imputation, feature scaling, hyperparameter tuning, and visualization.
- `data_sandbox.py`: Simple exploratory data analysis (EDA) script to support data profiling and class imbalance inspection.
- `images/` directory: Directory containing output figures.

---

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

# Approach and Rationale

The project followed a structured workflow to explore predictive modeling using a limited but heterogeneous dataset. The overall objective was to assess the feasibility of diagnosing cognitive status leveraging genetic variants, demographic features, and polygenic risk scores (PRS).

1. Exploratory Data Analysis (EDA): 
    
    - EDA was performed to inspect class distributions, feature 
    missingness, and data consistency.
    
    - Plots stored in `images/EDA/`.

2. Preprocessing:
    
    - Addressed missing data and encoded features.

    - `PRS` were imputed using a KNN-based strategy using the dataset SNPs to preserve inter-feature dependencies.

3. Modeling:

    - A baseline logistic regression model established initial performance.

    - A Random Forest classifier was then trained with class weighting and hyperparameter tuning.

4. Model Interpretation: 

    - SHAP values were computed for the Random Forest to support interpretability.

    - Feature importances and SHAP plots stored in `images/shap/` and `images/perm_imp/`.

The modeling approach began with a simple, well-established method to set baseline performance for comparison. Logistic regression was chosen initially for its robustness, interpretability, and fast training times.

As a more advanced alternative, a Random Forest classifier was selected due to its flexibility in capturing non-linear patterns in complex genetic and clinical data. Compared to logistic regression, Random Forests improved predictive accuracy and offered intuitive interpretability through feature importance metrics, making it a natural and effective next step.

## Preprocessing Strategy

- Missing Data:

    - Age imputed with the median.

    - Gender and SNPs imputed with the mode.

    - PRS imputed with KNN imputer based on neighbors including SNPs data to preserve correlations without discarding samples.

- Encoding:

    - Gender encoded as binary.

    - Target variable label-encoded (CN, MCI, AD).

    - SNPs encoded both as categorical and one-hot; categorical for tree-based models, one-hot for linear models.

- Feature Scaling:

    - Applied `StandardScaler` to numerical features (age, prs, feat1).

- Class Imbalance:

    - Managed via built-in `class_weight='balanced'` setting for both models.

# Models and Performance

- Logistic Regression:

    - Baseline with balanced class weights, multinomial setting.

    - Balanced accuracy of ~44%, with varying recall and precision across classes.

- Random Forest:

    - Tuned with class weights and hyperparameters to prevent overfitting.

    - Slight improvement (Balanced accuracy, 44.8%). Removing the ambiguous `feat1` feature slightly improved recall for Alzheimer's and MCI groups.
    
    - Balanced accuracy improved to 47.1%.

    
# Discussion and Next Steps

- Performance metrics:

    - Balanced accuracy, precision, and recall were the metrics prioritized due to class imbalance and eventual clinical relevance.

    - Overall performance remained modest (~47% balanced accuracy), with recall for Alzheimer’s improving after using Random Forests models and a simple feature selection.

    - These results also reflect the marked challenge of multiclass prediction with limited SNP information.

- Biological interpretation:

    - Permutation importance analysis identified `var8` as the most influential feature for model performance. This may suggest it represents a variant relevant to the clinical stage of the subjects in the dataset. `PRS` also showed relatively high importance, especially for AD and CN predictions. Additionally, `age`, `gender`, and `var4` stood out for their broader influence on some or all three classes.

    - Beeswarm plots (based on SHAP values) provided deeper insight into the role of each feature. The strong influence of `var8` on AD and CN classification is reflected in the clear formation of two distinct SHAP value clusters, each pushing predictions in opposite directions. For MCI, the pattern is less distinct, although a direct correlation with the age variable is visible.

    - It was also clearly observed that `gender = 0` (males) tends to shift predictions toward AD and MCI, while the opposite occurs for CN. This contradicts commonly reported epidemiological data, where AD incidence is generally higher in females. However, no strong conclusions must be taken on this from an overall small sample size with a slight imbalance towards male samples (~55%). 

    - Dependence plots offered further perspectives on feature behavior:

        - Among the five most important features for AD classification, var8 again stands out. Its _major/major_ genotype is associated with negative SHAP values, while the altered genotypes show positive SHAP values. The plot also highlights a correlation with `PRS`. In light of this, it may be speculated that `var8` is associated with or located near the APOE4 gene.

        - The `shap.dependence_plot` function detects, by default, which other feature most strongly interacts with the one being plotted. The generally low interaction values suggest the model did not fully capture strong non-linear feature interactions, which may be a focus point for future improvements.

- Key observations:

    - `PRS` imputation avoids sample loss and potential bias, this feature was one of the most important for classification performances.

    - `feat1` may introduce noise or irrelevant variability, its removal improved performance.

    - SHAP analysis highlights key features driving predictions.

- Eventual next steps:

    - Evaluate dimensionality reduction techniques.

    - Benchmark additional models (e.g., XGBoost, LightGBM).

    - Expand the feature set with external cohorts or richer phenotypic data.

