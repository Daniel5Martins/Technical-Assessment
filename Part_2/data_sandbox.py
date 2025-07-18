import matplotlib.pyplot as plt
import seaborn as sns


from ML_script import load_data

# Loading data
df = load_data()
#print(df.head())
#print(df.info())
# *** Few NAs, prs is the most problematic ***
#   age:                9 NA
#   gender:             1 NA
#   prs:                40 NA
#   vars (10 cols):     16/30/17/25/21/1/25/0/23/22 NA
#   is_smoker, feat1 & diagnosis:  0 NA

# --- Exploratory Data Analysis ---
if True:
    counts = df['diagnosis'].value_counts()
    print("Class Counts:")
    print(counts)
    # MCI: 277; AD: 220; CN: 175

    # Plot counts
    sns.countplot(x='diagnosis', data=df)
    plt.title("Class Distribution")
    plt.savefig('images/EDA/class_dist.png', dpi = 300)
    plt.close()
# *** Class imbalance must be addressed ***

# --- Decide on dropping lines with NA at the prs column ---
if True:
    df['PRS_missing'] = df['prs'].isna()

    # Show normalized phenotype distribution for PRS-missing vs not-missing
    distribution = df.groupby('PRS_missing')['diagnosis'].value_counts(normalize=True).unstack()
    print(distribution)
    # Greater proportion of NAs in MCI group (52.5% of the NAs vs. 40.5% of the data as a whole)
    # Lower proportion of NAs in AD group (20% of the NAs vs. 33.5% of the data as a whole)
    # To avoid biasing the dataset and losing possibly valuable samples, missing PRS will be imputed rather than dropped.

    # Histogram of PRS (excluding NaNs). kde presents a kernel distribution estimate for an approximate distribution shape.
    sns.histplot(df['prs'].dropna(), bins=30, kde=True)
    plt.title("Distribution of Polygenic Risk Scores (PRS)")
    plt.xlabel("PRS")
    plt.ylabel("Frequency")
    plt.savefig('images/EDA/PRS_distribution.png', dpi=300)
    plt.close()
    # Close to a normal distribution
# *** PRS will be imputed with a KNN-based approach in order to avoid using the phenotype (final goal is prediction) *** 

# --- Explore the possible meaning of "feat1" column ---
if False:
    # Simple correlation with PRS
    feat1_prs_corr = df[['feat1', 'prs']].corr()
    print(feat1_prs_corr)
    # Low correlation: 0.040833

if True:
    # Boxplot of feat1 by diagnosis
    sns.boxplot(x='diagnosis', y='feat1', data=df)
    plt.title("Distribution of feat1 by Diagnosis")
    plt.savefig('images/EDA/feat1_boxplot.png', dpi=300)
    plt.close()
    # No clear distinction across diagnosis classes

# *** Possibly an auxiliary anonymized feature (or noise), may be useful to add diversity. Compare predictive power with SHAP and perm_scores ***

