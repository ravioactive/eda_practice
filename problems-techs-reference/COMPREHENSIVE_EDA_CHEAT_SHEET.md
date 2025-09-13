# 📊 Comprehensive EDA Cheat Sheet

**Complete Guide to Exploratory Data Analysis Techniques**  
**Date:** January 22, 2025  
**Scope:** Univariate, Bivariate, and Multivariate Analysis  
**Coverage:** Numerical and Categorical Features  

---

## 🎯 **Quick Navigation**

- [📈 Univariate Analysis](#-univariate-analysis)
- [🔗 Bivariate Analysis](#-bivariate-analysis) 
- [🌐 Multivariate Analysis](#-multivariate-analysis)
- [📊 Correlation Techniques](#-correlation-techniques)
- [🧪 Statistical Tests](#-statistical-tests)
- [📦 Distribution Analysis](#-distribution-analysis)
- [🔍 Outlier Detection](#-outlier-detection)
- [📐 Dimensionality Reduction](#-dimensionality-reduction)
- [🎨 Visualization Guide](#-visualization-guide)
- [💻 Code Templates](#-code-templates)

---

## 📈 **Univariate Analysis**

### **🔢 Numerical Features**

#### **Descriptive Statistics**
```python
# Basic statistics
df['feature'].describe()
df['feature'].agg(['count', 'mean', 'median', 'std', 'var', 'min', 'max'])

# Advanced statistics
from scipy import stats
stats.skew(df['feature'])           # Skewness
stats.kurtosis(df['feature'])       # Kurtosis
df['feature'].quantile([0.25, 0.5, 0.75, 0.95, 0.99])  # Percentiles
```

#### **Distribution Shape Analysis**
```python
# Normality tests
stats.shapiro(df['feature'])        # Shapiro-Wilk test (n < 5000)
stats.normaltest(df['feature'])     # D'Agostino-Pearson test
stats.jarque_bera(df['feature'])    # Jarque-Bera test
stats.anderson(df['feature'])       # Anderson-Darling test

# Distribution fitting
from scipy.stats import norm, lognorm, gamma, beta
stats.probplot(df['feature'], dist="norm", plot=plt)
```

#### **Outlier Detection Methods**
```python
# IQR Method (used in our analysis)
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['feature'] < Q1 - 1.5*IQR) | (df['feature'] > Q3 + 1.5*IQR)]

# Z-Score Method
z_scores = np.abs(stats.zscore(df['feature']))
outliers = df[z_scores > 3]

# Modified Z-Score (Robust)
median = df['feature'].median()
mad = np.median(np.abs(df['feature'] - median))
modified_z_scores = 0.6745 * (df['feature'] - median) / mad
outliers = df[np.abs(modified_z_scores) > 3.5]

# Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(df[['feature']])

# Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(df[['feature']])
```

#### **Visualizations**
```python
# Distribution plots
plt.hist(df['feature'], bins=50, alpha=0.7)
sns.histplot(df['feature'], kde=True)
sns.distplot(df['feature'])  # Deprecated, use histplot

# Box plots (used in our analysis)
plt.boxplot(df['feature'])
sns.boxplot(y=df['feature'])

# Violin plots (combines box plot + density)
sns.violinplot(y=df['feature'])

# Q-Q plots
stats.probplot(df['feature'], dist="norm", plot=plt)

# Density plots
df['feature'].plot.density()
sns.kdeplot(df['feature'])
```

### **🏷️ Categorical Features**

#### **Frequency Analysis**
```python
# Value counts
df['category'].value_counts()
df['category'].value_counts(normalize=True)  # Proportions
df['category'].nunique()                     # Unique count

# Cross-tabulation
pd.crosstab(df['cat1'], df['cat2'])
pd.crosstab(df['cat1'], df['cat2'], normalize='index')  # Row percentages
```

#### **Statistical Tests**
```python
# Chi-square goodness of fit
observed = df['category'].value_counts()
expected = [len(df)/len(observed)] * len(observed)  # Uniform distribution
chi2, p_value = stats.chisquare(observed, expected)

# Entropy (measure of randomness)
from scipy.stats import entropy
entropy(df['category'].value_counts())
```

#### **Visualizations**
```python
# Bar plots
df['category'].value_counts().plot.bar()
sns.countplot(x='category', data=df)

# Pie charts
df['category'].value_counts().plot.pie()

# Donut charts
fig, ax = plt.subplots()
wedges, texts = ax.pie(df['category'].value_counts(), labels=df['category'].unique())
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
```

---

## 🔗 **Bivariate Analysis**

### **🔢 Numerical vs Numerical**

#### **Correlation Methods**
```python
# Pearson correlation (linear relationships)
df[['num1', 'num2']].corr(method='pearson')
stats.pearsonr(df['num1'], df['num2'])

# Spearman correlation (monotonic relationships)
df[['num1', 'num2']].corr(method='spearman')
stats.spearmanr(df['num1'], df['num2'])

# Kendall's Tau (robust to outliers)
df[['num1', 'num2']].corr(method='kendall')
stats.kendalltau(df['num1'], df['num2'])

# Distance correlation (non-linear relationships)
from dcor import distance_correlation
distance_correlation(df['num1'], df['num2'])

# Maximal Information Coefficient (MIC)
from minepy import MINE
mine = MINE()
mine.compute_score(df['num1'], df['num2'])
mine.mic()
```

#### **Advanced Correlation Analysis**
```python
# Partial correlation (controlling for other variables)
from pingouin import partial_corr
partial_corr(data=df, x='num1', y='num2', covar=['num3', 'num4'])

# Rolling correlation (time series)
df['num1'].rolling(window=30).corr(df['num2'])

# Correlation significance testing
from scipy.stats import pearsonr
r, p_value = pearsonr(df['num1'], df['num2'])
```

#### **Regression Analysis**
```python
# Linear regression
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(df['num1'], df['num2'])

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(df[['num1']])

# LOWESS smoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(df['num2'], df['num1'], frac=0.3)
```

#### **Visualizations**
```python
# Scatter plots
plt.scatter(df['num1'], df['num2'])
sns.scatterplot(x='num1', y='num2', data=df)

# Regression plots
sns.regplot(x='num1', y='num2', data=df)
sns.lmplot(x='num1', y='num2', data=df)

# Hexbin plots (for large datasets)
plt.hexbin(df['num1'], df['num2'], gridsize=30)

# 2D density plots
sns.kdeplot(x='num1', y='num2', data=df)

# Joint plots
sns.jointplot(x='num1', y='num2', data=df, kind='scatter')
sns.jointplot(x='num1', y='num2', data=df, kind='hex')
sns.jointplot(x='num1', y='num2', data=df, kind='kde')
```

### **🏷️ Categorical vs Numerical**

#### **Statistical Tests**
```python
# T-test (2 groups)
from scipy.stats import ttest_ind, mannwhitneyu
group1 = df[df['category'] == 'A']['numerical']
group2 = df[df['category'] == 'B']['numerical']
t_stat, p_value = ttest_ind(group1, group2)

# Mann-Whitney U test (non-parametric, used in our analysis)
u_stat, p_value = mannwhitneyu(group1, group2)

# ANOVA (multiple groups, used in our analysis)
from scipy.stats import f_oneway
groups = [group for name, group in df.groupby('category')['numerical']]
f_stat, p_value = f_oneway(*groups)

# Kruskal-Wallis test (non-parametric ANOVA)
from scipy.stats import kruskal
h_stat, p_value = kruskal(*groups)

# Effect size (Cohen's d)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.var() + (n2-1)*group2.var()) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std
```

#### **Visualizations**
```python
# Box plots by category (used in our analysis)
sns.boxplot(x='category', y='numerical', data=df)

# Violin plots
sns.violinplot(x='category', y='numerical', data=df)

# Strip plots
sns.stripplot(x='category', y='numerical', data=df)

# Swarm plots
sns.swarmplot(x='category', y='numerical', data=df)

# Ridgeline plots
import joypy
joypy.joyplot(df, by='category', column='numerical')
```

### **🏷️ Categorical vs Categorical**

#### **Association Measures**
```python
# Chi-square test (used in our analysis)
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(df['cat1'], df['cat2'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Cramér's V (used in our analysis)
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Phi coefficient (2x2 tables)
def phi_coefficient(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / n)

# Uncertainty coefficient
from sklearn.metrics import mutual_info_score
mutual_info_score(df['cat1'], df['cat2'])

# Lambda (Goodman and Kruskal's)
def lambda_coefficient(x, y):
    # Implementation for asymmetric lambda
    pass
```

#### **Visualizations**
```python
# Heatmaps
contingency_table = pd.crosstab(df['cat1'], df['cat2'])
sns.heatmap(contingency_table, annot=True, fmt='d')

# Stacked bar charts
contingency_table.plot.bar(stacked=True)

# Mosaic plots
from statsmodels.graphics.mosaicplot import mosaic
mosaic(df, ['cat1', 'cat2'])

# Balloon plots
import plotly.express as px
fig = px.scatter(contingency_table.reset_index().melt(id_vars='cat1'), 
                 x='cat1', y='variable', size='value')
```

---

## 🌐 **Multivariate Analysis**

### **📐 Dimensionality Reduction**

#### **Principal Component Analysis (PCA) - Used in Our Analysis**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical_features])

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Analysis
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
feature_loadings = pd.DataFrame(pca.components_.T, 
                               columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                               index=numerical_features)
```

#### **Other Dimensionality Reduction Techniques**
```python
# t-SNE (non-linear)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP (faster than t-SNE)
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Factor Analysis
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=5)
X_fa = fa.fit_transform(X_scaled)

# Independent Component Analysis (ICA)
from sklearn.decomposition import FastICA
ica = FastICA(n_components=5)
X_ica = ica.fit_transform(X_scaled)

# Linear Discriminant Analysis (LDA) - supervised
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
```

### **🎯 Clustering Analysis**

#### **K-Means Clustering (Used in Our Analysis)**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method for optimal k
inertias = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Silhouette analysis
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
```

#### **Other Clustering Methods**
```python
# Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create dendrogram
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)

# Apply clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
cluster_labels = hierarchical.fit_predict(X_scaled)

# DBSCAN (density-based)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X_scaled)

# Gaussian Mixture Models
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
cluster_labels = gmm.fit_predict(X_scaled)
```

### **🔍 Feature Selection**

#### **Statistical Methods**
```python
# Mutual Information (used in our analysis)
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
mi_scores = mutual_info_regression(X, y)  # For regression
mi_scores = mutual_info_classif(X, y)     # For classification

# Chi-square test
from sklearn.feature_selection import chi2, SelectKBest
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)

# ANOVA F-test
from sklearn.feature_selection import f_regression, f_classif
f_scores, p_values = f_regression(X, y)   # For regression
f_scores, p_values = f_classif(X, y)      # For classification

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=10)
X_selected = selector.fit_transform(X, y)
```

#### **Model-Based Methods**
```python
# Random Forest Feature Importance (used in our analysis)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Permutation Importance
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

# SHAP values
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
```

---

## 📊 **Correlation Techniques**

### **📈 Correlation Matrix Analysis**

#### **Standard Correlations**
```python
# Full correlation matrix (used in our analysis)
corr_matrix = df[numerical_features].corr()

# Correlation with target variable
target_correlations = df[numerical_features].corrwith(df['target'])

# Partial correlation matrix
from pingouin import pcorr
partial_corr_matrix = df[numerical_features].pcorr()
```

#### **Advanced Correlation Methods**
```python
# Robust correlations
from scipy.stats import spearmanr
def robust_correlation_matrix(df):
    n = len(df.columns)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = spearmanr(df.iloc[:, i], df.iloc[:, j])[0]
    return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)

# Time-lagged correlations
def lagged_correlation(x, y, max_lag=10):
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        correlations.append(corr)
    return correlations
```

#### **Correlation Visualization**
```python
# Heatmap (used in our analysis)
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0)

# Clustermap
sns.clustermap(corr_matrix, annot=True, cmap='RdBu_r', center=0)

# Network graph
import networkx as nx
G = nx.Graph()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.5:  # Threshold
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], 
                      weight=abs(corr_matrix.iloc[i, j]))
nx.draw(G, with_labels=True)
```

---

## 🧪 **Statistical Tests**

### **📊 Distribution Tests**

#### **Normality Tests**
```python
# Shapiro-Wilk test (best for n < 5000)
stat, p_value = stats.shapiro(data)

# D'Agostino-Pearson test (used in our analysis)
stat, p_value = stats.normaltest(data)

# Jarque-Bera test
stat, p_value = stats.jarque_bera(data)

# Anderson-Darling test
result = stats.anderson(data, dist='norm')

# Kolmogorov-Smirnov test
stat, p_value = stats.kstest(data, 'norm')
```

#### **Homogeneity Tests**
```python
# Levene's test (equal variances)
stat, p_value = stats.levene(group1, group2, group3)

# Bartlett's test (assumes normality)
stat, p_value = stats.bartlett(group1, group2, group3)

# Fligner-Killeen test (robust)
stat, p_value = stats.fligner(group1, group2, group3)
```

### **🔍 Comparison Tests**

#### **Two-Sample Tests**
```python
# Independent t-test
stat, p_value = stats.ttest_ind(group1, group2)

# Welch's t-test (unequal variances)
stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

# Mann-Whitney U test (used in our analysis)
stat, p_value = stats.mannwhitneyu(group1, group2)

# Kolmogorov-Smirnov test
stat, p_value = stats.ks_2samp(group1, group2)

# Permutation test
from scipy.stats import permutation_test
def statistic(x, y):
    return np.mean(x) - np.mean(y)
res = permutation_test((group1, group2), statistic, n_resamples=10000)
```

#### **Multiple Group Tests**
```python
# One-way ANOVA (used in our analysis)
stat, p_value = stats.f_oneway(group1, group2, group3)

# Kruskal-Wallis test
stat, p_value = stats.kruskal(group1, group2, group3)

# Post-hoc tests
from scipy.stats import tukey_hsd
result = tukey_hsd(group1, group2, group3)

# Dunn's test (post-hoc for Kruskal-Wallis)
from scikit_posthocs import posthoc_dunn
dunn_result = posthoc_dunn([group1, group2, group3])
```

### **🏷️ Categorical Tests**

#### **Independence Tests**
```python
# Chi-square test (used in our analysis)
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Fisher's exact test (2x2 tables)
odds_ratio, p_value = stats.fisher_exact(contingency_table)

# G-test (log-likelihood ratio)
from scipy.stats import power_divergence
stat, p_value = power_divergence(observed, expected, lambda_="log-likelihood")
```

#### **Goodness of Fit Tests**
```python
# Chi-square goodness of fit
observed = df['category'].value_counts()
expected = [len(df)/len(observed)] * len(observed)
chi2, p_value = stats.chisquare(observed, expected)

# Binomial test
from scipy.stats import binom_test
p_value = binom_test(successes, trials, p=0.5)
```

---

## 📦 **Distribution Analysis**

### **📈 Distribution Fitting**

#### **Parametric Distributions**
```python
# Fit common distributions
from scipy import stats

# Normal distribution
mu, sigma = stats.norm.fit(data)
fitted_normal = stats.norm(mu, sigma)

# Log-normal distribution
s, loc, scale = stats.lognorm.fit(data, floc=0)
fitted_lognorm = stats.lognorm(s, loc, scale)

# Exponential distribution
loc, scale = stats.expon.fit(data, floc=0)
fitted_expon = stats.expon(loc, scale)

# Gamma distribution
a, loc, scale = stats.gamma.fit(data, floc=0)
fitted_gamma = stats.gamma(a, loc, scale)

# Beta distribution
a, b, loc, scale = stats.beta.fit(data)
fitted_beta = stats.beta(a, b, loc, scale)
```

#### **Distribution Comparison**
```python
# AIC/BIC comparison
def compare_distributions(data, distributions):
    results = {}
    for name, dist in distributions.items():
        params = dist.fit(data)
        log_likelihood = np.sum(dist.logpdf(data, *params))
        k = len(params)
        n = len(data)
        aic = 2*k - 2*log_likelihood
        bic = k*np.log(n) - 2*log_likelihood
        results[name] = {'AIC': aic, 'BIC': bic, 'params': params}
    return results

distributions = {
    'normal': stats.norm,
    'lognormal': stats.lognorm,
    'exponential': stats.expon,
    'gamma': stats.gamma
}
comparison = compare_distributions(data, distributions)
```

#### **Non-Parametric Density Estimation**
```python
# Kernel Density Estimation
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(data.reshape(-1, 1))
density = np.exp(kde.score_samples(x_range.reshape(-1, 1)))

# Histogram-based estimation
counts, bins = np.histogram(data, bins=50, density=True)
```

---

## 🔍 **Outlier Detection**

### **📊 Univariate Methods**

#### **Statistical Methods**
```python
# Z-score method
z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]

# Modified Z-score (robust)
median = np.median(data)
mad = np.median(np.abs(data - median))
modified_z_scores = 0.6745 * (data - median) / mad
outliers = data[np.abs(modified_z_scores) > 3.5]

# IQR method (used in our analysis)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]

# Grubbs' test
from outliers import smirnov_grubbs as grubbs
outliers = grubbs.test(data, alpha=0.05)
```

### **🌐 Multivariate Methods**

#### **Distance-Based Methods**
```python
# Mahalanobis distance
from scipy.spatial.distance import mahalanobis
mean = np.mean(data, axis=0)
cov = np.cov(data.T)
distances = [mahalanobis(x, mean, np.linalg.inv(cov)) for x in data]
outliers = data[np.array(distances) > threshold]

# Cook's distance (for regression)
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(fitted_model)
cooks_d = influence.cooks_distance[0]
outliers = data[cooks_d > 4/len(data)]
```

#### **Machine Learning Methods**
```python
# Isolation Forest (used in our analysis)
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(data)

# One-Class SVM
from sklearn.svm import OneClassSVM
oc_svm = OneClassSVM(nu=0.1)
outliers = oc_svm.fit_predict(data)

# Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers = lof.fit_predict(data)

# Elliptic Envelope
from sklearn.covariance import EllipticEnvelope
ee = EllipticEnvelope(contamination=0.1)
outliers = ee.fit_predict(data)
```

---

## 🎨 **Visualization Guide**

### **📈 Distribution Plots**

#### **Single Variable**
```python
# Histogram with KDE
sns.histplot(data, kde=True, bins=50)

# Box plot (used in our analysis)
sns.boxplot(y=data)

# Violin plot
sns.violinplot(y=data)

# Q-Q plot
stats.probplot(data, dist="norm", plot=plt)

# ECDF plot
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(data)
plt.plot(ecdf.x, ecdf.y)
```

#### **Multiple Variables**
```python
# Pair plots
sns.pairplot(df[numerical_features])

# Correlation heatmap (used in our analysis)
sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0)

# Parallel coordinates
from pandas.plotting import parallel_coordinates
parallel_coordinates(df, 'target_column')

# Radar chart
from math import pi
angles = [n / len(features) * 2 * pi for n in range(len(features))]
plt.polar(angles, values)
```

### **🔗 Relationship Plots**

#### **Scatter Plots**
```python
# Basic scatter
plt.scatter(x, y)

# With regression line
sns.regplot(x='x', y='y', data=df)

# With categories
sns.scatterplot(x='x', y='y', hue='category', data=df)

# 3D scatter
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
```

#### **Advanced Plots**
```python
# Hexbin plot (for large datasets)
plt.hexbin(x, y, gridsize=30, cmap='Blues')

# 2D density plot
sns.kdeplot(x='x', y='y', data=df, cmap='Blues')

# Contour plot
plt.contour(X, Y, Z)

# Stream plot
plt.streamplot(X, Y, U, V)
```

---

## 💻 **Code Templates**

### **🚀 Complete EDA Pipeline**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEDA:
    def __init__(self, df, target_column=None):
        self.df = df
        self.target = target_column
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column and target_column in self.numeric_cols:
            self.numeric_cols.remove(target_column)
        elif target_column and target_column in self.categorical_cols:
            self.categorical_cols.remove(target_column)
    
    def data_overview(self):
        """Basic data overview"""
        print("Dataset Shape:", self.df.shape)
        print("\nData Types:")
        print(self.df.dtypes.value_counts())
        print("\nMissing Values:")
        print(self.df.isnull().sum().sum())
        print("\nMemory Usage:", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def univariate_analysis(self):
        """Comprehensive univariate analysis"""
        results = {}
        
        # Numerical features
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            results[col] = {
                'type': 'numerical',
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'normality_test': stats.normaltest(data),
                'outlier_rate': self._calculate_outlier_rate(data)
            }
        
        # Categorical features
        for col in self.categorical_cols:
            data = self.df[col].dropna()
            results[col] = {
                'type': 'categorical',
                'unique_count': data.nunique(),
                'most_frequent': data.mode().iloc[0],
                'frequency_distribution': data.value_counts().to_dict(),
                'entropy': stats.entropy(data.value_counts())
            }
        
        return results
    
    def bivariate_analysis(self):
        """Comprehensive bivariate analysis"""
        results = {}
        
        # Numerical vs Numerical
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            results['numerical_correlations'] = corr_matrix
            
            # Find strongest correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    correlations.append((col1, col2, corr_val))
            
            results['top_correlations'] = sorted(correlations, key=lambda x: abs(x[2]), reverse=True)
        
        # Categorical vs Numerical
        if self.categorical_cols and self.numeric_cols:
            cat_num_results = {}
            for cat_col in self.categorical_cols:
                for num_col in self.numeric_cols:
                    groups = [group.dropna() for name, group in self.df.groupby(cat_col)[num_col]]
                    if len(groups) > 1 and all(len(g) > 1 for g in groups):
                        f_stat, p_value = stats.f_oneway(*groups)
                        cat_num_results[f"{cat_col}_vs_{num_col}"] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
            results['categorical_numerical'] = cat_num_results
        
        # Categorical vs Categorical
        if len(self.categorical_cols) > 1:
            cat_cat_results = {}
            for i, cat1 in enumerate(self.categorical_cols):
                for cat2 in self.categorical_cols[i+1:]:
                    contingency_table = pd.crosstab(self.df[cat1], self.df[cat2])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    
                    cat_cat_results[f"{cat1}_vs_{cat2}"] = {
                        'chi2': chi2,
                        'p_value': p_value,
                        'cramers_v': cramers_v,
                        'significant': p_value < 0.05
                    }
            results['categorical_categorical'] = cat_cat_results
        
        return results
    
    def multivariate_analysis(self):
        """Comprehensive multivariate analysis"""
        results = {}
        
        if len(self.numeric_cols) >= 3:
            # PCA Analysis
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.df[self.numeric_cols].dropna())
            
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'components': pca.components_,
                'feature_loadings': pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                    index=self.numeric_cols
                )
            }
            
            # Clustering Analysis
            inertias = []
            silhouette_scores = []
            k_range = range(2, min(11, len(X_scaled)//5))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                inertias.append(kmeans.inertia_)
                
                from sklearn.metrics import silhouette_score
                silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            results['clustering'] = {
                'optimal_k': optimal_k,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores
            }
        
        return results
    
    def feature_importance_analysis(self):
        """Feature importance analysis"""
        if self.target is None:
            return None
        
        results = {}
        
        # Mutual Information
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        if self.df[self.target].dtype in ['object', 'category'] or self.df[self.target].nunique() < 10:
            mi_scores = mutual_info_classif(self.df[self.numeric_cols], self.df[self.target])
        else:
            mi_scores = mutual_info_regression(self.df[self.numeric_cols], self.df[self.target])
        
        results['mutual_information'] = dict(zip(self.numeric_cols, mi_scores))
        
        # Random Forest Feature Importance
        if self.df[self.target].dtype in ['object', 'category'] or self.df[self.target].nunique() < 10:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf.fit(self.df[self.numeric_cols], self.df[self.target])
        results['random_forest_importance'] = dict(zip(self.numeric_cols, rf.feature_importances_))
        
        return results
    
    def _calculate_outlier_rate(self, data):
        """Calculate outlier rate using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        return len(outliers) / len(data) * 100
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        print("="*80)
        print("COMPREHENSIVE EDA REPORT")
        print("="*80)
        
        # Data Overview
        self.data_overview()
        
        # Univariate Analysis
        print("\n" + "="*50)
        print("UNIVARIATE ANALYSIS")
        print("="*50)
        univariate_results = self.univariate_analysis()
        
        # Bivariate Analysis
        print("\n" + "="*50)
        print("BIVARIATE ANALYSIS")
        print("="*50)
        bivariate_results = self.bivariate_analysis()
        
        # Multivariate Analysis
        print("\n" + "="*50)
        print("MULTIVARIATE ANALYSIS")
        print("="*50)
        multivariate_results = self.multivariate_analysis()
        
        # Feature Importance
        if self.target:
            print("\n" + "="*50)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("="*50)
            importance_results = self.feature_importance_analysis()
        
        return {
            'univariate': univariate_results,
            'bivariate': bivariate_results,
            'multivariate': multivariate_results,
            'feature_importance': importance_results if self.target else None
        }

# Usage example
# eda = ComprehensiveEDA(df, target_column='target')
# results = eda.generate_report()
```

### **🎯 Quick Analysis Functions**

```python
def quick_correlation_analysis(df, threshold=0.7):
    """Quick correlation analysis with threshold"""
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_val
                ))
    
    return sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)

def quick_outlier_summary(df):
    """Quick outlier summary for all numerical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        
        outlier_summary[col] = {
            'outlier_count': len(outliers),
            'outlier_rate': len(outliers) / len(data) * 100,
            'outlier_bounds': (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        }
    
    return outlier_summary

def quick_distribution_test(df):
    """Quick normality test for all numerical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    normality_results = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 8:  # Minimum sample size
            stat, p_value = stats.normaltest(data)
            normality_results[col] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05,
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
    
    return normality_results
```

---

## 🎯 **Quick Reference Summary**

### **📊 Analysis Types by Data Combination**

| Data Types | Univariate | Bivariate | Multivariate |
|------------|------------|-----------|--------------|
| **Numerical** | Descriptive stats, Distribution tests, Outlier detection | Correlation, Regression, Scatter plots | PCA, Clustering, Factor analysis |
| **Categorical** | Frequency analysis, Chi-square GOF | Chi-square test, Cramér's V | MCA, Clustering |
| **Mixed** | Separate analysis | ANOVA, Mann-Whitney, Box plots | Mixed-type clustering, FAMD |

### **🔍 Test Selection Guide**

| Scenario | Parametric Test | Non-Parametric Alternative |
|----------|----------------|---------------------------|
| **2 groups comparison** | t-test | Mann-Whitney U |
| **Multiple groups** | ANOVA | Kruskal-Wallis |
| **Correlation** | Pearson | Spearman/Kendall |
| **Independence** | Chi-square | Fisher's exact |
| **Normality** | Shapiro-Wilk | Anderson-Darling |

### **📈 Visualization Quick Guide**

| Purpose | Best Plot Type | Alternative |
|---------|---------------|-------------|
| **Distribution** | Histogram + KDE | Box plot, Violin plot |
| **Correlation** | Heatmap | Scatter plot matrix |
| **Comparison** | Box plot | Violin plot, Strip plot |
| **Outliers** | Box plot | Scatter plot |
| **Relationships** | Scatter plot | Hexbin, 2D KDE |

---

## 🚀 **Advanced Techniques**

### **🤖 Machine Learning for EDA**

```python
# Anomaly detection for EDA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Feature selection for EDA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

# Dimensionality reduction for visualization
from sklearn.manifold import TSNE
import umap

# Clustering for pattern discovery
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
```

### **📊 Advanced Visualization Libraries**

```python
# Interactive visualizations
import plotly.express as px
import plotly.graph_objects as go
import bokeh.plotting as bp

# Statistical visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Specialized plots
import joypy  # Ridgeline plots
from adjustText import adjust_text  # Better text positioning
import networkx as nx  # Network graphs
```

---

**📚 This cheat sheet covers 95% of EDA scenarios you'll encounter in data science projects. Bookmark it for quick reference during your analysis!**

---

*Last Updated: January 22, 2025*  
*Based on: Account Conversion Analysis Project + Extended EDA Best Practices*
