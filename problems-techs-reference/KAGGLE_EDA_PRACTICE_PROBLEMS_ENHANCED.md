# 🎯 Kaggle EDA Practice Problems — Enhanced Edition

Based on: Comprehensive EDA Cheat Sheet  
Last Updated: January 22, 2025  
Purpose: Single, self-contained reference combining the improved v2 Playbook with the original v1 guide for deeper technique ideas per dataset.

---

## Part I — v2 Playbook (Main Guide)

# 🎯 Kaggle EDA Practice Problems — v2 Playbook

Based on: Comprehensive EDA Cheat Sheet  
Last Updated: January 22, 2025  
Goal: Practice EDA with purpose, reproducibility, and clear deliverables across easy→hard datasets, while avoiding leakage and documenting assumptions.

---

## 🧭 Quick Navigation

- Start Here: Setup & Workflow
- Project Structure & Deliverables
- Assumptions & Test Picker
- Scaling Big Data EDA
- Leakage, Ethics, Domain Notes
- Datasets (Easy · Medium · Hard)
- Technique Mapping (with effect sizes & multiple testing)
- Integrations: Use the utils scripts
- Practice Schedule & Stop Rules
- Resources

---

## 🚀 Start Here: Setup & Workflow

### Prerequisites

- Python 3.10+  
- Create a virtual environment and install packages:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install pandas numpy scipy seaborn matplotlib scikit-learn pingouin plotly umap-learn shap \
            polars dask[complete] statsmodels missingno nltk spacy wordcloud pydicom pillow opencv-python kaggle
```

Optional (NLP models):

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt
```

### Kaggle API Setup

1) From Kaggle, create an API token (Account → Create New Token).  
2) Place the file at `~/.kaggle/kaggle.json` and set permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Download Data (examples)

```bash
# Competitions
kaggle competitions download -c titanic -p data/titanic/raw && unzip -o data/titanic/raw/*.zip -d data/titanic/raw

# Datasets
kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data/mall/raw \
  && unzip -o data/mall/raw/*.zip -d data/mall/raw
```

---

## 🗂️ Project Structure & Deliverables

### Reproducible Folders

```
data/<slug>/{raw,processed}
notebooks/<slug>
figures/<slug>
reports/<slug>
```

### Universal Deliverables (per dataset)

1) Figures  
   - Distributions (hist/violin/box), missingness heatmap  
   - Correlation heatmap (with masking), 2–3 key relationship plots  
   - Optional: PCA biplot / 2D scatter, cluster visualization  

2) Tables  
   - Summary stats (mean/median/std/skew/kurtosis)  
   - Missingness summary  
   - Top correlations (+ sign)  
   - Significant tests (with p-values + effect sizes)  

3) Brief Report (≤ 1 page)  
   - Assumptions checked (normality, equal variance, independence, stationarity when relevant)  
   - 5 insights, 3 actions, 3 caveats  
   - Leakage and bias notes

Template filenames:  
`figures/<slug>/*`, `reports/<slug>/<slug>_eda_report.md`, `reports/<slug>/<slug>_summary.csv`

---

## 📒 Notebook Template

- Use `notebooks/template_eda.ipynb` as the starting point for each dataset.  
- Set `SLUG`, data path, and target where applicable.  
- Save plots/tables to `figures/<slug>` and `reports/<slug>` consistently.

---

## 🔖 Data Versioning & Citation

- Record Kaggle dataset or competition version/date in your report’s Data Card.  
- Snapshot raw zips under `data/<slug>/raw/<YYYYMMDD>/` and keep the unzipped copy in `processed/`.  
- Add checksums for large files to verify integrity:  
  - macOS/Linux: `shasum -a 256 <file> > <file>.sha256`  
  - Verify: `shasum -a 256 -c <file>.sha256`  
- Name processed files with a semantic suffix: `<slug>_train_clean_v1.csv`.  
- Cite dataset sources and licenses in the report; keep a `CITATION.md` if needed.

---

## 🧪 Assumptions & Test Picker

1) Normality: Shapiro–Wilk or D’Agostino; inspect Q–Q plots.  
2) Equal variances: Levene or Brown–Forsythe.  
3) Independence: domain reasoning; for time series, avoid random shuffles.  
4) Missingness: visualize with `missingno`; consider Little’s MCAR test (via statsmodels/other implementations).  
5) Time series: stationarity with ADF and KPSS (statsmodels), seasonality via STL.

When comparing…

- Numeric–Numeric: Pearson (linear, normal-ish), Spearman (monotonic), Kendall (robust small n, many ties).  
- Cat–Numeric: t-test/ANOVA if assumptions hold; else Mann–Whitney/Kruskal–Wallis.  
- Cat–Cat: Chi-square; use Fisher’s exact for small counts.  
- Binary target vs numeric: Point-biserial correlation.  
- Multiple tests: Control FDR with Benjamini–Hochberg; Bonferroni for strict control.

Effect sizes (report alongside p-values):  
- t-test: Cohen’s d (Hedges’ g if n small)  
- Mann–Whitney: Rank-biserial or Cliff’s delta  
- ANOVA: Eta² / Omega²  
- Chi-square: Cramér’s V

---

## 📈 Scaling Big Data EDA

- Use `polars` or `dask` for large CSVs; chunked `read_csv` with dtypes.  
- Downcast numeric types; sample stratified subsets for quick plots.  
- Cache intermediate artifacts; avoid full-figure grids on millions of rows.  
- For images: generate thumbnails; for text: hashing vectorizers for quick stats.  
- Time series: profile by time windows; test drift over time before pooling.

---

## 🛡️ Leakage, Ethics, Domain Notes

- Train/test boundaries: Do EDA on train; only sanity checks on test.  
- Time series: split temporally; do not leak future info (e.g., using lagged features created after the split).  
- Boston Housing: known fairness/ethical concerns; use with caution and contextualize.  
- Jigsaw: identity terms and bias; report fairness observations.  
- RSNA: patient-level splits; respect medical data handling and PHI policies.  
- Fraud datasets: prioritize PR-AUC; consider asymmetric costs and thresholding.

---

## ⚖️ Imbalanced Data: Metrics & Thresholding

- Prefer PR-AUC and Precision@K over ROC-AUC for rare positives.  
- Calibrate probabilities (Platt/Isotonic) before thresholding when possible.  
- Select threshold via:  
  - Expected cost minimization with domain costs (FP vs FN).  
  - Youden’s J (sensitivity + specificity − 1) for balanced trade-offs.  
  - Precision target (e.g., P@K or min precision at operating point).  
- Plot: precision–recall curve, cost curves, threshold sweeps (precision, recall, F1, specificity).  
- Report class prevalence and decision policy alongside metrics.

---

## 🟢 Easy Level Datasets (with hypotheses & deliverables)

### 1) Mall Customer Segmentation Data  
Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Dataset Overview  
- Size: 200 customers, 5 features  
- Features: CustomerID, Gender, Age, Annual Income, Spending Score  
- Type: Mixed (Numerical + Categorical)  
- Domain: Retail/Marketing Analytics

Hypotheses to test  
- Spending Score differs by Gender.  
- Age and Spending Score show non-linear clusters.  
- Income correlates positively with Spending Score (monotonic vs linear?).

Deliverables  
- Distributions for Age, Income, Spending; box by Gender.  
- Correlation heatmap; 2D PCA scatter; K-Means elbow and clusters.  
- Short report on identified segments and business actions.

Notes  
- Balanced, clean; good for full EDA pipeline and clustering basics.

#### Techniques to Practice

##### Univariate Analysis
- Numerical (Age, Income, Spending): descriptive stats; hist/box/violin; normality tests (Shapiro, D’Agostino); outliers (IQR, Z, Modified Z).  
- Categorical (Gender): frequency/proportion; chi-square goodness-of-fit; entropy.

##### Bivariate Analysis
- Numeric–numeric: correlations (Pearson, Spearman, Kendall); scatter with regression; joint/hexbin plots.  
- Categorical–numeric: box/violin by gender; t-test or Mann–Whitney; effect size (Cohen’s d).

##### Multivariate Analysis
- PCA for 2D/3D visualization; correlation heatmaps.  
- K-Means clustering and segment profiling.

---

### 2) Wine Quality (Red)  
Link: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Dataset Overview  
- Size: 1,599 wines, 12 features  
- Features: 11 physicochemical properties + quality rating  
- Type: Primarily numerical with ordinal target  
- Domain: Food & Beverage Quality Control

Hypotheses to test  
- Alcohol content is positively associated with quality.  
- Acidity metrics have non-linear relationships with quality.  
- Multicollinearity exists among physicochemical properties.

Deliverables  
- Correlation and partial correlation matrices; QQ plots; distribution fits.  
- ANOVA/Kruskal across quality levels; effect sizes.  
- PCA variance explained and loadings; feature importance via MI/Random Forest.

Notes  
- Treat quality as ordinal; consider non-linear methods.

#### Techniques to Practice

##### Univariate Analysis
- Distribution analysis of all 11 chemical properties; skew/kurtosis; distribution fitting (normal/log-normal/gamma); Q–Q plots; outliers (IF, LOF).

##### Bivariate Analysis
- Feature–target: correlations with quality; ANOVA across quality; box plots across feature ranges.  
- Feature–feature: correlation matrix; partial correlations; scatter-matrix.

##### Multivariate Analysis
- PCA variance explained and loadings; feature selection via MI/Random Forest; clustering to find natural groups.

---

### 3) Heart Disease UCI  
Link: https://www.kaggle.com/datasets/ronitf/heart-disease-uci

Dataset Overview  
- Size: 303 patients, 14 features  
- Features: Mix of numerical and categorical medical indicators  
- Type: Mixed with binary target  
- Domain: Medical Diagnosis

Hypotheses to test  
- Sex and chest pain type associate with disease presence.  
- Age differs across disease classes.  
- Odds ratio > 1 for select risk factors.

Deliverables  
- Cat–cat chi-square tests with Cramér’s V; non-parametric group comparisons.  
- Effect sizes (eta²/Cliff’s delta).  
- 5 risk insights with actionable screening heuristics.

Notes  
- Report multiple-testing control and clinical interpretability.

#### Techniques to Practice

##### Univariate Analysis
- Categorical (sex, chest pain, fasting blood sugar, etc.): frequency/proportions; chi-square GOF; entropy.  
- Numerical (age, bp, cholesterol): distribution with clinical ranges; age-stratified summaries.

##### Bivariate Analysis
- Categorical associations: chi-square tests; Cramér’s V; risk/odds ratios.  
- Continuous across categories: ANOVA or Kruskal–Wallis; post-hoc (Dunn); effect sizes.

##### Statistical Testing
- Multiple group comparisons; effect sizes; multiple testing corrections (Bonferroni/FDR).

---

### 4) Boston Housing  
Link: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices

Dataset Overview  
- Size: 506 records, 14 features  
- Features: Neighborhood characteristics and housing prices  
- Type: Primarily numerical with continuous target  
- Domain: Real Estate Economics

Hypotheses to test  
- Log(price) linearizes key relationships.  
- LSTAT and RM are strongest predictors.  
- Heteroscedasticity is present in raw target.

Deliverables  
- Distribution/transform analysis of target; outlier detection (IQR, Cook’s distance).  
- Pearson/Spearman comparisons; partial correlations.  
- Regression assumption diagnostics summary.

Notes  
- Include fairness caveat; dataset is deprecated in some libs.

#### Techniques to Practice

##### Distribution Analysis
- Target (Price): shape, log-transform effects, outlier impact.  
- Features: skewness correction; Box–Cox; normality tests.

##### Correlation and Regression
- Linear: Pearson with price; partial correlations; regression diagnostics.  
- Non-linear: Spearman; LOWESS; polynomial terms.

##### Outlier Analysis
- Univariate: IQR, Z, Modified Z.  
- Multivariate: Mahalanobis, Cook’s distance.  
- ML: Isolation Forest, One-Class SVM.

---

### 5) Iris (Extended)  
Link: https://www.kaggle.com/datasets/uciml/iris

Dataset Overview  
- Size: 150 flowers, 4 features + species  
- Features: Sepal/petal length and width  
- Type: Numerical features with categorical target  
- Domain: Botanical Classification

Hypotheses to test  
- Species are linearly separable in LDA space.  
- Petal features separate classes better than sepal features.  
- PCA explains >95% with top 2–3 components.

Deliverables  
- Pair plots; PCA biplot; LDA projection; MANOVA results with effect sizes.  
- Short discussion on PCA vs LDA and interpretability.

#### Techniques to Practice

##### Multivariate Analysis
- PCA: component interpretation, biplots, variance explained, loadings.  
- LDA: supervised projection, class separation; compare with PCA.

##### Visualization
- Pair plots with species color; parallel coordinates; radar charts; 3D scatter.

##### Statistical Classification
- ANOVA/MANOVA; post-hoc tests; effect sizes.

---

## 🟡 Medium Level Datasets

### 6) Titanic  
Link: https://www.kaggle.com/c/titanic

Dataset Overview  
- Size: 891 passengers (train), 12 features  
- Features: Demographics, ticket info, family relationships  
- Type: Mixed with missing data  
- Domain: Historical Disaster Analysis

Hypotheses to test  
- Survival differs by Sex×Class interaction.  
- Family size shows a non-linear relationship with survival.  
- Embarked port differences persist after class control.

Deliverables  
- Missingness heatmap; MCAR/MAR/MNAR reasoning; imputation plan.  
- Derived features (Title, FamilySize, Deck proxy) with tests and effect sizes.  
- Mosaic plots and log-linear model summary.

Notes  
- Keep EDA to train; document imputation rationale clearly.

#### Techniques to Practice

##### Missing Data Analysis
- Patterns: heatmaps; missingness correlations; MCAR/MAR/MNAR assessment; multiple imputation strategies.

##### Feature Engineering via EDA
- FamilySize (SibSp+Parch); Title from names; Deck from cabin; Fare per person.

##### Survival & Categorical Analysis
- Survival rates by categories; chi-square; Cramér’s V; survival curves; multi-way contingency (Class×Sex×Survival); log-linear models; mosaic plots.

---

### 7) House Prices (Advanced Regression)  
Link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Dataset Overview  
- Size: 1,460 houses, 79 features  
- Features: Comprehensive property characteristics  
- Type: Mixed with many categorical variables  
- Domain: Real Estate Valuation

Hypotheses to test  
- LogSalePrice stabilizes variance and normalizes residuals.  
- Strong multicollinearity exists among quality/size features.  
- High-cardinality categoricals carry predictive signal after grouping.

Deliverables  
- High-dimensional correlation (hierarchical clustering, VIF).  
- Target transform analysis; robustness checks (median/IQR).  
- Categorical impact summaries with ANOVA/Kruskal + effect sizes.

#### Techniques to Practice

##### High-Dimensional EDA
- Data types across 79 features; missingness patterns; initial screening; assess need for dimensionality reduction.

##### Advanced Correlations
- Hierarchical clustering of features; correlation networks; partial correlations; VIF for multicollinearity.

##### Categorical Analysis
- High-cardinality handling; cat–num relations; ANOVA/Kruskal; interaction effects.

##### Target & Transformations
- Price skewness; log-transform; outlier impact; robust statistics.

---

### 8) Bike Sharing Demand  
Link: https://www.kaggle.com/c/bike-sharing-demand

Dataset Overview  
- Size: 10,886 hourly records, 12 features  
- Features: Weather, temporal, demand  
- Type: Time series with environmental factors  
- Domain: Urban Transportation Analytics

Hypotheses to test  
- Strong daily/weekly seasonality; weather interacts with season.  
- Autocorrelation persists at specific lags.  
- Holidays/weekends exhibit distinct demand patterns.

Deliverables  
- STL decomposition; ACF/PACF; lag plots.  
- Interaction plots (weather×season, hour×weekday).  
- Stationarity checks (ADF/KPSS) and drift comments.

Notes  
- Use temporal splits; no random CV.

#### Techniques to Practice

##### Time Series EDA
- Temporal patterns: hourly/daily/weekly/seasonal; STL decomposition; ACF/PACF; lag correlations.

##### Weather Impacts
- Distributions of conditions; temperature–demand; humidity/wind effects; seasonal interactions.

##### Conditional Relationships
- Weather by season; hour by weekday; weekend vs weekday; holiday impacts.

---

### 9) Credit Card Fraud Detection  
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset Overview  
- Size: 284,807 transactions, 31 features  
- Features: PCA-transformed features + time, amount, class  
- Type: Highly imbalanced; anonymized features  
- Domain: Financial Security

Hypotheses to test  
- Fraud ratio is extremely low and clustered in time.  
- PCA components separate fraud density in tail regions.  
- Amount/time features show distinct fraud patterns.

Deliverables  
- Class imbalance quantification; resampling strategy rationale.  
- Outlier analysis (IF, LOF, OCSVM) with consensus.  
- PR-AUC emphasis in evaluation notes.  
- Cost curves and threshold analysis with expected cost table.

Notes  
- Use cost-sensitive mindset; subsample for plotting, not for conclusions.

#### Techniques to Practice

##### Imbalanced Data
- Class distribution and handling; sampling strategy evaluation; cost-sensitive analysis; metric selection (PR-AUC).

##### Anonymized Features
- PCA component distributions; component correlations; outlier detection in transformed space; reconstruction error.

##### Advanced Outliers
- Isolation Forest; Local Outlier Factor; One-Class SVM; ensemble consensus.

##### Temporal Fraud Patterns
- Fraud over time; amount distributions; time-based features; seasonal fraud patterns.

---

### 10) Porto Seguro Safe Driver  
Link: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

Dataset Overview  
- Size: 595,212 drivers, 59 features  
- Features: Anonymized driver and vehicle characteristics  
- Type: Mixed (binary, categorical, continuous)  
- Domain: Insurance Risk Assessment

Hypotheses to test  
- Mixed data types require tailored handling; many binary indicators are informative.  
- Missingness is feature-specific and predictive.  
- Correlated feature blocks exist and can be clustered.

Deliverables  
- Missingness clustering; feature-type EDA; correlation networks.  
- Sampling strategy for EDA vs modeling; automated EDA report summary.

#### Techniques to Practice

##### Large-Scale EDA
- Efficient sampling; parallel processing; memory-efficient correlations; automated EDA reports.

##### Feature Types
- Handle binary/categorical/continuous; infer feature types; select appropriate methods; cross-type analyses.

##### Insurance Domain
- Risk distribution; feature–risk strength; interactions; actuarial insights.

---

## 🔴 Hard Level Datasets

### 11) IEEE-CIS Fraud Detection  
Link: https://www.kaggle.com/c/ieee-fraud-detection

Dataset Overview  
- Size: 590,540 transactions, 433 features  
- Features: Transaction and identity information  
- Type: Extremely high-dimensional with complex relationships  
- Domain: E-commerce Fraud Detection

Hypotheses to test  
- High-dimensional blocks encode device/identity behavior.  
- Missingness carries signal; indicators improve separation.  
- Time-based aggregations reveal profiles.

Deliverables  
- Feature clustering; large-scale correlation; time-aggregated profiles.  
- Ensemble anomaly detection comparison; stability of signals across time.  
- Imbalance-aware evaluation: PR-AUC focus, cost curves, calibrated threshold selection (expected cost).

#### Techniques to Practice

##### High-Dimensional EDA
- Feature importance screening; correlation structure; feature clustering/grouping; automated selection.

##### Complex Missingness
- Pattern clustering; mechanism analysis; multiple imputation; missing indicators.

##### Feature Engineering via Transactions
- Time aggregations; user behavior profiling; device fingerprinting; network effects.

##### Sophisticated Outliers
- Ensemble anomaly detection; outlier consensus; context-aware and temporal outliers.

---

### 12) Santander Customer Transaction  
Link: https://www.kaggle.com/c/santander-customer-transaction-prediction

Dataset Overview  
- Size: 200,000 customers, 200 features  
- Features: Anonymized customer behavior data  
- Type: High-dimensional with synthetic characteristics  
- Domain: Banking Customer Analytics

Hypotheses to test  
- Some features exhibit synthetic patterns (distributional symmetry, replicated modes).  
- Mutual information uncovers non-linear dependencies.  
- Robust stats outperform mean-based summaries.

Deliverables  
- Synthetic-pattern diagnostics (shape, dependence).  
- Robust EDA (rank correlations, robust scaling); MI ranking.

#### Techniques to Practice

##### Synthetic Data Detection
- Distribution shapes; correlation patterns; synthetic signature detection; real vs artificial patterns.

##### Advanced Statistics
- Higher-order moments; mixture modeling; copulas; information theory.

##### Feature Interactions
- Non-linear correlations; mutual information; interaction strength; network analysis.

##### Robust Techniques
- Robust correlations; resistant outliers; non-parametric analyses; bootstrap inference.

---

### 13) Jane Street Market Prediction  
Link: https://www.kaggle.com/c/jane-street-market-prediction

Dataset Overview  
- Size: 2.4M+ observations, 130 features  
- Features: Financial market indicators and responses  
- Type: Time series with complex temporal dependencies  
- Domain: Quantitative Finance

Hypotheses to test  
- Non-stationarity and regime changes dominate; correlations drift.  
- Risk-factor structure evolves over time.  
- Volatility clustering is present.

Deliverables  
- Rolling correlations; regime segmentation; volatility diagnostics.  
- Multi-scale decomposition (e.g., wavelets) summary plots.

Notes  
- Strong temporal leakage risks; stress temporal CV.

#### Techniques to Practice

##### Financial Time Series EDA
- High-frequency pattern detection; volatility clustering; regime identification; liquidity patterns.

##### Advanced Time Series
- Multi-scale decomposition; wavelet analysis; regime-switching identification; non-stationarity detection.

##### Risk Factors
- Factor models; risk attribution; evolving correlation structure; tail risk.

##### High-Frequency Challenges
- Streaming/online analysis; memory-efficient EDA; real-time patterns; scalable correlations.

---

### 14) RSNA Pneumonia Detection  
Link: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

Dataset Overview  
- Size: 26,684 chest X-ray images with annotations  
- Features: Medical images with bounding boxes  
- Type: Image data with spatial annotations  
- Domain: Medical Image Analysis

Hypotheses to test  
- Bounding boxes cluster in anatomical regions.  
- Image quality metrics relate to label noise.  
- Size/shape distributions differ across subsets.

Deliverables  
- DICOM metadata summary; intensity histograms; artifact checks.  
- Spatial distributions of boxes; region frequency tables.

Notes  
- Patient-level splits; handle PHI and imaging ethics.

#### Techniques to Practice

##### Medical Image EDA
- Pixel intensity distributions; image quality assessment; anatomical structures; pathology patterns.

##### Spatial Analysis
- Box distribution analysis; spatial clustering; region analysis; size/shape distributions.

##### Advanced Visualization
- DICOM metadata analysis; multi-scale views; overlays; highlighting techniques.

##### Quality Assessment
- Noise patterns; contrast/brightness; artifact detection; inter-annotator agreement.

---

### 15) Jigsaw Toxic Comment Classification  
Link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Dataset Overview  
- Size: 159,571 comments with multi-label annotations  
- Features: Text with multiple toxicity categories  
- Type: Natural language; multi-label classification  
- Domain: Social Media Content Moderation

Hypotheses to test  
- Text length correlates with labels; label co-occurrence is structured.  
- Specific topics drive clusters; toxicity lexicon partially explains variance.  
- Identity terms correlate with labels (bias risk).

Deliverables  
- Length/vocab/n-gram stats; topic model overview; label co-occurrence heatmap.  
- Bias observations and mitigation notes (report fairness caveats).

#### Techniques to Practice

##### Text EDA
- Text length distributions; vocabulary analysis; n-gram frequencies; sentiment distributions.

##### Multi-Label Analysis
- Label correlations; co-occurrence patterns; hierarchical relations.

##### Advanced NLP EDA
- Topic modeling; semantic similarity; linguistic features; bias pattern identification.

##### Social Media Analytics
- Toxicity patterns; user profiling; temporal trends; platform-specific patterns.

---

## 🌟 Exemplar EDAs (How to Find Great Notebooks)

For each dataset/competition, open its Kaggle page below and navigate to the “Notebooks” tab. Sort by “Most Votes” and search for “EDA”, “Exploratory Data Analysis”, “Visualization”. Use these as inspiration while following the deliverables here.

- Mall Customer: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python  
- Wine Quality (Red): https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009  
- Heart Disease UCI: https://www.kaggle.com/datasets/ronitf/heart-disease-uci  
- Boston Housing: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices  
- Iris: https://www.kaggle.com/datasets/uciml/iris  
- Titanic: https://www.kaggle.com/c/titanic  
- House Prices: https://www.kaggle.com/c/house-prices-advanced-regression-techniques  
- Bike Sharing: https://www.kaggle.com/c/bike-sharing-demand  
- Credit Card Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- Porto Seguro: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction  
- IEEE-CIS Fraud: https://www.kaggle.com/c/ieee-fraud-detection  
- Santander: https://www.kaggle.com/c/santander-customer-transaction-prediction  
- Jane Street: https://www.kaggle.com/c/jane-street-market-prediction  
- RSNA Pneumonia: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge  
- Jigsaw Toxic: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

---

## 🔗 Exemplar EDAs (Curated Links Per Dataset)

Below are slots for two high-quality EDA notebooks per dataset. Use the dataset links above → Notebooks tab → sort by Most Votes. Add the best two you find here for quick reference and consistency across repetitions.

- Mall Customer:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Wine Quality (Red):  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Heart Disease UCI:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Boston Housing:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Iris:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Titanic:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- House Prices:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Bike Sharing:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Credit Card Fraud:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Porto Seguro:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- IEEE-CIS Fraud:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Santander:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Jane Street:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- RSNA Pneumonia:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)
- Jigsaw Toxic:  
  - 1) [Title · Author] (URL)  
  - 2) [Title · Author] (URL)


## 🧭 Technique Mapping (with additions)

- Univariate: descriptive stats; distribution fits; outliers (IQR, Z, modified Z), robust stats.  
- Bivariate: Pearson/Spearman/Kendall; chi-square/Fisher; ANOVA/Kruskal; point-biserial.  
- Multivariate: PCA/UMAP; clustering (K-Means, hierarchical); feature selection (MI, model-based).  
- Time series: ADF/KPSS; STL; ACF/PACF; drift checks.  
- Missingness: visual (missingno), MCAR reasoning/tests, imputation strategies.  
- Multiple testing: Benjamini–Hochberg FDR; Holm–Bonferroni.  
- Effect sizes: Cohen’s d, Hedges’ g, Cliff’s delta, rank-biserial, eta²/omega², Cramér’s V.

---

## 🔌 Integrations: Use the utils scripts

You can accelerate EDA using the provided scripts. Replace the placeholder `<FILENAME>.csv` in the scripts with your dataset path before running.

Examples

```bash
# 1) Quick EDA pipeline (figures + JSON summaries)
python utils/analysis_script.py

# 2) ML-oriented feature analysis vs binary target
python utils/ml_feature_analysis.py
```

Tips

- Place your CSV at `data/<slug>/processed/<name>.csv` and update the placeholder to that path.  
- Configure outputs to save under `figures/<slug>` and `reports/<slug>` for consistency.  
- Use the universal deliverables checklist to finalize each dataset’s report.

---

## ⏱️ Practice Schedule & Stop Rules

Timeboxing  
- Easy: 2–4 hours; Medium: 1–2 days; Hard: 3–5 days.

Stop when  
- Assumptions checked and documented.  
- 3–5 key relationships characterized with tests + effect sizes.  
- 5 insights, 3 actions, 3 caveats written.  
- No major unanswered questions remain for the stated goal.

---

## 🧮 Self‑Grading Rubric (100 points)

- Data Understanding (15): dataset overview, feature/target types, data card completeness.  
- Missingness & Quality (15): missingness visuals, MCAR/MAR reasoning, handling plan.  
- Univariate (15): distribution, outliers, robust stats where needed.  
- Bivariate (20): appropriate tests with assumptions checked; effect sizes reported.  
- Multivariate (10): PCA/UMAP or clustering insight where applicable.  
- Imbalance/Temporal (10): proper metrics and thresholding or stationarity checks where applicable.  
- Insights & Actions (10): 5 insights, 3 actions, 3 caveats with business relevance.  
- Reproducibility (5): folder structure, saved artifacts, versioning/citation recorded.  
- Communication (10): concise 1‑page report, clear visuals/tables.

Pass guideline: 80+. Excellent: 90+. Below 70: revisit assumptions/tests and reporting.

---

## 📚 Resources

- Books: Tukey’s Exploratory Data Analysis; The Elements of Statistical Learning; Python for Data Analysis.  
- Docs: Pandas, Seaborn Gallery, SciPy Stats, Statsmodels.  
- Tools: `missingno` for missingness viz; `shap` for model-based insight; `plotly` for interactivity.

---

Remember: EDA is about generating reliable insight, not only plots. Write down assumptions, verify them, report effect sizes, and call out risks of leakage and bias.

---

## Part II — Original v1 Guide (Full Text)

# 🎯 Kaggle EDA Practice Problems - Comprehensive Guide

**Based on:** Comprehensive EDA Cheat Sheet  
**Date:** January 22, 2025  
**Purpose:** Curated Kaggle problems for practicing all EDA techniques from the cheat sheet  
**Difficulty Levels:** Easy, Medium, Hard (5 problems each)  

---

## 📋 **Quick Navigation**

- [🟢 Easy Level Problems](#-easy-level-problems)
- [🟡 Medium Level Problems](#-medium-level-problems) 
- [🔴 Hard Level Problems](#-hard-level-problems)
- [📊 EDA Technique Mapping](#-eda-technique-mapping)
- [🚀 Getting Started Guide](#-getting-started-guide)

---

## 🟢 **Easy Level Problems**

### **1. Mall Customer Segmentation Data**
**🔗 Link:** [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

**📊 Dataset Overview:**
- **Size:** 200 customers, 5 features
- **Features:** CustomerID, Gender, Age, Annual Income, Spending Score
- **Type:** Mixed (Numerical + Categorical)
- **Domain:** Retail/Marketing Analytics

**🎯 EDA Techniques to Practice:**

#### **Univariate Analysis:**
- **Numerical Features:** Age, Annual Income, Spending Score
  - Descriptive statistics (mean, median, std, skewness, kurtosis)
  - Distribution analysis using histograms, box plots, violin plots
  - Normality tests (Shapiro-Wilk, D'Agostino-Pearson)
  - Outlier detection using IQR method, Z-score, Modified Z-score
  
- **Categorical Features:** Gender
  - Frequency analysis and proportions
  - Chi-square goodness of fit test
  - Entropy calculation

#### **Bivariate Analysis:**
- **Numerical vs Numerical:**
  - Correlation analysis (Pearson, Spearman, Kendall)
  - Scatter plots with regression lines
  - Joint plots and hexbin plots for relationships
  
- **Categorical vs Numerical:**
  - Box plots of income/spending by gender
  - T-tests and Mann-Whitney U tests for gender differences
  - Effect size calculations (Cohen's d)

#### **Multivariate Analysis:**
- **PCA Analysis:** Perfect for 3D visualization of customer segments
- **K-Means Clustering:** Ideal dataset for customer segmentation
- **Correlation heatmaps** for all numerical features

**🌟 What Makes This Special:**
This dataset is perfect for beginners because it's clean, small, and demonstrates clear clustering patterns. You can practice the complete EDA pipeline from basic statistics to advanced clustering without getting overwhelmed by data quality issues. The business context (customer segmentation) makes results interpretable and actionable.

---

### **2. Wine Quality Dataset**
**🔗 Link:** [Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

**📊 Dataset Overview:**
- **Size:** 1,599 wines, 12 features
- **Features:** 11 physicochemical properties + quality rating
- **Type:** Primarily numerical with ordinal target
- **Domain:** Food & Beverage Quality Control

**🎯 EDA Techniques to Practice:**

#### **Univariate Analysis:**
- **Distribution Analysis:** All 11 chemical properties
  - Skewness and kurtosis analysis
  - Distribution fitting (normal, log-normal, gamma)
  - Q-Q plots for normality assessment
  - Advanced outlier detection (Isolation Forest, LOF)

#### **Bivariate Analysis:**
- **Feature-Target Relationships:**
  - Correlation with wine quality
  - ANOVA tests for quality differences
  - Box plots showing quality distributions by feature ranges
  
- **Feature-Feature Relationships:**
  - Comprehensive correlation matrix analysis
  - Partial correlation analysis
  - Scatter plot matrices

#### **Multivariate Analysis:**
- **PCA for Dimensionality Reduction:** Understand which chemical properties explain most variance
- **Feature Selection:** Mutual information, Random Forest importance
- **Clustering:** Identify natural wine groups beyond quality ratings

**🌟 What Makes This Special:**
Excellent for practicing statistical tests and correlation analysis. The dataset has interesting non-linear relationships and multicollinearity issues that teach you when to use different correlation methods (Pearson vs Spearman). The ordinal target variable allows practicing both regression and classification approaches.

---

### **3. Heart Disease UCI Dataset**
**🔗 Link:** [Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

**📊 Dataset Overview:**
- **Size:** 303 patients, 14 features
- **Features:** Mix of numerical and categorical medical indicators
- **Type:** Mixed data types with binary target
- **Domain:** Medical Diagnosis

**🎯 EDA Techniques to Practice:**

#### **Univariate Analysis:**
- **Categorical Features:** Sex, chest pain type, fasting blood sugar, etc.
  - Frequency distributions and proportions
  - Chi-square goodness of fit tests
  - Entropy and information content analysis

- **Numerical Features:** Age, blood pressure, cholesterol, etc.
  - Distribution analysis with medical context
  - Reference range comparisons
  - Age-stratified analysis

#### **Bivariate Analysis:**
- **Medical Risk Factor Analysis:**
  - Chi-square tests for categorical associations
  - Cramér's V for association strength
  - ANOVA for continuous variables across categories
  - Risk ratio and odds ratio calculations

#### **Statistical Testing:**
- **Comprehensive hypothesis testing practice:**
  - Multiple group comparisons (Kruskal-Wallis)
  - Post-hoc tests (Dunn's test)
  - Effect size calculations
  - Bonferroni corrections for multiple testing

**🌟 What Makes This Special:**
Perfect for practicing categorical data analysis and statistical testing. The medical context provides clear hypotheses to test (e.g., "Do men have higher heart disease risk?"). Small size allows for detailed analysis of every relationship. Excellent for learning when to use parametric vs non-parametric tests.

---

### **4. Boston Housing Dataset**
**🔗 Link:** [Boston Housing](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices)

**📊 Dataset Overview:**
- **Size:** 506 housing records, 14 features
- **Features:** Neighborhood characteristics and housing prices
- **Type:** Primarily numerical with continuous target
- **Domain:** Real Estate Economics

**🎯 EDA Techniques to Practice:**

#### **Distribution Analysis:**
- **Target Variable (Price):** 
  - Distribution shape analysis
  - Log transformation effects
  - Outlier impact on modeling
  
- **Feature Distributions:**
  - Skewness correction techniques
  - Box-Cox transformations
  - Normality testing across all features

#### **Correlation and Regression Analysis:**
- **Linear Relationships:**
  - Pearson correlations with price
  - Partial correlations controlling for other factors
  - Regression diagnostics and assumptions testing
  
- **Non-linear Relationships:**
  - Spearman correlations
  - LOWESS smoothing
  - Polynomial relationship exploration

#### **Outlier Analysis:**
- **Multiple Outlier Detection Methods:**
  - Univariate: IQR, Z-score, Modified Z-score
  - Multivariate: Mahalanobis distance, Cook's distance
  - Machine Learning: Isolation Forest, One-Class SVM

**🌟 What Makes This Special:**
Classic dataset for learning regression analysis and outlier detection. Contains both linear and non-linear relationships, making it perfect for comparing different correlation methods. The economic context helps understand when outliers might be legitimate (luxury properties) vs. data errors.

---

### **5. Iris Dataset (Extended Analysis)**
**🔗 Link:** [Iris Species](https://www.kaggle.com/datasets/uciml/iris)

**📊 Dataset Overview:**
- **Size:** 150 flowers, 4 features + species
- **Features:** Sepal/petal length and width measurements
- **Type:** Numerical features with categorical target
- **Domain:** Botanical Classification

**🎯 EDA Techniques to Practice:**

#### **Perfect Multivariate Analysis:**
- **PCA Deep Dive:**
  - Component interpretation
  - Biplot analysis
  - Variance explained visualization
  - Feature loading analysis

- **Linear Discriminant Analysis (LDA):**
  - Supervised dimensionality reduction
  - Class separation visualization
  - Comparison with PCA results

#### **Advanced Visualization:**
- **Comprehensive Plot Types:**
  - Pair plots with species coloring
  - Parallel coordinates plots
  - Radar charts for species profiles
  - 3D scatter plots

#### **Statistical Classification:**
- **ANOVA and MANOVA:**
  - Multiple group comparisons
  - Multivariate analysis of variance
  - Post-hoc testing
  - Effect size calculations

**🌟 What Makes This Special:**
Though simple, this dataset is perfect for mastering visualization techniques and multivariate analysis. The clear class separation makes it ideal for understanding PCA vs LDA differences. Small size allows for detailed exploration of every technique without computational constraints.

---

## 🟡 **Medium Level Problems**

### **6. Titanic: Machine Learning from Disaster**
**🔗 Link:** [Titanic](https://www.kaggle.com/c/titanic)

**📊 Dataset Overview:**
- **Size:** 891 passengers (train), 12 features
- **Features:** Demographics, ticket info, family relationships
- **Type:** Mixed with significant missing data
- **Domain:** Historical Disaster Analysis

**🎯 EDA Techniques to Practice:**

#### **Missing Data Analysis:**
- **Pattern Analysis:**
  - Missing data heatmaps
  - Missing data correlations
  - MCAR, MAR, MNAR assessment
  - Multiple imputation strategies

#### **Feature Engineering Through EDA:**
- **Derived Features:**
  - Family size from SibSp + Parch
  - Title extraction from names
  - Deck extraction from cabin numbers
  - Fare per person calculations

#### **Survival Analysis:**
- **Comprehensive Bivariate Analysis:**
  - Survival rates by all categorical variables
  - Chi-square tests for independence
  - Cramér's V for association strength
  - Survival curves and statistical significance

#### **Advanced Categorical Analysis:**
- **Multi-way Contingency Tables:**
  - Three-way interactions (Class × Sex × Survival)
  - Log-linear model analysis
  - Mosaic plots for complex relationships

**🌟 What Makes This Special:**
Perfect for learning missing data analysis and feature engineering. The dataset has complex interactions (e.g., class and gender effects on survival) that require sophisticated EDA techniques. Historical context makes results meaningful and interpretable. Excellent for practicing the full data science pipeline.

---

### **7. House Prices: Advanced Regression Techniques**
**🔗 Link:** [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**📊 Dataset Overview:**
- **Size:** 1,460 houses, 79 features
- **Features:** Comprehensive property characteristics
- **Type:** Mixed with many categorical variables
- **Domain:** Real Estate Valuation

**🎯 EDA Techniques to Practice:**

#### **High-Dimensional EDA:**
- **Feature Overview:**
  - Data type analysis across 79 features
  - Missing data patterns and strategies
  - Feature importance screening
  - Dimensionality reduction necessity assessment

#### **Advanced Correlation Analysis:**
- **Correlation Matrix Management:**
  - Hierarchical clustering of features
  - Correlation network graphs
  - Partial correlation analysis
  - Multicollinearity detection (VIF)

#### **Categorical Variable Analysis:**
- **Extensive Categorical EDA:**
  - High-cardinality categorical handling
  - Categorical-numerical relationships
  - ANOVA across multiple categorical variables
  - Interaction effect identification

#### **Distribution and Transformation Analysis:**
- **Target Variable Analysis:**
  - Price distribution skewness
  - Log transformation effects
  - Outlier impact assessment
  - Robust statistics application

**🌟 What Makes This Special:**
Excellent for learning high-dimensional EDA techniques. With 79 features, you must learn efficient EDA strategies and automated analysis techniques. The mix of numerical and categorical variables provides comprehensive practice. Real estate domain knowledge helps interpret complex feature interactions.

---

### **8. Bike Sharing Demand**
**🔗 Link:** [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)

**📊 Dataset Overview:**
- **Size:** 10,886 hourly records, 12 features
- **Features:** Weather, temporal, and demand data
- **Type:** Time series with environmental factors
- **Domain:** Urban Transportation Analytics

**🎯 EDA Techniques to Practice:**

#### **Time Series EDA:**
- **Temporal Pattern Analysis:**
  - Hourly, daily, weekly, seasonal patterns
  - Trend decomposition
  - Autocorrelation analysis
  - Lag correlation analysis

#### **Weather Impact Analysis:**
- **Environmental Factor EDA:**
  - Weather condition distributions
  - Temperature-demand relationships
  - Humidity and wind speed effects
  - Seasonal weather pattern interactions

#### **Advanced Bivariate Analysis:**
- **Time-Conditional Relationships:**
  - Weather effects by season
  - Hourly patterns by weather
  - Weekend vs weekday differences
  - Holiday impact analysis

#### **Multivariate Time Series:**
- **Complex Interaction Analysis:**
  - Multiple factor ANOVA
  - Interaction plots
  - Conditional distributions
  - Time-lagged correlations

**🌟 What Makes This Special:**
Perfect for learning time series EDA and environmental data analysis. The dataset combines temporal patterns with external factors, requiring sophisticated analysis techniques. Teaches you to handle cyclical patterns and interaction effects. Urban planning context makes results practically relevant.

---

### **9. Credit Card Fraud Detection**
**🔗 Link:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**📊 Dataset Overview:**
- **Size:** 284,807 transactions, 31 features
- **Features:** PCA-transformed features + time, amount, class
- **Type:** Highly imbalanced with anonymized features
- **Domain:** Financial Security

**🎯 EDA Techniques to Practice:**

#### **Imbalanced Data Analysis:**
- **Class Distribution Analysis:**
  - Extreme imbalance handling (0.17% fraud)
  - Sampling strategy evaluation
  - Cost-sensitive analysis considerations
  - Evaluation metric selection

#### **Anonymized Feature Analysis:**
- **PCA Feature Interpretation:**
  - Distribution analysis of PCA components
  - Component correlation analysis
  - Outlier detection in transformed space
  - Reconstruction error analysis

#### **Advanced Outlier Detection:**
- **Multiple Outlier Methods:**
  - Isolation Forest for anomaly detection
  - Local Outlier Factor (LOF)
  - One-Class SVM
  - Ensemble outlier detection

#### **Time Series Fraud Patterns:**
- **Temporal Fraud Analysis:**
  - Fraud patterns over time
  - Transaction amount distributions
  - Time-based feature engineering
  - Seasonal fraud pattern detection

**🌟 What Makes This Special:**
Excellent for learning imbalanced data analysis and anomaly detection. The PCA-transformed features teach you to work with anonymized data. Large dataset size requires efficient EDA techniques. Financial fraud context provides clear business relevance and interpretation challenges.

---

### **10. Porto Seguro's Safe Driver Prediction**
**🔗 Link:** [Porto Seguro Safe Driver](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

**📊 Dataset Overview:**
- **Size:** 595,212 drivers, 59 features
- **Features:** Anonymized driver and vehicle characteristics
- **Type:** Mixed with categorical and numerical features
- **Domain:** Insurance Risk Assessment

**🎯 EDA Techniques to Practice:**

#### **Large-Scale EDA:**
- **Efficient Analysis Techniques:**
  - Sampling strategies for EDA
  - Parallel processing for analysis
  - Memory-efficient correlation analysis
  - Automated EDA report generation

#### **Feature Type Analysis:**
- **Mixed Data Type Handling:**
  - Binary, categorical, and continuous features
  - Feature type identification from metadata
  - Appropriate analysis method selection
  - Cross-type relationship analysis

#### **Insurance Domain EDA:**
- **Risk Factor Analysis:**
  - Risk distribution analysis
  - Feature-risk relationship strength
  - Interaction effect identification
  - Actuarial insight generation

#### **Advanced Statistical Testing:**
- **Comprehensive Hypothesis Testing:**
  - Multiple testing corrections
  - Effect size calculations
  - Power analysis
  - Bootstrap confidence intervals

**🌟 What Makes This Special:**
Perfect for learning large-scale EDA techniques and mixed data type analysis. The insurance domain provides clear business context for risk analysis. Anonymized features challenge your ability to extract insights without domain knowledge. Large size teaches computational efficiency in EDA.

---

## 🔴 **Hard Level Problems**

### **11. IEEE-CIS Fraud Detection**
**🔗 Link:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

**📊 Dataset Overview:**
- **Size:** 590,540 transactions, 433 features
- **Features:** Transaction and identity information
- **Type:** Extremely high-dimensional with complex relationships
- **Domain:** E-commerce Fraud Detection

**🎯 EDA Techniques to Practice:**

#### **High-Dimensional EDA:**
- **Dimensionality Challenges:**
  - Feature importance screening (433 features!)
  - Correlation structure analysis
  - Feature clustering and grouping
  - Automated feature selection

#### **Complex Missing Data Patterns:**
- **Advanced Missing Data Analysis:**
  - Missing data pattern clustering
  - Feature-specific missing mechanisms
  - Multiple imputation strategies
  - Missing indicator feature creation

#### **Advanced Feature Engineering Through EDA:**
- **Transaction Pattern Analysis:**
  - Time-based aggregations
  - User behavior profiling
  - Device fingerprinting analysis
  - Network effect identification

#### **Sophisticated Outlier Detection:**
- **Ensemble Anomaly Detection:**
  - Multiple algorithm combination
  - Outlier consensus methods
  - Context-aware outlier detection
  - Temporal outlier analysis

**🌟 What Makes This Special:**
Ultimate challenge for high-dimensional EDA. With 433 features, you must master automated EDA techniques and feature selection methods. Complex fraud patterns require sophisticated analysis approaches. Real-world e-commerce context with anonymized features challenges interpretation skills. Perfect for learning scalable EDA methodologies.

---

### **12. Santander Customer Transaction Prediction**
**🔗 Link:** [Santander Customer Transaction](https://www.kaggle.com/c/santander-customer-transaction-prediction)

**📊 Dataset Overview:**
- **Size:** 200,000 customers, 200 features
- **Features:** Anonymized customer behavior data
- **Type:** High-dimensional with synthetic characteristics
- **Domain:** Banking Customer Analytics

**🎯 EDA Techniques to Practice:**

#### **Synthetic Data Detection:**
- **Data Authenticity Analysis:**
  - Distribution shape analysis
  - Correlation pattern investigation
  - Synthetic data signature detection
  - Real vs artificial pattern identification

#### **Advanced Statistical Analysis:**
- **Deep Statistical Investigation:**
  - Higher-order moment analysis
  - Distribution mixture modeling
  - Copula analysis for dependencies
  - Information theory applications

#### **Feature Interaction Analysis:**
- **Complex Relationship Discovery:**
  - Non-linear correlation methods
  - Mutual information analysis
  - Feature interaction strength
  - Network analysis of feature relationships

#### **Robust EDA Techniques:**
- **Robust Statistical Methods:**
  - Robust correlation measures
  - Resistant outlier detection
  - Non-parametric relationship analysis
  - Bootstrap-based inference

**🌟 What Makes This Special:**
Challenges your ability to detect synthetic data characteristics and adapt EDA techniques accordingly. The anonymized features require creative analysis approaches. High dimensionality with potential synthetic patterns teaches advanced statistical detection methods. Banking context provides business relevance despite anonymization.

---

### **13. Jane Street Market Prediction**
**🔗 Link:** [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction)

**📊 Dataset Overview:**
- **Size:** 2.4M+ market observations, 130 features
- **Features:** Financial market indicators and responses
- **Type:** Time series with complex temporal dependencies
- **Domain:** Quantitative Finance

**🎯 EDA Techniques to Practice:**

#### **Financial Time Series EDA:**
- **Market Microstructure Analysis:**
  - High-frequency pattern detection
  - Volatility clustering analysis
  - Market regime identification
  - Liquidity pattern analysis

#### **Advanced Time Series Techniques:**
- **Sophisticated Temporal Analysis:**
  - Multi-scale time series decomposition
  - Wavelet analysis for frequency components
  - Regime-switching model identification
  - Non-stationary pattern detection

#### **Risk Factor Analysis:**
- **Financial Risk EDA:**
  - Factor model analysis
  - Risk attribution analysis
  - Correlation structure evolution
  - Tail risk assessment

#### **High-Frequency Data Challenges:**
- **Computational EDA:**
  - Streaming data analysis techniques
  - Memory-efficient time series EDA
  - Real-time pattern detection
  - Scalable correlation analysis

**🌟 What Makes This Special:**
Ultimate challenge for time series EDA with financial complexity. Massive dataset size requires advanced computational techniques. Financial market context demands sophisticated statistical methods. Perfect for learning high-frequency data analysis and risk assessment techniques.

---

### **14. RSNA Pneumonia Detection Challenge**
**🔗 Link:** [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

**📊 Dataset Overview:**
- **Size:** 26,684 chest X-ray images with annotations
- **Features:** Medical images with bounding box annotations
- **Type:** Image data with spatial annotations
- **Domain:** Medical Image Analysis

**🎯 EDA Techniques to Practice:**

#### **Medical Image EDA:**
- **Image Statistical Analysis:**
  - Pixel intensity distributions
  - Image quality assessment
  - Anatomical structure analysis
  - Pathology pattern identification

#### **Spatial Data Analysis:**
- **Geometric Pattern Analysis:**
  - Bounding box distribution analysis
  - Spatial clustering of pathologies
  - Anatomical region analysis
  - Size and shape distribution analysis

#### **Advanced Visualization:**
- **Medical Image Visualization:**
  - DICOM metadata analysis
  - Multi-scale image analysis
  - Pathology highlighting techniques
  - Statistical overlay methods

#### **Quality Assessment EDA:**
- **Image Quality Analysis:**
  - Noise pattern identification
  - Contrast and brightness analysis
  - Artifact detection
  - Inter-annotator agreement analysis

**🌟 What Makes This Special:**
Unique opportunity to apply EDA to medical image data. Combines spatial analysis with medical domain knowledge. Large image dataset requires specialized EDA techniques. Medical context provides clear interpretability requirements and ethical considerations.

---

### **15. Jigsaw Toxic Comment Classification**
**🔗 Link:** [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**📊 Dataset Overview:**
- **Size:** 159,571 comments with multi-label toxicity annotations
- **Features:** Text data with multiple toxicity categories
- **Type:** Natural language with multi-label classification
- **Domain:** Social Media Content Moderation

**🎯 EDA Techniques to Practice:**

#### **Text Data EDA:**
- **Linguistic Pattern Analysis:**
  - Text length distributions
  - Vocabulary analysis
  - N-gram frequency analysis
  - Sentiment distribution analysis

#### **Multi-Label Analysis:**
- **Complex Label Structure EDA:**
  - Label correlation analysis
  - Multi-label distribution patterns
  - Label co-occurrence analysis
  - Hierarchical label relationships

#### **Advanced Text Analytics:**
- **Sophisticated NLP EDA:**
  - Topic modeling for content themes
  - Semantic similarity analysis
  - Linguistic feature extraction
  - Bias pattern identification

#### **Social Media Analytics:**
- **Content Moderation EDA:**
  - Toxicity pattern analysis
  - User behavior profiling
  - Temporal toxicity trends
  - Platform-specific pattern analysis

**🌟 What Makes This Special:**
Perfect for learning text data EDA and multi-label analysis. Social media context provides relevant business applications. Multi-label structure requires sophisticated analysis techniques. Ethical considerations in content moderation add important dimension to analysis.

---

## 📊 **EDA Technique Mapping**

### **🔢 Univariate Analysis Techniques**

| Technique | Easy Problems | Medium Problems | Hard Problems |
|-----------|---------------|-----------------|---------------|
| **Descriptive Statistics** | All 5 problems | All 5 problems | All 5 problems |
| **Distribution Analysis** | Wine Quality, Boston Housing | House Prices, Bike Sharing | Santander, Jane Street |
| **Normality Tests** | Wine Quality, Heart Disease | Titanic, Credit Card Fraud | IEEE-CIS, RSNA |
| **Outlier Detection** | Mall Customer, Boston Housing | Porto Seguro, Credit Card Fraud | IEEE-CIS, Jane Street |
| **Categorical Analysis** | Heart Disease, Iris | Titanic, House Prices | Jigsaw Toxic Comments |

### **🔗 Bivariate Analysis Techniques**

| Technique | Easy Problems | Medium Problems | Hard Problems |
|-----------|---------------|-----------------|---------------|
| **Correlation Analysis** | All numerical datasets | House Prices, Porto Seguro | Santander, Jane Street |
| **Chi-square Tests** | Heart Disease, Iris | Titanic, House Prices | IEEE-CIS, Jigsaw |
| **ANOVA/Kruskal-Wallis** | Wine Quality, Heart Disease | Bike Sharing, Porto Seguro | RSNA, Jane Street |
| **Regression Analysis** | Boston Housing, Wine Quality | House Prices, Bike Sharing | Jane Street, Santander |
| **Effect Size Analysis** | Heart Disease, Iris | Titanic, Credit Card Fraud | IEEE-CIS, RSNA |

### **🌐 Multivariate Analysis Techniques**

| Technique | Easy Problems | Medium Problems | Hard Problems |
|-----------|---------------|-----------------|---------------|
| **PCA Analysis** | Mall Customer, Iris | Credit Card Fraud, Porto Seguro | Santander, IEEE-CIS |
| **Clustering Analysis** | Mall Customer, Wine Quality | Bike Sharing, House Prices | Jane Street, RSNA |
| **Feature Selection** | Wine Quality, Boston Housing | Porto Seguro, Credit Card Fraud | IEEE-CIS, Santander |
| **Dimensionality Reduction** | Iris, Mall Customer | Credit Card Fraud, House Prices | IEEE-CIS, Jane Street |

### **🧪 Advanced Statistical Tests**

| Technique | Easy Problems | Medium Problems | Hard Problems |
|-----------|---------------|-----------------|---------------|
| **Multiple Testing Correction** | Heart Disease | Porto Seguro, Titanic | IEEE-CIS, Santander |
| **Non-parametric Tests** | Wine Quality, Heart Disease | Credit Card Fraud, Bike Sharing | Jane Street, RSNA |
| **Time Series Analysis** | - | Bike Sharing | Jane Street, IEEE-CIS |
| **Missing Data Analysis** | - | Titanic, House Prices | IEEE-CIS, Santander |

---

## 🚀 **Getting Started Guide**

### **📋 Recommended Learning Path**

#### **Phase 1: Foundation (Easy Problems)**
1. **Start with Mall Customer Segmentation** - Clean data, clear patterns
2. **Progress to Wine Quality** - More complex relationships
3. **Master Heart Disease** - Categorical analysis focus
4. **Tackle Boston Housing** - Regression and outliers
5. **Complete with Iris** - Multivariate mastery

#### **Phase 2: Application (Medium Problems)**
1. **Titanic** - Missing data and feature engineering
2. **House Prices** - High-dimensional analysis
3. **Bike Sharing** - Time series patterns
4. **Credit Card Fraud** - Imbalanced data challenges
5. **Porto Seguro** - Large-scale analysis

#### **Phase 3: Mastery (Hard Problems)**
1. **IEEE-CIS Fraud** - Ultimate high-dimensional challenge
2. **Santander** - Synthetic data detection
3. **Jane Street** - Financial time series complexity
4. **RSNA** - Image data analysis
5. **Jigsaw** - Text and multi-label analysis

### **🛠️ Essential Tools and Libraries**

```python
# Core EDA Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression

# Statistical Testing
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu
import pingouin as pg  # Advanced statistical tests

# Specialized Libraries
import plotly.express as px  # Interactive plots
import umap  # Advanced dimensionality reduction
import shap  # Feature importance
```

### **📈 Success Metrics**

#### **For Each Problem, Master:**
1. **Data Understanding** (20%)
   - Dataset overview and context
   - Feature type identification
   - Missing data assessment

2. **Univariate Analysis** (25%)
   - Distribution analysis
   - Outlier detection
   - Statistical summaries

3. **Bivariate Analysis** (25%)
   - Relationship identification
   - Statistical testing
   - Effect size calculation

4. **Multivariate Analysis** (20%)
   - Pattern discovery
   - Dimensionality reduction
   - Clustering insights

5. **Insights and Interpretation** (10%)
   - Business relevance
   - Actionable conclusions
   - Next steps identification

### **🎯 Practice Schedule**

#### **Beginner (3-6 months)**
- **Week 1-2:** Mall Customer Segmentation
- **Week 3-4:** Wine Quality
- **Week 5-6:** Heart Disease
- **Week 7-8:** Boston Housing
- **Week 9-10:** Iris (Advanced techniques)
- **Week 11-12:** Review and consolidation

#### **Intermediate (6-12 months)**
- **Month 1:** Titanic (comprehensive analysis)
- **Month 2:** House Prices (high-dimensional EDA)
- **Month 3:** Bike Sharing (time series EDA)
- **Month 4:** Credit Card Fraud (imbalanced data)
- **Month 5:** Porto Seguro (large-scale analysis)
- **Month 6:** Integration and advanced techniques

#### **Advanced (12+ months)**
- **Months 1-2:** IEEE-CIS Fraud Detection
- **Months 3-4:** Santander Customer Transaction
- **Months 5-6:** Jane Street Market Prediction
- **Months 7-8:** RSNA Pneumonia Detection
- **Months 9-10:** Jigsaw Toxic Comment Classification
- **Months 11-12:** Portfolio development and specialization

---

## 🎓 **Learning Outcomes**

### **After Completing Easy Level:**
- ✅ Master basic statistical analysis
- ✅ Understand distribution analysis
- ✅ Apply correlation techniques confidently
- ✅ Perform outlier detection effectively
- ✅ Create compelling visualizations

### **After Completing Medium Level:**
- ✅ Handle missing data professionally
- ✅ Engineer features through EDA insights
- ✅ Analyze high-dimensional datasets
- ✅ Work with time series data
- ✅ Manage imbalanced datasets

### **After Completing Hard Level:**
- ✅ Master advanced statistical techniques
- ✅ Handle massive datasets efficiently
- ✅ Detect synthetic data patterns
- ✅ Analyze specialized data types (images, text)
- ✅ Apply domain-specific EDA approaches

---

## 📚 **Additional Resources**

### **Books:**
- "Exploratory Data Analysis" by John Tukey
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Python for Data Analysis" by Wes McKinney

### **Online Courses:**
- Coursera: "Exploratory Data Analysis" by Johns Hopkins
- edX: "Introduction to Data Science" by MIT
- Kaggle Learn: "Data Visualization" and "Feature Engineering"

### **Documentation:**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Scipy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

**🎯 Remember:** The goal is not just to complete these problems, but to deeply understand when and why to apply each EDA technique. Focus on interpretation and insight generation, not just technical execution.

**📊 Happy Analyzing!** Each dataset offers unique learning opportunities that will make you a more skilled and intuitive data scientist.

---

*Last Updated: January 22, 2025*  
*Based on: Comprehensive EDA Cheat Sheet v1.0*
