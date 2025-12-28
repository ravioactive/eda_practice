# Univariate Analysis - Extended Techniques for Specialized Problem Types

## 📊 **Techniques Beyond Customer Segmentation**

This document catalogs univariate analysis techniques that are **less applicable or emphasized** in the Customer Segmentation context but are **essential for other ML problem types**. These techniques expand the univariate analysis framework to be comprehensive across diverse data science applications.

---

## **📚 CONTEXT: Why These Extensions?**

The univariate analysis organization proposal focuses on techniques directly applicable to:
- Customer demographics (age, income, spending)
- Categorical attributes (gender, segments)
- Cross-sectional retail data

**Missing Coverage Includes:**
- Time series single-variable analysis
- Geospatial univariate patterns
- Survival/reliability single-variable analysis
- Text/NLP token-level analysis
- Image intensity/pixel analysis
- Signal processing single-channel analysis
- Compositional single-component analysis
- Circular/directional single-variable analysis
- High-dimensional marginal analysis
- Functional data analysis
- Causal univariate diagnostics
- Bayesian hierarchical univariate modeling

---

## **📁 EXTENDED FOLDER STRUCTURE**

```
univariate/
├── [EXISTING 01-07 folders from organization proposal]
│
├── 08_time_series_univariate/
│   ├── trend_and_seasonality_decomposition.ipynb
│   ├── stationarity_and_unit_roots.ipynb
│   ├── autocorrelation_and_spectral_analysis.ipynb
│   └── change_point_and_anomaly_detection.ipynb
│
├── 09_geospatial_univariate/
│   ├── spatial_distribution_analysis.ipynb
│   ├── spatial_clustering_and_hotspots.ipynb
│   ├── geostatistical_summary_statistics.ipynb
│   └── spatial_autocorrelation_univariate.ipynb
│
├── 10_survival_reliability_univariate/
│   ├── survival_curve_estimation.ipynb
│   ├── hazard_function_analysis.ipynb
│   ├── censoring_pattern_analysis.ipynb
│   └── reliability_metrics.ipynb
│
├── 11_text_nlp_univariate/
│   ├── token_frequency_analysis.ipynb
│   ├── vocabulary_richness_metrics.ipynb
│   ├── document_length_analysis.ipynb
│   └── lexical_diversity_measures.ipynb
│
├── 12_image_signal_univariate/
│   ├── intensity_histogram_analysis.ipynb
│   ├── spectral_power_distribution.ipynb
│   ├── signal_quality_metrics.ipynb
│   └── noise_characterization.ipynb
│
├── 13_compositional_univariate/
│   ├── single_component_analysis.ipynb
│   ├── proportion_transformation.ipynb
│   ├── compositional_variability.ipynb
│   └── reference_component_selection.ipynb
│
├── 14_circular_directional_univariate/
│   ├── circular_descriptive_statistics.ipynb
│   ├── directional_distribution_fitting.ipynb
│   ├── circular_hypothesis_testing.ipynb
│   └── temporal_angle_analysis.ipynb
│
├── 15_high_dimensional_marginal/
│   ├── marginal_screening_methods.ipynb
│   ├── univariate_feature_selection.ipynb
│   ├── multiple_testing_marginal.ipynb
│   └── false_discovery_control.ipynb
│
├── 16_functional_data_univariate/
│   ├── functional_descriptive_statistics.ipynb
│   ├── functional_depth_measures.ipynb
│   ├── curve_registration.ipynb
│   └── functional_pca_univariate.ipynb
│
└── 17_bayesian_hierarchical_univariate/
    ├── hierarchical_means_and_variances.ipynb
    ├── shrinkage_estimation.ipynb
    ├── random_effects_univariate.ipynb
    └── prior_sensitivity_analysis.ipynb
```

---

## **📋 DETAILED NOTEBOOK CONTENT SPECIFICATIONS**

### **08_time_series_univariate/ - Temporal Single-Variable Analysis**

#### **trend_and_seasonality_decomposition.ipynb**
**Primary Focus**: Decomposing time series into interpretable components
- **Classical decomposition**: Additive (Y = T + S + R) vs multiplicative (Y = T × S × R)
- **Moving average smoothing**: Simple, weighted, exponential
- **STL decomposition**: Seasonal-Trend decomposition using LOESS
- **X-11/X-13 methods**: Census Bureau seasonal adjustment
- **MSTL**: Multiple seasonal decomposition
- **Business context**: Sales trends, demand seasonality, economic cycles

#### **stationarity_and_unit_roots.ipynb**
**Primary Focus**: Testing and achieving stationarity for modeling
- **Augmented Dickey-Fuller (ADF)**: Unit root testing
- **Phillips-Perron (PP) test**: Robust unit root test
- **KPSS test**: Stationarity null hypothesis
- **Variance ratio test**: Random walk detection
- **Structural break tests**: Zivot-Andrews, Perron tests
- **Differencing strategies**: Order selection, seasonal differencing

#### **autocorrelation_and_spectral_analysis.ipynb**
**Primary Focus**: Temporal dependency and frequency domain analysis
- **Autocorrelation Function (ACF)**: Lag correlation patterns
- **Partial Autocorrelation (PACF)**: Direct lag effects
- **Ljung-Box Q-test**: White noise testing
- **Durbin-Watson statistic**: First-order autocorrelation
- **Periodogram**: Spectral power estimation
- **Welch's method**: Improved spectral estimation

#### **change_point_and_anomaly_detection.ipynb**
**Primary Focus**: Detecting structural changes and outliers in time series
- **CUSUM**: Cumulative sum control charts
- **Pettitt's test**: Non-parametric change point
- **BOCPD**: Bayesian Online Change Point Detection
- **Prophet anomaly detection**: Facebook's approach
- **Isolation Forest for time series**: Temporal anomaly scoring
- **Contextual anomalies**: Point vs collective anomalies

---

### **09_geospatial_univariate/ - Spatial Distribution Analysis**

#### **spatial_distribution_analysis.ipynb**
**Primary Focus**: Understanding geographic patterns in single variables
- **Spatial mean and median center**: Central tendency in space
- **Standard distance**: Spatial dispersion measure
- **Directional distribution**: Ellipse-based spread
- **Spatial range**: Bounding box analysis
- **Density mapping**: Kernel density for point patterns
- **Business context**: Customer location density, service coverage

#### **spatial_clustering_and_hotspots.ipynb**
**Primary Focus**: Identifying spatial concentrations
- **Getis-Ord Gi***: Hot spot analysis
- **Local Moran's I**: High-high, low-low clusters
- **DBSCAN spatial**: Density-based spatial clustering
- **Ripley's K**: Point pattern clustering
- **Nearest neighbor analysis**: Clustering vs dispersion
- **Kernel density hotspots**: Continuous surface analysis

#### **geostatistical_summary_statistics.ipynb**
**Primary Focus**: Spatial statistics for continuous surfaces
- **Semivariogram**: Spatial dependence structure
- **Nugget, sill, range**: Variogram parameters
- **Anisotropy**: Directional spatial variation
- **Cross-validation**: Spatial prediction accuracy
- **Spatial median**: Robust central location
- **Trimmed spatial mean**: Outlier-resistant center

#### **spatial_autocorrelation_univariate.ipynb**
**Primary Focus**: Testing spatial dependence in univariate data
- **Global Moran's I**: Overall spatial autocorrelation
- **Geary's C**: Alternative autocorrelation measure
- **Getis-Ord G**: Concentration of high/low values
- **Join count statistics**: Categorical spatial patterns
- **Spatial weight matrix construction**: Contiguity, distance-based
- **Permutation inference**: Significance testing

---

### **10_survival_reliability_univariate/ - Time-to-Event Analysis**

#### **survival_curve_estimation.ipynb**
**Primary Focus**: Estimating survival probabilities over time
- **Kaplan-Meier estimator**: Non-parametric survival curve
- **Nelson-Aalen estimator**: Cumulative hazard estimation
- **Life table method**: Actuarial survival estimation
- **Confidence intervals**: Greenwood's formula, log-log transform
- **Median survival time**: 50th percentile estimation
- **Restricted mean survival**: Area under survival curve

#### **hazard_function_analysis.ipynb**
**Primary Focus**: Understanding instantaneous risk over time
- **Hazard rate estimation**: Kernel smoothing approaches
- **Cumulative hazard**: Nelson-Aalen estimator
- **Hazard shapes**: Increasing, decreasing, bathtub, constant
- **Parametric hazard models**: Exponential, Weibull, Gompertz
- **Mean residual life**: Expected remaining time
- **Hazard ratios**: Effect size interpretation

#### **censoring_pattern_analysis.ipynb**
**Primary Focus**: Understanding and handling incomplete observations
- **Censoring types**: Right, left, interval censoring
- **Censoring mechanisms**: Random vs informative
- **Censoring distribution**: Kaplan-Meier for censoring
- **Truncation patterns**: Left, right truncation
- **Missing time patterns**: Administrative vs loss-to-follow-up
- **Sensitivity analysis**: Impact of censoring assumptions

#### **reliability_metrics.ipynb**
**Primary Focus**: Engineering reliability and quality metrics
- **MTTF (Mean Time To Failure)**: Average lifetime
- **MTBF (Mean Time Between Failures)**: Repairable systems
- **Failure rate**: λ(t) instantaneous failure
- **Reliability function**: R(t) = 1 - F(t)
- **B-life percentiles**: B10, B50 life estimation
- **Availability**: Uptime proportion

---

### **11_text_nlp_univariate/ - Single-Document/Token Analysis**

#### **token_frequency_analysis.ipynb**
**Primary Focus**: Word and token distribution analysis
- **Term frequency (TF)**: Raw and normalized counts
- **Zipf's law validation**: Frequency-rank relationship
- **Heaps' law**: Vocabulary growth rate
- **Stop word analysis**: Function word patterns
- **N-gram frequencies**: Bigram, trigram distributions
- **Hapax legomena**: Single-occurrence words

#### **vocabulary_richness_metrics.ipynb**
**Primary Focus**: Lexical diversity and vocabulary measures
- **Type-Token Ratio (TTR)**: Basic vocabulary diversity
- **MTLD**: Measure of Textual Lexical Diversity
- **VOCD-D**: Sophisticated TTR variant
- **Yule's K**: Vocabulary characteristic constant
- **Simpson's D for vocabulary**: Concentration measure
- **Vocabulary coverage**: Percentage of common words

#### **document_length_analysis.ipynb**
**Primary Focus**: Document and text length patterns
- **Word count distributions**: Document length patterns
- **Sentence length analysis**: Readability indicators
- **Character count**: Compression and encoding
- **Paragraph structure**: Document organization
- **Length normalization**: Adjusting for document size
- **Outlier documents**: Unusually short/long texts

#### **lexical_diversity_measures.ipynb**
**Primary Focus**: Comprehensive text diversity metrics
- **Honore's R**: Hapax-based richness
- **Sichel's S**: Dis legomena proportion
- **Brunet's W**: Vocabulary richness index
- **Carroll's CTTR**: Corrected TTR
- **Guiraud's R**: Root TTR
- **Herdan's C**: Log TTR variant

---

### **12_image_signal_univariate/ - Single-Channel Analysis**

#### **intensity_histogram_analysis.ipynb**
**Primary Focus**: Image intensity distribution analysis
- **Grayscale histogram**: Pixel intensity distribution
- **Color channel histograms**: R, G, B separate analysis
- **Histogram statistics**: Mean, variance, skewness of intensities
- **Dynamic range**: Intensity span utilization
- **Histogram equalization**: Contrast enhancement analysis
- **Cumulative histogram**: CDF-based analysis

#### **spectral_power_distribution.ipynb**
**Primary Focus**: Frequency content of signals and images
- **Power Spectral Density (PSD)**: Frequency power distribution
- **1/f noise characteristics**: Pink noise detection
- **Dominant frequency**: Peak frequency identification
- **Bandwidth measures**: Signal frequency spread
- **Spectral flatness**: Noise vs tonal content
- **Spectral centroid**: Frequency center of mass

#### **signal_quality_metrics.ipynb**
**Primary Focus**: Signal quality and integrity assessment
- **Signal-to-Noise Ratio (SNR)**: Quality measure
- **Peak Signal-to-Noise Ratio (PSNR)**: Image quality
- **Root Mean Square (RMS)**: Signal amplitude
- **Crest factor**: Peak-to-RMS ratio
- **Dynamic range**: Maximum to minimum ratio
- **Total Harmonic Distortion (THD)**: Harmonic content

#### **noise_characterization.ipynb**
**Primary Focus**: Understanding noise properties
- **Noise variance estimation**: Median Absolute Deviation
- **Noise distribution**: Gaussian, Poisson, salt-and-pepper
- **Autocorrelation of noise**: White vs colored noise
- **Noise power spectrum**: Frequency characteristics
- **SNR estimation**: Local and global methods
- **Noise floor**: Minimum detectable signal

---

### **13_compositional_univariate/ - Parts-of-Whole Analysis**

#### **single_component_analysis.ipynb**
**Primary Focus**: Analyzing individual components of compositions
- **Marginal distribution issues**: Why standard stats fail
- **Spurious correlation**: Closed data problems
- **Component variability**: Proper variance measures
- **Dominance analysis**: Major vs minor components
- **Zero handling**: Replacement strategies
- **Business context**: Market shares, budget allocations

#### **proportion_transformation.ipynb**
**Primary Focus**: Transforming compositional data for analysis
- **Log-ratio transformations**: ALR, CLR, ILR
- **Additive log-ratio (ALR)**: Reference component selection
- **Centered log-ratio (CLR)**: Symmetric transformation
- **Isometric log-ratio (ILR)**: Orthonormal coordinates
- **Back-transformation**: Returning to simplex
- **Choosing reference**: Criteria for ALR denominator

#### **compositional_variability.ipynb**
**Primary Focus**: Measuring spread in compositional data
- **Aitchison distance**: Proper compositional metric
- **Total variance**: Sum of log-ratio variances
- **Variation matrix**: Pairwise log-ratio variances
- **Variation array**: Comprehensive variability display
- **Center definition**: Closed geometric mean
- **Dispersion measures**: Compositional variance

#### **reference_component_selection.ipynb**
**Primary Focus**: Choosing appropriate reference for log-ratios
- **Stability criteria**: Low variance reference
- **Biological reference**: Housekeeping genes analogy
- **Statistical criteria**: Minimum variation selection
- **Multiple reference**: Geometric mean approaches
- **Sensitivity analysis**: Impact of reference choice
- **Domain knowledge**: Subject-matter guidance

---

### **14_circular_directional_univariate/ - Angular Data Analysis**

#### **circular_descriptive_statistics.ipynb**
**Primary Focus**: Summary statistics for circular/angular data
- **Circular mean**: Mean direction calculation
- **Mean resultant length**: Concentration measure (0 to 1)
- **Circular variance**: 1 - R̄ spread measure
- **Circular standard deviation**: Angular dispersion
- **Median direction**: Robust central direction
- **Circular range**: Angular spread measure

#### **directional_distribution_fitting.ipynb**
**Primary Focus**: Fitting distributions to angular data
- **Von Mises distribution**: "Normal" for circles
- **Wrapped distributions**: Wrapped normal, Cauchy
- **Cardioid distribution**: Asymmetric circular
- **Jones-Pewsey distribution**: Flexible circular
- **MLE for circular distributions**: Parameter estimation
- **Goodness-of-fit tests**: Circular KS, Kuiper's test

#### **circular_hypothesis_testing.ipynb**
**Primary Focus**: Statistical inference for angular data
- **Rayleigh test**: Uniformity vs unimodal
- **V-test**: Specified mean direction test
- **Watson's U² test**: Uniformity testing
- **Kuiper's test**: Circular KS alternative
- **Rao's spacing test**: Uniform distribution test
- **One-sample mean direction test**: Known direction comparison

#### **temporal_angle_analysis.ipynb**
**Primary Focus**: Time-of-day and seasonal circular patterns
- **Time-of-day analysis**: 24-hour circular analysis
- **Day-of-week**: Weekly circular patterns
- **Monthly patterns**: Annual cyclical analysis
- **Circular time series**: Temporal angular data
- **Rose diagrams**: Directional histograms
- **Business context**: Peak hours, seasonal peaks

---

### **15_high_dimensional_marginal/ - Marginal Analysis in High-D**

#### **marginal_screening_methods.ipynb**
**Primary Focus**: Univariate screening in high-dimensional data
- **Sure Independence Screening (SIS)**: Marginal correlation ranking
- **Distance correlation screening**: Non-linear marginal association
- **Mutual information screening**: Information-theoretic ranking
- **Variance screening**: Feature variability filtering
- **Sparsity screening**: Non-zero proportion filtering
- **Iterative SIS**: Multiple screening rounds

#### **univariate_feature_selection.ipynb**
**Primary Focus**: Single-feature selection methods
- **ANOVA F-test**: Numerical vs categorical target
- **Chi-square test**: Categorical vs categorical
- **Mutual information**: Universal association
- **Fisher score**: Between vs within variance ratio
- **Information gain**: Decision tree criterion
- **ReliefF univariate**: Instance-based weighting

#### **multiple_testing_marginal.ipynb**
**Primary Focus**: Controlling errors in many univariate tests
- **Bonferroni correction**: FWER control
- **Holm-Bonferroni**: Step-down FWER
- **Benjamini-Hochberg**: FDR control
- **Benjamini-Yekutieli**: Dependent FDR
- **q-value estimation**: Storey's approach
- **Permutation-based thresholds**: Empirical null

#### **false_discovery_control.ipynb**
**Primary Focus**: Advanced FDR methods for high-dimensional data
- **Local FDR (fdr)**: Posterior probability of null
- **Empirical null estimation**: Data-driven null distribution
- **Knockoff filter**: FDR with model-X knockoffs
- **Conditional FDR**: Stratified analysis
- **Weighted FDR**: Prior information incorporation
- **Adaptive procedures**: Data-driven α levels

---

### **16_functional_data_univariate/ - Curve and Function Analysis**

#### **functional_descriptive_statistics.ipynb**
**Primary Focus**: Summary measures for functional data
- **Pointwise mean function**: Average curve
- **Pointwise variance**: Variability at each point
- **Functional median**: Depth-based central curve
- **Functional mode**: Most common curve shape
- **Covariance function**: C(s,t) surface
- **Basis expansion**: Fourier, B-spline, wavelet

#### **functional_depth_measures.ipynb**
**Primary Focus**: Centrality and outlyingness for curves
- **Fraiman-Muniz depth**: Univariate integration
- **Modified band depth**: Simplified band depth
- **h-mode depth**: Modal depth for functions
- **Half-region depth**: Simplified Tukey depth
- **Functional boxplot**: Median, central region, outliers
- **Outlier detection**: Depth-based curve outliers

#### **curve_registration.ipynb**
**Primary Focus**: Aligning curves in time/phase
- **Landmark registration**: Aligning known features
- **Continuous registration**: Warping functions
- **Dynamic time warping**: Discrete alignment
- **Procrustes alignment**: Shape matching
- **Amplitude vs phase variability**: Separation
- **Registered mean**: After alignment

#### **functional_pca_univariate.ipynb**
**Primary Focus**: Dimensionality reduction for single curves
- **Functional PCA basics**: Eigenfunctions
- **Variance explained**: Proportion by component
- **Score interpretation**: Individual curve loadings
- **Component visualization**: Mode of variation plots
- **Regularized FPCA**: Smoothness constraints
- **Sparse FPCA**: Variable selection in functions

---

### **17_bayesian_hierarchical_univariate/ - Multilevel Single-Variable**

#### **hierarchical_means_and_variances.ipynb**
**Primary Focus**: Modeling grouped univariate data
- **Random intercepts model**: Group-specific means
- **Variance components**: Between vs within
- **Intraclass correlation (ICC)**: Clustering strength
- **Partial pooling**: Shrinkage toward grand mean
- **Group-level estimation**: Small sample improvements
- **Business context**: Store-level, region-level variation

#### **shrinkage_estimation.ipynb**
**Primary Focus**: Borrowing strength across groups
- **James-Stein estimation**: Simultaneous shrinkage
- **Empirical Bayes**: Estimated hyperparameters
- **Full Bayes**: Prior hyperparameter modeling
- **Shrinkage visualization**: Individual vs pooled
- **Mean squared error comparison**: Shrinkage benefits
- **Small sample groups**: Maximum benefit scenarios

#### **random_effects_univariate.ipynb**
**Primary Focus**: Modeling random group effects
- **Random effects distribution**: Normal, t, mixture
- **Variance parameter estimation**: Restricted MLE
- **BLUP**: Best Linear Unbiased Prediction
- **Prediction intervals**: Group-specific uncertainty
- **Model comparison**: Random vs fixed effects
- **Cross-validation**: Leave-group-out prediction

#### **prior_sensitivity_analysis.ipynb**
**Primary Focus**: Impact of prior choices on inference
- **Variance prior sensitivity**: Half-Cauchy, inverse-gamma
- **Mean prior sensitivity**: Informative vs vague
- **Prior predictive checks**: Prior implications
- **Posterior sensitivity**: Robustness to prior
- **Reference priors**: Jeffreys, reference analysis
- **Weakly informative priors**: Default recommendations

---

## **🎯 PROBLEM TYPE APPLICABILITY MATRIX**

| Extended Folder | Time Series | Geospatial | Survival | NLP | Image | Panel | High-D | Other |
|-----------------|:-----------:|:----------:|:--------:|:---:|:-----:|:-----:|:------:|:-----:|
| 08_time_series | ✅ Primary | ⚪ | ⚪ | ⚪ | ⚪ | 🟡 | ⚪ | ⚪ |
| 09_geospatial | ⚪ | ✅ Primary | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 10_survival | ⚪ | ⚪ | ✅ Primary | ⚪ | ⚪ | ⚪ | ⚪ | Reliability |
| 11_text_nlp | ⚪ | ⚪ | ⚪ | ✅ Primary | ⚪ | ⚪ | ⚪ | ⚪ |
| 12_image_signal | ⚪ | ⚪ | ⚪ | ⚪ | ✅ Primary | ⚪ | ⚪ | Audio/IoT |
| 13_compositional | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | Microbiome, Finance |
| 14_circular | 🟡 Time-of-day | 🟡 Direction | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | Wind, Navigation |
| 15_high_dimensional | ⚪ | ⚪ | ⚪ | 🟡 | 🟡 | ⚪ | ✅ Primary | Genomics |
| 16_functional | ✅ Curves | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | Growth curves |
| 17_bayesian_hierarchical | 🟡 | 🟡 | ⚪ | ⚪ | ⚪ | ✅ Primary | ⚪ | Education, Clinical |

**Legend:** ✅ Primary application | 🟡 Secondary application | ⚪ Not primary focus

---

## **📊 TECHNIQUE COUNT SUMMARY**

| Extended Folder | Notebooks | Techniques |
|-----------------|-----------|------------|
| 08_time_series_univariate | 4 | ~25 |
| 09_geospatial_univariate | 4 | ~24 |
| 10_survival_reliability | 4 | ~24 |
| 11_text_nlp_univariate | 4 | ~24 |
| 12_image_signal_univariate | 4 | ~24 |
| 13_compositional_univariate | 4 | ~20 |
| 14_circular_directional | 4 | ~24 |
| 15_high_dimensional_marginal | 4 | ~24 |
| 16_functional_data | 4 | ~24 |
| 17_bayesian_hierarchical | 4 | ~20 |
| **TOTAL EXTENDED** | **40** | **~233** |

**Combined with Organization Proposal:**
- **Original Framework**: 25 notebooks
- **Extended Techniques**: 40 notebooks
- **TOTAL COMPREHENSIVE**: 65 notebooks

---

## **🔗 RELATIONSHIP TO EXISTING COVERAGE**

### **Techniques That Bridge Original and Extended:**

| Original Folder | Extended Folder | Bridging Topics |
|-----------------|-----------------|-----------------|
| 02_statistical_inference | 17_bayesian_hierarchical | Bayesian methods scale up |
| 04_outlier_detection | 08_time_series | Temporal anomaly detection |
| 05_information_theory | 11_text_nlp | Entropy for vocabulary |
| 06_advanced_techniques | 16_functional | EVT for functional tails |
| 03_distribution | 14_circular | Non-Euclidean distributions |

### **Unique to Extended (No Overlap):**
- Spatial autocorrelation (09)
- Survival curves and hazard (10)
- Compositional constraints (13)
- Circular statistics (14)
- Functional depth (16)

---

## **📅 EXTENDED IMPLEMENTATION ROADMAP**

### **Phase 6: Temporal & Spatial Extensions (Weeks 9-10)**
1. **08_time_series_univariate/**: Decomposition, stationarity, spectral
2. **09_geospatial_univariate/**: Spatial distribution, hotspots, variograms

### **Phase 7: Domain-Specific Extensions (Weeks 11-12)**
1. **10_survival_reliability/**: Kaplan-Meier, hazard, censoring
2. **11_text_nlp_univariate/**: Token frequency, vocabulary metrics
3. **12_image_signal_univariate/**: Intensity, spectral, noise

### **Phase 8: Specialized Data Types (Weeks 13-14)**
1. **13_compositional_univariate/**: Log-ratios, compositional stats
2. **14_circular_directional/**: Angular statistics, von Mises

### **Phase 9: Advanced Extensions (Weeks 15-16)**
1. **15_high_dimensional_marginal/**: Screening, FDR control
2. **16_functional_data/**: Functional PCA, depth measures
3. **17_bayesian_hierarchical/**: Shrinkage, random effects

---

## **🔧 KEY PYTHON LIBRARIES BY EXTENSION**

```python
# Time Series
import statsmodels.tsa as tsa
from prophet import Prophet
import ruptures  # Change point detection

# Geospatial
import geopandas
from pysal import esda, weights
from pointpats import centrography

# Survival
import lifelines
from sksurv import survival

# Text/NLP
from collections import Counter
import nltk.text
from lexicalrichness import LexicalRichness

# Image/Signal
import cv2
from scipy.signal import welch, spectrogram
import pywt  # Wavelets

# Compositional
from skbio.stats.composition import clr, ilr
import compositions

# Circular
import pycircstat
from astropy.stats import circmean, circvar

# High-Dimensional
from sklearn.feature_selection import mutual_info_classif
import knockpy

# Functional Data
import skfda
from skfda.exploratory.depth import ModifiedBandDepth

# Bayesian Hierarchical
import pymc as pm
import bambi
import arviz
```

---

## **📋 NON-REDUNDANCY VERIFICATION**

### **Topics NOT Covered in Original Proposal:**

| Extended Topic | Verification |
|----------------|--------------|
| STL decomposition | Not in 06_advanced_techniques |
| ADF/KPSS tests | Not in 03_distribution |
| Spatial weight matrices | New domain |
| Kaplan-Meier | Not in any original |
| Type-Token Ratio | Not in 05_information_theory |
| Circular mean/variance | New statistics type |
| Log-ratio transforms | New constraint type |
| Functional PCA | Not in any original |
| Shrinkage estimation | Not in 02_statistical_inference |

---

*Last Updated: December 27, 2025*
*Purpose: Extended univariate techniques for comprehensive coverage across all ML problem types*

