# Univariate Analysis - Extended Techniques Learning Roadmap

## 📚 **Learning Guide for Extended Univariate Analysis Techniques**

This document provides a comprehensive learning and practice roadmap for the univariate analysis techniques identified as **missing from the Customer Segmentation focused proposal**. These techniques are essential for handling diverse ML problem types beyond simple tabular customer data.

**Prerequisites:** Completion of the original univariate analysis roadmap (Phases 1-5, Weeks 1-8).

---

## **📅 IMPLEMENTATION ROADMAP (CONTINUATION)**

### **Phase 6: Temporal & Spatial Domains (Weeks 9-11)**

#### **Week 9: Time Series Univariate Fundamentals**

**Day 1-2: Decomposition & Trend Analysis**
1. `08_time_series_univariate/trend_and_seasonality_decomposition.ipynb`
   - Classical additive and multiplicative decomposition
   - STL decomposition implementation
   - Practice dataset: [Store Sales](https://www.kaggle.com/c/store-sales-time-series-forecasting)

2. `08_time_series_univariate/stationarity_and_unit_roots.ipynb`
   - ADF, KPSS, and PP test implementation
   - Differencing strategies and order selection

**Day 3-4: Autocorrelation & Spectral**
3. `08_time_series_univariate/autocorrelation_and_spectral_analysis.ipynb`
   - ACF/PACF computation and interpretation
   - Periodogram and spectral density estimation
   - Practice dataset: [Air Quality](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set)

**Day 5-7: Change Points & Anomalies**
4. `08_time_series_univariate/change_point_and_anomaly_detection.ipynb`
   - CUSUM and Pettitt's test implementation
   - Isolation Forest for temporal data
   - Practice dataset: [Web Traffic](https://www.kaggle.com/c/web-traffic-time-series-forecasting)

---

#### **Week 10: Geospatial Univariate Analysis**

**Day 1-2: Spatial Distribution Basics**
1. `09_geospatial_univariate/spatial_distribution_analysis.ipynb`
   - Mean center and standard distance computation
   - Kernel density estimation for points
   - Practice dataset: [NYC Taxi](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

2. `09_geospatial_univariate/spatial_clustering_and_hotspots.ipynb`
   - Getis-Ord Gi* hot spot analysis
   - DBSCAN for spatial clustering

**Day 3-4: Geostatistics**
3. `09_geospatial_univariate/geostatistical_summary_statistics.ipynb`
   - Semivariogram estimation and interpretation
   - Practice dataset: [Earthquake Data](https://www.kaggle.com/usgs/earthquake-database)

4. `09_geospatial_univariate/spatial_autocorrelation_univariate.ipynb`
   - Global and Local Moran's I implementation
   - Spatial weight matrix construction

**Day 5-7: Integration Project**
5. Combined spatial-temporal analysis mini-project
   - Apply both domains to real-world dataset
   - Document findings and methodology

---

#### **Week 11: Survival/Reliability Analysis**

**Day 1-3: Survival Curve Estimation**
1. `10_survival_reliability_univariate/survival_curve_estimation.ipynb`
   - Kaplan-Meier estimator implementation
   - Confidence intervals and median survival
   - Practice dataset: [Telco Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

2. `10_survival_reliability_univariate/hazard_function_analysis.ipynb`
   - Nelson-Aalen cumulative hazard
   - Hazard shape interpretation

**Day 4-5: Censoring & Reliability**
3. `10_survival_reliability_univariate/censoring_pattern_analysis.ipynb`
   - Censoring type identification
   - Informative censoring diagnostics
   - Practice dataset: [Heart Failure](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

4. `10_survival_reliability_univariate/reliability_metrics.ipynb`
   - MTTF, MTBF calculation
   - B-life percentile estimation

**Day 6-7: Review & Practice**
5. Cross-domain integration review
6. Practice exercises from all three domains

---

### **Phase 7: Text/NLP & Signal Processing (Weeks 12-13)**

#### **Week 12: Text/NLP Univariate**

**Day 1-2: Token Frequency Analysis**
1. `11_text_nlp_univariate/token_frequency_analysis.ipynb`
   - TF computation and Zipf's law validation
   - N-gram frequency analysis
   - Practice dataset: [IMDB Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

2. `11_text_nlp_univariate/vocabulary_richness_metrics.ipynb`
   - TTR, MTLD, VOCD-D implementation
   - Yule's K and vocabulary coverage

**Day 3-4: Document & Lexical Analysis**
3. `11_text_nlp_univariate/document_length_analysis.ipynb`
   - Word/sentence count distributions
   - Outlier document detection
   - Practice dataset: [News Category](https://www.kaggle.com/rmisra/news-category-dataset)

4. `11_text_nlp_univariate/lexical_diversity_measures.ipynb`
   - Honore's R, Sichel's S implementation
   - Comprehensive diversity comparison

**Day 5-7: NLP Integration**
5. Combined NLP univariate analysis project
6. Vocabulary and length pattern analysis
7. Documentation and insights

---

#### **Week 13: Image/Signal Univariate**

**Day 1-2: Intensity Analysis**
1. `12_image_signal_univariate/intensity_histogram_analysis.ipynb`
   - Grayscale and color channel histograms
   - Histogram statistics and equalization
   - Practice dataset: [CIFAR-10](https://www.kaggle.com/c/cifar-10)

2. `12_image_signal_univariate/spectral_power_distribution.ipynb`
   - PSD estimation with Welch's method
   - Dominant frequency identification

**Day 3-4: Signal Quality & Noise**
3. `12_image_signal_univariate/signal_quality_metrics.ipynb`
   - SNR, PSNR, RMS computation
   - Practice dataset: [ECG Heartbeat](https://www.kaggle.com/shayanfazeli/heartbeat)

4. `12_image_signal_univariate/noise_characterization.ipynb`
   - Noise variance estimation
   - Noise distribution identification

**Day 5-7: Signal Processing Project**
5. Combined image/signal analysis
6. Multi-channel analysis exercise
7. Portfolio documentation

---

### **Phase 8: Specialized Data Types (Weeks 14-15)**

#### **Week 14: Compositional & Circular Data**

**Day 1-3: Compositional Univariate**
1. `13_compositional_univariate/single_component_analysis.ipynb`
   - Understanding compositional constraints
   - Spurious correlation demonstration
   - Practice dataset: [Microbiome Data](https://qiita.ucsd.edu/)

2. `13_compositional_univariate/proportion_transformation.ipynb`
   - ALR, CLR, ILR transformation implementation
   - Back-transformation and interpretation

3. `13_compositional_univariate/compositional_variability.ipynb`
   - Aitchison distance and total variance
   - Variation matrix construction

**Day 4-5: Circular Statistics**
4. `14_circular_directional_univariate/circular_descriptive_statistics.ipynb`
   - Circular mean and variance computation
   - Mean resultant length interpretation
   - Practice dataset: [Wind Direction](https://www.kaggle.com/datasets/sanjay3105/wind-directions-data)

5. `14_circular_directional_univariate/directional_distribution_fitting.ipynb`
   - Von Mises distribution fitting
   - Kuiper's goodness-of-fit test

**Day 6-7: Circular Inference**
6. `14_circular_directional_univariate/circular_hypothesis_testing.ipynb`
   - Rayleigh and V-test implementation
   - Watson's U² test

7. `14_circular_directional_univariate/temporal_angle_analysis.ipynb`
   - Time-of-day circular analysis
   - Seasonal circular patterns

---

#### **Week 15: High-Dimensional & Functional Data**

**Day 1-3: High-Dimensional Marginal**
1. `15_high_dimensional_marginal/marginal_screening_methods.ipynb`
   - Sure Independence Screening implementation
   - Distance correlation screening
   - Practice dataset: [Gene Expression](https://www.kaggle.com/crawford/gene-expression)

2. `15_high_dimensional_marginal/univariate_feature_selection.ipynb`
   - ANOVA F-test, mutual information
   - Fisher score computation

3. `15_high_dimensional_marginal/multiple_testing_marginal.ipynb`
   - Bonferroni, Holm, BH implementation
   - q-value estimation

**Day 4-5: Functional Data**
4. `16_functional_data_univariate/functional_descriptive_statistics.ipynb`
   - Pointwise mean and variance functions
   - Basis expansion with B-splines
   - Practice dataset: [Growth Curves](https://CRAN.R-project.org/package=fda)

5. `16_functional_data_univariate/functional_depth_measures.ipynb`
   - Modified band depth implementation
   - Functional boxplot construction

**Day 6-7: Functional Advanced**
6. `16_functional_data_univariate/curve_registration.ipynb`
   - Landmark registration basics
   - Dynamic time warping

7. `16_functional_data_univariate/functional_pca_univariate.ipynb`
   - FPCA implementation and interpretation
   - Mode of variation plots

---

### **Phase 9: Bayesian Methods & Synthesis (Week 16)**

#### **Week 16: Bayesian Hierarchical & Integration**

**Day 1-3: Bayesian Hierarchical Univariate**
1. `17_bayesian_hierarchical_univariate/hierarchical_means_and_variances.ipynb`
   - Random intercepts model with PyMC
   - ICC computation and interpretation
   - Practice dataset: [Student Performance](https://www.kaggle.com/uciml/student-alcohol-consumption)

2. `17_bayesian_hierarchical_univariate/shrinkage_estimation.ipynb`
   - James-Stein estimation
   - Empirical Bayes implementation

3. `17_bayesian_hierarchical_univariate/random_effects_univariate.ipynb`
   - BLUP computation
   - Prediction intervals for groups

**Day 4-5: Prior Sensitivity**
4. `17_bayesian_hierarchical_univariate/prior_sensitivity_analysis.ipynb`
   - Prior predictive checks
   - Variance prior comparison (Half-Cauchy vs Inverse-Gamma)

**Day 6-7: Integration and Portfolio**
5. Cross-domain integration project
   - Combine techniques across problem types
6. Portfolio completion and documentation
7. Self-assessment and gap identification

---

## **📋 TOPICS BY TECHNIQUE - LEARNING CURRICULUM**

### **1. TIME SERIES UNIVARIATE ANALYSIS**

#### **1.1 Decomposition Methods**
**What It Is:** Breaking down time series into trend, seasonality, and residual components for better understanding and modeling.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Classical Decomposition** | Additive (Y=T+S+R) vs multiplicative (Y=T×S×R) | Implement both, choose appropriately |
| **Moving Average Smoothing** | Simple, weighted, exponential | Apply different windows, interpret results |
| **STL Decomposition** | Seasonal-Trend using LOESS | Tune parameters, extract components |
| **MSTL** | Multiple seasonal patterns | Handle daily, weekly, annual seasonality |
| **Residual Analysis** | Randomness testing of remainders | Apply Ljung-Box, interpret ACF |
| **Trend Significance** | Testing for significant trend | Mann-Kendall, Sen's slope |

#### **1.2 Stationarity Testing**
**What It Is:** Assessing whether statistical properties remain constant over time, crucial for many time series models.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Augmented Dickey-Fuller** | Unit root test (H₀: non-stationary) | Interpret test statistic and p-value |
| **KPSS Test** | Stationarity test (H₀: stationary) | Understand opposite null hypothesis |
| **Phillips-Perron Test** | Robust unit root test | Compare with ADF results |
| **Differencing** | Achieving stationarity via d-order differencing | Select appropriate order |
| **Seasonal Differencing** | Removing seasonal unit roots | Combine with regular differencing |
| **Structural Breaks** | Testing for parameter changes | Zivot-Andrews, Chow test |

#### **1.3 Spectral Analysis**
**What It Is:** Analyzing time series in the frequency domain to identify cyclical patterns.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Periodogram** | Raw spectral power estimate | Compute and interpret peaks |
| **Welch's Method** | Improved spectral estimation | Apply overlapping segments |
| **Dominant Frequency** | Primary cyclical component | Identify and interpret |
| **Bandwidth** | Frequency resolution | Balance resolution and variance |
| **Spectral Density** | Continuous power distribution | Fit smooth spectral curve |

---

### **2. GEOSPATIAL UNIVARIATE ANALYSIS**

#### **2.1 Spatial Distribution Statistics**
**What It Is:** Measures of central tendency and dispersion for geographic point patterns.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Mean Center** | Geographic center of mass | Compute weighted and unweighted |
| **Median Center** | Robust central location | Iterative computation |
| **Standard Distance** | Spatial standard deviation | Interpret as circular extent |
| **Directional Distribution** | Standard deviational ellipse | Extract orientation and axes |
| **Convex Hull** | Minimum bounding polygon | Compute and analyze area |
| **Kernel Density** | Continuous density surface | Select bandwidth, interpret |

#### **2.2 Spatial Autocorrelation**
**What It Is:** Testing and measuring spatial dependency—whether nearby locations have similar values.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Spatial Weight Matrix** | Define neighbor relationships | Construct queen, rook, distance-based |
| **Global Moran's I** | Overall spatial autocorrelation | Interpret I, z-score, p-value |
| **Geary's C** | Alternative autocorrelation (0-2 scale) | Compare with Moran's I |
| **Local Moran's I (LISA)** | Cluster/outlier identification | Create cluster maps |
| **Getis-Ord G** | Hot/cold spot concentration | Interpret high/low clustering |
| **Permutation Inference** | Significance testing | Run Monte Carlo simulations |

---

### **3. SURVIVAL/RELIABILITY ANALYSIS**

#### **3.1 Survival Curve Estimation**
**What It Is:** Estimating the probability of surviving past time t, accounting for censored observations.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Kaplan-Meier Estimator** | Product-limit survival estimate | Compute S(t) step function |
| **Greenwood's Variance** | Standard error for KM | Compute confidence bands |
| **Median Survival** | 50th percentile of survival | Estimate with CI |
| **Restricted Mean** | Area under survival curve | Interpret as average survival |
| **Life Table Method** | Actuarial grouped estimation | Apply to interval data |
| **Nelson-Aalen** | Cumulative hazard H(t) | Compare with KM |

#### **3.2 Hazard Function Analysis**
**What It Is:** The instantaneous risk of event occurring at time t, given survival to t.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Hazard Rate** | h(t) = f(t)/S(t) | Interpret instantaneous risk |
| **Cumulative Hazard** | H(t) = ∫h(u)du | Nelson-Aalen estimation |
| **Hazard Shapes** | Increasing, decreasing, bathtub | Identify from data |
| **Parametric Hazards** | Weibull, exponential, Gompertz | Fit and compare |
| **Kernel Smoothing** | Non-parametric hazard | Apply bandwidth selection |

---

### **4. TEXT/NLP UNIVARIATE**

#### **4.1 Token Frequency Analysis**
**What It Is:** Analyzing the distribution of words/tokens in documents.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Term Frequency** | Raw and normalized word counts | Compute TF matrix |
| **Zipf's Law** | Frequency ∝ 1/rank | Validate power-law relationship |
| **Heaps' Law** | Vocabulary growth V(n) ∝ n^β | Estimate β parameter |
| **N-gram Frequency** | Bigram, trigram distributions | Analyze common phrases |
| **Hapax Legomena** | Single-occurrence words | Count and interpret |
| **Stop Word Ratio** | Function word proportion | Assess text composition |

#### **4.2 Vocabulary Richness**
**What It Is:** Measuring lexical diversity—how varied the vocabulary is.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Type-Token Ratio** | Unique words / total words | Recognize length dependence |
| **MTLD** | Measure of Textual Lexical Diversity | Implement forward/backward |
| **VOCD-D** | Vocabulary diversity statistic | Fit TTR curve |
| **Yule's K** | Vocabulary characteristic | Interpret richness |
| **Honore's R** | Hapax-based richness | Calculate and compare |
| **Simpson's D** | Vocabulary concentration | Connect to diversity theory |

---

### **5. COMPOSITIONAL UNIVARIATE**

#### **5.1 Log-Ratio Transformations**
**What It Is:** Transforming parts-of-whole data to enable standard statistical analysis.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Simplex Constraint** | Data sums to constant | Recognize compositional structure |
| **ALR Transform** | Log-ratio to reference component | Transform and back-transform |
| **CLR Transform** | Log-ratio to geometric mean | Handle zeros appropriately |
| **ILR Transform** | Orthonormal basis coordinates | Construct and interpret |
| **Zero Replacement** | Multiplicative/Bayesian methods | Apply appropriate strategies |
| **Reference Selection** | Choosing ALR denominator | Use stability criteria |

#### **5.2 Compositional Statistics**
**What It Is:** Proper descriptive statistics for compositional data.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Aitchison Distance** | Proper compositional metric | Calculate between samples |
| **Center (Barycenter)** | Closed geometric mean | Compute compositional center |
| **Total Variance** | Sum of CLR variances | Measure overall spread |
| **Variation Matrix** | Pairwise log-ratio variances | Construct and interpret |
| **Proportionality** | Ratio-based association | Calculate phi and rho |

---

### **6. CIRCULAR STATISTICS**

#### **6.1 Circular Descriptive Statistics**
**What It Is:** Summary statistics for data measured on a circle (angles, directions, times).

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Circular Mean** | Mean direction θ̄ = atan2(S, C) | Handle wraparound correctly |
| **Mean Resultant Length** | R̄ ∈ [0,1] concentration | Interpret dispersion |
| **Circular Variance** | V = 1 - R̄ | Connect to linear variance |
| **Circular SD** | σ = √(-2 ln R̄) | Alternative dispersion |
| **Median Direction** | Robust central direction | Compute and compare |
| **Circular Range** | Angular extent | Account for gaps |

#### **6.2 Circular Inference**
**What It Is:** Hypothesis testing and distribution fitting for circular data.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Rayleigh Test** | H₀: uniform vs unimodal | Interpret z-statistic |
| **V-Test** | Test against specified direction | Directional alternative |
| **Von Mises Fit** | "Circular normal" estimation | MLE for μ and κ |
| **Kuiper's Test** | Circular KS alternative | Compare distributions |
| **Watson's U²** | Uniformity test | Alternative to Rayleigh |

---

### **7. HIGH-DIMENSIONAL MARGINAL**

#### **7.1 Marginal Screening**
**What It Is:** Univariate analysis to pre-filter features in high-dimensional settings.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **SIS** | Sure Independence Screening | Rank by marginal correlation |
| **Distance Correlation** | Non-linear marginal association | Detect non-linear effects |
| **MI Screening** | Mutual information ranking | Apply to mixed data |
| **Variance Filter** | Remove low-variance features | Set appropriate threshold |
| **Iterative SIS** | Multiple screening rounds | Handle correlations |

#### **7.2 Multiple Testing**
**What It Is:** Controlling error rates when performing many univariate tests.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **FWER** | Family-wise error rate control | Understand P(≥1 false positive) |
| **Bonferroni** | Divide α by m | Apply and recognize conservatism |
| **Holm-Bonferroni** | Step-down sequential | Implement rejection order |
| **FDR** | False discovery rate | Understand E(FP/discoveries) |
| **Benjamini-Hochberg** | Step-up FDR procedure | Compute q-values |
| **q-value** | Storey's π₀ estimation | Use empirical null |

---

### **8. FUNCTIONAL DATA**

#### **8.1 Functional Descriptive Statistics**
**What It Is:** Summary measures when each observation is a curve or function.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Pointwise Mean** | μ(t) average at each t | Compute and plot |
| **Pointwise Variance** | σ²(t) variability at each t | Create variance bands |
| **Functional Median** | Depth-based central curve | Apply band depth |
| **Covariance Function** | C(s,t) surface | Estimate and visualize |
| **Basis Expansion** | B-spline, Fourier representation | Fit smooth functions |

#### **8.2 Functional Depth**
**What It Is:** Measuring centrality and outlyingness for curves.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Band Depth** | Proportion inside bands | Compute for all curves |
| **Modified Band Depth** | Simplified computation | Implement efficiently |
| **Functional Boxplot** | Median, envelope, outliers | Create and interpret |
| **Outlier Detection** | Low-depth curves | Identify anomalous curves |
| **Depth-based Ranking** | Order curves by centrality | Use for robust statistics |

---

### **9. BAYESIAN HIERARCHICAL**

#### **9.1 Hierarchical Modeling**
**What It Is:** Modeling grouped data with random group effects.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Random Intercepts** | Group-specific means | Fit with PyMC/Stan |
| **Variance Components** | Between vs within group | Estimate and interpret |
| **ICC** | ρ = σ²_b/(σ²_b + σ²_w) | Compute and interpret |
| **Partial Pooling** | Shrinkage toward grand mean | Visualize shrinkage |
| **Group Prediction** | BLUP for new groups | Generate predictions |

#### **9.2 Shrinkage Estimation**
**What It Is:** Borrowing strength across groups to improve estimates.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **James-Stein** | Simultaneous shrinkage | Implement for means |
| **Empirical Bayes** | Estimate hyperparameters | Two-stage approach |
| **Full Bayes** | Prior on hyperparameters | Full posterior sampling |
| **Shrinkage Visualization** | Individual → pooled | Create shrinkage plots |
| **MSE Comparison** | Shrinkage benefit | Show improvement |

---

## **📊 RECOMMENDED PRACTICE DATASETS BY DOMAIN**

| Domain | Dataset | Techniques Applicable |
|--------|---------|----------------------|
| **Time Series** | [Store Sales](https://www.kaggle.com/c/store-sales-time-series-forecasting) | Decomposition, stationarity |
| **Time Series** | [Air Quality](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set) | Spectral, change points |
| **Geospatial** | [NYC Taxi](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) | Spatial distribution, KDE |
| **Geospatial** | [Earthquake](https://www.kaggle.com/usgs/earthquake-database) | Spatial autocorrelation, hotspots |
| **Survival** | [Telco Churn](https://www.kaggle.com/blastchar/telco-customer-churn) | Kaplan-Meier, hazard |
| **Survival** | [Heart Failure](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) | Survival curves |
| **Text/NLP** | [IMDB Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | Token frequency, richness |
| **Text/NLP** | [News Category](https://www.kaggle.com/rmisra/news-category-dataset) | Vocabulary metrics |
| **Image/Signal** | [CIFAR-10](https://www.kaggle.com/c/cifar-10) | Intensity histograms |
| **Image/Signal** | [ECG Heartbeat](https://www.kaggle.com/shayanfazeli/heartbeat) | Spectral, SNR |
| **Compositional** | [Microbiome](https://qiita.ucsd.edu/) | Log-ratios, compositional stats |
| **Circular** | [Wind Direction](https://www.kaggle.com/datasets/sanjay3105/wind-directions-data) | Circular statistics |
| **High-Dim** | [Gene Expression](https://www.kaggle.com/crawford/gene-expression) | Screening, FDR |
| **Functional** | [Growth Curves](https://CRAN.R-project.org/package=fda) | FPCA, functional boxplot |
| **Hierarchical** | [Student Performance](https://www.kaggle.com/uciml/student-alcohol-consumption) | Random effects, shrinkage |

---

## **🔧 ESSENTIAL PYTHON LIBRARIES**

### **By Domain**

```python
# Time Series Univariate
import statsmodels.tsa.api as tsa
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import ruptures  # Change point detection
from scipy.signal import periodogram, welch

# Geospatial Univariate
import geopandas as gpd
from libpysal import weights
from esda import Moran, Geary, getisord
from pointpats import centrography, PointPattern

# Survival/Reliability
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
import reliability

# Text/NLP Univariate
from collections import Counter
import nltk
from lexicalrichness import LexicalRichness
from textstat import textstat

# Image/Signal
import cv2
from scipy.signal import welch, spectrogram
from scipy.fft import fft, fftfreq
import pywt  # Wavelets

# Compositional
from skbio.stats.composition import clr, ilr, closure
import compositions

# Circular Statistics
import pycircstat
from astropy.stats import circmean, circvar
from scipy.stats import vonmises

# High-Dimensional Marginal
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    f_classif, chi2
)
from statsmodels.stats.multitest import multipletests
import knockpy

# Functional Data
import skfda
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.preprocessing.smoothing import BasisSmoother

# Bayesian Hierarchical
import pymc as pm
import bambi as bmb
import arviz as az
```

---

## **📈 SKILL PROGRESSION CHECKLIST**

### **Foundational Level (After Phase 6)**
- [ ] Can decompose time series into trend, seasonality, residuals
- [ ] Can test for stationarity using ADF and KPSS
- [ ] Can compute spatial distribution statistics (mean center, standard distance)
- [ ] Can estimate Kaplan-Meier survival curves with confidence intervals
- [ ] Can calculate hazard rates and interpret shapes

### **Intermediate Level (After Phase 7-8)**
- [ ] Can perform spectral analysis and identify dominant frequencies
- [ ] Can compute spatial autocorrelation (Moran's I, Getis-Ord)
- [ ] Can analyze token frequency and vocabulary richness
- [ ] Can compute circular mean, variance, and fit von Mises
- [ ] Can apply log-ratio transformations for compositional data
- [ ] Can calculate image intensity statistics and SNR

### **Advanced Level (After Phase 9)**
- [ ] Can implement marginal screening for high-dimensional data
- [ ] Can apply FDR control for multiple testing
- [ ] Can compute functional depth and create functional boxplots
- [ ] Can fit hierarchical random effects models
- [ ] Can perform shrinkage estimation with empirical Bayes
- [ ] Can integrate multiple domain techniques for complex problems

---

## **🎯 EXPECTED OUTCOMES**

After completing this extended roadmap, you will be able to:

1. **Identify** the appropriate univariate analysis technique for any data type
2. **Decompose** time series into interpretable components
3. **Analyze** spatial patterns and autocorrelation in geographic data
4. **Estimate** survival curves and hazard functions from censored data
5. **Quantify** vocabulary richness and token distributions in text
6. **Handle** compositional data with proper log-ratio transformations
7. **Apply** circular statistics for directional and cyclic data
8. **Perform** marginal screening with proper multiple testing control
9. **Analyze** functional data with depth measures and FPCA
10. **Fit** hierarchical models with appropriate shrinkage

**Total Extended Timeline:** 8 additional weeks (Weeks 9-16)
**Combined with Original:** 16 weeks for complete univariate analysis mastery

---

## **📋 FINAL INTEGRATION PROJECT**

### **Capstone: Multi-Domain Univariate Analysis**

**Objective:** Apply techniques from at least 4 different extended domains to a real-world dataset.

**Example Project:** E-commerce Platform Analysis
1. **Time Series**: Analyze daily order volume (decomposition, stationarity)
2. **Geospatial**: Customer location density and hotspots
3. **Text/NLP**: Product review vocabulary analysis
4. **Survival**: Customer churn time-to-event analysis
5. **High-Dimensional**: Feature screening for prediction

**Deliverables:**
- Jupyter notebook with all analyses
- Executive summary of findings
- Method selection justification
- Recommendations for business action

---

*Last Updated: December 27, 2025*
*Purpose: Learning roadmap for comprehensive univariate analysis across all ML problem types*

