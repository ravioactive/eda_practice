# Bivariate Analysis - Extended Techniques for Other Problem Types

## 📊 **Overview: Missing Techniques in Customer Segmentation Context**

The original bivariate analysis proposal was tailored specifically for the **Customer Segmentation problem** from Kaggle. This document identifies **bivariate analysis techniques that were excluded or underemphasized** because they are less applicable to customer segmentation but are **critical for other common ML problem types**.

**Problem Types Addressed:**
- Time Series Analysis (extended beyond basic)
- Geospatial/Geographic Data
- Survival/Event Analysis
- Text/NLP Data
- Network/Graph Data
- Image Data
- Panel/Longitudinal Data
- High-Dimensional Data
- Compositional Data
- Circular/Directional Data
- Signal Processing
- Sequence/String Data

---

## **📁 EXTENDED BIVARIATE FOLDER STRUCTURE**

```
bivariate_extended/
├── 11_geospatial_bivariate/
│   ├── spatial_autocorrelation.ipynb
│   ├── spatial_regression.ipynb
│   ├── point_pattern_analysis.ipynb
│   └── distance_based_analysis.ipynb
│
├── 12_survival_event_analysis/
│   ├── survival_curve_comparisons.ipynb
│   ├── cox_proportional_hazards.ipynb
│   ├── competing_risks.ipynb
│   └── time_to_event_correlations.ipynb
│
├── 13_text_nlp_bivariate/
│   ├── text_numeric_correlations.ipynb
│   ├── document_similarity.ipynb
│   ├── topic_correlations.ipynb
│   └── embedding_relationships.ipynb
│
├── 14_network_graph_bivariate/
│   ├── node_attribute_correlations.ipynb
│   ├── edge_formation_analysis.ipynb
│   ├── network_assortativity.ipynb
│   └── bipartite_network_analysis.ipynb
│
├── 15_image_bivariate/
│   ├── feature_map_correlations.ipynb
│   ├── structural_similarity.ipynb
│   ├── cross_image_analysis.ipynb
│   └── pixel_intensity_relationships.ipynb
│
├── 16_panel_longitudinal_bivariate/
│   ├── within_between_correlations.ipynb
│   ├── intraclass_correlation.ipynb
│   ├── repeated_measures_analysis.ipynb
│   └── mixed_effects_bivariate.ipynb
│
├── 17_high_dimensional_bivariate/
│   ├── sparse_correlation_estimation.ipynb
│   ├── regularized_covariance.ipynb
│   ├── multiple_testing_correction.ipynb
│   └── false_discovery_control.ipynb
│
├── 18_compositional_data_bivariate/
│   ├── log_ratio_analysis.ipynb
│   ├── aitchison_distance.ipynb
│   ├── compositional_correlation.ipynb
│   └── simplex_geometry.ipynb
│
├── 19_circular_directional_bivariate/
│   ├── circular_correlation.ipynb
│   ├── angular_linear_correlation.ipynb
│   ├── directional_statistics.ipynb
│   └── circular_regression.ipynb
│
├── 20_signal_processing_bivariate/
│   ├── cross_spectral_analysis.ipynb
│   ├── coherence_analysis.ipynb
│   ├── phase_analysis.ipynb
│   └── wavelet_coherence.ipynb
│
├── 21_causal_inference_bivariate/
│   ├── propensity_score_matching.ipynb
│   ├── instrumental_variables.ipynb
│   ├── regression_discontinuity.ipynb
│   └── difference_in_differences.ipynb
│
├── 22_specialized_data_types/
│   ├── count_data_bivariate.ipynb
│   ├── ordinal_advanced_analysis.ipynb
│   ├── sequence_string_analysis.ipynb
│   └── missing_data_bivariate.ipynb
│
└── 23_bayesian_bivariate/
    ├── bayesian_correlation.ipynb
    ├── credible_intervals.ipynb
    ├── bayes_factors.ipynb
    └── hierarchical_relationships.ipynb
```

---

## **📋 DETAILED NOTEBOOK CONTENT SPECIFICATIONS**

### **11_geospatial_bivariate/ - Spatial Relationships**

**Why Missing from Customer Segmentation:** Customer segmentation dataset lacks geographic coordinates.

**Applicable Problem Types:** Location-based services, real estate, epidemiology, environmental science, transportation.

#### **spatial_autocorrelation.ipynb**
**Primary Focus**: Measuring spatial dependence between variables
- **Moran's I**: Global spatial autocorrelation measure
- **Local Moran's I (LISA)**: Local indicators of spatial association
- **Geary's C**: Alternative autocorrelation measure
- **Spatial lag analysis**: Neighbors' influence on values
- **Spatial weight matrices**: Queen, rook, distance-based contiguity

```python
# Key libraries and concepts
from libpysal.weights import Queen, KNN, DistanceBand
from esda.moran import Moran, Moran_Local
from esda.geary import Geary
import geopandas as gpd
```

#### **spatial_regression.ipynb**
**Primary Focus**: Modeling relationships with spatial dependence
- **Ordinary Least Squares (OLS) baseline**: Standard regression for comparison
- **Spatial Lag Model (SLM)**: Spatially lagged dependent variable
- **Spatial Error Model (SEM)**: Spatially correlated errors
- **Geographically Weighted Regression (GWR)**: Locally varying relationships
- **Model comparison**: Lagrange multiplier tests, AIC/BIC

```python
from spreg import OLS, GM_Lag, GM_Error
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
```

#### **point_pattern_analysis.ipynb**
**Primary Focus**: Spatial distribution of events
- **Ripley's K-function**: Multi-scale clustering analysis
- **L-function**: Normalized K-function
- **G-function**: Nearest neighbor distribution
- **Cross K-function**: Bivariate point patterns
- **Complete spatial randomness (CSR)**: Null hypothesis testing

```python
from pointpats import PointPattern, G, K, F, L
from pointpats.centrography import mean_center, std_distance
```

#### **distance_based_analysis.ipynb**
**Primary Focus**: Distance decay and proximity effects
- **Distance decay functions**: Inverse distance, exponential decay
- **Buffer analysis**: Variable relationships within distance bands
- **Spatial interpolation**: Kriging, IDW for continuous surfaces
- **Variogram analysis**: Spatial continuity modeling
- **Accessibility measures**: Network-based proximity

```python
from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging
import osmnx as ox  # For network distances
```

---

### **12_survival_event_analysis/ - Time-to-Event Relationships**

**Why Missing from Customer Segmentation:** No time-to-event or censoring in the dataset.

**Applicable Problem Types:** Medical outcomes, customer churn, equipment failure, subscription cancellation, loan defaults.

#### **survival_curve_comparisons.ipynb**
**Primary Focus**: Comparing survival functions across groups
- **Kaplan-Meier estimator**: Non-parametric survival curves
- **Log-rank test**: Comparing two or more survival curves
- **Gehan-Wilcoxon test**: Weight early observations more
- **Peto-Peto test**: Alternative log-rank weighting
- **Stratified log-rank test**: Controlling for confounders

```python
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes
```

#### **cox_proportional_hazards.ipynb**
**Primary Focus**: Semi-parametric survival regression
- **Cox PH model basics**: Hazard ratios, baseline hazard
- **Proportional hazards assumption**: Schoenfeld residuals test
- **Time-varying covariates**: Extended Cox models
- **Stratified Cox models**: Handling non-proportional hazards
- **Model diagnostics**: Concordance index, deviance residuals

```python
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
```

#### **competing_risks.ipynb**
**Primary Focus**: Multiple possible event types
- **Cumulative incidence functions**: Event-specific probabilities
- **Fine-Gray regression**: Subdistribution hazards
- **Cause-specific hazards**: Traditional approach
- **Gray's test**: Comparing cumulative incidence curves
- **Multi-state models**: Complex event transitions

```python
from lifelines import AalenJohansenFitter
from cmprsk import crr  # Competing risks regression
```

#### **time_to_event_correlations.ipynb**
**Primary Focus**: Correlating survival times and covariates
- **Accelerated failure time (AFT) models**: Parametric survival
- **Frailty models**: Random effects in survival
- **Concordance measures**: C-statistic, Somers' D for survival
- **Time-dependent AUC**: Discrimination over time
- **Brier score**: Calibration for survival predictions

```python
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
from sksurv.metrics import brier_score, concordance_index_censored
```

---

### **13_text_nlp_bivariate/ - Text-Variable Relationships**

**Why Missing from Customer Segmentation:** Dataset contains no text fields.

**Applicable Problem Types:** Sentiment analysis, topic modeling, document classification, review analysis, chatbot training.

#### **text_numeric_correlations.ipynb**
**Primary Focus**: Relating text features to numerical outcomes
- **TF-IDF correlation**: Word importance vs target variable
- **Sentiment score correlations**: Sentiment vs numerical outcomes
- **Readability correlations**: Flesch-Kincaid vs engagement metrics
- **Text length correlations**: Character/word counts vs outcomes
- **Named entity correlations**: Entity counts vs variables

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import textstat
import spacy
```

#### **document_similarity.ipynb**
**Primary Focus**: Measuring similarity between documents
- **Cosine similarity**: TF-IDF vector comparison
- **Jaccard similarity**: Set-based similarity
- **Soft cosine similarity**: Semantic word relationships
- **Word Mover's Distance**: Semantic document distance
- **BM25 similarity**: Information retrieval scoring

```python
from sklearn.metrics.pairwise import cosine_similarity
from gensim.similarities import SoftCosineSimilarity
from gensim.similarities import WmdSimilarity
```

#### **topic_correlations.ipynb**
**Primary Focus**: Relating topics to variables
- **Topic-variable correlation**: LDA topics vs numerical features
- **Topic co-occurrence**: Document topic correlations
- **Dynamic topic correlation**: Topic evolution over time
- **Topic coherence**: Internal topic consistency
- **Topic diversity**: Inter-topic distinctiveness

```python
from gensim.models import LdaModel, CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
```

#### **embedding_relationships.ipynb**
**Primary Focus**: Dense vector representation analysis
- **Word embedding correlations**: Word2Vec, GloVe similarity
- **Sentence embedding similarity**: BERT, Sentence-BERT
- **Embedding space analysis**: t-SNE, UMAP visualization
- **Cross-lingual embeddings**: Multilingual relationship analysis
- **Contextual embedding correlations**: Attention-based relationships

```python
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import transformers
```

---

### **14_network_graph_bivariate/ - Network Relationships**

**Why Missing from Customer Segmentation:** No network/relationship data between customers.

**Applicable Problem Types:** Social networks, fraud detection, recommendation systems, biological networks, supply chain.

#### **node_attribute_correlations.ipynb**
**Primary Focus**: Relating node properties
- **Attribute assortativity**: Correlation of connected nodes' attributes
- **Attribute-centrality correlation**: Properties vs network position
- **Homophily analysis**: Similar nodes connecting
- **Attribute influence**: Neighbors' attributes affecting node
- **Community-attribute relationships**: Group characteristics

```python
import networkx as nx
from networkx.algorithms.assortativity import attribute_assortativity_coefficient
from cdlib import algorithms as community_algorithms
```

#### **edge_formation_analysis.ipynb**
**Primary Focus**: Understanding what drives connections
- **Preferential attachment**: Degree-based edge formation
- **Triadic closure**: Triangle formation analysis
- **Link prediction features**: Common neighbors, Adamic-Adar, etc.
- **Temporal edge formation**: Dynamic network analysis
- **Edge weight analysis**: Connection strength factors

```python
from networkx.algorithms.link_prediction import (
    common_neighbors, adamic_adar_index, jaccard_coefficient
)
from networkx.algorithms.triads import triadic_census
```

#### **network_assortativity.ipynb**
**Primary Focus**: Mixing patterns in networks
- **Degree assortativity**: High-degree nodes connecting
- **Scalar assortativity**: Continuous attribute mixing
- **Categorical assortativity**: Discrete attribute mixing
- **Assortativity by layer**: Multiplex network analysis
- **Rich-club coefficient**: Elite node connectivity

```python
nx.degree_assortativity_coefficient(G)
nx.numeric_assortativity_coefficient(G, 'attribute')
nx.attribute_assortativity_coefficient(G, 'category')
```

#### **bipartite_network_analysis.ipynb**
**Primary Focus**: Two-type node networks
- **Bipartite projection**: One-mode network creation
- **Bipartite centrality**: Node importance in bipartite graphs
- **Bipartite clustering**: Community detection
- **Affiliation networks**: Group membership analysis
- **Bipartite link prediction**: Cross-type edge prediction

```python
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite import (
    projected_graph, closeness_centrality
)
```

---

### **15_image_bivariate/ - Image-Based Relationships**

**Why Missing from Customer Segmentation:** No image data in the dataset.

**Applicable Problem Types:** Medical imaging, satellite imagery, product photos, facial recognition, quality inspection.

#### **feature_map_correlations.ipynb**
**Primary Focus**: CNN feature relationships
- **Layer activation correlations**: Feature map dependencies
- **Gram matrix analysis**: Style/texture correlations (neural style transfer)
- **Attention map correlations**: Transformer attention patterns
- **Feature redundancy**: Highly correlated features in CNNs
- **Cross-layer correlations**: Feature evolution through network

```python
import torch
from torchvision import models
from captum.attr import LayerActivation
```

#### **structural_similarity.ipynb**
**Primary Focus**: Image comparison metrics
- **SSIM (Structural Similarity Index)**: Perceptual similarity
- **PSNR (Peak Signal-to-Noise Ratio)**: Reconstruction quality
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Deep perceptual
- **Feature-based matching**: SIFT, SURF, ORB correspondence
- **Histogram correlation**: Color/intensity distribution comparison

```python
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips  # Learned perceptual metric
import cv2
```

#### **cross_image_analysis.ipynb**
**Primary Focus**: Relationships between images
- **Image retrieval correlation**: Query-result similarity
- **Multi-view correlation**: Same object from different views
- **Temporal image correlation**: Video frame relationships
- **Cross-domain correspondence**: Paired image domains
- **Image clustering relationships**: Grouping similar images

```python
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import imagehash
```

#### **pixel_intensity_relationships.ipynb**
**Primary Focus**: Pixel-level analysis
- **Spatial cross-correlation**: Neighboring pixel relationships
- **Multi-channel correlation**: RGB, HSV channel dependencies
- **Texture correlation**: GLCM-based texture features
- **Edge correlation**: Gradient-based feature relationships
- **Frequency domain correlation**: FFT-based analysis

```python
from skimage.feature import graycomatrix, graycoprops
import scipy.fft as fft
import scipy.ndimage as ndimage
```

---

### **16_panel_longitudinal_bivariate/ - Repeated Measures Relationships**

**Why Missing from Customer Segmentation:** Cross-sectional data, no repeated observations per customer.

**Applicable Problem Types:** Clinical trials, economic panels, educational assessments, survey research, A/B testing with time.

#### **within_between_correlations.ipynb**
**Primary Focus**: Decomposing correlations in panel data
- **Within-subject correlation**: Individual-level over time
- **Between-subject correlation**: Cross-individual at same time
- **Total correlation decomposition**: Combined effects
- **Random effects correlation**: Variance component analysis
- **Fixed effects residual correlation**: After controlling for individuals

```python
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects
import pingouin as pg
```

#### **intraclass_correlation.ipynb**
**Primary Focus**: Reliability and clustering effects
- **ICC(1)**: One-way random effects
- **ICC(2)**: Two-way random effects
- **ICC(3)**: Two-way mixed effects
- **Agreement vs consistency**: ICC variants
- **Design effects**: Cluster-based sampling adjustments

```python
from pingouin import intraclass_corr
from statsmodels.stats.inter_rater import fleiss_kappa
import scipy.stats as stats
```

#### **repeated_measures_analysis.ipynb**
**Primary Focus**: Within-subject comparisons
- **Paired t-test**: Two time points
- **Repeated measures ANOVA**: Multiple time points
- **Friedman test**: Non-parametric repeated measures
- **Sphericity correction**: Greenhouse-Geisser, Huynh-Feldt
- **Contrast analysis**: Linear, quadratic trends

```python
from scipy.stats import friedmanchisquare, wilcoxon
import pingouin as pg
from statsmodels.stats.anova import AnovaRM
```

#### **mixed_effects_bivariate.ipynb**
**Primary Focus**: Random and fixed effect combinations
- **Random intercepts**: Subject-specific baselines
- **Random slopes**: Subject-specific relationships
- **Cross-level interactions**: Level 1 × Level 2 effects
- **Growth curve modeling**: Time trend estimation
- **Model comparison**: Likelihood ratio tests, ICC-based

```python
import statsmodels.formula.api as smf
from pymer4.models import Lmer
import lme4  # R interface through rpy2
```

---

### **17_high_dimensional_bivariate/ - Many-Variable Relationships**

**Why Missing from Customer Segmentation:** Only 4 features in the dataset.

**Applicable Problem Types:** Genomics, finance (many assets), sensor networks, recommendation systems, feature engineering.

#### **sparse_correlation_estimation.ipynb**
**Primary Focus**: Estimating correlations with sparsity
- **Graphical LASSO**: Sparse precision matrix estimation
- **CLIME estimator**: Constrained L1-minimization
- **Huge package methods**: High-dimensional undirected graph estimation
- **Thresholded sample correlation**: Hard/soft thresholding
- **Sparse canonical correlation**: High-dimensional CCA

```python
from sklearn.covariance import GraphicalLassoCV
from scipy.sparse import csr_matrix
from statsmodels.multivariate.cancorr import CanCorr
```

#### **regularized_covariance.ipynb**
**Primary Focus**: Robust high-dimensional covariance
- **Ledoit-Wolf shrinkage**: Optimal shrinkage intensity
- **Oracle approximating shrinkage (OAS)**: Alternative shrinkage
- **Minimum covariance determinant**: Robust estimation
- **Factor model covariance**: Low-rank approximation
- **Block diagonal estimation**: Structured covariance

```python
from sklearn.covariance import LedoitWolf, OAS, MinCovDet
from sklearn.decomposition import FactorAnalysis
```

#### **multiple_testing_correction.ipynb**
**Primary Focus**: Controlling errors in many tests
- **Bonferroni correction**: Family-wise error rate control
- **Holm-Bonferroni**: Stepwise Bonferroni
- **Benjamini-Hochberg**: False discovery rate control
- **Storey's q-value**: FDR estimation
- **Permutation-based correction**: Resampling approaches

```python
from statsmodels.stats.multitest import multipletests
from scipy.stats import false_discovery_control
import qvalue  # Python port of Storey's q-value
```

#### **false_discovery_control.ipynb**
**Primary Focus**: FDR-specific methods
- **Local FDR (lfdr)**: Probability of false discovery per test
- **Empirical Bayes FDR**: Data-driven null distribution
- **FDR regression**: Covariate-adjusted FDR
- **Group FDR**: Structured multiple testing
- **Knockoff filter**: Model-X knockoffs for FDR

```python
from knockpy import KnockoffFilter
import locfdr  # Local FDR estimation
```

---

### **18_compositional_data_bivariate/ - Parts-of-Whole Relationships**

**Why Missing from Customer Segmentation:** No compositional (proportional) data like market shares or chemical compositions.

**Applicable Problem Types:** Microbiome, geochemistry, market shares, time allocation, budget allocation.

#### **log_ratio_analysis.ipynb**
**Primary Focus**: Compositional data transformations
- **Additive log-ratio (ALR)**: Simple log-ratio with reference
- **Centered log-ratio (CLR)**: Geometric mean reference
- **Isometric log-ratio (ILR)**: Orthonormal coordinates
- **Pairwise log-ratios**: All component pairs
- **Interpretation**: Back-transformation to compositions

```python
from skbio.stats.composition import clr, ilr, alr
import composition_stats as comp
```

#### **aitchison_distance.ipynb**
**Primary Focus**: Measuring compositional dissimilarity
- **Aitchison distance**: Proper compositional metric
- **Aitchison inner product**: Compositional correlation
- **Variation matrix**: Pairwise log-ratio variances
- **Procrustes analysis**: Compositional shape comparison
- **PERMANOVA for compositions**: Testing group differences

```python
from skbio.stats.distance import permanova
from scipy.spatial.distance import squareform
import scikit_posthocs as sp
```

#### **compositional_correlation.ipynb**
**Primary Focus**: Correlation in compositional data
- **Spurious correlation problem**: Constant-sum constraint
- **Proportionality**: Alternative to correlation
- **Phi statistic**: Log-ratio variance-based measure
- **Rho statistic**: Proportionality coefficient
- **SparCC**: Sparse correlations for microbiome

```python
from propr import propr  # Proportionality
import sparcc3  # SparCC implementation
```

#### **simplex_geometry.ipynb**
**Primary Focus**: Geometric perspective on compositions
- **Ternary plots**: 3-component visualization
- **Simplex coordinates**: Geometric representation
- **Perturbation**: Compositional addition
- **Power transformation**: Compositional scaling
- **Subcomposition coherence**: Subset analysis

```python
import ternary
import matplotlib.pyplot as plt
from skbio.stats.composition import closure
```

---

### **19_circular_directional_bivariate/ - Angular Data Relationships**

**Why Missing from Customer Segmentation:** No directional or periodic data (angles, time of day as circular, compass directions).

**Applicable Problem Types:** Wind direction, time-of-day patterns, compass data, astronomical observations, biological rhythms.

#### **circular_correlation.ipynb**
**Primary Focus**: Correlation between circular variables
- **Circular-circular correlation**: r_cc coefficient
- **T-linear association**: Alternative circular correlation
- **Fisher-Lee correlation**: Embedding-based measure
- **Circular rank correlation**: Non-parametric variant
- **Concentration parameter estimation**: Von Mises fitting

```python
from pycircstat import corrcc, corrcl
from scipy.stats import circmean, circstd
import astropy.stats as astats
```

#### **angular_linear_correlation.ipynb**
**Primary Focus**: Circular-linear relationships
- **Circular-linear correlation (r_cl)**: Angle vs linear variable
- **Mardia-Jupp coefficient**: Alternative measure
- **Point-biserial circular correlation**: Binary × circular
- **Regression on circular predictors**: Linear outcome from angle
- **Harmonic regression**: Sine/cosine decomposition

```python
from pycircstat import corrcl
import numpy as np
# Harmonic regression: y = a + b*cos(θ) + c*sin(θ)
```

#### **directional_statistics.ipynb**
**Primary Focus**: Statistical tests for directional data
- **Rayleigh test**: Testing uniformity vs unimodal
- **Watson's test**: Two-sample circular comparison
- **Kuiper's test**: Circular Kolmogorov-Smirnov
- **Rao's spacing test**: Uniformity based on gaps
- **ANOVA for circular data**: Watson-Williams test

```python
from pycircstat import rayleigh, watson_u2, kuiper
from astropy.stats import rayleightest, kuiper
```

#### **circular_regression.ipynb**
**Primary Focus**: Predicting circular outcomes
- **Circular-circular regression**: Angle → Angle
- **Linear-circular regression**: Linear → Angle
- **Von Mises regression**: GLM for circular data
- **Projected normal regression**: Alternative framework
- **Mixed predictor regression**: Combined linear and circular predictors

```python
import circumplex  # Circular statistics
from pycircstat import regression
```

---

### **20_signal_processing_bivariate/ - Frequency Domain Relationships**

**Why Missing from Customer Segmentation:** No signal/waveform data requiring frequency analysis.

**Applicable Problem Types:** Audio processing, ECG/EEG analysis, vibration analysis, financial time series, sensor data.

#### **cross_spectral_analysis.ipynb**
**Primary Focus**: Frequency-domain relationships
- **Cross-spectral density (CSD)**: Shared frequency components
- **Power spectral density (PSD)**: Individual frequency content
- **Cross-spectrum estimation**: Welch, periodogram methods
- **Frequency-specific correlation**: Band-limited analysis
- **Spectrogram cross-correlation**: Time-frequency analysis

```python
from scipy.signal import csd, welch, coherence, spectrogram
import mne  # For EEG/MEG analysis
```

#### **coherence_analysis.ipynb**
**Primary Focus**: Frequency-resolved correlation
- **Magnitude-squared coherence**: Frequency correlation strength
- **Partial coherence**: Controlling for third signal
- **Multiple coherence**: One signal vs multiple
- **Imaginary coherence**: Robust to volume conduction
- **Coherence significance testing**: Bootstrap methods

```python
from scipy.signal import coherence
from mne.connectivity import spectral_connectivity
import nitime  # Time series analysis for neuroscience
```

#### **phase_analysis.ipynb**
**Primary Focus**: Phase relationships between signals
- **Phase coherence**: Phase locking value (PLV)
- **Phase-amplitude coupling**: PAC analysis
- **Instantaneous phase**: Hilbert transform
- **Phase synchronization**: Phase-locking analysis
- **Phase difference distribution**: Circular statistics

```python
from scipy.signal import hilbert
import mne
from pactools import Comodulogram
```

#### **wavelet_coherence.ipynb**
**Primary Focus**: Time-frequency coherence analysis
- **Continuous wavelet transform (CWT)**: Time-frequency decomposition
- **Wavelet coherence**: Localized coherence in time-frequency
- **Cross-wavelet transform**: Phase relationship evolution
- **Wavelet phase coherence**: Time-resolved phase locking
- **Scale-dependent correlation**: Multi-resolution analysis

```python
import pywt
from waveletcoherence import WaveletCoherence
import scaleogram
```

---

### **21_causal_inference_bivariate/ - Causal Relationship Estimation**

**Why Missing from Customer Segmentation:** Observational data without experimental design or causal structure.

**Applicable Problem Types:** Treatment effect estimation, policy evaluation, econometrics, epidemiology, marketing attribution.

#### **propensity_score_matching.ipynb**
**Primary Focus**: Matching for causal inference
- **Propensity score estimation**: Logistic regression, GBM
- **Matching methods**: Nearest neighbor, caliper, optimal
- **Balance diagnostics**: Standardized mean differences
- **Average treatment effect (ATE)**: Population-level effect
- **Sensitivity analysis**: Rosenbaum bounds

```python
from causalinference import CausalModel
from sklearn.neighbors import NearestNeighbors
import dowhy
```

#### **instrumental_variables.ipynb**
**Primary Focus**: Handling endogeneity
- **Two-stage least squares (2SLS)**: IV regression
- **Instrument validity**: Relevance and exclusion
- **Weak instruments**: F-statistic, Stock-Yogo thresholds
- **Overidentification tests**: Sargan-Hansen test
- **Local average treatment effect (LATE)**: Complier effects

```python
from linearmodels.iv import IV2SLS
from statsmodels.sandbox.regression.gmm import IV2SLS
import econml
```

#### **regression_discontinuity.ipynb**
**Primary Focus**: Sharp and fuzzy RD designs
- **Sharp RD**: Treatment at cutoff
- **Fuzzy RD**: Probabilistic treatment at cutoff
- **Bandwidth selection**: Optimal window size
- **Local polynomial regression**: Near-cutoff estimation
- **Placebo tests**: Falsification checks

```python
from rdrobust import rdrobust, rdbwselect
import rdd  # Regression discontinuity package
```

#### **difference_in_differences.ipynb**
**Primary Focus**: Panel data causal inference
- **Classic DiD**: Two periods, two groups
- **Staggered adoption**: Multiple treatment timing
- **Parallel trends assumption**: Validation methods
- **Synthetic control**: Data-driven counterfactual
- **Event study design**: Dynamic treatment effects

```python
from difference_in_differences import did
from synth_control import SyntheticControl
import econml.dml
```

---

### **22_specialized_data_types/ - Unique Data Type Relationships**

#### **count_data_bivariate.ipynb**
**Primary Focus**: Discrete count variable relationships
- **Poisson regression bivariate**: Count vs predictors
- **Negative binomial regression**: Overdispersed counts
- **Zero-inflated models**: Excess zeros handling
- **Hurdle models**: Two-stage count process
- **Rate ratio analysis**: Incidence rate comparisons

```python
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson
```

#### **ordinal_advanced_analysis.ipynb**
**Primary Focus**: Ordered categorical relationships
- **Polychoric correlation**: Ordinal-ordinal correlation
- **Polyserial correlation**: Ordinal-continuous correlation
- **Ordinal logistic regression**: Proportional odds model
- **Continuation ratio models**: Sequential odds
- **Adjacent category models**: Local odds

```python
from semopy import polychoric_corr
from statsmodels.miscmodels.ordinal_model import OrderedModel
import factor_analyzer
```

#### **sequence_string_analysis.ipynb**
**Primary Focus**: String and sequence relationships
- **Edit distance**: Levenshtein, Damerau-Levenshtein
- **Sequence alignment**: Needleman-Wunsch, Smith-Waterman
- **N-gram similarity**: Character/word n-gram overlap
- **Longest common subsequence**: String similarity
- **Sequence pattern mining**: Common patterns across sequences

```python
from Levenshtein import distance, ratio
from Bio import pairwise2  # Sequence alignment
from difflib import SequenceMatcher
```

#### **missing_data_bivariate.ipynb**
**Primary Focus**: Relationships in incomplete data
- **Missing data mechanisms**: MCAR, MAR, MNAR testing
- **Little's MCAR test**: Overall MCAR assessment
- **Pattern analysis**: Missingness correlations
- **Imputation impact**: Sensitivity of correlations
- **Multiple imputation inference**: Rubin's rules

```python
from missingno import matrix, heatmap
from sklearn.impute import SimpleImputer, KNNImputer
import mice  # Multiple imputation
```

---

### **23_bayesian_bivariate/ - Probabilistic Relationship Estimation**

**Why Missing from Customer Segmentation:** Focus was on frequentist methods.

**Applicable Problem Types:** All domains where uncertainty quantification, prior knowledge incorporation, or small samples are important.

#### **bayesian_correlation.ipynb**
**Primary Focus**: Probabilistic correlation estimation
- **Posterior correlation distribution**: Full uncertainty
- **Jeffrey's prior**: Non-informative correlation prior
- **Beta prior on Fisher-z**: Informative priors
- **Robust Bayesian correlation**: Heavy-tailed alternatives
- **Correlation difference testing**: Bayesian comparison

```python
import pymc as pm
import arviz as az
from scipy.stats import pearsonr
```

#### **credible_intervals.ipynb**
**Primary Focus**: Bayesian interval estimation
- **Highest density intervals (HDI)**: Most probable values
- **Equal-tailed intervals (ETI)**: Symmetric quantiles
- **Prediction intervals**: Future observation uncertainty
- **Posterior predictive checks**: Model validation
- **Interval hypothesis testing**: ROPE analysis

```python
import arviz as az
from pymc import sample_posterior_predictive
```

#### **bayes_factors.ipynb**
**Primary Focus**: Bayesian hypothesis testing
- **Bayes factor calculation**: Evidence for H1 vs H0
- **JZS Bayes factor**: Default correlation test
- **Savage-Dickey method**: Nested model comparison
- **Interpretation guidelines**: Jeffreys' scale
- **Prior sensitivity**: Robustness of conclusions

```python
from pingouin import bayesfactor_pearson
import bayes_factor  # Custom implementations
```

#### **hierarchical_relationships.ipynb**
**Primary Focus**: Multilevel Bayesian modeling
- **Random effects correlation**: Group-level relationships
- **Varying slopes models**: Relationship heterogeneity
- **Partial pooling**: Shrinkage estimation
- **Cross-level interactions**: Moderating effects
- **Model comparison**: WAIC, LOO-CV

```python
import pymc as pm
import bambi as bmb
from arviz import compare
```

---

## **🎯 PROBLEM TYPE APPLICABILITY MATRIX**

| Folder | Time Series | Geospatial | Survival | NLP | Network | Image | Panel | High-Dim |
|--------|-------------|------------|----------|-----|---------|-------|-------|----------|
| 11_geospatial | ○ | ● | ○ | ○ | ◐ | ◐ | ◐ | ○ |
| 12_survival | ◐ | ○ | ● | ○ | ○ | ○ | ◐ | ○ |
| 13_text_nlp | ○ | ○ | ○ | ● | ◐ | ○ | ○ | ◐ |
| 14_network | ○ | ◐ | ○ | ◐ | ● | ○ | ○ | ◐ |
| 15_image | ○ | ◐ | ○ | ○ | ○ | ● | ○ | ◐ |
| 16_panel | ● | ◐ | ◐ | ○ | ○ | ○ | ● | ○ |
| 17_high_dim | ◐ | ○ | ◐ | ◐ | ◐ | ◐ | ○ | ● |
| 18_compositional | ○ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ |
| 19_circular | ● | ● | ○ | ○ | ○ | ○ | ○ | ○ |
| 20_signal | ● | ○ | ○ | ◐ | ○ | ◐ | ○ | ◐ |
| 21_causal | ◐ | ○ | ◐ | ○ | ○ | ○ | ● | ○ |
| 22_specialized | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |
| 23_bayesian | ● | ● | ● | ● | ● | ● | ● | ● |

**Legend:** ● Primary applicability | ◐ Secondary/moderate applicability | ○ Limited applicability

---

## **📊 SUMMARY OF EXTENDED COVERAGE**

| Category | Notebooks | Key Techniques | Primary Use Cases |
|----------|-----------|----------------|-------------------|
| **Geospatial** | 4 | Moran's I, GWR, Kriging | Location-based analytics |
| **Survival** | 4 | Kaplan-Meier, Cox PH, Competing risks | Medical, churn, reliability |
| **Text/NLP** | 4 | Document similarity, topic correlation | NLP, sentiment analysis |
| **Network** | 4 | Assortativity, link prediction | Social networks, fraud |
| **Image** | 4 | SSIM, feature maps, cross-image | Computer vision, medical imaging |
| **Panel/Longitudinal** | 4 | ICC, within/between, mixed effects | Clinical trials, surveys |
| **High-Dimensional** | 4 | Graphical LASSO, FDR control | Genomics, finance |
| **Compositional** | 4 | Log-ratios, Aitchison distance | Microbiome, geochemistry |
| **Circular/Directional** | 4 | Circular correlation, von Mises | Wind, time patterns |
| **Signal Processing** | 4 | Coherence, wavelet, phase | Audio, EEG, sensors |
| **Causal Inference** | 4 | PSM, IV, RDD, DiD | Policy, treatment effects |
| **Specialized Types** | 4 | Count, ordinal, sequence, missing | Domain-specific |
| **Bayesian** | 4 | Posterior, Bayes factor, hierarchical | Uncertainty quantification |

**Total Extended Notebooks:** 52 additional notebooks

**Combined with Original Proposal:** 40 + 52 = **92 bivariate analysis notebooks**

---

*Last Updated: December 27, 2025*
*Purpose: Comprehensive bivariate analysis coverage for all common ML problem types*

