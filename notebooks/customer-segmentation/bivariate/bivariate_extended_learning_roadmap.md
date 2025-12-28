# Bivariate Analysis - Extended Techniques Learning Roadmap

## 📚 **Learning Guide for Extended Bivariate Analysis Techniques**

This document provides a comprehensive learning and practice roadmap for the bivariate analysis techniques identified as **missing from the Customer Segmentation focused proposal**. These techniques are essential for handling diverse ML problem types beyond simple tabular customer data.

**Prerequisites:** Completion of the original bivariate analysis roadmap (Weeks 1-7).

---

## **📅 IMPLEMENTATION ROADMAP (CONTINUATION)**

### **Phase 5: Specialized Data Domains (Weeks 8-10)**

#### **Week 8: Geospatial & Survival Analysis**

**Day 1-2: Geospatial Bivariate Fundamentals**
1. `11_geospatial_bivariate/spatial_autocorrelation.ipynb`
   - Moran's I and Geary's C implementation
   - Spatial weight matrix construction
   - Practice dataset: [NYC Taxi Fares](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

2. `11_geospatial_bivariate/spatial_regression.ipynb`
   - Spatial lag and error models
   - Geographically Weighted Regression basics

**Day 3-4: Spatial Advanced & Point Patterns**
3. `11_geospatial_bivariate/point_pattern_analysis.ipynb`
   - Ripley's K and L functions
   - Practice dataset: [Earthquake Data](https://www.kaggle.com/usgs/earthquake-database)

4. `11_geospatial_bivariate/distance_based_analysis.ipynb`
   - Kriging and spatial interpolation
   - Distance decay modeling

**Day 5-7: Survival Analysis Bivariate**
5. `12_survival_event_analysis/survival_curve_comparisons.ipynb`
   - Kaplan-Meier curves and log-rank test
   - Practice dataset: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

6. `12_survival_event_analysis/cox_proportional_hazards.ipynb`
   - Cox PH modeling and hazard ratios
   - Proportional hazards assumption testing

7. `12_survival_event_analysis/competing_risks.ipynb`
   - Cumulative incidence functions
   - Fine-Gray regression basics

---

#### **Week 9: Text/NLP & Network Analysis**

**Day 1-3: Text-Variable Relationships**
1. `13_text_nlp_bivariate/text_numeric_correlations.ipynb`
   - TF-IDF correlation with outcomes
   - Practice dataset: [IMDB Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

2. `13_text_nlp_bivariate/document_similarity.ipynb`
   - Cosine similarity, Word Mover's Distance
   - Practice dataset: [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset)

3. `13_text_nlp_bivariate/topic_correlations.ipynb`
   - LDA topic-variable relationships
   - Topic coherence analysis

**Day 4-5: Network Bivariate Analysis**
4. `14_network_graph_bivariate/node_attribute_correlations.ipynb`
   - Assortativity and homophily
   - Practice dataset: [Social Network Analysis](https://www.kaggle.com/mdecrevoisier/social-network-analysis)

5. `14_network_graph_bivariate/edge_formation_analysis.ipynb`
   - Link prediction features
   - Preferential attachment analysis

**Day 6-7: Network Advanced**
6. `14_network_graph_bivariate/network_assortativity.ipynb`
   - Degree and attribute assortativity
   - Rich-club coefficient

7. `14_network_graph_bivariate/bipartite_network_analysis.ipynb`
   - Bipartite projections
   - Affiliation network analysis

---

#### **Week 10: Image & Panel Data Analysis**

**Day 1-3: Image Bivariate Methods**
1. `15_image_bivariate/structural_similarity.ipynb`
   - SSIM and PSNR implementation
   - Practice dataset: [CIFAR-10](https://www.kaggle.com/c/cifar-10)

2. `15_image_bivariate/feature_map_correlations.ipynb`
   - CNN layer activation analysis
   - Gram matrix for style correlation

3. `15_image_bivariate/cross_image_analysis.ipynb`
   - Image retrieval and clustering
   - Multi-view correspondence

**Day 4-7: Panel/Longitudinal Data**
4. `16_panel_longitudinal_bivariate/within_between_correlations.ipynb`
   - Decomposing panel correlations
   - Practice dataset: [World Bank Indicators](https://www.kaggle.com/worldbank/world-development-indicators)

5. `16_panel_longitudinal_bivariate/intraclass_correlation.ipynb`
   - ICC variants and interpretation
   - Design effect calculations

6. `16_panel_longitudinal_bivariate/repeated_measures_analysis.ipynb`
   - Repeated measures ANOVA
   - Sphericity corrections

7. `16_panel_longitudinal_bivariate/mixed_effects_bivariate.ipynb`
   - Random slopes and intercepts
   - Growth curve modeling

---

### **Phase 6: Advanced Specialized Methods (Weeks 11-13)**

#### **Week 11: High-Dimensional & Compositional Data**

**Day 1-3: High-Dimensional Bivariate**
1. `17_high_dimensional_bivariate/sparse_correlation_estimation.ipynb`
   - Graphical LASSO implementation
   - Practice dataset: [Gene Expression](https://www.kaggle.com/crawford/gene-expression)

2. `17_high_dimensional_bivariate/regularized_covariance.ipynb`
   - Ledoit-Wolf shrinkage
   - Minimum covariance determinant

3. `17_high_dimensional_bivariate/multiple_testing_correction.ipynb`
   - Bonferroni, Holm, BH procedures
   - Practice: GWAS-style analysis

**Day 4-5: False Discovery Control**
4. `17_high_dimensional_bivariate/false_discovery_control.ipynb`
   - Local FDR estimation
   - Knockoff filter introduction

**Day 6-7: Compositional Data**
5. `18_compositional_data_bivariate/log_ratio_analysis.ipynb`
   - CLR, ILR, ALR transformations
   - Practice dataset: [Microbiome Data](https://www.kaggle.com/datasets/headsortails/microbiome-human-gut)

6. `18_compositional_data_bivariate/compositional_correlation.ipynb`
   - Proportionality measures
   - SparCC for microbiome

7. `18_compositional_data_bivariate/simplex_geometry.ipynb`
   - Ternary plots and simplex operations

---

#### **Week 12: Circular/Directional & Signal Processing**

**Day 1-3: Circular Statistics**
1. `19_circular_directional_bivariate/circular_correlation.ipynb`
   - Circular-circular correlation
   - Practice dataset: [Wind Direction Data](https://www.kaggle.com/datasets/sanjay3105/wind-directions-data)

2. `19_circular_directional_bivariate/angular_linear_correlation.ipynb`
   - Circular-linear relationships
   - Harmonic regression

3. `19_circular_directional_bivariate/directional_statistics.ipynb`
   - Rayleigh and Watson tests
   - Von Mises distribution fitting

**Day 4-7: Signal Processing Bivariate**
4. `20_signal_processing_bivariate/cross_spectral_analysis.ipynb`
   - Cross-spectral density estimation
   - Practice dataset: [ECG Heartbeat](https://www.kaggle.com/shayanfazeli/heartbeat)

5. `20_signal_processing_bivariate/coherence_analysis.ipynb`
   - Magnitude-squared coherence
   - Partial coherence implementation

6. `20_signal_processing_bivariate/phase_analysis.ipynb`
   - Phase locking value
   - Hilbert transform analysis

7. `20_signal_processing_bivariate/wavelet_coherence.ipynb`
   - Wavelet transform coherence
   - Time-frequency analysis

---

#### **Week 13: Causal Inference & Specialized Types**

**Day 1-4: Causal Inference Methods**
1. `21_causal_inference_bivariate/propensity_score_matching.ipynb`
   - PSM implementation and diagnostics
   - Practice dataset: [LaLonde Job Training](https://users.nber.org/~rdehejia/nswdata.html)

2. `21_causal_inference_bivariate/instrumental_variables.ipynb`
   - 2SLS regression
   - Weak instrument testing

3. `21_causal_inference_bivariate/regression_discontinuity.ipynb`
   - Sharp and fuzzy RD designs
   - Bandwidth selection

4. `21_causal_inference_bivariate/difference_in_differences.ipynb`
   - Classic DiD and staggered adoption
   - Parallel trends validation

**Day 5-7: Specialized Data Types**
5. `22_specialized_data_types/count_data_bivariate.ipynb`
   - Poisson and negative binomial regression
   - Zero-inflated models

6. `22_specialized_data_types/ordinal_advanced_analysis.ipynb`
   - Polychoric/polyserial correlation
   - Ordinal logistic regression

7. `22_specialized_data_types/missing_data_bivariate.ipynb`
   - MCAR testing
   - Multiple imputation impact

---

### **Phase 7: Bayesian Methods & Integration (Week 14)**

#### **Week 14: Bayesian Bivariate & Synthesis**

**Day 1-3: Bayesian Correlation Methods**
1. `23_bayesian_bivariate/bayesian_correlation.ipynb`
   - Posterior correlation estimation
   - PyMC implementation

2. `23_bayesian_bivariate/credible_intervals.ipynb`
   - HDI and ETI construction
   - Posterior predictive checks

3. `23_bayesian_bivariate/bayes_factors.ipynb`
   - JZS Bayes factor for correlation
   - Prior sensitivity analysis

**Day 4-5: Hierarchical Relationships**
4. `23_bayesian_bivariate/hierarchical_relationships.ipynb`
   - Varying slopes and intercepts
   - Cross-level interactions

**Day 6-7: Integration and Review**
5. Cross-domain integration project
   - Combine techniques across problem types
   - Portfolio completion

---

## **📋 TOPICS BY TECHNIQUE - LEARNING CURRICULUM**

### **1. GEOSPATIAL BIVARIATE ANALYSIS**

#### **1.1 Spatial Autocorrelation**
**What It Is:** Measures the degree to which nearby observations are correlated, violating the independence assumption of standard statistics.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Spatial Weight Matrices** | Define neighbor relationships (queen, rook, k-nearest, distance bands) | Construct W matrices for different spatial structures |
| **Moran's I Global** | Overall spatial autocorrelation measure (-1 to +1) | Interpret significance and implement from scratch |
| **Local Moran's I (LISA)** | Identify local clusters and outliers | Create LISA cluster maps |
| **Geary's C** | Alternative autocorrelation (0-2, 1=no correlation) | Compare with Moran's I for interpretation |
| **Getis-Ord G and G*** | Hot/cold spot detection | Implement and visualize hot spots |
| **Moran Scatterplot** | Visualize spatial autocorrelation | Interpret quadrants (HH, LL, HL, LH) |

#### **1.2 Spatial Regression**
**What It Is:** Regression models that account for spatial dependence in the data or errors.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **OLS Diagnostics** | Test for spatial dependence in OLS residuals | Run Lagrange multiplier tests |
| **Spatial Lag Model (SLM)** | Spatially lagged dependent variable (Wy) | Estimate and interpret spatial lag coefficient (ρ) |
| **Spatial Error Model (SEM)** | Spatially correlated errors (λ) | Distinguish from SLM using LM tests |
| **GWR (Geographically Weighted Regression)** | Locally varying regression coefficients | Implement and map local R² and coefficients |
| **Bandwidth Selection** | Optimal neighborhood size for GWR | Apply CV-based and AICc methods |
| **Spatial Durbin Model** | Combined lag and error effects | Know when to use vs SLM/SEM |

#### **1.3 Point Pattern Analysis**
**What It Is:** Analysis of the spatial distribution of events/locations to detect clustering, dispersion, or randomness.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Complete Spatial Randomness (CSR)** | Null hypothesis of random distribution | Generate CSR simulations for comparison |
| **Ripley's K-function** | Multi-scale clustering detection | Interpret K(r) above/below CSR envelope |
| **L-function** | Variance-stabilized K-function | Use L(r)-r for easier interpretation |
| **G-function** | Nearest neighbor distribution | Detect clustering via G(r) |
| **F-function** | Empty space function | Complement to G-function |
| **Cross-K Function** | Bivariate point pattern relationship | Analyze two types of points together |

---

### **2. SURVIVAL/EVENT ANALYSIS**

#### **2.1 Survival Curve Comparisons**
**What It Is:** Methods to compare time-to-event distributions across groups, handling censored observations.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Kaplan-Meier Estimator** | Non-parametric survival function estimation | Compute S(t) and confidence bands |
| **Censoring Types** | Right, left, interval censoring | Handle each type appropriately |
| **Log-Rank Test** | Compare survival curves (χ² based) | Test with 2+ groups, interpret p-value |
| **Gehan-Wilcoxon Test** | Early difference emphasis | Know when to prefer over log-rank |
| **Peto-Peto Test** | Alternative weighting scheme | Compare with other tests |
| **Stratified Log-Rank** | Control for confounders | Apply stratification correctly |

#### **2.2 Cox Proportional Hazards**
**What It Is:** Semi-parametric regression relating covariates to hazard rate.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Hazard Function** | Instantaneous risk of event | Interpret h(t) and hazard ratios |
| **Cox PH Model** | Baseline hazard × exp(βX) | Fit model, interpret coefficients |
| **Partial Likelihood** | Estimation without baseline hazard | Understand why PH works without h₀(t) |
| **PH Assumption Testing** | Schoenfeld residuals | Test and diagnose violations |
| **Time-Varying Covariates** | Covariates changing over follow-up | Implement extended Cox model |
| **Concordance Index (C-index)** | Discrimination measure | Calculate and interpret (0.5-1) |

#### **2.3 Competing Risks**
**What It Is:** Analysis when multiple event types can occur, and one prevents the other.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Cumulative Incidence Function (CIF)** | Probability of specific event by time t | Calculate CIF for each event type |
| **Cause-Specific Hazard** | Traditional approach treating other events as censored | Fit and interpret CSH models |
| **Fine-Gray Model** | Subdistribution hazard regression | Implement and compare with CSH |
| **Gray's Test** | Compare CIF across groups | Apply and interpret |
| **Multi-State Models** | Complex event transitions | Model state progressions |

---

### **3. TEXT/NLP BIVARIATE ANALYSIS**

#### **3.1 Text-Numeric Correlations**
**What It Is:** Relating text features (sentiment, topics, entities) to numerical outcomes.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **TF-IDF Vectors** | Term frequency-inverse document frequency | Create sparse document vectors |
| **TF-IDF Correlation** | Word importance vs outcomes | Identify predictive words |
| **Sentiment Scores** | Polarity and subjectivity | Correlate with numerical targets |
| **Readability Metrics** | Flesch-Kincaid, Gunning Fog | Relate complexity to engagement |
| **Text Length Effects** | Word/character counts | Analyze length-outcome relationships |
| **Named Entity Correlations** | Entity counts by type | Extract and correlate NER features |

#### **3.2 Document Similarity**
**What It Is:** Measuring how similar two documents are using various metrics.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Cosine Similarity** | Angle between document vectors | Implement for TF-IDF and embeddings |
| **Jaccard Similarity** | Set overlap of words | Compare with cosine for different use cases |
| **Word Mover's Distance (WMD)** | Semantic distance using word vectors | Implement with pre-trained embeddings |
| **Soft Cosine Similarity** | Incorporate word similarity in cosine | Use word2vec/GloVe for word relationships |
| **BM25** | Probabilistic retrieval scoring | Implement for document ranking |
| **Sentence Embeddings** | Dense vector representations (BERT, etc.) | Use transformer-based similarity |

#### **3.3 Topic Modeling Correlations**
**What It Is:** Relating latent topics to other variables.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **LDA (Latent Dirichlet Allocation)** | Generative topic model | Fit LDA and extract topic distributions |
| **Topic-Variable Correlation** | Topics vs numerical features | Correlate topic proportions with outcomes |
| **Topic Coherence** | Internal topic quality | Compute c_v, u_mass coherence |
| **Topic Diversity** | Inter-topic distinctiveness | Measure and optimize |
| **Dynamic Topics** | Topic evolution over time | Track topic trends |
| **Hierarchical Topics** | Nested topic structure | Model with hLDA |

---

### **4. NETWORK/GRAPH BIVARIATE ANALYSIS**

#### **4.1 Node Attribute Correlations**
**What It Is:** Analyzing relationships between node properties and network position.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Attribute Assortativity** | Correlation of connected nodes' attributes | Calculate for continuous and categorical |
| **Homophily** | Like connects to like | Measure and test for homophily |
| **Centrality-Attribute Correlation** | Network importance vs properties | Correlate degree, betweenness with attributes |
| **Community-Attribute Relationships** | Group characteristics | Analyze attributes by community |
| **Influence Spread** | Neighbor effects on attributes | Model cascade and contagion |

#### **4.2 Edge Formation Analysis**
**What It Is:** Understanding what drives connection formation in networks.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Preferential Attachment** | Rich get richer (degree-based) | Measure and test PA mechanism |
| **Triadic Closure** | Friends of friends become friends | Calculate clustering coefficient |
| **Common Neighbors** | Shared connections predict links | Implement link prediction score |
| **Adamic-Adar Index** | Weighted common neighbors | Compare with simple CN |
| **Jaccard Coefficient** | Normalized neighbor overlap | Implement for link prediction |
| **Resource Allocation** | Inverse degree weighting | Compare prediction methods |

#### **4.3 Assortativity**
**What It Is:** The tendency of nodes to connect with similar (or dissimilar) nodes.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Degree Assortativity** | Do high-degree connect to high-degree? | Calculate r coefficient (-1 to +1) |
| **Numeric Assortativity** | Continuous attribute mixing | Generalize beyond degree |
| **Categorical Assortativity** | Discrete attribute mixing | Handle nominal categories |
| **Rich-Club Coefficient** | Elite node connectivity | Identify and measure rich-club |
| **Disassortativity** | Opposite nodes connecting | Interpret negative assortativity |

---

### **5. PANEL/LONGITUDINAL BIVARIATE**

#### **5.1 Within-Between Decomposition**
**What It Is:** Separating correlations that occur within vs across subjects/units.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Within-Subject Correlation** | Changes over time within individuals | Calculate using de-meaned data |
| **Between-Subject Correlation** | Cross-sectional at individual means | Use subject-level averages |
| **Simpson's Paradox** | When within ≠ between | Recognize and handle reversals |
| **Contextual Effects** | Group mean effects on individual outcomes | Test contextual vs compositional |
| **Hauser Model** | Decomposition framework | Implement full decomposition |

#### **5.2 Intraclass Correlation (ICC)**
**What It Is:** The proportion of variance attributable to groups/clusters.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **ICC(1)** | One-way random effects | Measure group-level variance share |
| **ICC(2)** | Two-way random effects | Include rater/time effects |
| **ICC(3)** | Two-way mixed effects | Fixed raters, random subjects |
| **Agreement vs Consistency** | Absolute vs relative consistency | Choose appropriate ICC form |
| **Design Effect** | Sample size adjustment for clustering | Calculate DEFF = 1 + (n-1)×ICC |
| **Reliability Coefficient** | Cronbach's alpha relationship | Connect ICC to scale reliability |

#### **5.3 Mixed Effects Models**
**What It Is:** Models with both fixed (population) and random (subject-specific) effects.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Random Intercepts** | Subject-specific baselines | Fit and interpret variance components |
| **Random Slopes** | Subject-specific relationships | Model heterogeneous effects |
| **Cross-Level Interactions** | Level 2 moderates Level 1 | Test and interpret interactions |
| **Growth Curve Models** | Time trends with random effects | Model linear and nonlinear growth |
| **Model Comparison** | Likelihood ratio tests, AIC, BIC | Choose between nesting models |
| **Centering** | Grand-mean vs group-mean | Apply appropriate centering strategy |

---

### **6. HIGH-DIMENSIONAL BIVARIATE**

#### **6.1 Sparse Correlation Estimation**
**What It Is:** Estimating correlation/covariance when p >> n or when true structure is sparse.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Graphical LASSO** | L1-penalized precision matrix | Fit and interpret sparse inverse covariance |
| **CLIME Estimator** | Alternative sparse precision | Compare with graphical LASSO |
| **Thresholding Methods** | Hard/soft threshold sample correlation | Apply and tune thresholds |
| **Stability Selection** | Robustness of selected edges | Implement subsampling approach |
| **Cross-Validation** | Tuning regularization | Select optimal λ |

#### **6.2 Multiple Testing Correction**
**What It Is:** Controlling error rates when testing many hypotheses simultaneously.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **FWER (Family-Wise Error Rate)** | P(≥1 false positive) control | Understand when to control FWER |
| **Bonferroni Correction** | Divide α by m | Apply and recognize conservativeness |
| **Holm-Bonferroni** | Step-down procedure | Implement sequential rejection |
| **FDR (False Discovery Rate)** | Expected proportion of false discoveries | Prefer for exploratory analysis |
| **Benjamini-Hochberg** | Step-up FDR procedure | Apply and interpret q-values |
| **Storey's q-value** | Estimating π₀ for FDR | Use empirical null estimation |

---

### **7. COMPOSITIONAL DATA**

#### **7.1 Log-Ratio Transformations**
**What It Is:** Transforming parts-of-whole data to enable standard statistical analysis.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Simplex Constraint** | Data sums to constant | Recognize compositional structure |
| **ALR (Additive Log-Ratio)** | Log-ratio to reference component | Transform and back-transform |
| **CLR (Centered Log-Ratio)** | Log-ratio to geometric mean | Handle zeros, interpret in CLR space |
| **ILR (Isometric Log-Ratio)** | Orthonormal coordinates | Create orthonormal basis |
| **Zero Handling** | Multiplicative replacement | Apply Bayesian or geometric approaches |

#### **7.2 Compositional Correlation**
**What It Is:** Proper correlation measures for compositional data.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Spurious Correlation** | Why Pearson fails on compositions | Demonstrate with examples |
| **Proportionality** | Ratio-based association | Calculate phi and rho |
| **Variation Matrix** | Pairwise log-ratio variance | Construct and interpret |
| **SparCC** | Sparse correlations for microbiome | Implement for high-dimensional compositions |
| **SECOM** | Spearman-based compositional correlation | Compare with SparCC |

---

### **8. CIRCULAR/DIRECTIONAL DATA**

#### **8.1 Circular Statistics Fundamentals**
**What It Is:** Statistics for data measured on a circle (angles, directions, times of day).

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Circular Mean** | Mean direction accounting for wraparound | Calculate using trigonometric approach |
| **Circular Variance** | Spread around the circle (0 to 1) | Interpret R̄ (mean resultant length) |
| **Von Mises Distribution** | "Normal" for circular data | Fit and estimate κ concentration |
| **Circular-Circular Correlation** | Two angle relationship | Calculate r_cc coefficient |
| **Circular-Linear Correlation** | Angle vs linear variable | Calculate r_cl coefficient |
| **Rayleigh Test** | Test for uniformity vs unimodal | Apply and interpret |

---

### **9. SIGNAL PROCESSING BIVARIATE**

#### **9.1 Spectral Analysis**
**What It Is:** Frequency-domain analysis of relationships between signals.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Power Spectral Density (PSD)** | Power at each frequency | Estimate using Welch's method |
| **Cross-Spectral Density (CSD)** | Shared frequency content | Calculate and interpret |
| **Coherence** | Frequency-resolved correlation | Compute magnitude-squared coherence |
| **Phase Spectrum** | Phase relationship at each frequency | Extract and interpret phase |
| **Bandwidth Selection** | Frequency resolution vs variance | Balance using segment length |

#### **9.2 Time-Frequency Analysis**
**What It Is:** Analyzing how frequency content changes over time.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Short-Time Fourier Transform (STFT)** | Windowed frequency analysis | Create and interpret spectrograms |
| **Wavelet Transform** | Multi-resolution analysis | Apply continuous and discrete wavelets |
| **Wavelet Coherence** | Time-frequency coherence | Identify transient relationships |
| **Phase-Amplitude Coupling** | High-freq amplitude modulated by low-freq phase | Compute PAC metrics |

---

### **10. CAUSAL INFERENCE BIVARIATE**

#### **10.1 Propensity Score Methods**
**What It Is:** Using treatment probability to control for confounding.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Propensity Score Estimation** | Model treatment assignment | Fit logistic/GBM for PS |
| **Matching Methods** | 1:1, k:1, caliper matching | Implement nearest neighbor matching |
| **Balance Diagnostics** | Standardized mean differences | Check SMD < 0.1 after matching |
| **ATE vs ATT** | Population vs treated effect | Estimate and interpret both |
| **Sensitivity Analysis** | Unobserved confounding bounds | Apply Rosenbaum bounds |

#### **10.2 Instrumental Variables**
**What It Is:** Using external variation to identify causal effects with endogeneity.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Instrument Requirements** | Relevance and exclusion | Evaluate instrument validity |
| **Two-Stage Least Squares (2SLS)** | IV regression method | Implement and interpret |
| **Weak Instruments** | First-stage F < 10 problem | Test with Stock-Yogo thresholds |
| **Overidentification Tests** | Sargan-Hansen test | Apply with multiple instruments |
| **LATE Interpretation** | Local average treatment effect | Understand complier effects |

#### **10.3 Regression Discontinuity**
**What It Is:** Exploiting treatment thresholds for causal identification.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Sharp RD** | Treatment at exact cutoff | Implement with local polynomial |
| **Fuzzy RD** | Probabilistic treatment at cutoff | Combine with IV estimation |
| **Bandwidth Selection** | Optimal window around cutoff | Apply MSE-optimal methods |
| **Continuity Assumption** | Potential outcomes continuous at cutoff | Check and validate |
| **Placebo Tests** | Falsification at fake cutoffs | Demonstrate robustness |

---

### **11. BAYESIAN BIVARIATE**

#### **11.1 Bayesian Correlation**
**What It Is:** Probabilistic estimation of correlations with uncertainty quantification.

**Topics to Master:**
| Topic | Description | Completion Criteria |
|-------|-------------|---------------------|
| **Posterior Distribution** | Full correlation uncertainty | Sample from posterior |
| **Prior Selection** | Jeffreys, uniform, informative | Compare prior impacts |
| **Credible Intervals** | HDI vs ETI | Compute and interpret |
| **Bayes Factor** | Evidence for correlation ≠ 0 | Calculate JZS Bayes factor |
| **ROPE Analysis** | Region of practical equivalence | Test practical significance |
| **Posterior Predictive Checks** | Model validation | Generate and compare to data |

---

## **📊 RECOMMENDED PRACTICE DATASETS BY DOMAIN**

| Domain | Dataset | Techniques Applicable |
|--------|---------|----------------------|
| **Geospatial** | [NYC Taxi Fares](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) | Spatial regression, distance decay |
| **Geospatial** | [Airbnb Listings](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) | Spatial autocorrelation, GWR |
| **Survival** | [Telco Churn](https://www.kaggle.com/blastchar/telco-customer-churn) | Kaplan-Meier, Cox PH |
| **Survival** | [Heart Failure Prediction](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) | Survival curves, hazard ratios |
| **Text/NLP** | [IMDB Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | Text-numeric correlation |
| **Text/NLP** | [News Category](https://www.kaggle.com/rmisra/news-category-dataset) | Topic modeling, document similarity |
| **Network** | [Social Networks](https://snap.stanford.edu/data/) | Assortativity, link prediction |
| **Network** | [Bitcoin Transaction](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | Node correlation, edge formation |
| **Image** | [CIFAR-10](https://www.kaggle.com/c/cifar-10) | Feature map correlation, SSIM |
| **Panel** | [World Development Indicators](https://www.kaggle.com/worldbank/world-development-indicators) | ICC, mixed effects |
| **High-Dim** | [Gene Expression](https://www.kaggle.com/crawford/gene-expression) | Graphical LASSO, FDR |
| **Compositional** | [Microbiome Data](https://qiita.ucsd.edu/) | Log-ratios, SparCC |
| **Circular** | [Wind Direction](https://www.kaggle.com/datasets/sanjay3105/wind-directions-data) | Circular statistics |
| **Signal** | [ECG Heartbeat](https://www.kaggle.com/shayanfazeli/heartbeat) | Coherence, spectral analysis |
| **Causal** | [LaLonde Job Training](https://users.nber.org/~rdehejia/nswdata.html) | PSM, DiD |

---

## **🔧 ESSENTIAL PYTHON LIBRARIES**

### **By Domain**

```python
# Geospatial
libpysal, esda, spreg, mgwr, geopandas, pointpats, pykrige

# Survival Analysis
lifelines, scikit-survival, cmprsk

# Text/NLP
scikit-learn (TfidfVectorizer), gensim, spacy, sentence-transformers, textblob

# Network Analysis
networkx, igraph, cdlib, node2vec

# Image Analysis
scikit-image, opencv-python, lpips, torchvision

# Panel/Longitudinal
linearmodels, pymer4, statsmodels

# High-Dimensional
sklearn.covariance, knockpy

# Compositional
skbio, propr, sparcc3

# Circular Statistics
pycircstat, astropy.stats

# Signal Processing
scipy.signal, mne, pywt, nitime

# Causal Inference
causalinference, econml, dowhy, rdrobust

# Bayesian
pymc, arviz, bambi, pingouin
```

---

## **📈 SKILL PROGRESSION CHECKLIST**

### **Foundational Level (After Phase 5)**
- [ ] Can perform spatial autocorrelation analysis
- [ ] Can compare survival curves and fit Cox PH models
- [ ] Can correlate text features with numerical outcomes
- [ ] Can calculate network assortativity measures
- [ ] Can analyze repeated measures data with ICC

### **Intermediate Level (After Phase 6)**
- [ ] Can implement graphical LASSO for sparse correlations
- [ ] Can apply FDR correction for multiple testing
- [ ] Can transform and analyze compositional data
- [ ] Can compute circular statistics and correlations
- [ ] Can perform spectral and coherence analysis

### **Advanced Level (After Phase 7)**
- [ ] Can implement propensity score matching
- [ ] Can design and analyze regression discontinuity
- [ ] Can apply Bayesian correlation estimation
- [ ] Can combine techniques across domains
- [ ] Can select appropriate method for novel problem types

---

## **🎯 EXPECTED OUTCOMES**

After completing this extended roadmap, you will be able to:

1. **Identify** the appropriate bivariate analysis technique for any data type
2. **Implement** specialized methods for geospatial, survival, text, network, and image data
3. **Apply** proper correction methods for high-dimensional data
4. **Handle** specialized data types (compositional, circular, panel)
5. **Estimate** causal effects using quasi-experimental methods
6. **Quantify** uncertainty using Bayesian approaches
7. **Integrate** multiple techniques for complex, multi-domain problems

**Total Extended Timeline:** 7 additional weeks (Weeks 8-14)
**Combined with Original:** 14 weeks for complete bivariate analysis mastery

---

*Last Updated: December 27, 2025*
*Purpose: Learning roadmap for comprehensive bivariate analysis across all ML problem types*

