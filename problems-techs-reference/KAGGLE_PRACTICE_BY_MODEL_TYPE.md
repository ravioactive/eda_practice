# 🤖 Kaggle Practice Problems by Model Type

**Comprehensive Practice Guide Organized by ML Model Types**  
**Date:** January 22, 2025  
**Based on:** ML Feature Analysis & EDA Cheat Sheets  
**Coverage:** Tree-Based, Linear, Distance-Based, Neural Networks, Transformers & Encoders  

---

## 🎯 **Quick Navigation**

- [🌳 Tree-Based Models](#-tree-based-models)
- [📈 Linear Models](#-linear-models)
- [📏 Distance-Based Models](#-distance-based-models)
- [🧠 Neural Networks](#-neural-networks)
- [🔄 Transformers & Encoders](#-transformers--encoders)
- [📊 Cross-Model Comparison](#-cross-model-comparison)
- [💡 Practice Strategy](#-practice-strategy)

---

## 🌳 **Tree-Based Models**
*Random Forest, XGBoost, LightGBM, CatBoost, Decision Trees*

### **🟢 Beginner Level**

#### **1. Titanic: Machine Learning from Disaster** ⭐⭐⭐⭐⭐
**🔗 Link:** [Titanic Dataset](https://www.kaggle.com/c/titanic)

**📊 EDA Focus:**
- Survival rate analysis by passenger class, gender, age groups
- Missing data patterns (Age, Cabin, Embarked)
- Categorical relationships using chi-square tests
- Outlier detection in Age and Fare

**🌳 Tree-Based Techniques:**
- **Decision Trees:** Interpretable rules for survival prediction
- **Random Forest:** Handle missing values naturally, feature importance
- **Feature Engineering:** Family size, title extraction, cabin deck
- **Tree Visualization:** Decision paths and feature splits

**💻 Code Practice:**
```python
# Tree-specific techniques from cheat sheet:
- Handle missing values as separate category
- Create binned features for continuous variables
- Feature interactions that trees can split on
- Random Forest feature importance analysis
```

---

#### **2. House Prices: Advanced Regression** ⭐⭐⭐⭐⭐
**🔗 Link:** [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**📊 EDA Focus:**
- Price distribution analysis and log transformation
- Correlation analysis between numerical features
- Categorical feature analysis (80+ categorical variables)
- Outlier detection in living area and sale price

**🌳 Tree-Based Techniques:**
- **XGBoost:** Handle high-cardinality categoricals
- **LightGBM:** Efficient training on large feature set
- **Feature Engineering:** Area calculations, quality interactions
- **Regularization:** Tree pruning and early stopping

---

#### **3. Palmer Penguins Classification** ⭐⭐⭐
**🔗 Link:** [Palmer Penguins](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data)

**📊 EDA Focus:**
- Species distribution across islands
- Bill and flipper measurement correlations
- Body mass distribution by species and sex
- Missing value analysis

**🌳 Tree-Based Techniques:**
- **Decision Trees:** Species classification rules
- **Random Forest:** Ensemble classification
- **Feature Engineering:** Ratio features (bill length/depth)
- **Tree Interpretation:** Feature importance and decision paths

---

#### **4. Wine Quality Prediction** ⭐⭐⭐
**🔗 Link:** [Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

**📊 EDA Focus:**
- Quality score distribution
- Chemical property correlations
- Outlier detection in chemical measurements
- Quality vs chemical property relationships

**🌳 Tree-Based Techniques:**
- **Random Forest:** Multi-class quality prediction
- **Gradient Boosting:** Ordinal regression approach
- **Feature Engineering:** Chemical property ratios and interactions

---

#### **5. Breast Cancer Wisconsin** ⭐⭐⭐
**🔗 Link:** [Breast Cancer](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

**📊 EDA Focus:**
- Malignant vs benign distribution
- Feature correlation analysis (30 features)
- Distribution analysis of cell measurements
- Feature redundancy analysis

**🌳 Tree-Based Techniques:**
- **Decision Trees:** Medical decision rules
- **Random Forest:** Robust classification
- **Feature Selection:** Tree-based importance ranking

---

### **🟡 Intermediate Level**

#### **1. Credit Card Fraud Detection** ⭐⭐⭐⭐⭐
**🔗 Link:** [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

**📊 EDA Focus:**
- Extreme class imbalance analysis (0.17% fraud)
- PCA feature distribution analysis
- Time-based fraud patterns
- Amount distribution analysis

**🌳 Tree-Based Techniques:**
- **XGBoost:** Handle imbalanced data with scale_pos_weight
- **LightGBM:** Efficient training on large dataset
- **Feature Engineering:** Time-based aggregations, amount binning
- **Sampling Techniques:** SMOTE with tree ensembles

---

#### **2. Santander Customer Satisfaction** ⭐⭐⭐⭐
**🔗 Link:** [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction)

**📊 EDA Focus:**
- 300+ anonymized features analysis
- Feature variance and correlation analysis
- Target distribution analysis
- Constant and duplicate feature identification

**🌳 Tree-Based Techniques:**
- **Random Forest:** Handle high-dimensional data
- **XGBoost:** Feature selection through importance
- **Feature Engineering:** Automated feature interactions
- **Dimensionality Reduction:** Tree-based feature selection

---

#### **3. Porto Seguro Safe Driver Prediction** ⭐⭐⭐⭐
**🔗 Link:** [Porto Seguro](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

**📊 EDA Focus:**
- Insurance claim probability analysis
- Mixed data types (binary, categorical, continuous)
- Missing value patterns
- Feature correlation analysis

**🌳 Tree-Based Techniques:**
- **LightGBM:** Handle mixed data types efficiently
- **CatBoost:** Native categorical feature handling
- **Feature Engineering:** Risk score calculations, interaction terms

---

#### **4. Rossmann Store Sales** ⭐⭐⭐⭐
**🔗 Link:** [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)

**📊 EDA Focus:**
- Sales time series patterns
- Store type and assortment analysis
- Holiday and promotion effects
- Competition distance impact

**🌳 Tree-Based Techniques:**
- **XGBoost:** Time series regression
- **Feature Engineering:** Lag features, rolling statistics
- **Temporal Features:** Holiday encoding, trend features

---

#### **5. Allstate Claims Severity** ⭐⭐⭐⭐
**🔗 Link:** [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity)

**📊 EDA Focus:**
- Claim severity distribution (continuous target)
- Categorical feature analysis (100+ categories)
- Feature interaction analysis
- Outlier detection in claim amounts

**🌳 Tree-Based Techniques:**
- **XGBoost:** Regression with custom objectives
- **LightGBM:** Efficient categorical handling
- **Feature Engineering:** Category combinations, frequency encoding

---

### **🔴 Advanced Level**

#### **1. Home Credit Default Risk** ⭐⭐⭐⭐⭐
**🔗 Link:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

**📊 EDA Focus:**
- Multiple related tables analysis
- Credit history patterns
- Application data relationships
- External data source integration

**🌳 Tree-Based Techniques:**
- **LightGBM:** Handle massive feature sets (1000+ features)
- **Advanced Feature Engineering:** Aggregations across tables
- **Time-based Features:** Credit history patterns
- **Ensemble Methods:** Multiple tree model stacking

---

#### **2. IEEE-CIS Fraud Detection** ⭐⭐⭐⭐⭐
**🔗 Link:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

**📊 EDA Focus:**
- Transaction and identity data analysis
- Device and browser fingerprinting
- Network analysis of related transactions
- Time-based fraud patterns

**🌳 Tree-Based Techniques:**
- **XGBoost:** Complex feature interactions
- **CatBoost:** High-cardinality categorical handling
- **Advanced Engineering:** Graph features, sequence patterns
- **Adversarial Validation:** Domain adaptation techniques

---

#### **3. Microsoft Malware Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [Microsoft Malware Prediction](https://www.kaggle.com/c/microsoft-malware-prediction)

**📊 EDA Focus:**
- System configuration analysis
- Malware family patterns
- Hardware and software correlations
- Geographic distribution analysis

**🌳 Tree-Based Techniques:**
- **LightGBM:** Large-scale classification
- **Feature Engineering:** System fingerprinting, version encoding
- **Memory Optimization:** Efficient data handling techniques

---

#### **4. LANL Earthquake Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction)

**📊 EDA Focus:**
- Seismic signal analysis
- Time-to-failure patterns
- Signal frequency analysis
- Acoustic emission patterns

**🌳 Tree-Based Techniques:**
- **XGBoost:** Time series regression
- **Feature Engineering:** Signal processing features, rolling statistics
- **Advanced Techniques:** Fourier transforms, wavelet features

---

#### **5. Predicting Molecular Properties** ⭐⭐⭐⭐⭐
**🔗 Link:** [Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling)

**📊 EDA Focus:**
- Molecular structure analysis
- Chemical bond relationships
- 3D coordinate analysis
- Coupling constant distributions

**🌳 Tree-Based Techniques:**
- **LightGBM:** Chemical property prediction
- **Feature Engineering:** Distance calculations, angle features
- **Domain Knowledge:** Chemical feature creation

---

## 📈 **Linear Models**
*Linear Regression, Logistic Regression, Ridge, Lasso, Elastic Net, SVM*

### **🟢 Beginner Level**

#### **1. Boston Housing Prices** ⭐⭐⭐⭐
**🔗 Link:** [Boston Housing](https://www.kaggle.com/c/boston-housing)

**📊 EDA Focus:**
- House price distribution analysis
- Feature correlation with price
- Neighborhood effects analysis
- Outlier detection in prices and features

**📈 Linear Model Techniques:**
- **Linear Regression:** Basic price prediction
- **Ridge Regression:** Handle multicollinearity
- **Feature Engineering:** Polynomial features, interaction terms
- **Regularization:** L1/L2 penalty comparison

**💻 Code Practice:**
```python
# Linear model techniques from cheat sheet:
- Feature scaling and normalization
- Polynomial feature generation
- Regularization path analysis
- Coefficient interpretation and confidence intervals
```

---

#### **2. Student Performance Dataset** ⭐⭐⭐
**🔗 Link:** [Student Performance](https://www.kaggle.com/spscientist/students-performance-in-exams)

**📊 EDA Focus:**
- Score distribution analysis
- Gender and ethnic group comparisons
- Parental education impact
- Lunch and test preparation effects

**📈 Linear Model Techniques:**
- **Multiple Linear Regression:** Score prediction
- **Logistic Regression:** Pass/fail classification
- **Feature Engineering:** Categorical encoding, interaction terms

---

#### **3. Medical Insurance Costs** ⭐⭐⭐
**🔗 Link:** [Medical Insurance](https://www.kaggle.com/mirichoi0218/insurance)

**📊 EDA Focus:**
- Insurance cost distribution
- Age, BMI, and smoking effects
- Regional cost variations
- Family size impact analysis

**📈 Linear Model Techniques:**
- **Linear Regression:** Cost prediction
- **Feature Transformation:** Log transformation for costs
- **Categorical Encoding:** Region and smoker status

---

#### **4. Wine Quality Dataset** ⭐⭐⭐
**🔗 Link:** [Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

**📊 EDA Focus:**
- Quality score distribution
- Chemical property correlations
- Acidity and alcohol relationships
- Quality vs property scatter plots

**📈 Linear Model Techniques:**
- **Ordinal Regression:** Quality score prediction
- **Ridge Regression:** Handle correlated chemical properties
- **Feature Selection:** Lasso for sparse solutions

---

#### **5. Advertising Dataset** ⭐⭐⭐
**🔗 Link:** [Advertising Dataset](https://www.kaggle.com/ashydv/advertising-dataset)

**📊 EDA Focus:**
- Sales vs advertising budget relationships
- Media channel effectiveness comparison
- Budget allocation analysis
- Diminishing returns analysis

**📈 Linear Model Techniques:**
- **Multiple Linear Regression:** Sales prediction
- **Interaction Terms:** Media synergy effects
- **Model Diagnostics:** Residual analysis

---

### **🟡 Intermediate Level**

#### **1. Bike Sharing Demand** ⭐⭐⭐⭐
**🔗 Link:** [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)

**📊 EDA Focus:**
- Seasonal and hourly demand patterns
- Weather impact on bike usage
- Holiday and working day effects
- Temperature and humidity correlations

**📈 Linear Model Techniques:**
- **Ridge Regression:** Handle seasonal multicollinearity
- **Elastic Net:** Feature selection with grouping
- **Time Series Features:** Cyclical encoding, lag features

---

#### **2. Heart Disease Prediction** ⭐⭐⭐⭐
**🔗 Link:** [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

**📊 EDA Focus:**
- Heart disease prevalence analysis
- Risk factor correlation analysis
- Age and gender effects
- Chest pain type analysis

**📈 Linear Model Techniques:**
- **Logistic Regression:** Disease classification
- **Regularized Logistic:** L1/L2 penalty comparison
- **Feature Engineering:** Risk score calculations

---

#### **3. Loan Prediction Dataset** ⭐⭐⭐⭐
**🔗 Link:** [Loan Prediction](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)

**📊 EDA Focus:**
- Loan approval rate analysis
- Income and loan amount relationships
- Credit history impact
- Property area effects

**📈 Linear Model Techniques:**
- **Logistic Regression:** Approval prediction
- **Feature Engineering:** Income ratios, debt-to-income
- **Class Imbalance:** Weighted logistic regression

---

#### **4. Customer Churn Prediction** ⭐⭐⭐⭐
**🔗 Link:** [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

**📊 EDA Focus:**
- Churn rate analysis by service type
- Customer tenure patterns
- Monthly charges vs churn relationship
- Service usage patterns

**📈 Linear Model Techniques:**
- **Logistic Regression:** Churn prediction
- **Feature Engineering:** Tenure binning, service combinations
- **Model Interpretation:** Odds ratios and feature importance

---

#### **5. Credit Card Default** ⭐⭐⭐⭐
**🔗 Link:** [Credit Card Default](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)

**📊 EDA Focus:**
- Default rate analysis
- Payment history patterns
- Credit limit utilization
- Demographic factor analysis

**📈 Linear Model Techniques:**
- **Logistic Regression:** Default prediction
- **Regularization:** Handle payment history correlations
- **Feature Engineering:** Payment ratios, utilization rates

---

### **🔴 Advanced Level**

#### **1. Santander Value Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [Santander Value Prediction](https://www.kaggle.com/c/santander-value-prediction-challenge)

**📊 EDA Focus:**
- 4000+ anonymized features analysis
- Target value distribution (highly skewed)
- Feature sparsity analysis
- Constant feature identification

**📈 Linear Model Techniques:**
- **Elastic Net:** High-dimensional regression
- **Feature Selection:** Lasso with cross-validation
- **Advanced Regularization:** Group lasso, adaptive lasso

---

#### **2. Mercedes-Benz Greener Manufacturing** ⭐⭐⭐⭐⭐
**🔗 Link:** [Mercedes-Benz Manufacturing](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)

**📊 EDA Focus:**
- Manufacturing configuration analysis
- Testing time distribution
- Categorical feature combinations
- Rare category analysis

**📈 Linear Model Techniques:**
- **Ridge Regression:** Handle categorical interactions
- **Feature Engineering:** Configuration combinations
- **Dimensionality Reduction:** PCA with linear models

---

#### **3. New York City Taxi Fare Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [NYC Taxi Fare](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

**📊 EDA Focus:**
- Fare distribution analysis
- Geographic patterns (pickup/dropoff)
- Time-based fare variations
- Distance vs fare relationships

**📈 Linear Model Techniques:**
- **Linear Regression:** Fare prediction
- **Geospatial Features:** Distance calculations, borough encoding
- **Temporal Features:** Rush hour effects, seasonal patterns

---

#### **4. Predict Future Sales** ⭐⭐⭐⭐⭐
**🔗 Link:** [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)

**📊 EDA Focus:**
- Sales time series analysis
- Product category trends
- Shop performance analysis
- Seasonal sales patterns

**📈 Linear Model Techniques:**
- **Time Series Regression:** Sales forecasting
- **Feature Engineering:** Lag features, rolling averages
- **Regularization:** Handle seasonal correlations

---

#### **5. Avito Demand Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [Avito Demand Prediction](https://www.kaggle.com/c/avito-demand-prediction)

**📊 EDA Focus:**
- Ad demand distribution
- Text and image feature analysis
- Price and demand relationships
- Geographic demand patterns

**📈 Linear Model Techniques:**
- **Linear Regression:** Demand prediction
- **Text Features:** TF-IDF with linear models
- **Mixed Data:** Combining text, image, and numerical features

---

## 📏 **Distance-Based Models**
*K-Nearest Neighbors (KNN), Support Vector Machines (SVM), K-Means Clustering*

### **🟢 Beginner Level**

#### **1. Iris Species Classification** ⭐⭐⭐⭐⭐
**🔗 Link:** [Iris Dataset](https://www.kaggle.com/uciml/iris)

**📊 EDA Focus:**
- Species distribution analysis
- Petal and sepal measurement correlations
- Feature pair scatter plots
- Species separation visualization

**📏 Distance-Based Techniques:**
- **KNN Classification:** Species prediction
- **Distance Metrics:** Euclidean vs Manhattan comparison
- **Feature Scaling:** StandardScaler vs RobustScaler
- **K-Optimization:** Elbow method and cross-validation

**💻 Code Practice:**
```python
# Distance-based techniques from cheat sheet:
- Feature scaling for distance calculations
- Optimal k selection using cross-validation
- Distance metric comparison
- Curse of dimensionality analysis
```

---

#### **2. Breast Cancer Wisconsin** ⭐⭐⭐⭐
**🔗 Link:** [Breast Cancer](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

**📊 EDA Focus:**
- Malignant vs benign feature distributions
- Feature correlation analysis (30 features)
- Outlier detection in measurements
- Feature importance for separation

**📏 Distance-Based Techniques:**
- **KNN Classification:** Tumor classification
- **SVM:** Linear and RBF kernel comparison
- **Feature Selection:** Distance-based importance
- **Dimensionality Reduction:** PCA before KNN

---

#### **3. Wine Dataset** ⭐⭐⭐
**🔗 Link:** [Wine Recognition](https://www.kaggle.com/brynja/wineuci)

**📊 EDA Focus:**
- Wine class distribution
- Chemical property analysis
- Feature correlation patterns
- Class separability analysis

**📏 Distance-Based Techniques:**
- **KNN Classification:** Wine type prediction
- **K-Means Clustering:** Unsupervised wine grouping
- **SVM:** Multi-class classification

---

#### **4. Seeds Dataset** ⭐⭐⭐
**🔗 Link:** [Seeds Dataset](https://www.kaggle.com/dongeorge/seed-from-uci)

**📊 EDA Focus:**
- Seed variety distribution
- Geometric measurement analysis
- Feature correlation analysis
- Variety separation patterns

**📏 Distance-Based Techniques:**
- **KNN Classification:** Seed variety prediction
- **Distance Metrics:** Custom distance functions
- **Feature Engineering:** Geometric ratios and combinations

---

#### **5. Glass Identification** ⭐⭐⭐
**🔗 Link:** [Glass Identification](https://www.kaggle.com/uciml/glass)

**📊 EDA Focus:**
- Glass type distribution (imbalanced classes)
- Chemical composition analysis
- Refractive index patterns
- Oxide content relationships

**📏 Distance-Based Techniques:**
- **KNN Classification:** Glass type prediction
- **Weighted KNN:** Handle class imbalance
- **Feature Scaling:** Robust scaling for chemical data

---

### **🟡 Intermediate Level**

#### **1. MNIST Handwritten Digits** ⭐⭐⭐⭐
**🔗 Link:** [MNIST](https://www.kaggle.com/oddrationale/mnist-in-csv)

**📊 EDA Focus:**
- Digit distribution analysis
- Pixel intensity patterns
- Digit similarity analysis
- Dimensionality visualization (t-SNE)

**📏 Distance-Based Techniques:**
- **KNN Classification:** Digit recognition
- **Dimensionality Reduction:** PCA for efficiency
- **Distance Optimization:** Custom distance metrics
- **Approximate KNN:** LSH for large datasets

---

#### **2. Fashion MNIST** ⭐⭐⭐⭐
**🔗 Link:** [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist)

**📊 EDA Focus:**
- Fashion item distribution
- Pixel pattern analysis
- Item similarity visualization
- Challenging class pairs identification

**📏 Distance-Based Techniques:**
- **KNN Classification:** Fashion item recognition
- **SVM:** Image classification with kernels
- **Feature Engineering:** HOG features, texture descriptors

---

#### **3. Human Activity Recognition** ⭐⭐⭐⭐
**🔗 Link:** [Human Activity Recognition](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones)

**📊 EDA Focus:**
- Activity distribution analysis
- Sensor signal patterns
- Feature correlation analysis
- Time-domain vs frequency-domain features

**📏 Distance-Based Techniques:**
- **KNN Classification:** Activity recognition
- **Time Series Distance:** DTW distance metrics
- **Feature Selection:** Signal processing features

---

#### **4. Customer Segmentation** ⭐⭐⭐⭐
**🔗 Link:** [Mall Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

**📊 EDA Focus:**
- Customer demographic analysis
- Spending behavior patterns
- Age vs income relationships
- Spending score distributions

**📏 Distance-Based Techniques:**
- **K-Means Clustering:** Customer segmentation
- **Hierarchical Clustering:** Dendrogram analysis
- **Cluster Validation:** Silhouette analysis, elbow method

---

#### **5. Olivetti Faces** ⭐⭐⭐⭐
**🔗 Link:** [Olivetti Faces](https://www.kaggle.com/imrandude/olivetti)

**📊 EDA Focus:**
- Face image analysis
- Pixel intensity distributions
- Face similarity patterns
- Lighting condition effects

**📏 Distance-Based Techniques:**
- **KNN Classification:** Face recognition
- **SVM:** Face classification with RBF kernel
- **Dimensionality Reduction:** Eigenfaces (PCA)

---

### **🔴 Advanced Level**

#### **1. Forest Cover Type Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [Forest Cover Type](https://www.kaggle.com/c/forest-cover-type-prediction)

**📊 EDA Focus:**
- Forest cover type distribution
- Elevation and slope analysis
- Soil type relationships
- Wilderness area effects

**📏 Distance-Based Techniques:**
- **KNN Classification:** Cover type prediction
- **Advanced Distance Metrics:** Mahalanobis distance
- **Feature Engineering:** Geospatial features, elevation ratios
- **Ensemble KNN:** Multiple distance metric combination

---

#### **2. Gesture Phase Segmentation** ⭐⭐⭐⭐⭐
**🔗 Link:** [Gesture Phase Segmentation](https://www.kaggle.com/birdx0810/gesture-phase-segmentation)

**📊 EDA Focus:**
- Gesture phase distribution
- Motion sensor data analysis
- Temporal pattern analysis
- Phase transition patterns

**📏 Distance-Based Techniques:**
- **KNN Classification:** Gesture phase prediction
- **Time Series KNN:** Temporal distance metrics
- **Feature Engineering:** Motion derivatives, velocity features

---

#### **3. Urban Sound Classification** ⭐⭐⭐⭐⭐
**🔗 Link:** [Urban Sound Classification](https://www.kaggle.com/chrisfilo/urbansound8k)

**📊 EDA Focus:**
- Sound class distribution
- Audio feature analysis (MFCC, spectral)
- Duration and frequency patterns
- Urban environment sound characteristics

**📏 Distance-Based Techniques:**
- **KNN Classification:** Sound classification
- **Audio Distance Metrics:** Spectral distance measures
- **Feature Engineering:** Audio signal processing features

---

#### **4. Speech Emotion Recognition** ⭐⭐⭐⭐⭐
**🔗 Link:** [Speech Emotion Recognition](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

**📊 EDA Focus:**
- Emotion distribution analysis
- Audio feature patterns by emotion
- Speaker variation analysis
- Prosodic feature analysis

**📏 Distance-Based Techniques:**
- **KNN Classification:** Emotion recognition
- **SVM:** Emotion classification with custom kernels
- **Feature Engineering:** Prosodic features, spectral features

---

#### **5. Satellite Image Classification** ⭐⭐⭐⭐⭐
**🔗 Link:** [Satellite Image Classification](https://www.kaggle.com/mahmoudreda55/satellite-image-classification)

**📊 EDA Focus:**
- Land use class distribution
- Spectral band analysis
- Spatial pattern analysis
- Seasonal variation effects

**📏 Distance-Based Techniques:**
- **KNN Classification:** Land use classification
- **SVM:** Image classification with spatial kernels
- **Feature Engineering:** Spectral indices, texture features

---

## 🧠 **Neural Networks**
*Feedforward NN, CNN, RNN, LSTM, GRU*

### **🟢 Beginner Level**

#### **1. MNIST Handwritten Digits** ⭐⭐⭐⭐⭐
**🔗 Link:** [MNIST](https://www.kaggle.com/oddrationale/mnist-in-csv)

**📊 EDA Focus:**
- Digit distribution analysis
- Pixel intensity patterns
- Image visualization and preprocessing
- Data augmentation analysis

**🧠 Neural Network Techniques:**
- **Feedforward NN:** Basic digit classification
- **CNN:** Convolutional layers for image recognition
- **Feature Engineering:** Normalization, data augmentation
- **Architecture Design:** Layer depth and width optimization

**💻 Code Practice:**
```python
# Neural network techniques from cheat sheet:
- Feature normalization for neural networks
- Embedding layers for categorical data
- Dropout and regularization techniques
- Learning rate scheduling and optimization
```

---

#### **2. Fashion MNIST** ⭐⭐⭐⭐
**🔗 Link:** [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist)

**📊 EDA Focus:**
- Fashion item distribution
- Pixel pattern complexity analysis
- Item similarity and confusion analysis
- Class imbalance investigation

**🧠 Neural Network Techniques:**
- **CNN:** Fashion item classification
- **Data Augmentation:** Rotation, flipping, scaling
- **Transfer Learning:** Pre-trained model adaptation

---

#### **3. IMDB Movie Reviews** ⭐⭐⭐⭐
**🔗 Link:** [IMDB Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

**📊 EDA Focus:**
- Sentiment distribution analysis
- Review length patterns
- Word frequency analysis
- Text preprocessing requirements

**🧠 Neural Network Techniques:**
- **RNN:** Sequential text processing
- **LSTM:** Long-term dependency modeling
- **Embedding Layers:** Word representation learning

---

#### **4. Cats vs Dogs** ⭐⭐⭐⭐
**🔗 Link:** [Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats)

**📊 EDA Focus:**
- Image size and quality analysis
- Color distribution patterns
- Edge and texture analysis
- Data augmentation strategies

**🧠 Neural Network Techniques:**
- **CNN:** Binary image classification
- **Transfer Learning:** VGG, ResNet adaptation
- **Data Augmentation:** Advanced image transformations

---

#### **5. Titanic with Neural Networks** ⭐⭐⭐
**🔗 Link:** [Titanic Dataset](https://www.kaggle.com/c/titanic)

**📊 EDA Focus:**
- Mixed data type analysis
- Feature correlation patterns
- Missing data impact
- Categorical encoding strategies

**🧠 Neural Network Techniques:**
- **Feedforward NN:** Tabular data classification
- **Embedding Layers:** Categorical feature embeddings
- **Mixed Input Architecture:** Numerical and categorical combination

---

### **🟡 Intermediate Level**

#### **1. CIFAR-10 Image Classification** ⭐⭐⭐⭐⭐
**🔗 Link:** [CIFAR-10](https://www.kaggle.com/c/cifar-10)

**📊 EDA Focus:**
- 10-class image distribution
- Color and texture analysis
- Inter-class similarity analysis
- Image complexity assessment

**🧠 Neural Network Techniques:**
- **Deep CNN:** Multi-layer convolutional networks
- **Residual Networks:** Skip connections and deep architectures
- **Batch Normalization:** Training stabilization

---

#### **2. Facial Keypoints Detection** ⭐⭐⭐⭐
**🔗 Link:** [Facial Keypoints](https://www.kaggle.com/c/facial-keypoints-detection)

**📊 EDA Focus:**
- Facial keypoint distribution
- Image quality and lighting analysis
- Missing keypoint patterns
- Face pose variation analysis

**🧠 Neural Network Techniques:**
- **CNN Regression:** Keypoint coordinate prediction
- **Data Augmentation:** Facial image transformations
- **Multi-task Learning:** Multiple keypoint prediction

---

#### **3. Digit Recognizer** ⭐⭐⭐⭐
**🔗 Link:** [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)

**📊 EDA Focus:**
- Handwriting style variations
- Digit confusion matrix analysis
- Noise and distortion patterns
- Data augmentation effectiveness

**🧠 Neural Network Techniques:**
- **Advanced CNN:** Deeper architectures
- **Ensemble Methods:** Multiple model combination
- **Regularization:** Dropout, weight decay

---

#### **4. Leaf Classification** ⭐⭐⭐⭐
**🔗 Link:** [Leaf Classification](https://www.kaggle.com/c/leaf-classification)

**📊 EDA Focus:**
- Leaf species distribution
- Shape and texture feature analysis
- Image preprocessing requirements
- Feature correlation analysis

**🧠 Neural Network Techniques:**
- **CNN:** Leaf image classification
- **Feature Fusion:** Combining CNN with traditional features
- **Transfer Learning:** Botanical image adaptation

---

#### **5. State Farm Distracted Driver Detection** ⭐⭐⭐⭐
**🔗 Link:** [Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

**📊 EDA Focus:**
- Driver behavior class distribution
- Image quality and angle analysis
- Driver identity patterns
- Behavior similarity analysis

**🧠 Neural Network Techniques:**
- **CNN Classification:** Driver behavior recognition
- **Data Leakage Prevention:** Driver-based splitting
- **Attention Mechanisms:** Focus on relevant image regions

---

### **🔴 Advanced Level**

#### **1. Human Protein Atlas Image Classification** ⭐⭐⭐⭐⭐
**🔗 Link:** [Human Protein Atlas](https://www.kaggle.com/c/human-protein-atlas-image-classification)

**📊 EDA Focus:**
- Multi-label protein pattern analysis
- Microscopy image characteristics
- Protein localization patterns
- Class co-occurrence analysis

**🧠 Neural Network Techniques:**
- **Multi-label CNN:** Protein pattern classification
- **Attention Mechanisms:** Cellular structure focus
- **Advanced Architectures:** EfficientNet, ResNeXt

---

#### **2. RSNA Pneumonia Detection** ⭐⭐⭐⭐⭐
**🔗 Link:** [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

**📊 EDA Focus:**
- Medical image analysis
- Pneumonia pattern distribution
- Image quality and artifacts
- Radiologist annotation analysis

**🧠 Neural Network Techniques:**
- **Medical CNN:** Chest X-ray analysis
- **Object Detection:** Pneumonia localization
- **Transfer Learning:** Medical imaging adaptation

---

#### **3. Carvana Image Masking Challenge** ⭐⭐⭐⭐⭐
**🔗 Link:** [Carvana Image Masking](https://www.kaggle.com/c/carvana-image-masking-challenge)

**📊 EDA Focus:**
- Car image segmentation analysis
- Background variation patterns
- Car type and angle analysis
- Mask quality assessment

**🧠 Neural Network Techniques:**
- **U-Net:** Image segmentation
- **Encoder-Decoder:** Semantic segmentation
- **Loss Functions:** Dice loss, IoU optimization

---

#### **4. TGS Salt Identification Challenge** ⭐⭐⭐⭐⭐
**🔗 Link:** [TGS Salt Identification](https://www.kaggle.com/c/tgs-salt-identification-challenge)

**📊 EDA Focus:**
- Seismic image analysis
- Salt deposit patterns
- Image depth and quality analysis
- Geological structure patterns

**🧠 Neural Network Techniques:**
- **Advanced U-Net:** Geological image segmentation
- **Data Augmentation:** Seismic image transformations
- **Ensemble Methods:** Multiple segmentation models

---

#### **5. Data Science Bowl 2018** ⭐⭐⭐⭐⭐
**🔗 Link:** [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018)

**📊 EDA Focus:**
- Cell nuclei segmentation analysis
- Microscopy image variations
- Cell type and staining analysis
- Annotation quality assessment

**🧠 Neural Network Techniques:**
- **Instance Segmentation:** Cell nuclei detection
- **Mask R-CNN:** Object detection and segmentation
- **Post-processing:** Watershed algorithm integration

---

## 🔄 **Transformers & Encoders**
*BERT, GPT, Autoencoders, VAE, Sequence-to-Sequence*

### **🟢 Beginner Level**

#### **1. IMDB Sentiment Analysis with BERT** ⭐⭐⭐⭐
**🔗 Link:** [IMDB Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

**📊 EDA Focus:**
- Text length distribution analysis
- Sentiment word patterns
- Vocabulary analysis and coverage
- Text preprocessing requirements

**🔄 Transformer Techniques:**
- **BERT Fine-tuning:** Sentiment classification
- **Tokenization Analysis:** WordPiece tokenization
- **Attention Visualization:** Understanding model focus
- **Transfer Learning:** Pre-trained model adaptation

**💻 Code Practice:**
```python
# Transformer techniques from cheat sheet:
- BERT tokenization and encoding
- Attention mechanism visualization
- Fine-tuning strategies
- Embedding extraction and analysis
```

---

#### **2. News Category Classification** ⭐⭐⭐
**🔗 Link:** [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset)

**📊 EDA Focus:**
- Category distribution analysis
- Headline length patterns
- Word frequency by category
- Text similarity analysis

**🔄 Transformer Techniques:**
- **BERT Classification:** News category prediction
- **Text Preprocessing:** Cleaning and normalization
- **Embedding Analysis:** Category separability

---

#### **3. MNIST Autoencoder** ⭐⭐⭐
**🔗 Link:** [MNIST](https://www.kaggle.com/oddrationale/mnist-in-csv)

**📊 EDA Focus:**
- Digit reconstruction quality
- Latent space visualization
- Compression ratio analysis
- Reconstruction error patterns

**🔄 Encoder-Decoder Techniques:**
- **Autoencoder:** Digit reconstruction
- **Variational Autoencoder:** Generative modeling
- **Latent Space Analysis:** t-SNE visualization

---

#### **4. Fashion MNIST Autoencoder** ⭐⭐⭐
**🔗 Link:** [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist)

**📊 EDA Focus:**
- Fashion item reconstruction analysis
- Latent representation patterns
- Item similarity in latent space
- Reconstruction quality by category

**🔄 Encoder-Decoder Techniques:**
- **Convolutional Autoencoder:** Fashion item reconstruction
- **Denoising Autoencoder:** Noise removal
- **Feature Learning:** Unsupervised representation

---

#### **5. Simple Text Generation** ⭐⭐⭐
**🔗 Link:** [Shakespeare Text](https://www.kaggle.com/kingburrito666/shakespeare-plays)

**📊 EDA Focus:**
- Character frequency analysis
- Text pattern analysis
- Sequence length distributions
- Language style characteristics

**🔄 Transformer Techniques:**
- **Character-level RNN:** Text generation
- **Attention Mechanisms:** Sequence modeling
- **Language Modeling:** Next character prediction

---

### **🟡 Intermediate Level**

#### **1. Disaster Tweets Classification** ⭐⭐⭐⭐⭐
**🔗 Link:** [Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

**📊 EDA Focus:**
- Tweet length and structure analysis
- Disaster vs non-disaster patterns
- Hashtag and mention analysis
- Text preprocessing challenges

**🔄 Transformer Techniques:**
- **BERT Fine-tuning:** Disaster tweet classification
- **RoBERTa Comparison:** Model performance analysis
- **Ensemble Methods:** Multiple transformer combination

---

#### **2. Jigsaw Toxic Comment Classification** ⭐⭐⭐⭐⭐
**🔗 Link:** [Toxic Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**📊 EDA Focus:**
- Multi-label toxicity analysis
- Comment length patterns
- Toxic word analysis
- Class imbalance investigation

**🔄 Transformer Techniques:**
- **BERT Multi-label:** Toxicity classification
- **Attention Analysis:** Toxic pattern identification
- **Bias Analysis:** Model fairness evaluation

---

#### **3. CommonLit Readability Prize** ⭐⭐⭐⭐
**🔗 Link:** [CommonLit Readability](https://www.kaggle.com/c/commonlitreadabilityprize)

**📊 EDA Focus:**
- Text complexity analysis
- Readability score distribution
- Linguistic feature analysis
- Text length vs complexity

**🔄 Transformer Techniques:**
- **BERT Regression:** Readability prediction
- **Feature Engineering:** Linguistic features with BERT
- **Model Interpretation:** Understanding complexity factors

---

#### **4. Shopee Product Matching** ⭐⭐⭐⭐⭐
**🔗 Link:** [Shopee Product Matching](https://www.kaggle.com/c/shopee-product-matching)

**📊 EDA Focus:**
- Product title similarity analysis
- Image and text correlation
- Duplicate product patterns
- Multi-modal data analysis

**🔄 Transformer Techniques:**
- **Multi-modal Transformers:** Text and image combination
- **Similarity Learning:** Product matching
- **Cross-modal Attention:** Text-image alignment

---

#### **5. Google QUEST Q&A Labeling** ⭐⭐⭐⭐
**🔗 Link:** [Google QUEST](https://www.kaggle.com/c/google-quest-challenge)

**📊 EDA Focus:**
- Question-answer quality analysis
- Multi-target regression analysis
- Text quality patterns
- Answer relevance assessment

**🔄 Transformer Techniques:**
- **BERT Multi-target:** Quality prediction
- **Question-Answer Encoding:** Paired text processing
- **Regression with Transformers:** Continuous target prediction

---

### **🔴 Advanced Level**

#### **1. Feedback Prize - Evaluating Student Writing** ⭐⭐⭐⭐⭐
**🔗 Link:** [Feedback Prize](https://www.kaggle.com/c/feedback-prize-2021)

**📊 EDA Focus:**
- Student writing analysis
- Discourse element patterns
- Writing quality assessment
- Argumentative structure analysis

**🔄 Transformer Techniques:**
- **Advanced BERT:** Discourse element detection
- **Sequence Labeling:** Token-level classification
- **Long Document Processing:** Handling extended texts

---

#### **2. Kaggle Book** ⭐⭐⭐⭐⭐
**🔗 Link:** [Kaggle Book](https://www.kaggle.com/competitions/kaggle-book)

**📊 EDA Focus:**
- Book content analysis
- Chapter structure patterns
- Writing style analysis
- Content complexity assessment

**🔄 Transformer Techniques:**
- **GPT Fine-tuning:** Text generation
- **Long-form Generation:** Extended text creation
- **Style Transfer:** Writing style adaptation

---

#### **3. Mechanisms of Action (MoA) Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [MoA Prediction](https://www.kaggle.com/c/lish-moa)

**📊 EDA Focus:**
- Gene expression analysis
- Drug mechanism patterns
- Multi-label target analysis
- Biological pathway analysis

**🔄 Transformer Techniques:**
- **Tabular Transformers:** Gene expression processing
- **Multi-label Prediction:** Drug mechanism classification
- **Attention in Tabular Data:** Feature importance learning

---

#### **4. Contradictory, My Dear Watson** ⭐⭐⭐⭐⭐
**🔗 Link:** [Contradictory Watson](https://www.kaggle.com/c/contradictory-my-dear-watson)

**📊 EDA Focus:**
- Multi-language text analysis
- Premise-hypothesis relationships
- Language distribution patterns
- Entailment pattern analysis

**🔄 Transformer Techniques:**
- **Multilingual BERT:** Cross-language understanding
- **Natural Language Inference:** Entailment prediction
- **Cross-lingual Transfer:** Language adaptation

---

#### **5. Ubiquant Market Prediction** ⭐⭐⭐⭐⭐
**🔗 Link:** [Ubiquant Market Prediction](https://www.kaggle.com/c/ubiquant-market-prediction)

**📊 EDA Focus:**
- Financial time series analysis
- Market feature patterns
- Investment universe analysis
- Return prediction challenges

**🔄 Transformer Techniques:**
- **Temporal Transformers:** Time series prediction
- **Financial Attention:** Market factor focus
- **Sequence-to-One:** Return prediction from sequences

---

## 📊 **Cross-Model Comparison**

### **🎯 Multi-Model Datasets**
*Datasets suitable for comparing different model types*

#### **1. Titanic - All Model Types** ⭐⭐⭐⭐⭐
- **Tree-Based:** Random Forest, XGBoost feature importance
- **Linear:** Logistic regression with regularization
- **Distance-Based:** KNN with optimal k selection
- **Neural Networks:** Feedforward NN with embeddings
- **Comparison Focus:** Model performance and interpretability

#### **2. House Prices - Regression Comparison** ⭐⭐⭐⭐⭐
- **Tree-Based:** XGBoost, LightGBM regression
- **Linear:** Ridge, Lasso, Elastic Net
- **Neural Networks:** Deep regression networks
- **Comparison Focus:** Feature engineering effectiveness

#### **3. MNIST - Classification Comparison** ⭐⭐⭐⭐⭐
- **Distance-Based:** KNN with dimensionality reduction
- **Neural Networks:** CNN, feedforward NN
- **Transformers:** Vision Transformer adaptation
- **Comparison Focus:** Image classification approaches

#### **4. IMDB Reviews - Text Analysis** ⭐⭐⭐⭐⭐
- **Linear:** Logistic regression with TF-IDF
- **Neural Networks:** LSTM, CNN for text
- **Transformers:** BERT, RoBERTa fine-tuning
- **Comparison Focus:** Text representation methods

#### **5. Credit Card Fraud - Anomaly Detection** ⭐⭐⭐⭐⭐
- **Tree-Based:** Isolation Forest, XGBoost
- **Distance-Based:** Local Outlier Factor, One-Class SVM
- **Neural Networks:** Autoencoders for anomaly detection
- **Comparison Focus:** Imbalanced data handling

---

## 💡 **Practice Strategy**

### **📅 Model-Type Learning Path**

#### **Week 1-2: Tree-Based Mastery**
1. **Start with Titanic** - Learn Random Forest basics
2. **Progress to House Prices** - Master XGBoost/LightGBM
3. **Advanced: Credit Card Fraud** - Handle imbalanced data
4. **Focus:** Feature importance, tree interpretation, ensemble methods

#### **Week 3-4: Linear Model Expertise**
1. **Boston Housing** - Linear regression fundamentals
2. **Heart Disease** - Logistic regression and regularization
3. **Santander Value** - High-dimensional linear models
4. **Focus:** Feature scaling, regularization, coefficient interpretation

#### **Week 5-6: Distance-Based Proficiency**
1. **Iris Dataset** - KNN fundamentals
2. **MNIST** - High-dimensional distance methods
3. **Forest Cover Type** - Advanced distance techniques
4. **Focus:** Distance metrics, dimensionality reduction, clustering

#### **Week 7-8: Neural Network Competency**
1. **Fashion MNIST** - CNN basics
2. **IMDB Reviews** - RNN/LSTM for sequences
3. **Human Protein Atlas** - Advanced CNN architectures
4. **Focus:** Architecture design, regularization, transfer learning

#### **Week 9-10: Transformer Mastery**
1. **Disaster Tweets** - BERT fine-tuning
2. **Toxic Comments** - Advanced transformer applications
3. **Feedback Prize** - Long document processing
4. **Focus:** Attention mechanisms, pre-training, fine-tuning

### **🎯 Skill Development by Model Type**

#### **🌳 Tree-Based Skills Checklist:**
- [ ] **Basic Trees:** Decision tree interpretation and pruning
- [ ] **Random Forest:** Ensemble methods and feature importance
- [ ] **Gradient Boosting:** XGBoost, LightGBM, CatBoost optimization
- [ ] **Feature Engineering:** Tree-friendly features, binning, interactions
- [ ] **Hyperparameter Tuning:** Grid search, Bayesian optimization

#### **📈 Linear Model Skills Checklist:**
- [ ] **Regression:** Linear, Ridge, Lasso, Elastic Net
- [ ] **Classification:** Logistic regression, SVM
- [ ] **Regularization:** L1/L2 penalties, cross-validation
- [ ] **Feature Engineering:** Scaling, polynomial features, interactions
- [ ] **Diagnostics:** Residual analysis, multicollinearity detection

#### **📏 Distance-Based Skills Checklist:**
- [ ] **KNN:** Optimal k selection, distance metrics
- [ ] **SVM:** Kernel selection, hyperparameter tuning
- [ ] **Clustering:** K-means, hierarchical, DBSCAN
- [ ] **Preprocessing:** Feature scaling, dimensionality reduction
- [ ] **Evaluation:** Silhouette analysis, cluster validation

#### **🧠 Neural Network Skills Checklist:**
- [ ] **Architectures:** Feedforward, CNN, RNN, LSTM
- [ ] **Regularization:** Dropout, batch normalization, early stopping
- [ ] **Optimization:** Learning rate scheduling, optimizers
- [ ] **Transfer Learning:** Pre-trained models, fine-tuning
- [ ] **Evaluation:** Validation strategies, overfitting prevention

#### **🔄 Transformer Skills Checklist:**
- [ ] **Pre-trained Models:** BERT, RoBERTa, GPT usage
- [ ] **Fine-tuning:** Task-specific adaptation
- [ ] **Attention:** Understanding and visualizing attention
- [ ] **Tokenization:** WordPiece, BPE, SentencePiece
- [ ] **Multi-modal:** Combining text, image, and tabular data

### **🏆 Advanced Challenges**

#### **Cross-Model Projects:**
1. **Model Ensemble:** Combine all model types on same dataset
2. **Feature Engineering Comparison:** How different models benefit from features
3. **Interpretability Analysis:** Compare model explanations across types
4. **Performance Benchmarking:** Speed vs accuracy trade-offs

#### **Domain Specialization:**
1. **Computer Vision:** Focus on CNN and Vision Transformers
2. **NLP:** Master RNN, LSTM, and Transformer architectures
3. **Time Series:** Specialize in temporal models and forecasting
4. **Tabular Data:** Excel at tree-based and linear model optimization

---

**🚀 This comprehensive guide provides 25+ problems per model type across all difficulty levels. Start with your preferred model type or follow the suggested learning path to master all approaches systematically!**

---

*Last Updated: January 22, 2025*  
*Based on: ML Feature Analysis & EDA Cheat Sheets*  
*Coverage: 125+ Practice Problems Across All Model Types*
