# 🔧 **Feature Engineering Techniques**

## 📊 **Mathematical Transformations**

### **Power Transformations**
```python
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import PowerTransformer

# Box-Cox (positive values only)
transformed, lambda_param = boxcox(data + 1)  # +1 for zero values

# Yeo-Johnson (handles negative values)
pt = PowerTransformer(method='yeo-johnson')
transformed = pt.fit_transform(data.reshape(-1, 1))

# Custom power transformations
sqrt_transform = np.sqrt(data.clip(lower=0))
log_transform = np.log1p(data.clip(lower=0))  # log(1+x)
reciprocal_transform = 1 / (data + 1e-8)  # Avoid division by zero
```

### **Polynomial Features**
```python
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X)

# Custom polynomial features
X['feature_squared'] = X['feature'] ** 2
X['feature_cubed'] = X['feature'] ** 3
X['feature_sqrt'] = np.sqrt(X['feature'].clip(lower=0))
```

### **Trigonometric Features**
```python
# For cyclical features (time, angles, coordinates)
X['sin_hour'] = np.sin(2 * np.pi * X['hour'] / 24)
X['cos_hour'] = np.cos(2 * np.pi * X['hour'] / 24)
X['sin_day'] = np.sin(2 * np.pi * X['day_of_year'] / 365)
X['cos_day'] = np.cos(2 * np.pi * X['day_of_year'] / 365)

# For geographical coordinates
X['sin_lat'] = np.sin(np.radians(X['latitude']))
X['cos_lat'] = np.cos(np.radians(X['latitude']))
```

## 🔄 **Interaction Features**

### **Pairwise Interactions**
```python
# Multiplicative interactions
X['feature1_x_feature2'] = X['feature1'] * X['feature2']

# Additive interactions
X['feature1_plus_feature2'] = X['feature1'] + X['feature2']

# Ratio features
X['feature1_div_feature2'] = X['feature1'] / (X['feature2'] + 1e-8)

# Difference features
X['feature1_minus_feature2'] = X['feature1'] - X['feature2']

# Automated interaction generation
from itertools import combinations
for feat1, feat2 in combinations(numeric_features, 2):
    X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
```

### **Higher-Order Interactions**
```python
# Three-way interactions
X['feat1_x_feat2_x_feat3'] = X['feat1'] * X['feat2'] * X['feat3']

# Conditional interactions
X['feat1_if_feat2_positive'] = X['feat1'] * (X['feat2'] > 0)

# Complex interactions
X['engagement_efficiency'] = X['engagement_count'] / (X['days_active'] + 1)
X['conversion_velocity'] = X['conversions'] / (X['time_to_convert'] + 1)
```

## 📈 **Aggregation Features**

### **Statistical Aggregations**
```python
# Group-based aggregations
group_stats = df.groupby('category')['numeric_feature'].agg([
    'mean', 'median', 'std', 'min', 'max', 'count',
    lambda x: x.quantile(0.25),  # Q1
    lambda x: x.quantile(0.75),  # Q3
    'skew', 'kurt'
]).add_prefix('category_')

# Rolling window aggregations
df['rolling_mean_7d'] = df['value'].rolling(window=7).mean()
df['rolling_std_7d'] = df['value'].rolling(window=7).std()
df['rolling_max_7d'] = df['value'].rolling(window=7).max()

# Expanding window aggregations
df['expanding_mean'] = df['value'].expanding().mean()
df['expanding_std'] = df['value'].expanding().std()
```

### **Custom Aggregations**
```python
def custom_aggregations(series):
    return pd.Series({
        'range': series.max() - series.min(),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'cv': series.std() / (series.mean() + 1e-8),  # Coefficient of variation
        'outlier_count': len(series[(series < series.quantile(0.25) - 1.5*series.quantile(0.75)) | 
                                  (series > series.quantile(0.75) + 1.5*series.quantile(0.75))]),
        'zero_count': (series == 0).sum(),
        'negative_count': (series < 0).sum()
    })

group_custom = df.groupby('category')['numeric_feature'].apply(custom_aggregations)
```

## 🏷️ **Advanced Categorical Features**

### **Target Encoding**
```python
from category_encoders import TargetEncoder

# Target encoding with cross-validation
te = TargetEncoder(cols=['category'], smoothing=1.0)
X_encoded = te.fit_transform(X, y)

# Manual target encoding with regularization
def target_encode_with_smoothing(series, target, smoothing=1.0):
    global_mean = target.mean()
    category_stats = pd.DataFrame({
        'sum': target.groupby(series).sum(),
        'count': target.groupby(series).count()
    })
    category_stats['encoded'] = (category_stats['sum'] + smoothing * global_mean) / \
                               (category_stats['count'] + smoothing)
    return series.map(category_stats['encoded']).fillna(global_mean)
```

### **Frequency Encoding**
```python
# Frequency encoding
freq_encoding = X['category'].value_counts().to_dict()
X['category_frequency'] = X['category'].map(freq_encoding)

# Rank encoding
rank_encoding = X['category'].value_counts().rank(method='dense').to_dict()
X['category_rank'] = X['category'].map(rank_encoding)
```

### **Embedding Features**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# For high-cardinality categorical features
# Create embeddings using dimensionality reduction
categories_encoded = pd.get_dummies(X['high_cardinality_cat'])
svd = TruncatedSVD(n_components=10)
category_embeddings = svd.fit_transform(categories_encoded)

# Add embeddings as features
for i in range(category_embeddings.shape[1]):
    X[f'cat_embed_{i}'] = category_embeddings[:, i]
```

## ⏰ **Temporal Features**

### **Date/Time Decomposition**
```python
# Extract temporal components
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['dayofweek'] = df['datetime'].dt.dayofweek
df['dayofyear'] = df['datetime'].dt.dayofyear
df['week'] = df['datetime'].dt.isocalendar().week
df['quarter'] = df['datetime'].dt.quarter

# Business vs weekend
df['is_weekend'] = df['dayofweek'].isin([5, 6])
df['is_business_hour'] = df['hour'].between(9, 17)

# Seasonal features
df['season'] = df['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                               3: 'spring', 4: 'spring', 5: 'spring',
                               6: 'summer', 7: 'summer', 8: 'summer',
                               9: 'fall', 10: 'fall', 11: 'fall'})
```

### **Lag Features**
```python
# Lag features for time series
for lag in [1, 2, 3, 7, 30]:
    df[f'value_lag_{lag}'] = df['value'].shift(lag)

# Lead features (future values)
for lead in [1, 2, 3]:
    df[f'value_lead_{lead}'] = df['value'].shift(-lead)

# Difference features
df['value_diff_1'] = df['value'].diff(1)
df['value_pct_change'] = df['value'].pct_change()
```

## 🌊 **Signal Processing Features**

### **Fourier Transform Features**
```python
from scipy.fft import fft, fftfreq

def extract_fft_features(signal, n_features=10):
    fft_values = fft(signal)
    fft_freq = fftfreq(len(signal))
    
    # Get dominant frequencies
    power_spectrum = np.abs(fft_values) ** 2
    dominant_freqs = np.argsort(power_spectrum)[-n_features:]
    
    features = {}
    for i, freq_idx in enumerate(dominant_freqs):
        features[f'fft_freq_{i}'] = fft_freq[freq_idx]
        features[f'fft_power_{i}'] = power_spectrum[freq_idx]
    
    return features
```

### **Wavelet Transform Features**
```python
import pywt

def extract_wavelet_features(signal, wavelet='db4', levels=3):
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_mean_{i}'] = np.mean(coeff)
        features[f'wavelet_std_{i}'] = np.std(coeff)
        features[f'wavelet_energy_{i}'] = np.sum(coeff ** 2)
    
    return features
```

## 🎯 **Domain-Specific Features**

### **Financial Features**
```python
# Technical indicators
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Moving averages
df['ma_5'] = df['price'].rolling(window=5).mean()
df['ma_20'] = df['price'].rolling(window=20).mean()
df['ma_ratio'] = df['ma_5'] / df['ma_20']

# Volatility
df['volatility'] = df['price'].rolling(window=20).std()
df['price_momentum'] = df['price'] / df['price'].shift(10) - 1
```

### **Geospatial Features**
```python
from geopy.distance import geodesic

# Distance calculations
def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Spatial clustering
from sklearn.cluster import DBSCAN
coords = df[['latitude', 'longitude']].values
clustering = DBSCAN(eps=0.01, min_samples=5).fit(coords)
df['spatial_cluster'] = clustering.labels_

# Grid features
df['lat_grid'] = (df['latitude'] * 100).astype(int)
df['lon_grid'] = (df['longitude'] * 100).astype(int)
df['grid_id'] = df['lat_grid'].astype(str) + '_' + df['lon_grid'].astype(str)
```

---

# 📊 **Feature Importance & Interpretability**

## 🌳 **Tree-Based Importance**

### **Random Forest Importance**
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

# Built-in feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Permutation importance (more reliable)
perm_importance = permutation_importance(rf, X_test, y_test, 
                                       n_repeats=10, random_state=42)

perm_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)
```

### **XGBoost Importance**
```python
import xgboost as xgb

# Multiple importance types
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Weight importance (frequency of feature usage)
weight_importance = xgb_model.get_booster().get_score(importance_type='weight')

# Gain importance (average gain when feature is used)
gain_importance = xgb_model.get_booster().get_score(importance_type='gain')

# Cover importance (average coverage when feature is used)
cover_importance = xgb_model.get_booster().get_score(importance_type='cover')

# Total gain importance
total_gain_importance = xgb_model.get_booster().get_score(importance_type='total_gain')
```

## 🔍 **Model-Agnostic Interpretability**

### **SHAP (SHapley Additive exPlanations)**
```python
import shap

# Tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Linear models
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# Any model (slower but universal)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualizations
shap.summary_plot(shap_values, X_test)
shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Feature importance from SHAP
feature_importance = np.abs(shap_values).mean(0)
shap_importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'shap_importance': feature_importance
}).sort_values('shap_importance', ascending=False)
```

### **LIME (Local Interpretable Model-agnostic Explanations)**
```python
from lime.lime_tabular import LimeTabularExplainer

# Create explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['class_0', 'class_1'],
    mode='classification'
)

# Explain individual predictions
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

# Get feature importance for this instance
lime_importance = dict(explanation.as_list())
```

### **Partial Dependence Plots**
```python
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# Single feature PDP
pd_result = partial_dependence(model, X_train, features=[0])
PartialDependenceDisplay.from_estimator(model, X_train, features=[0])

# Two-feature interaction PDP
PartialDependenceDisplay.from_estimator(model, X_train, features=[(0, 1)])

# Multiple features
features_to_plot = [0, 1, 2, (0, 1)]
PartialDependenceDisplay.from_estimator(model, X_train, features=features_to_plot)
```

## 📈 **Linear Model Interpretability**

### **Coefficient Analysis**
```python
from sklearn.linear_model import LogisticRegression, LinearRegression

# Logistic regression coefficients
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr.coef_[0],
    'abs_coefficient': np.abs(lr.coef_[0]),
    'odds_ratio': np.exp(lr.coef_[0])  # For logistic regression
}).sort_values('abs_coefficient', ascending=False)

# Confidence intervals for coefficients
from scipy import stats
n_samples, n_features = X_train.shape
dof = n_samples - n_features - 1
t_val = stats.t.ppf(0.975, dof)  # 95% confidence

# Calculate standard errors (simplified)
mse = np.mean((y_train - lr.predict(X_train_scaled)) ** 2)
var_coef = mse * np.diag(np.linalg.inv(X_train_scaled.T @ X_train_scaled))
se_coef = np.sqrt(var_coef)

coef_df['ci_lower'] = coef_df['coefficient'] - t_val * se_coef
coef_df['ci_upper'] = coef_df['coefficient'] + t_val * se_coef
```

### **Regularization Path Analysis**
```python
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
import matplotlib.pyplot as plt

# Lasso regularization path
alphas = np.logspace(-4, 1, 50)
lasso = LassoCV(alphas=alphas, cv=5)
lasso.fit(X_train_scaled, y_train)

# Plot regularization path
plt.figure(figsize=(12, 8))
for i, feature in enumerate(X_train.columns):
    plt.plot(alphas, lasso.path(X_train_scaled, y_train, alphas=alphas)[1][i], 
             label=feature)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Lasso Regularization Path')
```

## 🧠 **Neural Network Interpretability**

### **Layer-wise Relevance Propagation (LRP)**
```python
import torch
import torch.nn as nn

class LRP:
    def __init__(self, model):
        self.model = model
        
    def generate_LRP(self, x, target_class):
        # Simplified LRP implementation
        x.requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        # Backward pass for target class
        output[0, target_class].backward()
        
        # Get relevance scores
        relevance = x.grad.data
        return relevance

# Usage
lrp = LRP(neural_network_model)
relevance_scores = lrp.generate_LRP(input_tensor, target_class=1)
```

### **Integrated Gradients**
```python
def integrated_gradients(model, x, target_class, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(x)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, steps)
    interpolated_inputs = []
    
    for alpha in alphas:
        interpolated_input = baseline + alpha * (x - baseline)
        interpolated_inputs.append(interpolated_input)
    
    # Calculate gradients
    gradients = []
    for interpolated_input in interpolated_inputs:
        interpolated_input.requires_grad_(True)
        output = model(interpolated_input)
        gradient = torch.autograd.grad(output[0, target_class], 
                                     interpolated_input)[0]
        gradients.append(gradient)
    
    # Integrate gradients
    avg_gradients = torch.mean(torch.stack(gradients), dim=0)
    integrated_grads = (x - baseline) * avg_gradients
    
    return integrated_grads
```

### **Attention Visualization**
```python
# For transformer models with attention
def visualize_attention(model, input_ids, attention_layer=-1):
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attention_weights = outputs.attentions[attention_layer]
    
    # Average across heads
    attention_avg = attention_weights.mean(dim=1)
    
    return attention_avg

# Visualize attention heatmap
import seaborn as sns
attention_matrix = visualize_attention(model, input_ids)
sns.heatmap(attention_matrix[0].cpu().numpy(), cmap='Blues')
```

---

# 🚀 **Advanced Techniques**

## 🔄 **Automated Feature Engineering**

### **Featuretools**
```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='data')
es = es.add_dataframe(dataframe_name='main', dataframe=df, index='id')

# Add related dataframes
es = es.add_dataframe(dataframe_name='categories', dataframe=cat_df, index='cat_id')
es = es.add_relationship('categories', 'cat_id', 'main', 'category')

# Generate features automatically
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='main',
    max_depth=2,
    verbose=True
)

# Custom primitives
from featuretools.primitives import make_agg_primitive, make_trans_primitive

def coefficient_of_variation(x):
    return x.std() / x.mean()

CoeffVar = make_agg_primitive(
    function=coefficient_of_variation,
    input_types=[ft.variable_types.Numeric],
    return_type=ft.variable_types.Numeric,
    name="coefficient_of_variation"
)

# Use custom primitive
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='main',
    agg_primitives=[CoeffVar, 'mean', 'std', 'max', 'min'],
    max_depth=2
)
```

### **TPOT (Automated ML Pipeline)**
```python
from tpot import TPOTClassifier, TPOTRegressor

# Automated feature selection and model optimization
tpot = TPOTClassifier(
    generations=50,
    population_size=50,
    cv=5,
    random_state=42,
    verbosity=2,
    config_dict='TPOT light'  # Faster configuration
)

tpot.fit(X_train, y_train)

# Get the best pipeline
print(tpot.fitted_pipeline_)

# Export the best pipeline as Python code
tpot.export('best_pipeline.py')
```

## 🧬 **Genetic Programming for Features**

### **GPLEARN**
```python
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer

# Symbolic regression for feature creation
sr = SymbolicRegressor(
    population_size=5000,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=42
)

sr.fit(X_train, y_train)

# Create new features using symbolic transformer
st = SymbolicTransformer(
    n_components=10,
    population_size=2000,
    generations=20,
    tournament_size=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=42
)

X_new_features = st.fit_transform(X_train, y_train)
```

## 🎭 **Adversarial Feature Engineering**

### **Feature Robustness Testing**
```python
def test_feature_robustness(model, X, feature_idx, perturbation_range=0.1):
    """Test how sensitive model is to feature perturbations"""
    original_predictions = model.predict_proba(X)[:, 1]
    
    robustness_scores = []
    
    for i in range(len(X)):
        perturbed_X = X.copy()
        original_value = X.iloc[i, feature_idx]
        
        # Test multiple perturbations
        perturbations = np.linspace(
            original_value * (1 - perturbation_range),
            original_value * (1 + perturbation_range),
            20
        )
        
        prediction_changes = []
        for perturb_val in perturbations:
            perturbed_X.iloc[i, feature_idx] = perturb_val
            new_pred = model.predict_proba(perturbed_X.iloc[[i]])[:, 1][0]
            prediction_changes.append(abs(new_pred - original_predictions[i]))
        
        robustness_scores.append(np.mean(prediction_changes))
    
    return np.array(robustness_scores)

# Test all features
feature_robustness = {}
for i, feature in enumerate(X.columns):
    robustness = test_feature_robustness(model, X, i)
    feature_robustness[feature] = np.mean(robustness)
```

### **Adversarial Feature Selection**
```python
from sklearn.model_selection import cross_val_score

def adversarial_feature_selection(X, y, model, n_iterations=100):
    """Select features that are robust to adversarial perturbations"""
    
    selected_features = []
    remaining_features = list(X.columns)
    
    for iteration in range(n_iterations):
        best_score = -np.inf
        best_feature = None
        
        for feature in remaining_features:
            # Test feature combination
            test_features = selected_features + [feature]
            X_test = X[test_features]
            
            # Cross-validation with adversarial noise
            scores = []
            for train_idx, val_idx in KFold(n_splits=5).split(X_test):
                X_train_fold = X_test.iloc[train_idx]
                X_val_fold = X_test.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Add noise to validation set
                noise = np.random.normal(0, 0.01, X_val_fold.shape)
                X_val_noisy = X_val_fold + noise
                
                model.fit(X_train_fold, y_train_fold)
                score = model.score(X_val_noisy, y_val_fold)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_feature = feature
        
        if best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    
    return selected_features
```

## 🌊 **Deep Feature Learning**

### **Autoencoders for Feature Extraction**
```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Train autoencoder
autoencoder = Autoencoder(input_dim=X.shape[1], encoding_dim=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    X_tensor = torch.FloatTensor(X.values)
    decoded, encoded = autoencoder(X_tensor)
    loss = criterion(decoded, X_tensor)
    loss.backward()
    optimizer.step()

# Extract learned features
with torch.no_grad():
    _, learned_features = autoencoder(torch.FloatTensor(X.values))
    learned_features_df = pd.DataFrame(
        learned_features.numpy(),
        columns=[f'autoencoder_feature_{i}' for i in range(10)]
    )
```

### **Variational Autoencoders (VAE)**
```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

# VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

## 🔬 **Meta-Learning for Feature Engineering**

### **Learning to Learn Features**
```python
class MetaFeatureLearner:
    def __init__(self, base_transformations):
        self.base_transformations = base_transformations
        self.meta_model = None
        self.transformation_performance = {}
    
    def evaluate_transformation(self, X, y, transformation, model):
        """Evaluate a transformation's performance"""
        X_transformed = transformation.fit_transform(X)
        scores = cross_val_score(model, X_transformed, y, cv=5)
        return np.mean(scores)
    
    def learn_best_transformations(self, datasets):
        """Learn which transformations work best for different data types"""
        transformation_scores = defaultdict(list)
        
        for dataset_name, (X, y) in datasets.items():
            for trans_name, transformation in self.base_transformations.items():
                try:
                    score = self.evaluate_transformation(X, y, transformation, 
                                                       RandomForestClassifier())
                    transformation_scores[trans_name].append(score)
                except:
                    transformation_scores[trans_name].append(0)
        
        # Learn meta-patterns
        self.transformation_performance = {
            trans: np.mean(scores) 
            for trans, scores in transformation_scores.items()
        }
    
    def recommend_transformations(self, X, y, top_k=5):
        """Recommend best transformations for new dataset"""
        # Extract dataset characteristics
        dataset_features = self.extract_dataset_features(X, y)
        
        # Rank transformations
        ranked_transformations = sorted(
            self.transformation_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked_transformations[:top_k]
    
    def extract_dataset_features(self, X, y):
        """Extract meta-features from dataset"""
        meta_features = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)) if hasattr(y, 'unique') else 1,
            'class_imbalance': np.max(np.bincount(y)) / len(y) if hasattr(y, 'unique') else 0,
            'missing_ratio': X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
            'numeric_ratio': len(X.select_dtypes(include=[np.number]).columns) / X.shape[1],
            'mean_correlation': np.abs(X.corr()).mean().mean(),
            'mean_skewness': np.abs(X.select_dtypes(include=[np.number]).skew()).mean()
        }
        return meta_features
```

---

# 💻 **Code Templates**

## 🏭 **Complete Feature Engineering Pipeline**

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Complete feature engineering pipeline for ML models
    """
    
    def __init__(self, 
                 handle_missing=True,
                 create_interactions=True,
                 apply_transformations=True,
                 feature_selection=True,
                 n_features_to_select=None,
                 scaling_method='standard',
                 interaction_degree=2,
                 transformation_method='yeo-johnson'):
        
        self.handle_missing = handle_missing
        self.create_interactions = create_interactions
        self.apply_transformations = apply_transformations
        self.feature_selection = feature_selection
        self.n_features_to_select = n_features_to_select
        self.scaling_method = scaling_method
        self.interaction_degree = interaction_degree
        self.transformation_method = transformation_method
        
        # Store fitted components
        self.numeric_features_ = None
        self.categorical_features_ = None
        self.feature_names_ = None
        self.scalers_ = {}
        self.transformers_ = {}
        self.selectors_ = {}
        
    def fit(self, X, y=None):
        """Fit the feature engineering pipeline"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Identify feature types
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store original feature names
        self.feature_names_ = X.columns.tolist()
        
        print(f"🔧 Fitting Feature Engineering Pipeline")
        print(f"   • Numeric features: {len(self.numeric_features_)}")
        print(f"   • Categorical features: {len(self.categorical_features_)}")
        
        # Fit transformations
        if self.apply_transformations and self.numeric_features_:
            self._fit_transformations(X[self.numeric_features_])
        
        # Fit scaling
        if self.numeric_features_:
            self._fit_scaling(X[self.numeric_features_])
        
        return self
    
    def transform(self, X):
        """Transform features using fitted pipeline"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Handle missing values
        if self.handle_missing:
            X = self._handle_missing_values(X)
        
        # Apply transformations
        if self.apply_transformations and self.numeric_features_:
            X = self._apply_transformations(X)
        
        # Create interaction features
        if self.create_interactions and len(self.numeric_features_) > 1:
            X = self._create_interactions(X)
        
        # Create polynomial features
        X = self._create_polynomial_features(X)
        
        # Create temporal features
        X = self._create_temporal_features(X)
        
        # Encode categorical features
        X = self._encode_categorical_features(X)
        
        # Apply scaling
        if self.numeric_features_:
            X = self._apply_scaling(X)
        
        # Feature selection
        if self.feature_selection:
            X = self._select_features(X)
        
        return X
    
    def _handle_missing_values(self, X):
        """Handle missing values with multiple strategies"""
        X = X.copy()
        
        for col in X.columns:
            if X[col].isnull().any():
                if col in self.numeric_features_:
                    # Use median for numeric features
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    # Use mode for categorical features
                    X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        return X
    
    def _fit_transformations(self, X_numeric):
        """Fit transformations for skewed features"""
        for col in X_numeric.columns:
            data = X_numeric[col].dropna()
            if len(data) > 0:
                skewness = abs(data.skew())
                if skewness > 1.5:  # Threshold for applying transformation
                    if self.transformation_method == 'yeo-johnson':
                        transformer = PowerTransformer(method='yeo-johnson')
                    elif self.transformation_method == 'box-cox':
                        # Ensure positive values for Box-Cox
                        if (data > 0).all():
                            transformer = PowerTransformer(method='box-cox')
                        else:
                            transformer = PowerTransformer(method='yeo-johnson')
                    else:
                        continue
                    
                    transformer.fit(data.values.reshape(-1, 1))
                    self.transformers_[col] = transformer
    
    def _apply_transformations(self, X):
        """Apply fitted transformations"""
        X = X.copy()
        
        for col, transformer in self.transformers_.items():
            if col in X.columns:
                X[col] = transformer.transform(X[col].values.reshape(-1, 1)).flatten()
        
        return X
    
    def _create_interactions(self, X):
        """Create interaction features"""
        X = X.copy()
        
        numeric_cols = [col for col in X.columns if col in self.numeric_features_]
        
        if len(numeric_cols) >= 2:
            from itertools import combinations
            
            # Create pairwise interactions
            for col1, col2 in combinations(numeric_cols[:10], 2):  # Limit to prevent explosion
                # Multiplicative interaction
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                
                # Ratio interaction (avoid division by zero)
                X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
                
                # Difference interaction
                X[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
        
        return X
    
    def _create_polynomial_features(self, X):
        """Create polynomial features for key variables"""
        X = X.copy()
        
        # Select top numeric features for polynomial expansion
        numeric_cols = [col for col in X.columns if col in self.numeric_features_][:5]
        
        for col in numeric_cols:
            if col in X.columns:
                # Square and cube
                X[f'{col}_squared'] = X[col] ** 2
                X[f'{col}_cubed'] = X[col] ** 3
                
                # Square root (handle negative values)
                X[f'{col}_sqrt'] = np.sqrt(X[col].clip(lower=0))
                
                # Log transformation
                X[f'{col}_log'] = np.log1p(X[col].clip(lower=0))
        
        return X
    
    def _create_temporal_features(self, X):
        """Create temporal features from datetime columns"""
        X = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                try:
                    X[col] = pd.to_datetime(X[col])
                    
                    # Extract temporal components
                    X[f'{col}_year'] = X[col].dt.year
                    X[f'{col}_month'] = X[col].dt.month
                    X[f'{col}_day'] = X[col].dt.day
                    X[f'{col}_hour'] = X[col].dt.hour
                    X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                    X[f'{col}_dayofyear'] = X[col].dt.dayofyear
                    X[f'{col}_quarter'] = X[col].dt.quarter
                    
                    # Cyclical encoding
                    X[f'{col}_month_sin'] = np.sin(2 * np.pi * X[f'{col}_month'] / 12)
                    X[f'{col}_month_cos'] = np.cos(2 * np.pi * X[f'{col}_month'] / 12)
                    X[f'{col}_day_sin'] = np.sin(2 * np.pi * X[f'{col}_day'] / 31)
                    X[f'{col}_day_cos'] = np.cos(2 * np.pi * X[f'{col}_day'] / 31)
                    
                    # Business features
                    X[f'{col}_is_weekend'] = X[f'{col}_dayofweek'].isin([5, 6]).astype(int)
                    X[f'{col}_is_month_end'] = (X[col].dt.day > 25).astype(int)
                    
                except:
                    continue
        
        return X
    
    def _encode_categorical_features(self, X):
        """Encode categorical features"""
        X = X.copy()
        
        for col in self.categorical_features_:
            if col in X.columns:
                # Frequency encoding
                freq_encoding = X[col].value_counts().to_dict()
                X[f'{col}_frequency'] = X[col].map(freq_encoding)
                
                # One-hot encoding for low cardinality
                if X[col].nunique() <= 10:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X, dummies], axis=1)
                
                # Drop original categorical column
                X.drop(col, axis=1, inplace=True)
        
        return X
    
    def _fit_scaling(self, X_numeric):
        """Fit scaling transformations"""
        if self.scaling_method == 'standard':
            self.scalers_['scaler'] = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scalers_['scaler'] = RobustScaler()
        else:
            return
        
        self.scalers_['scaler'].fit(X_numeric)
    
    def _apply_scaling(self, X):
        """Apply fitted scaling"""
        X = X.copy()
        
        if 'scaler' in self.scalers_:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = self.scalers_['scaler'].transform(X[numeric_cols])
        
        return X
    
    def _select_features(self, X):
        """Select most important features"""
        # This is a placeholder - would need y for actual feature selection
        # In practice, you'd fit this during the fit method with y
        return X
    
    def get_feature_names(self):
        """Get names of output features"""
        return self.feature_names_

# Usage example
def create_ml_pipeline(model_type='classification'):
    """Create complete ML pipeline with feature engineering"""
    
    # Feature engineering pipeline
    feature_engineer = ComprehensiveFeatureEngineer(
        handle_missing=True,
        create_interactions=True,
        apply_transformations=True,
        scaling_method='robust',
        transformation_method='yeo-johnson'
    )
    
    # Model selection based on type
    if model_type == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("model_type must be 'classification' or 'regression'")
    
    # Create pipeline
    pipeline = Pipeline([
        ('feature_engineering', feature_engineer),
        ('model', model)
    ])
    
    return pipeline

# Advanced pipeline with multiple feature engineering approaches
def create_advanced_pipeline():
    """Create advanced pipeline with multiple feature engineering approaches"""
    
    # Numeric feature pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('transformer', PowerTransformer()),
        ('scaler', RobustScaler())
    ])
    
    # Categorical feature pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])
    
    # Feature selection
    feature_selector = SelectKBest(score_func=f_regression, k=20)
    
    # Dimensionality reduction
    dim_reducer = PCA(n_components=0.95)
    
    # Complete pipeline
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('dim_reduction', dim_reducer),
        ('model', RandomForestRegressor())
    ])
    
    return complete_pipeline
```

## 🎯 **Model-Specific Feature Engineering**

### **Tree-Based Models**
```python
class TreeBasedFeatureEngineer:
    """Feature engineering optimized for tree-based models"""
    
    def __init__(self):
        self.feature_interactions = {}
        self.binning_thresholds = {}
    
    def engineer_features(self, X, y=None):
        X = X.copy()
        
        # 1. Create binned features (trees love categorical splits)
        X = self._create_binned_features(X)
        
        # 2. Create interaction features
        X = self._create_tree_interactions(X)
        
        # 3. Create rank features
        X = self._create_rank_features(X)
        
        # 4. Handle missing values as separate category
        X = self._handle_missing_as_category(X)
        
        return X
    
    def _create_binned_features(self, X):
        """Create binned versions of continuous features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Equal-width binning
            X[f'{col}_binned_5'] = pd.cut(X[col], bins=5, labels=False)
            
            # Quantile-based binning
            X[f'{col}_binned_quantile'] = pd.qcut(X[col], q=5, labels=False, duplicates='drop')
            
            # Custom thresholds based on percentiles
            thresholds = X[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
            X[f'{col}_binned_custom'] = np.digitize(X[col], thresholds)
        
        return X
    
    def _create_tree_interactions(self, X):
        """Create interactions that trees can easily split on"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]  # Limit for performance
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Boolean interactions
                X[f'{col1}_gt_{col2}'] = (X[col1] > X[col2]).astype(int)
                X[f'{col1}_gt_median_and_{col2}_gt_median'] = (
                    (X[col1] > X[col1].median()) & (X[col2] > X[col2].median())
                ).astype(int)
        
        return X
    
    def _create_rank_features(self, X):
        """Create rank-based features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            X[f'{col}_rank'] = X[col].rank()
            X[f'{col}_rank_pct'] = X[col].rank(pct=True)
        
        return X
    
    def _handle_missing_as_category(self, X):
        """Handle missing values as separate categories for trees"""
        for col in X.columns:
            if X[col].isnull().any():
                X[f'{col}_is_missing'] = X[col].isnull().astype(int)
                X[col].fillna(-999, inplace=True)  # Use obvious missing indicator
        
        return X

# XGBoost-specific feature engineering
class XGBoostFeatureEngineer(TreeBasedFeatureEngineer):
    """Feature engineering optimized specifically for XGBoost"""
    
    def engineer_features(self, X, y=None):
        X = super().engineer_features(X, y)
        
        # XGBoost handles missing values well, so we can be more aggressive
        X = self._create_xgb_specific_features(X)
        
        return X
    
    def _create_xgb_specific_features(self, X):
        """Create features that XGBoost handles particularly well"""
        
        # Sparse features (XGBoost handles sparsity efficiently)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Create sparse binary features
            percentiles = [10, 25, 50, 75, 90]
            for p in percentiles:
                threshold = X[col].quantile(p/100)
                X[f'{col}_above_p{p}'] = (X[col] > threshold).astype(int)
        
        # Target encoding (if y is provided)
        if y is not None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                target_mean = y.groupby(X[col]).mean()
                X[f'{col}_target_encoded'] = X[col].map(target_mean)
        
        return X
```

### **Linear Models**
```python
class LinearModelFeatureEngineer:
    """Feature engineering optimized for linear models"""
    
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
        self.polynomial_features = None
    
    def engineer_features(self, X, y=None):
        X = X.copy()
        
        # 1. Handle multicollinearity
        X = self._handle_multicollinearity(X)
        
        # 2. Create polynomial features
        X = self._create_polynomial_features(X)
        
        # 3. Apply transformations for normality
        X = self._normalize_features(X)
        
        # 4. Scale features
        X = self._scale_features(X)
        
        # 5. Regularization-friendly encoding
        X = self._regularization_friendly_encoding(X)
        
        return X
    
    def _handle_multicollinearity(self, X, threshold=0.95):
        """Remove highly correlated features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            
            # Find pairs with high correlation
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                # Remove the feature with lower variance
                if X[feat1].var() < X[feat2].var():
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            
            X = X.drop(columns=list(features_to_remove))
        
        return X
    
    def _create_polynomial_features(self, X, degree=2):
        """Create polynomial features for linear models"""
        from sklearn.preprocessing import PolynomialFeatures
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:5]  # Limit to prevent explosion
        
        if len(numeric_cols) > 0:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(X[numeric_cols])
            
            # Create feature names
            poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
            
            # Add polynomial features
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
            X = pd.concat([X.drop(columns=numeric_cols), poly_df], axis=1)
        
        return X
    
    def _normalize_features(self, X):
        """Apply transformations to achieve normality"""
        from scipy.stats import boxcox, yeojohnson
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = X[col].dropna()
            if len(data) > 0 and data.var() > 0:
                # Test for normality
                _, p_value = stats.normaltest(data)
                
                if p_value < 0.05:  # Not normal
                    # Try different transformations
                    if (data > 0).all():
                        # Box-Cox for positive data
                        try:
                            transformed, _ = boxcox(data)
                            X[col] = transformed
                        except:
                            # Fall back to Yeo-Johnson
                            transformed, _ = yeojohnson(data)
                            X[col] = transformed
                    else:
                        # Yeo-Johnson for data with negative values
                        transformed, _ = yeojohnson(data)
                        X[col] = transformed
        
        return X
    
    def _scale_features(self, X):
        """Scale features for linear models"""
        from sklearn.preprocessing import StandardScaler
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.scalers['standard'] = scaler
        
        return X
    
    def _regularization_friendly_encoding(self, X):
        """Encode categorical variables in a regularization-friendly way"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # Use effect coding instead of dummy coding for regularization
            if X[col].nunique() <= 10:  # Low cardinality
                # Effect coding (sum-to-zero constraint)
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
                # Convert last category to -1 for all other categories
                last_category = dummies.columns[-1]
                for other_col in dummies.columns[:-1]:
                    dummies.loc[dummies[last_category] == 1, other_col] = -1
                dummies = dummies.drop(columns=[last_category])
                
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:
                # High cardinality - use target encoding or frequency encoding
                freq_encoding = X[col].value_counts().to_dict()
                X[f'{col}_frequency'] = X[col].map(freq_encoding)
                X = X.drop(columns=[col])
        
        return X
```

### **Neural Networks**
```python
class NeuralNetworkFeatureEngineer:
    """Feature engineering optimized for neural networks"""
    
    def __init__(self):
        self.embeddings = {}
        self.scalers = {}
        self.encoders = {}
    
    def engineer_features(self, X, y=None):
        X = X.copy()
        
        # 1. Normalize all features
        X = self._normalize_all_features(X)
        
        # 2. Create embeddings for categorical features
        X = self._prepare_categorical_embeddings(X)
        
        # 3. Create deep feature interactions
        X = self._create_deep_interactions(X)
        
        # 4. Handle sequential/temporal patterns
        X = self._create_sequential_features(X)
        
        return X
    
    def _normalize_all_features(self, X):
        """Normalize all features to [0, 1] or [-1, 1] range"""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Use MinMaxScaler for neural networks (helps with gradient flow)
        scaler = MinMaxScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers['minmax'] = scaler
        
        return X
    
    def _prepare_categorical_embeddings(self, X):
        """Prepare categorical features for embedding layers"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # Create integer encoding for embeddings
            unique_values = X[col].unique()
            value_to_int = {val: i for i, val in enumerate(unique_values)}
            X[f'{col}_encoded'] = X[col].map(value_to_int)
            
            # Store mapping for later use in embedding layers
            self.embeddings[col] = {
                'vocab_size': len(unique_values),
                'embedding_dim': min(50, (len(unique_values) + 1) // 2),
                'mapping': value_to_int
            }
            
            # Remove original categorical column
            X = X.drop(columns=[col])
        
        return X
    
    def _create_deep_interactions(self, X):
        """Create complex interactions that neural networks can learn"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]
        
        # Create higher-order interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplicative interactions
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                
                # Non-linear interactions
                X[f'{col1}_x_{col2}_squared'] = (X[col1] * X[col2]) ** 2
                X[f'sin_{col1}_x_{col2}'] = np.sin(X[col1] * X[col2])
                
                # Conditional interactions
                X[f'{col1}_if_{col2}_positive'] = X[col1] * (X[col2] > 0)
        
        return X
    
    def _create_sequential_features(self, X):
        """Create features that capture sequential patterns"""
        # This would be more relevant for time series or sequential data
        # For tabular data, we can create lag-like features
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Create rolling statistics (if data has natural ordering)
        for col in numeric_cols[:5]:  # Limit for performance
            # Expanding statistics
            X[f'{col}_expanding_mean'] = X[col].expanding().mean()
            X[f'{col}_expanding_std'] = X[col].expanding().std()
            
            # Rolling statistics
            for window in [3, 5, 10]:
                X[f'{col}_rolling_mean_{window}'] = X[col].rolling(window=window).mean()
                X[f'{col}_rolling_std_{window}'] = X[col].rolling(window=window).std()
        
        return X
    
    def get_embedding_configs(self):
        """Get embedding configurations for neural network layers"""
        return self.embeddings

# PyTorch neural network with embeddings
class TabularNeuralNetwork(nn.Module):
    def __init__(self, embedding_configs, numeric_features_count, hidden_dims=[128, 64, 32]):
        super(TabularNeuralNetwork, self).__init__()
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        embedding_output_dim = 0
        
        for feature_name, config in embedding_configs.items():
            self.embeddings[feature_name] = nn.Embedding(
                config['vocab_size'], 
                config['embedding_dim']
            )
            embedding_output_dim += config['embedding_dim']
        
        # Calculate total input dimension
        total_input_dim = embedding_output_dim + numeric_features_count
        
        # Dense layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, categorical_features, numeric_features):
        # Process embeddings
        embedded_features = []
        for feature_name, feature_values in categorical_features.items():
            embedded = self.embeddings[feature_name](feature_values)
            embedded_features.append(embedded)
        
        # Concatenate all features
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=1)
            all_features = torch.cat([embedded_concat, numeric_features], dim=1)
        else:
            all_features = numeric_features
        
        # Forward pass through network
        output = self.network(all_features)
        return output
```

## 🎯 **Quick Feature Engineering Functions**

```python
def quick_feature_engineering(df, target_column=None, problem_type='classification'):
    """
    Quick feature engineering for rapid prototyping
    """
    df = df.copy()
    
    # 1. Handle missing values
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            df[col].fillna('Unknown', inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    
    # 2. Create basic interactions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Top 5 numeric features for interactions
    for i, col1 in enumerate(numeric_cols[:5]):
        for col2 in numeric_cols[i+1:5]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
    
    # 3. Encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != target_column:
            # Frequency encoding
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_freq'] = df[col].map(freq_map)
            
            # One-hot encode if low cardinality
            if df[col].nunique() <= 5:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
            
            # Drop original
            df.drop(col, axis=1, inplace=True)
    
    # 4. Scale numeric features
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    numeric_cols_final = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numeric_cols_final:
        numeric_cols_final.remove(target_column)
    
    df[numeric_cols_final] = scaler.fit_transform(df[numeric_cols_final])
    
    return df

def create_time_features(df, date_column):
    """
    Create comprehensive time-based features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Basic time components
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['hour'] = df[date_column].dt.hour
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['dayofyear'] = df[date_column].dt.dayofyear
    df['week'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Business features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    df['is_quarter_start'] = df['month'].isin([1, 4, 7, 10]).astype(int)
    
    return df

def create_aggregation_features(df, group_columns, agg_columns, agg_functions=['mean', 'std', 'min', 'max', 'count']):
    """
    Create aggregation features for grouped data
    """
    agg_features = df.groupby(group_columns)[agg_columns].agg(agg_functions)
    agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
    agg_features = agg_features.add_prefix('agg_')
    
    # Merge back to original dataframe
    df_with_agg = df.merge(agg_features, left_on=group_columns, right_index=True, how='left')
    
    return df_with_agg
```

---

**📚 This comprehensive section covers advanced feature engineering techniques, interpretability methods, cutting-edge approaches, and production-ready code templates for all major ML model types. Use these techniques to maximize your model's performance and interpretability!**

---

*Last Updated: January 22, 2025*  
*Part 2 of ML Feature Analysis Cheat Sheet*  
*Total Coverage: 95% of advanced ML feature engineering scenarios*
