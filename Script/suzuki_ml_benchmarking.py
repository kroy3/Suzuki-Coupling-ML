"""
Suzuki-Miyaura Coupling Yield Prediction: Comprehensive Benchmarking Study
Authors: Kushal Raj Roy, Susen Das
Affiliation: University of Houston

This script generates synthetic but realistic Suzuki-Miyaura HTE data and benchmarks
multiple ML models for yield prediction across different metal catalysts.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# DATASET GENERATION: Chemistry-Informed Suzuki-Miyaura Reactions
# ============================================================================

def generate_suzuki_dataset():
    """
    Generate realistic Suzuki-Miyaura coupling dataset based on known reactivity patterns
    Modeled after Perera et al. Science 2018, 359, 429-434
    """
    
    # Define reaction components
    metals = ['Pd', 'Ni', 'Ru', 'Fe', 'Cu']
    
    ligands = ['PPh3', 'PCy3', 'SPhos', 'XPhos', 'RuPhos', 'XantPhos', 
               'dppf', 'BINAP', 'P(o-tol)3', 'JohnPhos', 'tBu3P', 'cataCXium']
    
    bases = ['K2CO3', 'K3PO4', 'Cs2CO3', 'NaOtBu', 'NaOH', 'KOH', 'Et3N', 'DBU']
    
    solvents = ['THF', 'Toluene', 'Dioxane', 'DMF']
    
    leaving_groups = ['Br', 'I', 'Cl', 'OTf']
    
    substrate_types = ['aryl', 'heteroaryl', 'vinyl']
    
    # Reactivity scales (0-1, higher = better)
    ligand_scores = {
        'PPh3': 0.6, 'PCy3': 0.75, 'SPhos': 0.9, 'XPhos': 0.95, 
        'RuPhos': 0.9, 'XantPhos': 0.8, 'dppf': 0.75, 'BINAP': 0.85,
        'P(o-tol)3': 0.7, 'JohnPhos': 0.8, 'tBu3P': 0.85, 'cataCXium': 0.8
    }
    
    base_scores = {
        'K2CO3': 0.7, 'K3PO4': 0.75, 'Cs2CO3': 0.9, 'NaOtBu': 0.95,
        'NaOH': 0.6, 'KOH': 0.65, 'Et3N': 0.4, 'DBU': 0.7
    }
    
    solvent_scores = {
        'THF': 0.9, 'Toluene': 0.6, 'Dioxane': 0.85, 'DMF': 0.8
    }
    
    lg_scores = {
        'I': 1.0, 'OTf': 0.9, 'Br': 0.8, 'Cl': 0.5
    }
    
    substrate_scores = {
        'aryl': 1.0, 'heteroaryl': 0.8, 'vinyl': 0.9
    }
    
    # Metal-specific preferences
    metal_preferences = {
        'Pd': {'base_mult': 1.0, 'lg_pref': ['I', 'Br', 'OTf'], 'opt_temp': 90, 'chloride_penalty': 0.4},
        'Ni': {'base_mult': 1.1, 'lg_pref': ['Br', 'Cl', 'I'], 'opt_temp': 100, 'chloride_penalty': 0.1},
        'Ru': {'base_mult': 0.9, 'lg_pref': ['I', 'OTf'], 'opt_temp': 110, 'chloride_penalty': 0.5},
        'Fe': {'base_mult': 0.85, 'lg_pref': ['I', 'Br'], 'opt_temp': 80, 'chloride_penalty': 0.6},
        'Cu': {'base_mult': 0.8, 'lg_pref': ['I', 'Br'], 'opt_temp': 120, 'chloride_penalty': 0.7}
    }
    
    data = []
    
    # Generate factorial combinations
    n_reactions = 5760  # Perera  et al. 
    
    for i in range(n_reactions):
        # Random selection
        metal = np.random.choice(metals)
        ligand = np.random.choice(ligands)
        base = np.random.choice(bases)
        solvent = np.random.choice(solvents)
        lg = np.random.choice(leaving_groups)
        substrate = np.random.choice(substrate_types)
        
        # Continuous parameters
        temperature = np.random.uniform(60, 130)  # °C
        time = np.random.uniform(2, 24)  # hours
        loading = np.random.uniform(0.5, 10)  # mol%
        
        # Steric and electronic parameters
        steric_hindrance = np.random.uniform(0, 1)
        electronic_effect = np.random.uniform(-1, 1)  # EWG negative, EDG positive
        
        # Calculate base yield using chemistry-informed rules
        metal_pref = metal_preferences[metal]
        
        # Base contribution
        base_yield = 0.3 + 0.4 * ligand_scores[ligand]
        base_yield += 0.2 * base_scores[base] * metal_pref['base_mult']
        base_yield += 0.1 * solvent_scores[solvent]
        
        # Leaving group contribution
        lg_contrib = lg_scores[lg]
        if lg == 'Cl':
            lg_contrib *= (1 - metal_pref['chloride_penalty'])
        if lg in metal_pref['lg_pref']:
            lg_contrib *= 1.2
        base_yield += 0.15 * lg_contrib
        
        # Substrate contribution
        base_yield += 0.1 * substrate_scores[substrate]
        
        # Temperature effect (Gaussian around optimum)
        temp_effect = np.exp(-((temperature - metal_pref['opt_temp'])**2) / 500)
        base_yield *= (0.7 + 0.3 * temp_effect)
        
        # Time effect (asymptotic approach to completion)
        time_effect = 1 - np.exp(-time / 8)
        base_yield *= time_effect
        
        # Loading effect (optimal around 2-5 mol%)
        if loading < 2:
            loading_effect = loading / 2
        elif loading > 5:
            loading_effect = 1 - (loading - 5) / 10
        else:
            loading_effect = 1.0
        base_yield *= loading_effect
        
        # Steric hindrance penalty
        base_yield *= (1 - 0.3 * steric_hindrance)
        
        # Electronic effect (EWG accelerate, EDG slow down)
        base_yield *= (1 + 0.1 * electronic_effect)
        
        # Convert to percentage and add noise
        yield_val = base_yield * 100
        yield_val = np.clip(yield_val + np.random.normal(0, 5), 0, 100)
        
        # Determine success (>50% yield)
        success = 1 if yield_val > 50 else 0
        
        data.append({
            'metal': metal,
            'ligand': ligand,
            'base': base,
            'solvent': solvent,
            'leaving_group': lg,
            'substrate_type': substrate,
            'temperature': temperature,
            'time': time,
            'catalyst_loading': loading,
            'steric_hindrance': steric_hindrance,
            'electronic_effect': electronic_effect,
            'yield': yield_val,
            'success': success
        })
    
    df = pd.DataFrame(data)
    
    print(f"Generated {len(df)} reactions")
    print(f"Mean yield: {df['yield'].mean():.1f}%")
    print(f"Success rate: {df['success'].mean()*100:.1f}%")
    print(f"\nYield by metal:")
    print(df.groupby('metal')['yield'].agg(['mean', 'count']))
    
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def prepare_features(df):
    """Encode categorical features and scale numerical features"""
    
    df_encoded = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['metal', 'ligand', 'base', 'solvent', 'leaving_group', 'substrate_type']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # Select features
    feature_cols = [col + '_encoded' for col in categorical_cols] + [
        'temperature', 'time', 'catalyst_loading', 'steric_hindrance', 'electronic_effect'
    ]
    
    X = df_encoded[feature_cols]
    y = df_encoded['yield']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['temperature', 'time', 'catalyst_loading', 'steric_hindrance', 'electronic_effect']
    numerical_indices = [feature_cols.index(col) for col in numerical_cols]
    
    X_scaled = X.copy()
    X_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X.iloc[:, numerical_indices])
    
    return X_scaled, y, feature_cols, label_encoders, scaler

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and compare performance"""
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, 
                                               min_samples_split=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                       learning_rate=0.1, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                                       max_iter=500, random_state=42, early_stopping=True)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='r2', n_jobs=-1)
        
        results.append({
            'Model': name,
            'R² (Train)': r2_train,
            'R² (Test)': r2_test,
            'RMSE (Train)': rmse_train,
            'RMSE (Test)': rmse_test,
            'MAE (Train)': mae_train,
            'MAE (Test)': mae_test,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std()
        })
        
        print(f"  Test R²: {r2_test:.3f}, RMSE: {rmse_test:.2f}%, MAE: {mae_test:.2f}%")
        print(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    results_df = pd.DataFrame(results)
    return results_df, trained_models

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(model, feature_names):
    """Extract and rank feature importances"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("Suzuki-Miyaura ML Benchmarking Study")
    print("University of Houston")
    print("="*70)
    
    # Generate dataset
    print("\n1. Generating dataset...")
    df = generate_suzuki_dataset()
    
    # Save raw data
    df.to_csv('/mnt/user-data/outputs/suzuki_dataset_raw.csv', index=False)
    print("✓ Raw dataset saved")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y, feature_cols, encoders, scaler = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train models
    print("\n3. Training and evaluating models...")
    results_df, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_df.to_csv('/mnt/user-data/outputs/model_performance_results.csv', index=False)
    print("\n✓ Results saved")
    
    # Display results table
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Feature importance for best model
    print("\n4. Analyzing feature importance (XGBoost)...")
    importance_df = analyze_feature_importance(models['XGBoost'], feature_cols)
    if importance_df is not None:
        importance_df.to_csv('/mnt/user-data/outputs/feature_importance.csv', index=False)
        print("\n✓ Top 10 features:")
        print(importance_df.head(10).to_string(index=False))
    
    # Metal-specific analysis
    print("\n5. Catalyst performance by metal...")
    metal_stats = df.groupby('metal').agg({
        'yield': ['mean', 'std', 'count'],
        'success': 'mean'
    }).round(2)
    metal_stats.columns = ['Mean Yield', 'Std Yield', 'N Reactions', 'Success Rate']
    metal_stats['Success Rate'] = (metal_stats['Success Rate'] * 100).round(1)
    metal_stats = metal_stats.sort_values('Mean Yield', ascending=False)
    
    print(metal_stats)
    metal_stats.to_csv('/mnt/user-data/outputs/metal_catalyst_performance.csv')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - suzuki_dataset_raw.csv (full dataset)")
    print("  - model_performance_results.csv (benchmarking results)")
    print("  - feature_importance.csv (feature rankings)")
    print("  - metal_catalyst_performance.csv (catalyst analysis)")
