import pandas as pd
import numpy as np
from joblib import load, dump
import warnings
warnings.filterwarnings('ignore')

# First, let's create a simplified version of the training script
def train_model_simple():
    """
    Simple training script - run this ONCE to create the model
    """
    print("=== Training Enhanced AURA Model ===")
    
    # Load your data (replace with your actual file path)
    df = pd.read_csv("labeled_fraud_data.csv", compression='gzip')
    print(f"Loaded {len(df)} records")
    
    # Basic feature engineering using only available columns
    def create_features(df):
        df_feat = df.copy()
        
        # Core DIF-style ratios
        df_feat['charitable_ratio'] = df_feat['e19800'] / np.maximum(df_feat['c00100'], 1)
        df_feat['medical_ratio'] = df_feat['e17500'] / np.maximum(df_feat['c00100'], 1)
        df_feat['mortgage_ratio'] = df_feat['e19200'] / np.maximum(df_feat['c00100'], 1)
        df_feat['itemized_ratio'] = df_feat['c04470'] / np.maximum(df_feat['c00100'], 1)
        df_feat['business_ratio'] = df_feat['e00900'] / np.maximum(df_feat['e00200'], 1)
        
        # Rounded number detection
        for col in ['e19800', 'e17500', 'e19200', 'e00200']:
            if col in df_feat.columns:
                df_feat[f'{col}_rounded'] = ((df_feat[col] % 100 == 0) & (df_feat[col] > 0)).astype(int)
        
        # Income per exemption
        df_feat['income_per_exemption'] = df_feat['c00100'] / np.maximum(df_feat['XTOT'], 1)
        
        # EITC ratio
        df_feat['eitc_ratio'] = df_feat['eitc'] / np.maximum(df_feat['c00100'], 1)
        
        # Business income indicator
        df_feat['has_business'] = (df_feat['e00900'] > 0).astype(int)
        df_feat['business_loss'] = (df_feat['e00900'] < 0).astype(int)
        
        # High deduction indicators
        df_feat['high_charitable'] = (df_feat['charitable_ratio'] > 0.1).astype(int)
        df_feat['high_medical'] = (df_feat['medical_ratio'] > 0.075).astype(int)
        
        # Multiple rounded fields
        rounded_cols = [f'{col}_rounded' for col in ['e19800', 'e17500', 'e19200', 'e00200'] 
                       if f'{col}_rounded' in df_feat.columns]
        df_feat['total_rounded'] = df_feat[rounded_cols].sum(axis=1)
        
        return df_feat
    
    # Create features
    df_features = create_features(df)
    
    # Prepare training data
    drop_cols = ["is_fraud", "fraud_type", "fraud_severity"]
    X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
    y = df_features["is_fraud"]
    
    # Fill any missing values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Simple train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import RobustScaler
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train_scaled)
    
    # Test performance
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    iso_scores = -iso.decision_function(X_test_scaled)
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    
    # Ensemble
    ensemble_scores = 0.7 * rf_proba + 0.3 * iso_scores_norm
    
    # Evaluate
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, ensemble_scores)
    print(f"Model AUC: {auc:.4f}")
    
    # Save models and scaler
    models = {
        'rf': rf,
        'iso': iso,
        'scaler': scaler,
        'feature_columns': X.columns.tolist()
    }
    
    dump(models, "enhanced_fraud_model.pkl")
    print("Model saved as 'enhanced_fraud_model.pkl'")
    
    return models

# Simple prediction script (like your original)
def predict_fraud_simple(data_path="labeled_fraud_data.csv", model_path="enhanced_fraud_model.pkl"):
    """
    Simple prediction script - similar to your original script.py
    """
    print("=== Loading Model and Data ===")
    
    # Load trained models
    try:
        models = load(model_path)
        rf = models['rf']
        iso = models['iso'] 
        scaler = models['scaler']
        feature_columns = models['feature_columns']
        print("✓ Models loaded successfully")
    except FileNotFoundError:
        print("❌ Model not found. Run train_model_simple() first!")
        return None
    
    # Load new data
    new_data = pd.read_csv(data_path, compression='gzip')
    
    # For demo, let's use non-fraud cases (like your original script)
    if 'is_fraud' in new_data.columns:
        new_data_clean = new_data[new_data["is_fraud"]==1].copy()
        print(f"Using {len(new_data_clean)} non-fraud cases for prediction")
    else:
        new_data_clean = new_data.copy()
        print(f"Processing {len(new_data_clean)} records")
    
    # Remove fraud labels for prediction
    drop_cols = ["is_fraud", "fraud_type", "fraud_severity"]
    for col in drop_cols:
        if col in new_data_clean.columns:
            new_data_clean = new_data_clean.drop(columns=[col])
    
    # Create same features as training
    def create_features(df):
        df_feat = df.copy()
        
        # Core DIF-style ratios
        df_feat['charitable_ratio'] = df_feat['e19800'] / np.maximum(df_feat['c00100'], 1)
        df_feat['medical_ratio'] = df_feat['e17500'] / np.maximum(df_feat['c00100'], 1) 
        df_feat['mortgage_ratio'] = df_feat['e19200'] / np.maximum(df_feat['c00100'], 1)
        df_feat['itemized_ratio'] = df_feat['c04470'] / np.maximum(df_feat['c00100'], 1)
        df_feat['business_ratio'] = df_feat['e00900'] / np.maximum(df_feat['e00200'], 1)
        
        # Rounded number detection
        for col in ['e19800', 'e17500', 'e19200', 'e00200']:
            if col in df_feat.columns:
                df_feat[f'{col}_rounded'] = ((df_feat[col] % 100 == 0) & (df_feat[col] > 0)).astype(int)
        
        # Income per exemption
        df_feat['income_per_exemption'] = df_feat['c00100'] / np.maximum(df_feat['XTOT'], 1)
        
        # EITC ratio
        df_feat['eitc_ratio'] = df_feat['eitc'] / np.maximum(df_feat['c00100'], 1)
        
        # Business income indicator
        df_feat['has_business'] = (df_feat['e00900'] > 0).astype(int)
        df_feat['business_loss'] = (df_feat['e00900'] < 0).astype(int)
        
        # High deduction indicators
        df_feat['high_charitable'] = (df_feat['charitable_ratio'] > 0.1).astype(int)
        df_feat['high_medical'] = (df_feat['medical_ratio'] > 0.075).astype(int)
        
        # Multiple rounded fields
        rounded_cols = [f'{col}_rounded' for col in ['e19800', 'e17500', 'e19200', 'e00200'] 
                       if f'{col}_rounded' in df_feat.columns]
        df_feat['total_rounded'] = df_feat[rounded_cols].sum(axis=1)
        
        return df_feat
    
    print("=== Creating Features ===")
    new_data_features = create_features(new_data_clean)
    
    # Prepare features for prediction
    X = new_data_features[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
    
    print("=== Generating Predictions ===")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # RF prediction
    rf_proba = rf.predict_proba(X_scaled)[:, 1]
    
    # Isolation Forest anomaly score
    iso_scores = -iso.decision_function(X_scaled)
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    
    # Ensemble score
    ensemble_scores = 0.7 * rf_proba + 0.3 * iso_scores_norm
    
    # Add results to dataframe
    results = new_data_clean.copy()
    results['fraud_risk_score'] = ensemble_scores
    results['rf_probability'] = rf_proba
    results['anomaly_score'] = iso_scores_norm
    
    # Add risk categories
    results['risk_level'] = pd.cut(
        results['fraud_risk_score'],
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
        labels=['Low', 'Medium-Low', 'Medium', 'High', 'Very High']
    )
    
    # Simple output like your original script
    print("=== Results ===")
    print(f"Total records processed: {len(results)}")
    print(f"Average fraud risk score: {results['fraud_risk_score'].mean():.4f}")
    print(f"High risk cases (>0.7): {(results['fraud_risk_score'] > 0.7).sum()}")
    print(f"Very high risk cases (>0.9): {(results['fraud_risk_score'] > 0.9).sum()}")
    
    print("\nRisk Level Distribution:")
    risk_counts = results['risk_level'].value_counts().sort_index()
    for level, count in risk_counts.items():
        print(f"  {level}: {count} ({count/len(results)*100:.1f}%)")
    
    print("\nTop 10 Highest Risk Cases:")
    top_cases = results.nlargest(10, 'fraud_risk_score')
    for i, (idx, row) in enumerate(top_cases.iterrows(), 1):
        risk_score = row['fraud_risk_score']
        agi = row.get('c00100', 0)
        charitable = row.get('e19800', 0)
        medical = row.get('e17500', 0)
        print(f"{i:2d}. Risk: {risk_score:.3f} | AGI: ${agi:>8,.0f} | Charitable: ${charitable:>6,.0f} | Medical: ${medical:>6,.0f}")
    
    # Show fraud risk scores for first 20 cases (like your original)
    print(f"\nFirst 20 Fraud Risk Scores:")
    print(results[['fraud_risk_score']].head(20))
    
    return results

# Main execution (like your original script structure)
if __name__ == "__main__":
    
    # Step 1: Train the model (run once)
    print("Step 1: Training model...")
    try:
        models = load("enhanced_fraud_model.pkl")
        print("✓ Model already exists, skipping training")
    except FileNotFoundError:
        print("Model not found, training new model...")
        models = train_model_simple()
    
    # Step 2: Run predictions (like your original script)
    print("\nStep 2: Running predictions...")
    results = predict_fraud_simple()
    
    if results is not None:
        # Save results
        results.to_csv("fraud_risk_results.csv", index=False)
        print(f"\n✓ Results saved to 'fraud_risk_results.csv'")
        print(f"✓ Total records with fraud risk scores: {len(results)}")
