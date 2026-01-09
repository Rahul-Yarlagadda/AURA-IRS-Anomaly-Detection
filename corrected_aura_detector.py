import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

class CorrectedAURAStyleDetector:
    """
    AURA-style fraud detection using ONLY available columns from message.txt
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_weights = {}
        self.income_bracket_profiles = {}
        
    def engineer_available_features(self, df):
        """
        Create AURA-style features using ONLY columns available in message.txt
        """
        df_enhanced = df.copy()
        
        # ===== 1. CORE DIF-STYLE RATIO FEATURES =====
        # These are the heart of IRS DIF system - comparing taxpayer to peers
        
        # Charitable deduction ratio (major audit trigger)
        df_enhanced['charitable_to_income_ratio'] = df_enhanced['e19800'] / np.maximum(df_enhanced['c00100'], 1)
        
        # Medical deduction ratio (high scrutiny area) 
        df_enhanced['medical_to_income_ratio'] = df_enhanced['e17500'] / np.maximum(df_enhanced['c00100'], 1)
        
        # Mortgage interest ratio
        df_enhanced['mortgage_to_income_ratio'] = df_enhanced['e19200'] / np.maximum(df_enhanced['c00100'], 1)
        
        # Business income to wage ratio (self-employment vs W2)
        df_enhanced['business_to_wage_ratio'] = df_enhanced['e00900'] / np.maximum(df_enhanced['e00200'], 1)
        
        # Total itemized deduction ratio
        df_enhanced['itemized_to_income_ratio'] = df_enhanced['c04470'] / np.maximum(df_enhanced['c00100'], 1)
        
        # ===== 2. ROUNDED NUMBER DETECTION =====
        # IRS flags suspiciously rounded amounts as potential estimates
        
        def detect_rounded_numbers(series, thresholds=[100, 500, 1000]):
            """Detect various levels of rounded numbers"""
            rounded_score = 0
            for threshold in thresholds:
                if series % threshold == 0 and series > 0:
                    rounded_score += 1
            return rounded_score
        
        # Apply to key deduction fields
        key_fields = ['e19800', 'e17500', 'e19200', 'e00200', 'e00900']
        for field in key_fields:
            if field in df_enhanced.columns:
                df_enhanced[f'{field}_rounded'] = df_enhanced[field].apply(
                    lambda x: detect_rounded_numbers(x) if pd.notna(x) else 0
                )
        
        # Total rounded fields score
        rounded_cols = [f'{field}_rounded' for field in key_fields if f'{field}_rounded' in df_enhanced.columns]
        df_enhanced['total_rounded_score'] = df_enhanced[rounded_cols].sum(axis=1)
        
        # ===== 3. STATISTICAL OUTLIER DETECTION =====
        # Calculate Z-scores for key financial metrics
        
        financial_metrics = ['c00100', 'e19800', 'e17500', 'e19200', 'e00200', 'e00900', 'c04470']
        for metric in financial_metrics:
            if metric in df_enhanced.columns:
                mean_val = df_enhanced[metric].mean()
                std_val = df_enhanced[metric].std()
                if std_val > 0:
                    df_enhanced[f'{metric}_zscore'] = np.abs((df_enhanced[metric] - mean_val) / std_val)
                else:
                    df_enhanced[f'{metric}_zscore'] = 0
        
        # Count of high Z-scores (>2 standard deviations)
        zscore_cols = [f'{metric}_zscore' for metric in financial_metrics if f'{metric}_zscore' in df_enhanced.columns]
        df_enhanced['high_zscore_count'] = (df_enhanced[zscore_cols] > 2).sum(axis=1)
        
        # ===== 4. LIFESTYLE INCONSISTENCY FEATURES =====
        # Income vs family size
        df_enhanced['income_per_exemption'] = df_enhanced['c00100'] / np.maximum(df_enhanced['XTOT'], 1)
        
        # Low income but many dependents (potential EITC fraud)
        df_enhanced['low_income_many_dependents'] = (
            (df_enhanced['c00100'] < 30000) & (df_enhanced['XTOT'] > 3)
        ).astype(int)
        
        # ===== 5. TAX CREDIT PATTERNS =====
        # EITC anomalies (high IRS focus area)
        df_enhanced['eitc_to_income_ratio'] = df_enhanced['eitc'] / np.maximum(df_enhanced['c00100'], 1)
        
        # Multiple credit usage pattern
        credit_fields = ['eitc', 'c07180', 'c07200', 'c07100', 'c08000']
        credit_count = 0
        total_credits = 0
        for credit in credit_fields:
            if credit in df_enhanced.columns:
                credit_count += (df_enhanced[credit] > 0).astype(int)
                total_credits += df_enhanced[credit]
        
        df_enhanced['multiple_credits_count'] = credit_count
        df_enhanced['total_credits_to_income'] = total_credits / np.maximum(df_enhanced['c00100'], 1)
        
        # ===== 6. BUSINESS INCOME PATTERNS =====
        # Self-employment red flags
        df_enhanced['has_business_income'] = (df_enhanced['e00900'] > 0).astype(int)
        df_enhanced['business_loss_indicator'] = (df_enhanced['e00900'] < 0).astype(int)
        
        # Business expense ratio (if business expenses available)
        if 'e09900' in df_enhanced.columns:
            df_enhanced['business_expense_ratio'] = df_enhanced['e09900'] / np.maximum(df_enhanced['e00900'], 1)
        
        # ===== 7. FILING STATUS AND DEMOGRAPHIC PATTERNS =====
        # Filing status optimization indicators
        df_enhanced['married_filing_separate'] = (df_enhanced['MARS'] == 3).astype(int)
        df_enhanced['head_of_household'] = (df_enhanced['MARS'] == 4).astype(int)
        df_enhanced['single_filer'] = (df_enhanced['MARS'] == 1).astype(int)
        
        # Age-related patterns
        df_enhanced['elderly_taxpayer'] = (df_enhanced['age_head'] >= 65).astype(int)
        df_enhanced['young_taxpayer'] = (df_enhanced['age_head'] < 25).astype(int)
        
        # ===== 8. WITHHOLDING AND REFUND PATTERNS =====
        # Large refund relative to income
        df_enhanced['refund_to_income_ratio'] = df_enhanced['refund'] / np.maximum(df_enhanced['c00100'], 1)
        df_enhanced['large_refund_indicator'] = (df_enhanced['refund'] > df_enhanced['c00100'] * 0.3).astype(int)
        
        # Withholding patterns
        df_enhanced['withholding_ratio'] = (df_enhanced['e07240'] + df_enhanced['e07260']) / np.maximum(df_enhanced['c00100'], 1)
        
        # ===== 9. INCOME SOURCE DIVERSITY =====
        # Multiple income sources complexity
        income_sources = ['e00200', 'e00300', 'e00600', 'e00700', 'e00800', 'e00900']
        income_source_count = 0
        for source in income_sources:
            if source in df_enhanced.columns:
                income_source_count += (df_enhanced[source] > 0).astype(int)
        
        df_enhanced['income_source_diversity'] = income_source_count
        
        # ===== 10. DEDUCTION OPTIMIZATION PATTERNS =====
        # Itemized vs standard deduction benefit
        df_enhanced['itemized_benefit'] = np.maximum(0, df_enhanced['c04470'] - df_enhanced['standard'])
        df_enhanced['itemized_benefit_ratio'] = df_enhanced['itemized_benefit'] / np.maximum(df_enhanced['c00100'], 1)
        
        # Close to standard deduction threshold (potential optimization)
        df_enhanced['close_to_standard'] = (
            np.abs(df_enhanced['c04470'] - df_enhanced['standard']) < (df_enhanced['standard'] * 0.1)
        ).astype(int)

        # ===== 12. AUDIT ADJUSTMENT FEATURES =====
        # If the dataset includes prior IRS audit adjustment data, integrate it as a compliance signal
        if 'audit_adjustment_amount' in df_enhanced.columns:
            # Normalize by income to prevent scale dominance
            df_enhanced['audit_adjustment_ratio'] = (
                df_enhanced['audit_adjustment_amount'] / np.maximum(df_enhanced['c00100'], 1)
            )
            
            # Binary indicator for any prior audit adjustment
            df_enhanced['had_audit_adjustment'] = (df_enhanced['audit_adjustment_amount'] > 0).astype(int)

        
        # ===== 11. GEOGRAPHIC RISK FACTORS =====
        # Income relative to geographic area median (using FIPS)
        if 'fips' in df_enhanced.columns:
            fips_income_median = df_enhanced.groupby('fips')['c00100'].transform('median')
            df_enhanced['income_vs_area_median'] = df_enhanced['c00100'] / np.maximum(fips_income_median, 1)
        
        # ===== 12. ALTERNATIVE MINIMUM TAX INDICATORS =====
        # AMT liability patterns
        df_enhanced['amt_liability'] = df_enhanced['c09600']
        df_enhanced['has_amt'] = (df_enhanced['c09600'] > 0).astype(int)
        
        return df_enhanced
    
    def create_income_bracket_profiles(self, df):
        """
        Create DIF-style peer group profiles by income bracket
        """
        # Define income brackets similar to IRS methodology
        df['income_bracket'] = pd.cut(
            df['c00100'], 
            bins=[-np.inf, 25000, 50000, 100000, 250000, 500000, np.inf],
            labels=['under_25k', '25k_50k', '50k_100k', '100k_250k', '250k_500k', 'over_500k']
        )
        
        profiles = {}
        
        for bracket in df['income_bracket'].unique():
            if pd.isna(bracket):
                continue
                
            bracket_data = df[df['income_bracket'] == bracket]
            
            if len(bracket_data) > 10:  # Ensure sufficient sample size
                profiles[bracket] = {
                    # Charitable giving patterns
                    'charitable_ratio_mean': bracket_data['charitable_to_income_ratio'].mean(),
                    'charitable_ratio_std': bracket_data['charitable_to_income_ratio'].std(),
                    'charitable_ratio_75th': bracket_data['charitable_to_income_ratio'].quantile(0.75),
                    
                    # Medical deduction patterns  
                    'medical_ratio_mean': bracket_data['medical_to_income_ratio'].mean(),
                    'medical_ratio_std': bracket_data['medical_to_income_ratio'].std(),
                    'medical_ratio_75th': bracket_data['medical_to_income_ratio'].quantile(0.75),
                    
                    # Itemized deduction patterns
                    'itemized_ratio_mean': bracket_data['itemized_to_income_ratio'].mean(),
                    'itemized_ratio_std': bracket_data['itemized_to_income_ratio'].std(),
                    'itemized_ratio_75th': bracket_data['itemized_to_income_ratio'].quantile(0.75),
                    
                    # Business income patterns
                    'business_income_pct': (bracket_data['e00900'] > 0).mean(),
                    
                    # Credit usage patterns
                    'eitc_usage_pct': (bracket_data['eitc'] > 0).mean(),
                    'multiple_credits_mean': bracket_data['multiple_credits_count'].mean(),
                }
        
        return profiles
    
    def calculate_dif_style_risk_score(self, df, profiles):
        """
        Calculate composite risk score similar to IRS DIF methodology
        """
        df_scored = df.copy()
        df_scored['dif_risk_score'] = 0
        
        for idx, row in df.iterrows():
            bracket = row['income_bracket']
            if pd.isna(bracket) or bracket not in profiles:
                continue
                
            profile = profiles[bracket]
            risk_score = 0
            
            # 1. Charitable deduction deviation (weight: 20 points max)
            if profile['charitable_ratio_std'] > 0:
                charitable_z = abs(row['charitable_to_income_ratio'] - profile['charitable_ratio_mean']) / profile['charitable_ratio_std']
                risk_score += min(charitable_z * 5, 20)
            
            # High charitable deduction penalty
            if row['charitable_to_income_ratio'] > profile['charitable_ratio_75th'] * 2:
                risk_score += 15
            
            # 2. Medical deduction deviation (weight: 15 points max)
            if profile['medical_ratio_std'] > 0:
                medical_z = abs(row['medical_to_income_ratio'] - profile['medical_ratio_mean']) / profile['medical_ratio_std']
                risk_score += min(medical_z * 4, 15)
            
            # 3. Itemized deduction deviation (weight: 25 points max)
            if profile['itemized_ratio_std'] > 0:
                itemized_z = abs(row['itemized_to_income_ratio'] - profile['itemized_ratio_mean']) / profile['itemized_ratio_std']
                risk_score += min(itemized_z * 6, 25)
            
            # 4. Business income anomalies (weight: 20 points max)
            if row['business_loss_indicator'] == 1:
                risk_score += 15  # Business losses are high audit risk
            
            if row['has_business_income'] == 1 and profile['business_income_pct'] < 0.1:
                risk_score += 10  # Unusual business income for income bracket
            
            # 5. EITC anomalies (weight: 25 points max)
            if row['eitc'] > 0 and profile['eitc_usage_pct'] < 0.1:
                risk_score += 20  # EITC claim unusual for income bracket
            
            if row['eitc_to_income_ratio'] > 0.3:
                risk_score += 15  # Very high EITC relative to income
            
            # 6. Rounded numbers penalty (weight: 15 points max)
            risk_score += min(row['total_rounded_score'] * 3, 15)
            
            # 7. Multiple high Z-scores penalty (weight: 20 points max)
            risk_score += min(row['high_zscore_count'] * 5, 20)
            
            # 8. Lifestyle inconsistency (weight: 10 points max)
            if row['low_income_many_dependents'] == 1:
                risk_score += 10
            
            # 9. Large refund relative to income (weight: 10 points max)
            if row['large_refund_indicator'] == 1:
                risk_score += 10
            
            # 10. Multiple credits optimization (weight: 10 points max)
            if row['multiple_credits_count'] > profile['multiple_credits_mean'] + 2:
                risk_score += 10
            
            df_scored.loc[idx, 'dif_risk_score'] = min(risk_score, 200)  # Cap at 200
        
        return df_scored
    
    def train_enhanced_ensemble(self, X, y):
        """
        Train ensemble with DIF-inspired methodology
        """
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Use RobustScaler (better for financial outliers than StandardScaler)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # 1. Random Forest with class balancing
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # Handle imbalanced fraud data
            bootstrap=True
        )
        rf.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf
        
        # 2. Logistic Regression (DIF-style linear scoring)
        lr = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            penalty='l2',
            C=1.0
        )
        lr.fit(X_train_scaled, y_train)
        self.models['logistic'] = lr
        
        # 3. Isolation Forest (unsupervised anomaly detection)
        # iso = IsolationForest(
        #     contamination=0.08,  # Expect ~8% anomalies
        #     random_state=42,
        #     n_jobs=-1,
        #     bootstrap=True,
        #     n_estimators=200
        # )
        iso = IsolationForest(
            contamination=0.2,  # Expect ~20% anomalies (match your dataset)
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            n_estimators=200
        )

        iso.fit(X_train_scaled)
        self.models['isolation_forest'] = iso
        
        # Store feature importance for interpretation
        self.feature_weights = dict(zip(X.columns, rf.feature_importances_))
        
        return X_test, y_test, X_test_scaled
    
    def predict_audit_risk(self, X_scaled):
        """
        Generate ensemble predictions with audit-focused scoring
        """
        # Random Forest probabilities
        rf_proba = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
        
        # Logistic Regression probabilities  
        lr_proba = self.models['logistic'].predict_proba(X_scaled)[:, 1]
        
        # Isolation Forest anomaly scores (normalized)
        iso_scores = -self.models['isolation_forest'].decision_function(X_scaled)
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        
        # Ensemble weighting (optimized for audit efficiency)
        # ensemble_scores = (
        #     0.5 * rf_proba +           # Primary supervised signal
        #     0.3 * lr_proba +           # Linear DIF-style component
        #     0.2 * iso_scores_norm      # Unsupervised anomaly component
        # )

        ensemble_scores = (
            0.4 * rf_proba +      # was 0.5
            0.4 * lr_proba +      # was 0.3
            0.2 * iso_scores_norm
        )

        
        return ensemble_scores, rf_proba, lr_proba, iso_scores_norm
    
    def evaluate_audit_efficiency(self, y_true, risk_scores):
        """
        Evaluate using audit efficiency metrics (not just classification accuracy)
        """
        results = {}
        
        # Test different audit selection thresholds
        # thresholds = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

        
        for threshold in thresholds:
            selected_for_audit = risk_scores >= threshold
            
            if selected_for_audit.sum() == 0:
                continue
                
            # Audit efficiency metrics
            true_positives = ((selected_for_audit) & (y_true == 1)).sum()
            total_audits = selected_for_audit.sum()
            total_fraud_cases = (y_true == 1).sum()
            
            audit_success_rate = true_positives / max(total_audits, 1)  # Precision for audit
            fraud_detection_rate = true_positives / max(total_fraud_cases, 1)  # Recall
            audit_burden = total_audits / len(y_true)  # % of population audited
            
            results[threshold] = {
                'audit_success_rate': audit_success_rate,
                'fraud_detection_rate': fraud_detection_rate, 
                'audit_burden': audit_burden,
                'total_audits': total_audits,
                'fraud_cases_found': true_positives
            }
        
        return results
    
    def get_top_risk_indicators(self, top_n=15):
        """
        Get most important risk factors for audit selection
        """
        if not self.feature_weights:
            return []
        
        sorted_features = sorted(
            self.feature_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:top_n]

# # Main training function using ONLY available columns
# def train_corrected_aura_model(data_path="labeled_fraud_data.csv"):
#     """
#     Train the corrected AURA-style model using only available columns
#     """
#     print("=== Training Corrected AURA-Style Fraud Detector ===")
    
#     # Load data
#     df = pd.read_csv(data_path, compression='gzip')
#     print(f"Loaded {len(df)} records")
    
#     # Initialize detector
#     detector = CorrectedAURAStyleDetector()
    
#     # Engineer features using ONLY available columns
#     print("Engineering features using available columns only...")
#     df_enhanced = detector.engineer_available_features(df)
    
#     # Create income bracket profiles
#     print("Creating income bracket peer profiles...")
#     profiles = detector.create_income_bracket_profiles(df_enhanced)
#     detector.income_bracket_profiles = profiles
    
#     # Calculate DIF-style risk scores
#     print("Calculating DIF-style risk scores...")
#     df_scored = detector.calculate_dif_style_risk_score(df_enhanced, profiles)
    
#     # Prepare features and target
#     exclude_cols = ["is_fraud", "fraud_type", "fraud_severity", "income_bracket"]
#     feature_cols = [col for col in df_scored.columns if col not in exclude_cols]
    
#     X = df_scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
#     y = df_scored["is_fraud"]
    
#     print(f"Training with {len(X.columns)} engineered features")
    
#     # Train ensemble
#     X_test, y_test, X_test_scaled = detector.train_enhanced_ensemble(X, y)
    
#     # Generate predictions
#     ensemble_scores, rf_proba, lr_proba, iso_scores = detector.predict_audit_risk(X_test_scaled)
    
#     # Evaluate audit efficiency
#     print("\n=== Audit Efficiency Results ===")
#     efficiency_results = detector.evaluate_audit_efficiency(y_test, ensemble_scores)
    
#     print("Threshold | Success Rate | Detection Rate | Audit Burden | Total Audits")
#     print("-" * 70)
#     for threshold, metrics in efficiency_results.items():
#         print(f"{threshold:8.2f} | {metrics['audit_success_rate']:11.3f} | "
#               f"{metrics['fraud_detection_rate']:13.3f} | {metrics['audit_burden']:11.3f} | "
#               f"{metrics['total_audits']:11d}")
    
#     # Show top risk indicators
#     print(f"\n=== Top Risk Indicators ===")
#     top_indicators = detector.get_top_risk_indicators()
#     for i, (indicator, importance) in enumerate(top_indicators, 1):
#         print(f"{i:2d}. {indicator:<30} {importance:.4f}")
    
#     # Overall performance
#     auc_score = roc_auc_score(y_test, ensemble_scores)
#     print(f"\nOverall AUC Score: {auc_score:.4f}")
    
#     # Save model
#     dump(detector, "corrected_aura_detector.pkl")
#     print("\nModel saved as 'corrected_aura_detector.pkl'")
    
#     return detector, efficiency_results

def train_corrected_aura_model(data_path="labeled_fraud_data.csv"):
    """
    Train the corrected AURA-style model using only pre-audit features.
    Uses audit_adjustment_amount only to define the target (non-compliance label),
    not as an input feature.
    """
    print("=== Training Corrected AURA-Style Fraud Detector ===")

    # Load data
    df = pd.read_csv(data_path, compression='gzip')
    print(f"Loaded {len(df):,} records")


    # ===== 1. Check audit_adjustment_amount presence =====
    if 'audit_adjustment_amount' not in df.columns:
        raise ValueError("Training data must include 'audit_adjustment_amount' column for labeling.")

    # ===== 2. Create non-compliance label =====
    df['is_non_compliant'] = (df['audit_adjustment_amount'] > 0).astype(int)

    # ===== 3. Engineer features (excluding post-audit columns) =====
    print("Engineering features using available columns only...")
    detector = CorrectedAURAStyleDetector()
    df_enhanced = detector.engineer_available_features(df)

    # Drop audit_adjustment_amount from features â€” avoid label leakage
    if 'audit_adjustment_amount' in df_enhanced.columns:
        df_enhanced = df_enhanced.drop(columns=['audit_adjustment_amount'])

    # ===== 4. Create income bracket profiles =====
    print("Creating income bracket peer profiles...")
    profiles = detector.create_income_bracket_profiles(df_enhanced)
    detector.income_bracket_profiles = profiles

    # ===== 5. Calculate DIF-style risk scores =====
    print("Calculating DIF-style risk scores...")
    df_scored = detector.calculate_dif_style_risk_score(df_enhanced, profiles)

    # ===== 6. Prepare features and labels =====
    X = df_scored.drop(columns=['is_non_compliant'], errors='ignore')
    y = df['is_non_compliant']

    leakage_features = [
    'audit_adjustment_ratio',
    'had_audit_adjustment',
    'audit_adjustment_amount',
    'is_fraud',  # if present
]

    # Drop them safely if they exist
    X = X.drop(columns=[c for c in leakage_features if c in X.columns], errors='ignore')

    # Clean and prepare numeric data
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert strings like 'none' to NaN
    X = X.fillna(0)                              # Replace NaN with 0

    print(f"Training with {X.shape[1]} engineered features")

    # ===== 7. Train the ensemble =====
    X_test, y_test, X_test_scaled = detector.train_enhanced_ensemble(X, y)

    # ===== 8. Evaluate audit efficiency =====
    # results = detector.evaluate_audit_efficiency(y_test, detector.models['random_forest'].predict_proba(X_test_scaled)[:, 1])
    # Use full ensemble (RF + LR + ISO) for risk scoring
    ensemble_scores, _, _, _ = detector.predict_audit_risk(X_test_scaled)
    results = detector.evaluate_audit_efficiency(y_test, ensemble_scores)


    print("\n=== Audit Efficiency Results ===")
    print("Threshold | Success Rate | Detection Rate | Audit Burden | Total Audits")
    print("----------------------------------------------------------------------")
    for th, metrics in results.items():
        print(f"    {th:.2f} | {metrics['audit_success_rate']:.3f} | {metrics['fraud_detection_rate']:.3f} | "
              f"{metrics['audit_burden']:.3f} | {metrics['total_audits']:,}")

    # ===== 9. Top risk indicators =====
    print("\n=== Top Risk Indicators ===")
    top_features = detector.get_top_risk_indicators(15)
    for i, (f, imp) in enumerate(top_features, 1):
        print(f"{i:2d}. {f:30s} {imp:.4f}")

    # ===== 10. AUC and model save =====
    rf_proba = detector.models['random_forest'].predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, rf_proba)
    print(f"\nOverall AUC Score: {auc:.4f}")

    dump(detector, "corrected_aura_detector.pkl")
    print("\nModel saved as 'corrected_aura_detector.pkl'")
    return detector

if __name__ == "__main__":
    detector = train_corrected_aura_model()



# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, IsolationForest
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.linear_model import LogisticRegression
# from joblib import dump, load
# import warnings
# warnings.filterwarnings('ignore')

# class CorrectedAURAStyleDetector:
#     """
#     AURA-style fraud detection using ONLY available columns from message.txt
#     """
    
#     def __init__(self):
#         self.models = {}
#         self.scalers = {}
#         self.feature_weights = {}
#         self.income_bracket_profiles = {}
        
#     def engineer_available_features(self, df):
#         """
#         Create AURA-style features using ONLY columns available in message.txt
#         """
#         df_enhanced = df.copy()
        
#         # ===== 1. CORE DIF-STYLE RATIO FEATURES =====
#         # These are the heart of IRS DIF system - comparing taxpayer to peers
        
#         # Charitable deduction ratio (major audit trigger)
#         df_enhanced['charitable_to_income_ratio'] = df_enhanced['e19800'] / np.maximum(df_enhanced['c00100'], 1)
        
#         # Medical deduction ratio (high scrutiny area) 
#         df_enhanced['medical_to_income_ratio'] = df_enhanced['e17500'] / np.maximum(df_enhanced['c00100'], 1)
        
#         # Mortgage interest ratio
#         df_enhanced['mortgage_to_income_ratio'] = df_enhanced['e19200'] / np.maximum(df_enhanced['c00100'], 1)
        
#         # Business income to wage ratio (self-employment vs W2)
#         df_enhanced['business_to_wage_ratio'] = df_enhanced['e00900'] / np.maximum(df_enhanced['e00200'], 1)
        
#         # Total itemized deduction ratio
#         df_enhanced['itemized_to_income_ratio'] = df_enhanced['c04470'] / np.maximum(df_enhanced['c00100'], 1)
        
#         # ===== 2. ROUNDED NUMBER DETECTION =====
#         # IRS flags suspiciously rounded amounts as potential estimates
        
#         def detect_rounded_numbers(series, thresholds=[100, 500, 1000]):
#             """Detect various levels of rounded numbers"""
#             rounded_score = 0
#             for threshold in thresholds:
#                 if series % threshold == 0 and series > 0:
#                     rounded_score += 1
#             return rounded_score
        
#         # Apply to key deduction fields
#         key_fields = ['e19800', 'e17500', 'e19200', 'e00200', 'e00900']
#         for field in key_fields:
#             if field in df_enhanced.columns:
#                 df_enhanced[f'{field}_rounded'] = df_enhanced[field].apply(
#                     lambda x: detect_rounded_numbers(x) if pd.notna(x) else 0
#                 )
        
#         # Total rounded fields score
#         rounded_cols = [f'{field}_rounded' for field in key_fields if f'{field}_rounded' in df_enhanced.columns]
#         df_enhanced['total_rounded_score'] = df_enhanced[rounded_cols].sum(axis=1)
        
#         # ===== 3. STATISTICAL OUTLIER DETECTION =====
#         # Calculate Z-scores for key financial metrics
        
#         financial_metrics = ['c00100', 'e19800', 'e17500', 'e19200', 'e00200', 'e00900', 'c04470']
#         for metric in financial_metrics:
#             if metric in df_enhanced.columns:
#                 mean_val = df_enhanced[metric].mean()
#                 std_val = df_enhanced[metric].std()
#                 if std_val > 0:
#                     df_enhanced[f'{metric}_zscore'] = np.abs((df_enhanced[metric] - mean_val) / std_val)
#                 else:
#                     df_enhanced[f'{metric}_zscore'] = 0
        
#         # Count of high Z-scores (>2 standard deviations)
#         zscore_cols = [f'{metric}_zscore' for metric in financial_metrics if f'{metric}_zscore' in df_enhanced.columns]
#         df_enhanced['high_zscore_count'] = (df_enhanced[zscore_cols] > 2).sum(axis=1)
        
#         # ===== 4. LIFESTYLE INCONSISTENCY FEATURES =====
#         # Income vs family size
#         df_enhanced['income_per_exemption'] = df_enhanced['c00100'] / np.maximum(df_enhanced['XTOT'], 1)
        
#         # Low income but many dependents (potential EITC fraud)
#         df_enhanced['low_income_many_dependents'] = (
#             (df_enhanced['c00100'] < 30000) & (df_enhanced['XTOT'] > 3)
#         ).astype(int)
        
#         # ===== 5. TAX CREDIT PATTERNS =====
#         # EITC anomalies (high IRS focus area)
#         df_enhanced['eitc_to_income_ratio'] = df_enhanced['eitc'] / np.maximum(df_enhanced['c00100'], 1)
        
#         # Multiple credit usage pattern
#         credit_fields = ['eitc', 'c07180', 'c07200', 'c07100', 'c08000']
#         credit_count = 0
#         total_credits = 0
#         for credit in credit_fields:
#             if credit in df_enhanced.columns:
#                 credit_count += (df_enhanced[credit] > 0).astype(int)
#                 total_credits += df_enhanced[credit]
        
#         df_enhanced['multiple_credits_count'] = credit_count
#         df_enhanced['total_credits_to_income'] = total_credits / np.maximum(df_enhanced['c00100'], 1)
        
#         # ===== 6. BUSINESS INCOME PATTERNS =====
#         # Self-employment red flags
#         df_enhanced['has_business_income'] = (df_enhanced['e00900'] > 0).astype(int)
#         df_enhanced['business_loss_indicator'] = (df_enhanced['e00900'] < 0).astype(int)
        
#         # Business expense ratio (if business expenses available)
#         if 'e09900' in df_enhanced.columns:
#             df_enhanced['business_expense_ratio'] = df_enhanced['e09900'] / np.maximum(df_enhanced['e00900'], 1)
        
#         # ===== 7. FILING STATUS AND DEMOGRAPHIC PATTERNS =====
#         # Filing status optimization indicators
#         df_enhanced['married_filing_separate'] = (df_enhanced['MARS'] == 3).astype(int)
#         df_enhanced['head_of_household'] = (df_enhanced['MARS'] == 4).astype(int)
#         df_enhanced['single_filer'] = (df_enhanced['MARS'] == 1).astype(int)
        
#         # Age-related patterns
#         df_enhanced['elderly_taxpayer'] = (df_enhanced['age_head'] >= 65).astype(int)
#         df_enhanced['young_taxpayer'] = (df_enhanced['age_head'] < 25).astype(int)
        
#         # ===== 8. WITHHOLDING AND REFUND PATTERNS =====
#         # Large refund relative to income
#         df_enhanced['refund_to_income_ratio'] = df_enhanced['refund'] / np.maximum(df_enhanced['c00100'], 1)
#         df_enhanced['large_refund_indicator'] = (df_enhanced['refund'] > df_enhanced['c00100'] * 0.3).astype(int)
        
#         # Withholding patterns
#         df_enhanced['withholding_ratio'] = (df_enhanced['e07240'] + df_enhanced['e07260']) / np.maximum(df_enhanced['c00100'], 1)
        
#         # ===== 9. INCOME SOURCE DIVERSITY =====
#         # Multiple income sources complexity
#         income_sources = ['e00200', 'e00300', 'e00600', 'e00700', 'e00800', 'e00900']
#         income_source_count = 0
#         for source in income_sources:
#             if source in df_enhanced.columns:
#                 income_source_count += (df_enhanced[source] > 0).astype(int)
        
#         df_enhanced['income_source_diversity'] = income_source_count
        
#         # ===== 10. DEDUCTION OPTIMIZATION PATTERNS =====
#         # Itemized vs standard deduction benefit
#         df_enhanced['itemized_benefit'] = np.maximum(0, df_enhanced['c04470'] - df_enhanced['standard'])
#         df_enhanced['itemized_benefit_ratio'] = df_enhanced['itemized_benefit'] / np.maximum(df_enhanced['c00100'], 1)
        
#         # Close to standard deduction threshold (potential optimization)
#         df_enhanced['close_to_standard'] = (
#             np.abs(df_enhanced['c04470'] - df_enhanced['standard']) < (df_enhanced['standard'] * 0.1)
#         ).astype(int)

#         # ===== 12. AUDIT ADJUSTMENT FEATURES =====
#         # If the dataset includes prior IRS audit adjustment data, integrate it as a compliance signal
#         if 'audit_adjustment_amount' in df_enhanced.columns:
#             # Normalize by income to prevent scale dominance
#             df_enhanced['audit_adjustment_ratio'] = (
#                 df_enhanced['audit_adjustment_amount'] / np.maximum(df_enhanced['c00100'], 1)
#             )
            
#             # Binary indicator for any prior audit adjustment
#             df_enhanced['had_audit_adjustment'] = (df_enhanced['audit_adjustment_amount'] > 0).astype(int)

        
#         # ===== 11. GEOGRAPHIC RISK FACTORS =====
#         # Income relative to geographic area median (using FIPS)
#         if 'fips' in df_enhanced.columns:
#             fips_income_median = df_enhanced.groupby('fips')['c00100'].transform('median')
#             df_enhanced['income_vs_area_median'] = df_enhanced['c00100'] / np.maximum(fips_income_median, 1)
        
#         # ===== 12. ALTERNATIVE MINIMUM TAX INDICATORS =====
#         # AMT liability patterns
#         df_enhanced['amt_liability'] = df_enhanced['c09600']
#         df_enhanced['has_amt'] = (df_enhanced['c09600'] > 0).astype(int)
        
#         return df_enhanced
    
#     def create_income_bracket_profiles(self, df):
#         """
#         Create DIF-style peer group profiles by income bracket
#         """
#         # Define income brackets similar to IRS methodology
#         df['income_bracket'] = pd.cut(
#             df['c00100'], 
#             bins=[-np.inf, 25000, 50000, 100000, 250000, 500000, np.inf],
#             labels=['under_25k', '25k_50k', '50k_100k', '100k_250k', '250k_500k', 'over_500k']
#         )
        
#         profiles = {}
        
#         for bracket in df['income_bracket'].unique():
#             if pd.isna(bracket):
#                 continue
                
#             bracket_data = df[df['income_bracket'] == bracket]
            
#             if len(bracket_data) > 10:  # Ensure sufficient sample size
#                 profiles[bracket] = {
#                     # Charitable giving patterns
#                     'charitable_ratio_mean': bracket_data['charitable_to_income_ratio'].mean(),
#                     'charitable_ratio_std': bracket_data['charitable_to_income_ratio'].std(),
#                     'charitable_ratio_75th': bracket_data['charitable_to_income_ratio'].quantile(0.75),
                    
#                     # Medical deduction patterns  
#                     'medical_ratio_mean': bracket_data['medical_to_income_ratio'].mean(),
#                     'medical_ratio_std': bracket_data['medical_to_income_ratio'].std(),
#                     'medical_ratio_75th': bracket_data['medical_to_income_ratio'].quantile(0.75),
                    
#                     # Itemized deduction patterns
#                     'itemized_ratio_mean': bracket_data['itemized_to_income_ratio'].mean(),
#                     'itemized_ratio_std': bracket_data['itemized_to_income_ratio'].std(),
#                     'itemized_ratio_75th': bracket_data['itemized_to_income_ratio'].quantile(0.75),
                    
#                     # Business income patterns
#                     'business_income_pct': (bracket_data['e00900'] > 0).mean(),
                    
#                     # Credit usage patterns
#                     'eitc_usage_pct': (bracket_data['eitc'] > 0).mean(),
#                     'multiple_credits_mean': bracket_data['multiple_credits_count'].mean(),
#                 }
        
#         return profiles
    
#     def calculate_dif_style_risk_score(self, df, profiles):
#         """
#         Calculate composite risk score similar to IRS DIF methodology
#         """
#         df_scored = df.copy()
#         df_scored['dif_risk_score'] = 0
        
#         for idx, row in df.iterrows():
#             bracket = row['income_bracket']
#             if pd.isna(bracket) or bracket not in profiles:
#                 continue
                
#             profile = profiles[bracket]
#             risk_score = 0
            
#             # 1. Charitable deduction deviation (weight: 20 points max)
#             if profile['charitable_ratio_std'] > 0:
#                 charitable_z = abs(row['charitable_to_income_ratio'] - profile['charitable_ratio_mean']) / profile['charitable_ratio_std']
#                 risk_score += min(charitable_z * 5, 20)
            
#             # High charitable deduction penalty
#             if row['charitable_to_income_ratio'] > profile['charitable_ratio_75th'] * 2:
#                 risk_score += 15
            
#             # 2. Medical deduction deviation (weight: 15 points max)
#             if profile['medical_ratio_std'] > 0:
#                 medical_z = abs(row['medical_to_income_ratio'] - profile['medical_ratio_mean']) / profile['medical_ratio_std']
#                 risk_score += min(medical_z * 4, 15)
            
#             # 3. Itemized deduction deviation (weight: 25 points max)
#             if profile['itemized_ratio_std'] > 0:
#                 itemized_z = abs(row['itemized_to_income_ratio'] - profile['itemized_ratio_mean']) / profile['itemized_ratio_std']
#                 risk_score += min(itemized_z * 6, 25)
            
#             # 4. Business income anomalies (weight: 20 points max)
#             if row['business_loss_indicator'] == 1:
#                 risk_score += 15  # Business losses are high audit risk
            
#             if row['has_business_income'] == 1 and profile['business_income_pct'] < 0.1:
#                 risk_score += 10  # Unusual business income for income bracket
            
#             # 5. EITC anomalies (weight: 25 points max)
#             if row['eitc'] > 0 and profile['eitc_usage_pct'] < 0.1:
#                 risk_score += 20  # EITC claim unusual for income bracket
            
#             if row['eitc_to_income_ratio'] > 0.3:
#                 risk_score += 15  # Very high EITC relative to income
            
#             # 6. Rounded numbers penalty (weight: 15 points max)
#             risk_score += min(row['total_rounded_score'] * 3, 15)
            
#             # 7. Multiple high Z-scores penalty (weight: 20 points max)
#             risk_score += min(row['high_zscore_count'] * 5, 20)
            
#             # 8. Lifestyle inconsistency (weight: 10 points max)
#             if row['low_income_many_dependents'] == 1:
#                 risk_score += 10
            
#             # 9. Large refund relative to income (weight: 10 points max)
#             if row['large_refund_indicator'] == 1:
#                 risk_score += 10
            
#             # 10. Multiple credits optimization (weight: 10 points max)
#             if row['multiple_credits_count'] > profile['multiple_credits_mean'] + 2:
#                 risk_score += 10
            
#             df_scored.loc[idx, 'dif_risk_score'] = min(risk_score, 200)  # Cap at 200
        
#         return df_scored
    
#     def train_enhanced_ensemble(self, X, y):
#         """
#         Train ensemble with DIF-inspired methodology
#         """
#         # Split data with stratification
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, stratify=y, random_state=42
#         )
        
#         # Use RobustScaler (better for financial outliers than StandardScaler)
#         scaler = RobustScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#         self.scalers['main'] = scaler
        
#         # 1. Random Forest with class balancing
#         rf = RandomForestClassifier(
#             n_estimators=300,
#             max_depth=12,
#             min_samples_split=10,
#             min_samples_leaf=5,
#             random_state=42,
#             n_jobs=-1,
#             class_weight='balanced',  # Handle imbalanced fraud data
#             bootstrap=True
#         )
#         rf.fit(X_train_scaled, y_train)
#         self.models['random_forest'] = rf
        
#         # 2. Logistic Regression (DIF-style linear scoring)
#         lr = LogisticRegression(
#             random_state=42,
#             class_weight='balanced',
#             max_iter=1000,
#             penalty='l2',
#             C=1.0
#         )
#         lr.fit(X_train_scaled, y_train)
#         self.models['logistic'] = lr
        
#         # 3. Isolation Forest (unsupervised anomaly detection)
#         iso = IsolationForest(
#         contamination=0.2,  # Expect ~20% anomalies (match your dataset)
#         random_state=42,
#         n_jobs=-1,
#         bootstrap=True,
#         n_estimators=200
#         )

#         iso.fit(X_train_scaled)
#         self.models['isolation_forest'] = iso
        
#         # Store feature importance for interpretation
#         self.feature_weights = dict(zip(X.columns, rf.feature_importances_))
        
#         return X_test, y_test, X_test_scaled
    
#     def predict_audit_risk(self, X_scaled):
#         """
#         Generate ensemble predictions with audit-focused scoring
#         """
#         # Random Forest probabilities
#         rf_proba = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
        
#         # Logistic Regression probabilities  
#         lr_proba = self.models['logistic'].predict_proba(X_scaled)[:, 1]
        
#         # Isolation Forest anomaly scores (normalized)
#         iso_scores = -self.models['isolation_forest'].decision_function(X_scaled)
#         iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        
#         # Ensemble weighting (optimized for audit efficiency)
#         ensemble_scores = (
#         0.4 * rf_proba +      # was 0.5
#         0.4 * lr_proba +      # was 0.3
#         0.2 * iso_scores_norm
#         )

        
#         return ensemble_scores, rf_proba, lr_proba, iso_scores_norm
    
#     def evaluate_audit_efficiency(self, y_true, risk_scores):
#         """
#         Evaluate using audit efficiency metrics (not just classification accuracy)
#         """
#         results = {}
        
#         # Test different audit selection thresholds
#         thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

        
#         for threshold in thresholds:
#             selected_for_audit = risk_scores >= threshold
            
#             if selected_for_audit.sum() == 0:
#                 continue
                
#             # Audit efficiency metrics
#             true_positives = ((selected_for_audit) & (y_true == 1)).sum()
#             total_audits = selected_for_audit.sum()
#             total_fraud_cases = (y_true == 1).sum()
            
#             audit_success_rate = true_positives / max(total_audits, 1)  # Precision for audit
#             fraud_detection_rate = true_positives / max(total_fraud_cases, 1)  # Recall
#             audit_burden = total_audits / len(y_true)  # % of population audited
            
#             results[threshold] = {
#                 'audit_success_rate': audit_success_rate,
#                 'fraud_detection_rate': fraud_detection_rate, 
#                 'audit_burden': audit_burden,
#                 'total_audits': total_audits,
#                 'fraud_cases_found': true_positives
#             }
        
#         return results
    
#     def get_top_risk_indicators(self, top_n=15):
#         """
#         Get most important risk factors for audit selection
#         """
#         if not self.feature_weights:
#             return []
        
#         sorted_features = sorted(
#             self.feature_weights.items(), 
#             key=lambda x: x[1], 
#             reverse=True
#         )
        
#         return sorted_features[:top_n]

# # Main training function using ONLY available columns
# def train_corrected_aura_model(data_path="labeled_fraud_data.csv"):
#     """
#     Train the corrected AURA-style model using only available columns
#     """
#     print("=== Training Corrected AURA-Style Fraud Detector ===")
    
#     # Load data
#     df = pd.read_csv(data_path, compression='gzip')
#     print(f"Loaded {len(df)} records")
    
#     # Initialize detector
#     detector = CorrectedAURAStyleDetector()
    
#     # Engineer features using ONLY available columns
#     print("Engineering features using available columns only...")
#     df_enhanced = detector.engineer_available_features(df)
    
#     # Create income bracket profiles
#     print("Creating income bracket peer profiles...")
#     profiles = detector.create_income_bracket_profiles(df_enhanced)
#     detector.income_bracket_profiles = profiles
    
#     # Calculate DIF-style risk scores
#     print("Calculating DIF-style risk scores...")
#     df_scored = detector.calculate_dif_style_risk_score(df_enhanced, profiles)
    
#     # Prepare features and target
#     exclude_cols = ["is_fraud", "fraud_type", "fraud_severity", "income_bracket"]
#     feature_cols = [col for col in df_scored.columns if col not in exclude_cols]
    
#     X = df_scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
#     y = df_scored["is_fraud"]
    
#     print(f"Training with {len(X.columns)} engineered features")
    
#     # Train ensemble
#     X_test, y_test, X_test_scaled = detector.train_enhanced_ensemble(X, y)
    
#     # Generate predictions
#     ensemble_scores, rf_proba, lr_proba, iso_scores = detector.predict_audit_risk(X_test_scaled)
    
#     # Evaluate audit efficiency
#     print("\n=== Audit Efficiency Results ===")
#     efficiency_results = detector.evaluate_audit_efficiency(y_test, ensemble_scores)
    
#     print("Threshold | Success Rate | Detection Rate | Audit Burden | Total Audits")
#     print("-" * 70)
#     for threshold, metrics in efficiency_results.items():
#         print(f"{threshold:8.2f} | {metrics['audit_success_rate']:11.3f} | "
#               f"{metrics['fraud_detection_rate']:13.3f} | {metrics['audit_burden']:11.3f} | "
#               f"{metrics['total_audits']:11d}")
    
#     # Show top risk indicators
#     print(f"\n=== Top Risk Indicators ===")
#     top_indicators = detector.get_top_risk_indicators()
#     for i, (indicator, importance) in enumerate(top_indicators, 1):
#         print(f"{i:2d}. {indicator:<30} {importance:.4f}")
    
#     # Overall performance
#     auc_score = roc_auc_score(y_test, ensemble_scores)
#     print(f"\nOverall AUC Score: {auc_score:.4f}")
    
#     # Save model
#     dump(detector, "corrected_aura_detector.pkl")
#     print("\nModel saved as 'corrected_aura_detector.pkl'")
    
#     return detector, efficiency_results

# # def train_corrected_aura_model(data_path="labeled_fraud_data.csv"):
# #     """
# #     Train the corrected AURA-style model using only pre-audit features.
# #     Uses audit_adjustment_amount only to define the target (non-compliance label),
# #     not as an input feature.
# #     """
# #     print("=== Training Corrected AURA-Style Fraud Detector ===")

# #     # Load data
# #     df = pd.read_csv(data_path, compression='gzip')
# #     print(f"Loaded {len(df):,} records")


# #     # ===== 1. Check audit_adjustment_amount presence =====
# #     if 'audit_adjustment_amount' not in df.columns:
# #         raise ValueError("Training data must include 'audit_adjustment_amount' column for labeling.")

# #     # ===== 2. Create non-compliance label =====
# #     df['is_non_compliant'] = (df['audit_adjustment_amount'] > 0).astype(int)

# #     # ===== 3. Engineer features (excluding post-audit columns) =====
# #     print("Engineering features using available columns only...")
# #     detector = CorrectedAURAStyleDetector()
# #     df_enhanced = detector.engineer_available_features(df)

# #     # Drop audit_adjustment_amount from features — avoid label leakage
# #     if 'audit_adjustment_amount' in df_enhanced.columns:
# #         df_enhanced = df_enhanced.drop(columns=['audit_adjustment_amount'])

# #     # ===== 4. Create income bracket profiles =====
# #     print("Creating income bracket peer profiles...")
# #     profiles = detector.create_income_bracket_profiles(df_enhanced)
# #     detector.income_bracket_profiles = profiles

# #     # ===== 5. Calculate DIF-style risk scores =====
# #     print("Calculating DIF-style risk scores...")
# #     df_scored = detector.calculate_dif_style_risk_score(df_enhanced, profiles)

# #     # ===== 6. Prepare features and labels =====
# #     X = df_scored.drop(columns=['is_non_compliant'], errors='ignore')
# #     y = df['is_non_compliant']

# #     leakage_features = [
# #     'audit_adjustment_ratio',
# #     'had_audit_adjustment',
# #     'audit_adjustment_amount',
# #     'is_fraud',  # if present
# # ]

# #     # Drop them safely if they exist
# #     X = X.drop(columns=[c for c in leakage_features if c in X.columns], errors='ignore')

# #     # Clean and prepare numeric data
# #     X = X.apply(pd.to_numeric, errors='coerce')  # Convert strings like 'none' to NaN
# #     X = X.fillna(0)                              # Replace NaN with 0

# #     print(f"Training with {X.shape[1]} engineered features")

# #     # ===== 7. Train the ensemble =====
# #     X_test, y_test, X_test_scaled = detector.train_enhanced_ensemble(X, y)

# #     # ===== 8. Evaluate audit efficiency =====
# #     # Use full ensemble (RF + LR + ISO) for risk scoring
# #     ensemble_scores, _, _, _ = detector.predict_audit_risk(X_test_scaled)
# #     results = detector.evaluate_audit_efficiency(y_test, ensemble_scores)


# #     print("\n=== Audit Efficiency Results ===")
# #     print("Threshold | Success Rate | Detection Rate | Audit Burden | Total Audits")
# #     print("----------------------------------------------------------------------")
# #     for th, metrics in results.items():
# #         print(f"    {th:.2f} | {metrics['audit_success_rate']:.3f} | {metrics['fraud_detection_rate']:.3f} | "
# #               f"{metrics['audit_burden']:.3f} | {metrics['total_audits']:,}")

# #     # ===== 9. Top risk indicators =====
# #     print("\n=== Top Risk Indicators ===")
# #     top_features = detector.get_top_risk_indicators(15)
# #     for i, (f, imp) in enumerate(top_features, 1):
# #         print(f"{i:2d}. {f:30s} {imp:.4f}")

# #     # ===== 10. AUC and model save =====
# #     rf_proba = detector.models['random_forest'].predict_proba(X_test_scaled)[:, 1]
# #     auc = roc_auc_score(y_test, rf_proba)
# #     print(f"\nOverall AUC Score: {auc:.4f}")

# #     dump(detector, "corrected_aura_detector.pkl")
# #     print("\nModel saved as 'corrected_aura_detector.pkl'")
# #     return detector

# if __name__ == "__main__":
#     detector = train_corrected_aura_model()


