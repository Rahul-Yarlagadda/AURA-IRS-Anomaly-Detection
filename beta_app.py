import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import your detector class
from corrected_aura_detector import CorrectedAURAStyleDetector

# Page configuration
st.set_page_config(
    page_title="AURA Tax Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .risk-very-high {
        background-color: #ffebee;
        padding: 10px;
        border-left: 4px solid #d32f2f;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #fff3e0;
        padding: 10px;
        border-left: 4px solid #f57c00;
        margin: 10px 0;
    }
    .case-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
    }
    .explanation-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .red-flag {
        color: #d32f2f;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'selected_case' not in st.session_state:
    st.session_state.selected_case = None

def load_trained_model():
    """Load the trained AURA detector"""
    try:
        detector = load("corrected_aura_detector.pkl")
        return detector, True
    except FileNotFoundError:
        return None, False

def process_uploaded_file(uploaded_file):
    """Process uploaded Excel/CSV file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def run_fraud_detection(df, detector):
    """Run fraud detection on uploaded data"""
    try:
        with st.spinner("🔍 Analyzing tax returns... This may take a few minutes."):
            # Remove fraud labels if present
            drop_cols = ["is_fraud", "fraud_type", "fraud_severity"]
            df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            # Engineer features
            progress_bar = st.progress(0)
            st.text("Engineering features...")
            df_enhanced = detector.engineer_available_features(df_clean)
            progress_bar.progress(33)
            
            # Create income bracket column (CRITICAL - must happen before DIF scoring)
            st.text("Creating income brackets...")
            df_enhanced['income_bracket'] = pd.cut(
                df_enhanced['c00100'], 
                bins=[-np.inf, 25000, 50000, 100000, 250000, 500000, np.inf],
                labels=['under_25k', '25k_50k', '50k_100k', '100k_250k', '250k_500k', 'over_500k']
            )
            
            # Create income bracket profiles if not exists
            if not detector.income_bracket_profiles:
                st.text("Creating income bracket profiles...")
                profiles = detector.create_income_bracket_profiles(df_enhanced)
                detector.income_bracket_profiles = profiles
            progress_bar.progress(50)
            
            # Calculate DIF scores
            st.text("Calculating DIF-style risk scores...")
            df_scored = detector.calculate_dif_style_risk_score(df_enhanced, detector.income_bracket_profiles)
            progress_bar.progress(66)
            
            # Prepare features
            exclude_cols = ["income_bracket"]
            feature_cols = [col for col in df_scored.columns if col not in exclude_cols]
            X = df_scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # CRITICAL: Align features with trained model
            # Get the features the model was trained on
            if hasattr(detector.scalers['main'], 'feature_names_in_'):
                trained_features = detector.scalers['main'].feature_names_in_
            else:
                # Fallback: get from random forest model
                trained_features = detector.models['random_forest'].feature_names_in_
            
            # Add missing columns with zeros
            for feature in trained_features:
                if feature not in X.columns:
                    st.warning(f"Adding missing feature: {feature}")
                    X[feature] = 0
            
            # Remove extra columns not in training
            extra_cols = [col for col in X.columns if col not in trained_features]
            if extra_cols:
                X = X.drop(columns=extra_cols)
            
            # Ensure column order matches training
            X = X[trained_features]
            
            # Scale and predict
            st.text("Generating predictions...")
            X_scaled = detector.scalers['main'].transform(X)
            ensemble_scores, rf_proba, lr_proba, iso_scores = detector.predict_audit_risk(X_scaled)
            progress_bar.progress(100)
            
            # Add results to original dataframe
            results = df_clean.copy()
            results['fraud_risk_score'] = ensemble_scores
            results['rf_probability'] = rf_proba
            results['lr_probability'] = lr_proba
            results['anomaly_score'] = iso_scores
            results['dif_score'] = df_scored['dif_risk_score'].values
            
            # Risk categories
            results['risk_level'] = pd.cut(
                results['fraud_risk_score'],
                bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                labels=['Low', 'Medium-Low', 'Medium', 'High', 'Very High']
            )
            
            # Flag non-compliant
            results['non_compliant'] = (results['fraud_risk_score'] >= 0.7).astype(int)
            
            st.text("✓ Predictions complete!")
            return results, X_scaled, list(trained_features)
            
    except Exception as e:
        st.error(f"Error during fraud detection: {str(e)}")
        st.exception(e)
        return None, None, None

def calculate_shap_values(detector, X_scaled):
    """Calculate SHAP values"""
    with st.spinner("📊 Calculating SHAP explanations... This may take a moment."):
        explainer = shap.TreeExplainer(detector.models['random_forest'])
        shap_values = explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values, explainer

def get_red_flags(row, feature_names, shap_vals):
    """Identify specific red flags for a case"""
    red_flags = []
    
    # Charitable deductions
    charitable_ratio = row.get('e19800', 0) / max(row.get('c00100', 1), 1)
    if charitable_ratio > 0.1:
        red_flags.append(f"🚩 High charitable deductions ({charitable_ratio*100:.1f}% of income)")
    
    # Medical deductions
    medical_ratio = row.get('e17500', 0) / max(row.get('c00100', 1), 1)
    if medical_ratio > 0.075:
        red_flags.append(f"🚩 High medical deductions ({medical_ratio*100:.1f}% of income)")
    
    # Rounded numbers
    rounded_count = 0
    for col in ['e19800', 'e17500', 'e19200', 'e00200']:
        if col in row and row[col] > 0 and row[col] % 100 == 0:
            rounded_count += 1
    if rounded_count >= 2:
        red_flags.append(f"🚩 Multiple rounded amounts detected ({rounded_count} fields)")
    
    # Business losses
    if row.get('e00900', 0) < 0:
        red_flags.append(f"🚩 Business losses reported (${abs(row.get('e00900', 0)):,.0f})")
    
    # EITC
    eitc_ratio = row.get('eitc', 0) / max(row.get('c00100', 1), 1)
    if eitc_ratio > 0.3:
        red_flags.append(f"🚩 High EITC claim relative to income ({eitc_ratio*100:.1f}%)")
    
    # Income per exemption
    income_per_exemption = row.get('c00100', 0) / max(row.get('XTOT', 1), 1)
    if income_per_exemption < 15000 and row.get('c00100', 0) > 0:
        red_flags.append(f"🚩 Low income per household member (${income_per_exemption:,.0f})")
    
    # Large refund
    refund_ratio = row.get('refund', 0) / max(row.get('c00100', 1), 1)
    if refund_ratio > 0.3:
        red_flags.append(f"🚩 Large refund relative to income ({refund_ratio*100:.1f}%)")
    
    return red_flags

def generate_shap_explanation(row, shap_vals, feature_names, top_n=5):
    """Generate SHAP-based explanation"""
    # Ensure shap_vals is a 1D array
    if isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim > 1:
            shap_vals = shap_vals.flatten()
    
    # Get feature values
    feature_values = []
    for f in feature_names:
        if f in row.index:
            feature_values.append(row[f])
        else:
            feature_values.append(0)
    
    # Create list of (feature, shap_value, actual_value)
    feature_contributions = []
    for i, (feature, shap_val, value) in enumerate(zip(feature_names, shap_vals, feature_values)):
        # Convert to scalar if needed
        if isinstance(shap_val, np.ndarray):
            shap_val = float(shap_val.flatten()[0])
        else:
            shap_val = float(shap_val)
        
        if isinstance(value, np.ndarray):
            value = float(value.flatten()[0])
        else:
            value = float(value)
            
        feature_contributions.append((feature, shap_val, value))
    
    # Sort by absolute SHAP value
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    explanations = []
    for i, (feature, shap_val, value) in enumerate(feature_contributions[:top_n], 1):
        impact = "increases" if shap_val > 0 else "decreases"
        explanations.append({
            'rank': i,
            'feature': feature,
            'shap_value': shap_val,
            'impact': impact,
            'value': value
        })
    
    return explanations

# ========== PAGE: UPLOAD ==========
def page_upload():
    st.title("🔍 AURA Tax Fraud Detection System")
    st.markdown("### Upload Tax Return Data for Analysis")
    
    # Load model
    detector, model_loaded = load_trained_model()
    
    if not model_loaded:
        st.error("❌ Model not found! Please train the model first.")
        st.code("python3 corrected_aura_detector.py", language="bash")
        return
    
    st.session_state.detector = detector
    st.success("✅ Model loaded successfully")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file with tax return data",
        type=['csv', 'xlsx', 'xls'],
        help="File should contain standard tax return fields (c00100, e19800, e17500, etc.)"
    )
    
    if uploaded_file is None:
        st.info("👆 Upload a file to begin analysis")
        
        with st.expander("📋 Required Data Format"):
            st.markdown("""
            Your file should contain tax return data with these key columns:
            - **c00100**: Adjusted Gross Income
            - **e19800**: Charitable contributions
            - **e17500**: Medical deductions
            - **e19200**: Mortgage interest
            - **e00900**: Business income
            - **e00200**: Wages and salaries
            - **XTOT**: Total exemptions
            - **eitc**: Earned Income Tax Credit
            - **MARS**: Filing status
            - **age_head**: Primary taxpayer age
            - And other standard tax form fields...
            """)
        return
    
    # Process file
    df = process_uploaded_file(uploaded_file)
    
    if df is None:
        return
    
    st.success(f"✅ Loaded {len(df):,} tax returns")
    
    # Show preview
    with st.expander("👀 Preview Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Fast mode option
    use_fast_mode = st.checkbox(
        "⚡ Fast Mode (Skip SHAP - recommended for >1000 returns)",
        value=len(df) > 1000,
        help="SHAP explanations will be calculated on-demand when viewing individual cases"
    )
    
    # Run analysis
    if st.button("🚀 Run Fraud Detection Analysis", type="primary", use_container_width=True):
        try:
            # Save detector to session state
            st.session_state.detector = detector
            
            results, X_scaled, feature_names = run_fraud_detection(df, detector)
            
            if results is None or len(results) == 0:
                st.error("❌ Analysis failed to produce results")
                return
            
            st.success("✅ Model predictions complete!")
            
            # Save to session state FIRST
            st.session_state.results_df = results
            st.session_state.X_scaled = X_scaled
            st.session_state.feature_names = feature_names
            
            # Calculate SHAP (optional)
            if not use_fast_mode:
                try:
                    with st.spinner("Calculating SHAP explanations... (this may take a few minutes)"):
                        shap_values, explainer = calculate_shap_values(detector, X_scaled)
                        st.session_state.shap_values = shap_values
                        st.session_state.shap_explainer = explainer
                        st.success("✅ SHAP calculations complete!")
                except Exception as shap_error:
                    st.warning(f"⚠️ SHAP calculation skipped: {shap_error}")
                    st.session_state.shap_values = None
                    st.session_state.shap_explainer = None
            else:
                st.info("⚡ Fast mode enabled - SHAP will be calculated on-demand")
                st.session_state.shap_values = None
                st.session_state.shap_explainer = None
            
            st.success("✅ Analysis Complete! Navigating to summary...")
            
            # Manual navigation option
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Summary", type="primary", use_container_width=True):
                    st.session_state.page = 'summary'
                    st.rerun()
            with col2:
                if st.button("📋 View Non-Compliant Cases", use_container_width=True):
                    st.session_state.page = 'non_compliant_list'
                    st.rerun()
            
            # Also try auto-navigation
            import time
            time.sleep(0.5)
            st.session_state.page = 'summary'
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed with error: {str(e)}")
            st.exception(e)
            return

# ========== PAGE: SUMMARY ==========
def page_summary():
    st.title("📊 Analysis Summary & Results")
    
    if st.session_state.results_df is None:
        st.warning("No results available. Please upload data first.")
        if st.button("← Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    results = st.session_state.results_df
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("← Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()
    
    # Key Metrics
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_returns = len(results)
    non_compliant_count = results['non_compliant'].sum()
    non_compliant_pct = (non_compliant_count / total_returns * 100) if total_returns > 0 else 0
    avg_risk = results['fraud_risk_score'].mean()
    very_high_risk = (results['fraud_risk_score'] >= 0.9).sum()
    
    with col1:
        st.metric("Total Returns Analyzed", f"{total_returns:,}")
    
    with col2:
        st.metric("Non-Compliant Filings", f"{non_compliant_count:,}", 
                 delta=f"{non_compliant_pct:.1f}% of total", delta_color="inverse")
    
    with col3:
        st.metric("Average Risk Score", f"{avg_risk:.3f}")
    
    with col4:
        st.metric("Very High Risk (>0.9)", f"{very_high_risk:,}")
    
    st.markdown("---")
    
    # Executive Summary with SHAP Insights
    st.header("💡 Executive Summary")
    
    summary_col1, summary_col2 = st.columns([2, 1])
    
    with summary_col1:
        st.markdown(f"""
        ### Analysis Overview
        
        Out of **{total_returns:,}** tax returns analyzed:
        - **{non_compliant_count:,} returns ({non_compliant_pct:.1f}%)** were flagged as non-compliant (risk score ≥ 0.70)
        - **{very_high_risk:,} returns** are considered very high risk (score ≥ 0.90)
        - The average fraud risk score across all returns is **{avg_risk:.3f}**
        
        ### Risk Distribution
        """)
        
        risk_dist = results['risk_level'].value_counts().sort_index()
        for level, count in risk_dist.items():
            pct = count / total_returns * 100
            st.markdown(f"- **{level}**: {count:,} returns ({pct:.1f}%)")
    
    with summary_col2:
        # Risk level pie chart
        fig_pie = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="Risk Distribution",
            color_discrete_map={
                'Low': '#4caf50',
                'Medium-Low': '#8bc34a',
                'Medium': '#ffc107',
                'High': '#ff9800',
                'Very High': '#f44336'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # SHAP Global Insights
    st.header("🎯 Key Risk Factors (SHAP Analysis)")
    
    if st.session_state.shap_values is not None and st.session_state.shap_explainer is not None:
        st.markdown("""
        The AI model identified the following factors as most important in determining fraud risk across all returns:
        """)
        
        # Feature importance from SHAP
        shap_abs_mean = np.abs(st.session_state.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': shap_abs_mean
        }).sort_values('Importance', ascending=False).head(10)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                feature_importance.style.format({'Importance': '{:.4f}'}),
                use_container_width=True,
                height=400
            )
        
        with col2:
            # SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                shap.summary_plot(
                    st.session_state.shap_values[:1000],  # Sample for speed
                    features=results[st.session_state.feature_names].iloc[:1000].values,
                    feature_names=st.session_state.feature_names,
                    show=False,
                    max_display=10
                )
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate SHAP summary plot: {str(e)}")
            finally:
                plt.close()
    else:
        st.info("""
        💡 **Key Risk Factors** (based on model feature importance):
        
        The model uses DIF-style statistical analysis to identify:
        - Charitable donation ratios
        - Medical deduction patterns
        - Rounded number detection
        - Business income anomalies
        - EITC claim patterns
        - Income per exemption ratios
        - Itemized deduction optimization
        
        *Note: Detailed SHAP analysis was skipped in fast mode. Individual cases will still show AI explanations.*
        """)
    
    st.markdown("---")
    
    # Risk Distribution Chart
    st.header("📊 Risk Score Distribution")
    
    fig_hist = px.histogram(
        results,
        x='fraud_risk_score',
        nbins=50,
        title="Distribution of Fraud Risk Scores",
        labels={'fraud_risk_score': 'Fraud Risk Score', 'count': 'Number of Returns'}
    )
    fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                      annotation_text="Non-Compliant Threshold (0.7)")
    fig_hist.add_vline(x=0.9, line_dash="dash", line_color="darkred",
                      annotation_text="Very High Risk (0.9)")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Navigation to non-compliant list
    st.header("🔍 Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 View All Non-Compliant Filings", type="primary", use_container_width=True):
            st.session_state.page = 'non_compliant_list'
            st.rerun()
    
    with col2:
        # Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Full Results (CSV)",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv",
            use_container_width=True
        )

# ========== PAGE: NON-COMPLIANT LIST ==========
def page_non_compliant_list():
    st.title("📋 Non-Compliant Tax Filings")
    
    if st.session_state.results_df is None:
        st.warning("No results available.")
        if st.button("← Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("← Back to Summary"):
            st.session_state.page = 'summary'
            st.rerun()
    
    results = st.session_state.results_df
    non_compliant = results[results['non_compliant'] == 1].sort_values('fraud_risk_score', ascending=False)
    
    st.markdown(f"### Found **{len(non_compliant):,}** non-compliant tax filings (Risk Score ≥ 0.70)")
    
    if len(non_compliant) == 0:
        st.success("🎉 No non-compliant filings detected!")
        return
    
    # Filters
    st.subheader("🔍 Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_risk = st.slider("Minimum Risk Score", 0.0, 1.0, 0.7, 0.05)
    
    with filter_col2:
        risk_levels = st.multiselect(
            "Risk Levels",
            options=['High', 'Very High'],
            default=['High', 'Very High']
        )
    
    with filter_col3:
        if 'c00100' in non_compliant.columns:
            income_range = st.select_slider(
                "Income Range",
                options=['All', '<$25k', '$25k-50k', '$50k-100k', '$100k-250k', '>$250k'],
                value='All'
            )
    
    # Apply filters
    filtered = non_compliant[
        (non_compliant['fraud_risk_score'] >= min_risk) &
        (non_compliant['risk_level'].isin(risk_levels))
    ]
    
    st.markdown(f"**Showing {len(filtered):,} of {len(non_compliant):,} non-compliant filings**")
    
    st.markdown("---")
    
    # Display list
    st.subheader("📊 Non-Compliant Filings List")
    st.markdown("*Click on a case to view detailed analysis*")
    
    # Create display dataframe
    display_cols = ['fraud_risk_score', 'risk_level', 'dif_score']
    
    # Add financial columns if available
    for col in ['c00100', 'e19800', 'e17500', 'e19200', 'eitc', 'XTOT', 'age_head']:
        if col in filtered.columns:
            display_cols.append(col)
    
    # Column name mapping
    col_names = {
        'fraud_risk_score': 'Risk Score',
        'risk_level': 'Risk Level',
        'dif_score': 'DIF Score',
        'c00100': 'AGI ($)',
        'e19800': 'Charitable ($)',
        'e17500': 'Medical ($)',
        'e19200': 'Mortgage ($)',
        'eitc': 'EITC ($)',
        'XTOT': 'Exemptions',
        'age_head': 'Age'
    }
    
    display_df = filtered[display_cols].copy()
    display_df = display_df.rename(columns=col_names)
    
    # Display each case as a clickable card
    for idx, row in filtered.iterrows():
        risk_score = row['fraud_risk_score']
        risk_level = row['risk_level']
        
        # Color coding
        if risk_score >= 0.9:
            card_class = "risk-very-high"
            emoji = "🔴"
        else:
            card_class = "risk-high"
            emoji = "🟠"
        
        with st.container():
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"### {emoji} Case #{idx}")
                st.markdown(f"**Risk Score:** {risk_score:.3f}")
                st.markdown(f"**Risk Level:** {risk_level}")
            
            with col2:
                if 'c00100' in row:
                    st.markdown(f"**AGI:** ${row.get('c00100', 0):,.0f}")
                if 'e19800' in row:
                    st.markdown(f"**Charitable:** ${row.get('e19800', 0):,.0f}")
                if 'e17500' in row:
                    st.markdown(f"**Medical:** ${row.get('e17500', 0):,.0f}")
            
            with col3:
                if st.button("View Details →", key=f"btn_{idx}", use_container_width=True):
                    st.session_state.selected_case = idx
                    st.session_state.page = 'case_detail'
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")  # Spacing
    
    # Download non-compliant only
    st.markdown("---")
    csv_non_compliant = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Non-Compliant Cases (CSV)",
        data=csv_non_compliant,
        file_name="non_compliant_cases.csv",
        mime="text/csv"
    )

# ========== PAGE: CASE DETAIL ==========
def page_case_detail():
    st.title("🔬 Detailed Case Analysis")
    
    if st.session_state.selected_case is None or st.session_state.results_df is None:
        st.warning("No case selected.")
        if st.button("← Back to List"):
            st.session_state.page = 'non_compliant_list'
            st.rerun()
        return
    
    # Navigation
    if st.button("← Back to Non-Compliant List"):
        st.session_state.page = 'non_compliant_list'
        st.rerun()
    
    idx = st.session_state.selected_case
    results = st.session_state.results_df
    case = results.loc[idx]
    
    # Get SHAP values for this case (calculate on-demand if needed)
    shap_idx = results.index.get_loc(idx)
    shap_explainer_local = None
    
    if st.session_state.shap_values is None:
        # Calculate SHAP on-demand for just this case
        with st.spinner("Calculating SHAP explanation for this case..."):
            try:
                if st.session_state.X_scaled is not None:
                    shap_explainer_local = shap.TreeExplainer(st.session_state.detector.models['random_forest'])
                    # Only calculate for this one case
                    shap_vals = shap_explainer_local.shap_values(st.session_state.X_scaled[shap_idx:shap_idx+1])
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1][0]
                    else:
                        shap_vals = shap_vals[0]
                    base_value = shap_explainer_local.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1]
                else:
                    shap_vals = None
                    base_value = 0.5
                    shap_explainer_local = None
            except Exception as e:
                st.warning(f"Could not calculate SHAP: {str(e)}")
                shap_vals = None
                base_value = 0.5
                shap_explainer_local = None
    else:
        # Use pre-calculated SHAP values
        shap_vals = st.session_state.shap_values[shap_idx]
        shap_explainer_local = st.session_state.shap_explainer
        if shap_explainer_local is not None:
            base_value = shap_explainer_local.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
        else:
            base_value = 0.5
    
    # Header
    risk_score = case['fraud_risk_score']
    if risk_score >= 0.9:
        st.error(f"🔴 **VERY HIGH RISK CASE** - Risk Score: {risk_score:.3f}")
    else:
        st.warning(f"🟠 **HIGH RISK CASE** - Risk Score: {risk_score:.3f}")
    
    st.markdown(f"### Case #{idx} - Detailed Analysis")
    
    st.markdown("---")
    
    # Case Overview
    st.header("📋 Case Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fraud Risk Score", f"{risk_score:.3f}")
        st.metric("Risk Level", case['risk_level'])
    
    with col2:
        st.metric("DIF Score", f"{case.get('dif_score', 0):.1f}")
        st.metric("RF Probability", f"{case.get('rf_probability', 0):.3f}")
    
    with col3:
        if 'c00100' in case:
            st.metric("Adjusted Gross Income", f"${case['c00100']:,.0f}")
        if 'XTOT' in case:
            st.metric("Exemptions", f"{int(case['XTOT'])}")
    
    with col4:
        if 'age_head' in case:
            st.metric("Age", f"{int(case['age_head'])}")
        if 'MARS' in case:
            filing_status = {1: 'Single', 2: 'Married Joint', 3: 'Married Separate', 4: 'Head of Household'}
            st.metric("Filing Status", filing_status.get(int(case['MARS']), 'Unknown'))
    
    st.markdown("---")
    
    # Red Flags
    st.header("🚩 Identified Red Flags")
    
    red_flags = get_red_flags(case, st.session_state.feature_names, shap_vals if shap_vals is not None else [])
    
    if red_flags:
        for flag in red_flags:
            st.markdown(f'<div class="explanation-box">{flag}</div>', unsafe_allow_html=True)
    else:
        st.info("No specific red flags identified - flagged based on statistical patterns")
    
    st.markdown("---")
    
    # Financial Details
    st.header("💰 Financial Details")
    
    financial_data = {}
    financial_fields = {
        'c00100': 'Adjusted Gross Income',
        'e00200': 'Wages & Salaries',
        'e00900': 'Business Income',
        'e19800': 'Charitable Contributions',
        'e17500': 'Medical Deductions',
        'e19200': 'Mortgage Interest',
        'c04470': 'Itemized Deductions',
        'eitc': 'Earned Income Tax Credit',
        'refund': 'Tax Refund'
    }
    
    for field, label in financial_fields.items():
        if field in case:
            financial_data[label] = f"${case[field]:,.0f}"
    
    col1, col2 = st.columns(2)
    
    items = list(financial_data.items())
    mid = len(items) // 2
    
    with col1:
        for label, value in items[:mid]:
            st.markdown(f"**{label}:** {value}")
    
    with col2:
        for label, value in items[mid:]:
            st.markdown(f"**{label}:** {value}")
    
    st.markdown("---")
    
    # SHAP Explanation (only if available)
    if shap_vals is not None and st.session_state.feature_names is not None:
        st.header("🤖 AI Explanation (SHAP Analysis)")
        
        st.markdown("""
        The model identified these factors as contributing most to the risk score for this case.
        Positive values increase risk, negative values decrease risk.
        """)
        
        explanations = generate_shap_explanation(case, shap_vals, st.session_state.feature_names, top_n=10)
    
    # Display explanations as a table
    exp_df = pd.DataFrame(explanations)
    exp_df['shap_value'] = exp_df['shap_value'].apply(lambda x: f"{x:+.4f}")
    exp_df['feature'] = exp_df['feature'].apply(lambda x: x.replace('_', ' ').title())
    
    st.dataframe(
        exp_df[['rank', 'feature', 'shap_value', 'impact']],
        column_config={
            'rank': 'Rank',
            'feature': 'Factor',
            'shap_value': 'SHAP Value',
            'impact': 'Impact on Risk'
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # SHAP Waterfall Plot
    st.header("📊 SHAP Waterfall Plot")
    st.markdown("*Visual breakdown of how each factor contributed to the final risk score*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get base value
        base_value = st.session_state.shap_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=case[st.session_state.feature_names].values,
                feature_names=st.session_state.feature_names
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### How to Read This Chart")
        st.markdown("""
        - **Starting point (E[f(x)])**: Average risk score across all returns
        - **Red bars**: Factors that INCREASE risk
        - **Blue bars**: Factors that DECREASE risk
        - **Ending point (f(x))**: Final risk score for this case
        
        Each bar shows how much that specific factor pushed the risk score up or down.
        """)
    
    st.markdown("---")
    
    # Comparison to Peers
    st.header("👥 Comparison to Peer Group")
    
    # Determine income bracket
    income = case.get('c00100', 0)
    if income < 25000:
        bracket = "Under $25k"
    elif income < 50000:
        bracket = "$25k-$50k"
    elif income < 100000:
        bracket = "$50k-$100k"
    elif income < 250000:
        bracket = "$100k-$250k"
    elif income < 500000:
        bracket = "$250k-$500k"
    else:
        bracket = "Over $500k"
    
    st.markdown(f"**Income Bracket:** {bracket}")
    
    # Calculate peer statistics
    peer_group = results[
        (results['c00100'] >= income * 0.8) & 
        (results['c00100'] <= income * 1.2)
    ]
    
    if len(peer_group) > 1:
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("### This Case vs. Peers")
            
            metrics = {
                'Charitable Ratio': (case.get('e19800', 0) / max(case.get('c00100', 1), 1), 
                                    (peer_group['e19800'] / peer_group['c00100'].replace(0, 1)).mean()),
                'Medical Ratio': (case.get('e17500', 0) / max(case.get('c00100', 1), 1),
                                 (peer_group['e17500'] / peer_group['c00100'].replace(0, 1)).mean()),
                'EITC Ratio': (case.get('eitc', 0) / max(case.get('c00100', 1), 1),
                              (peer_group['eitc'] / peer_group['c00100'].replace(0, 1)).mean())
            }
            
            for metric_name, (case_val, peer_avg) in metrics.items():
                if peer_avg > 0:
                    diff_pct = ((case_val - peer_avg) / peer_avg * 100) if peer_avg > 0 else 0
                    if abs(diff_pct) > 50:
                        color = "🔴" if case_val > peer_avg else "🟢"
                    else:
                        color = "⚪"
                    st.markdown(f"{color} **{metric_name}**: {case_val:.2%} (Peer avg: {peer_avg:.2%}, {diff_pct:+.0f}% diff)")
        
        with comp_col2:
            # Risk score distribution for peer group
            fig_peer = go.Figure()
            
            fig_peer.add_trace(go.Histogram(
                x=peer_group['fraud_risk_score'],
                name='Peer Group',
                nbinsx=30,
                marker_color='lightblue'
            ))
            
            fig_peer.add_vline(
                x=risk_score,
                line_dash="dash",
                line_color="red",
                annotation_text="This Case"
            )
            
            fig_peer.update_layout(
                title=f"Risk Scores in {bracket} Bracket",
                xaxis_title="Risk Score",
                yaxis_title="Count",
                showlegend=False
            )
            
            st.plotly_chart(fig_peer, use_container_width=True)
    
    st.markdown("---")
    
    # Audit Recommendation
    st.header("✅ Audit Recommendation")
    
    if risk_score >= 0.9:
        st.error("""
        ### 🚨 PRIORITY AUDIT RECOMMENDED
        
        This case exhibits **very high risk** indicators and should be prioritized for immediate audit.
        
        **Recommended Actions:**
        1. Full examination of all deductions and credits claimed
        2. Verification of income sources and amounts
        3. Review of supporting documentation for all claims
        4. Potential for significant revenue recovery
        """)
    elif risk_score >= 0.7:
        st.warning("""
        ### ⚠️ AUDIT RECOMMENDED
        
        This case shows **high risk** patterns and warrants audit examination.
        
        **Recommended Actions:**
        1. Review specific red flags identified above
        2. Request documentation for unusual deductions
        3. Verify income reporting accuracy
        4. Consider for routine audit queue
        """)
    else:
        st.info("""
        ### ℹ️ MONITORING RECOMMENDED
        
        While below the standard audit threshold, this case shows some concerning patterns.
        
        **Recommended Actions:**
        1. Monitor for patterns in future filings
        2. Flag for review if similar patterns continue
        """)
    
    st.markdown("---")
    
    # Export Case Report
    st.header("💾 Export Case Report")
    
    # Create detailed case report
    report_data = {
        'Case_ID': [idx],
        'Risk_Score': [risk_score],
        'Risk_Level': [case['risk_level']],
        'DIF_Score': [case.get('dif_score', 0)],
        'AGI': [case.get('c00100', 0)],
        'Red_Flags': ['; '.join(red_flags) if red_flags else 'None specific'],
        'Audit_Recommendation': ['Priority' if risk_score >= 0.9 else 'Standard' if risk_score >= 0.7 else 'Monitor']
    }
    
    # Add top 5 SHAP factors
    for i, exp in enumerate(explanations[:5], 1):
        report_data[f'Factor_{i}'] = [exp['feature']]
        report_data[f'Factor_{i}_Impact'] = [exp['shap_value']]
    
    report_df = pd.DataFrame(report_data)
    
    csv_report = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Case Report (CSV)",
        data=csv_report,
        file_name=f"case_{idx}_detailed_report.csv",
        mime="text/csv"
    )

# ========== MAIN APP ROUTER ==========
def main():
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1976d2/ffffff?text=AURA+System", use_column_width=True)
        
        st.markdown("---")
        st.header("Navigation")
        
        # Page buttons
        pages = {
            'upload': '📤 Upload Data',
            'summary': '📊 Summary',
            'non_compliant_list': '📋 Non-Compliant',
            'case_detail': '🔬 Case Detail'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True,
                        type="primary" if st.session_state.page == page_key else "secondary"):
                if page_key == 'upload' or st.session_state.results_df is not None:
                    st.session_state.page = page_key
                    st.rerun()
        
        st.markdown("---")
        
        # Stats in sidebar
        if st.session_state.results_df is not None:
            st.markdown("### Quick Stats")
            results = st.session_state.results_df
            st.metric("Total Returns", f"{len(results):,}")
            st.metric("Non-Compliant", f"{results['non_compliant'].sum():,}")
            st.metric("Avg Risk", f"{results['fraud_risk_score'].mean():.3f}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **AURA Tax Fraud Detection**
        
        AI-powered system using:
        - Machine Learning (Random Forest)
        - Anomaly Detection (Isolation Forest)
        - Explainable AI (SHAP)
        - DIF-style risk scoring
        """)
    
    # Route to appropriate page
    if st.session_state.page == 'upload':
        page_upload()
    elif st.session_state.page == 'summary':
        page_summary()
    elif st.session_state.page == 'non_compliant_list':
        page_non_compliant_list()
    elif st.session_state.page == 'case_detail':
        page_case_detail()

if __name__ == "__main__":
    main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from joblib import load
# import shap
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# # Import your detector class
# from corrected_aura_detector import CorrectedAURAStyleDetector
# from corrected_aura_detector_v2 import CorrectedAURAStyleDetectorV2

# # Page configuration
# st.set_page_config(
#     page_title="AURA Tax Fraud Detection",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .big-metric {
#         font-size: 28px;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     .risk-very-high {
#         background-color: #ffebee;
#         padding: 10px;
#         border-left: 4px solid #d32f2f;
#         margin: 10px 0;
#     }
#     .risk-high {
#         background-color: #fff3e0;
#         padding: 10px;
#         border-left: 4px solid #f57c00;
#         margin: 10px 0;
#     }
#     .case-card {
#         background-color: #f5f5f5;
#         padding: 15px;
#         border-radius: 8px;
#         margin: 10px 0;
#         border-left: 4px solid #1976d2;
#     }
#     .explanation-box {
#         background-color: #e3f2fd;
#         padding: 15px;
#         border-radius: 8px;
#         margin: 10px 0;
#     }
#     .red-flag {
#         color: #d32f2f;
#         font-weight: bold;
#     }
#     .stButton>button {
#         width: 100%;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'page' not in st.session_state:
#     st.session_state.page = 'upload'
# if 'results_df' not in st.session_state:
#     st.session_state.results_df = None
# if 'detector' not in st.session_state:
#     st.session_state.detector = None
# if 'shap_values' not in st.session_state:
#     st.session_state.shap_values = None
# if 'shap_explainer' not in st.session_state:
#     st.session_state.shap_explainer = None
# if 'feature_names' not in st.session_state:
#     st.session_state.feature_names = None
# if 'X_scaled' not in st.session_state:
#     st.session_state.X_scaled = None
# if 'selected_case' not in st.session_state:
#     st.session_state.selected_case = None

# def load_trained_model():
#     """Load the trained AURA detector"""
#     try:
#         detector = load("corrected_aura_detector.pkl")
#         return detector, True
#     except FileNotFoundError:
#         return None, False

# def process_uploaded_file(uploaded_file):
#     """Process uploaded Excel/CSV file"""
#     try:
#         if uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         elif uploaded_file.name.endswith(('.xls', '.xlsx')):
#             df = pd.read_excel(uploaded_file)
#         else:
#             st.error("Unsupported file format. Please upload CSV or Excel")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error reading file: {str(e)}")
#         return None

# def run_fraud_detection(df, detector):
#     """Run fraud detection on uploaded data"""
#     try:
#         with st.spinner("🔍 Analyzing tax returns... This may take a few minutes."):
#             # Remove fraud labels if present
#             drop_cols = ["is_fraud", "fraud_type", "fraud_severity"]
#             df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
#             # Engineer features
#             progress_bar = st.progress(0)
#             st.text("Engineering features...")
#             df_enhanced = detector.engineer_available_features(df_clean)
#             progress_bar.progress(33)
            
#             # Create income bracket column (CRITICAL - must happen before DIF scoring)
#             st.text("Creating income brackets...")
#             df_enhanced['income_bracket'] = pd.cut(
#                 df_enhanced['c00100'], 
#                 bins=[-np.inf, 25000, 50000, 100000, 250000, 500000, np.inf],
#                 labels=['under_25k', '25k_50k', '50k_100k', '100k_250k', '250k_500k', 'over_500k']
#             )
            
#             # Create income bracket profiles if not exists
#             if not detector.income_bracket_profiles:
#                 st.text("Creating income bracket profiles...")
#                 profiles = detector.create_income_bracket_profiles(df_enhanced)
#                 detector.income_bracket_profiles = profiles
#             progress_bar.progress(50)
            
#             # Calculate DIF scores
#             st.text("Calculating DIF-style risk scores...")
#             df_scored = detector.calculate_dif_style_risk_score(df_enhanced, detector.income_bracket_profiles)
#             progress_bar.progress(66)
            
#             # Prepare features
#             exclude_cols = ["income_bracket"]
#             feature_cols = [col for col in df_scored.columns if col not in exclude_cols]
#             X = df_scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            
#             # CRITICAL: Align features with trained model
#             # Get the features the model was trained on
#             if hasattr(detector.scalers['main'], 'feature_names_in_'):
#                 trained_features = detector.scalers['main'].feature_names_in_
#             else:
#                 # Fallback: get from random forest model
#                 trained_features = detector.models['random_forest'].feature_names_in_
            
#             # Add missing columns with zeros
#             for feature in trained_features:
#                 if feature not in X.columns:
#                     st.warning(f"Adding missing feature: {feature}")
#                     X[feature] = 0
            
#             # Remove extra columns not in training
#             extra_cols = [col for col in X.columns if col not in trained_features]
#             if extra_cols:
#                 X = X.drop(columns=extra_cols)
            
#             # Ensure column order matches training
#             X = X[trained_features]
            
#             # Scale and predict
#             st.text("Generating predictions...")
#             X_scaled = detector.scalers['main'].transform(X)
#             ensemble_scores, rf_proba, lr_proba, iso_scores = detector.predict_audit_risk(X_scaled)
#             progress_bar.progress(100)
            
#             # Add results to original dataframe
#             results = df_clean.copy()
#             results['fraud_risk_score'] = ensemble_scores
#             results['rf_probability'] = rf_proba
#             results['lr_probability'] = lr_proba
#             results['anomaly_score'] = iso_scores
#             results['dif_score'] = df_scored['dif_risk_score'].values
            
#             # Risk categories
#             results['risk_level'] = pd.cut(
#                 results['fraud_risk_score'],
#                 bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
#                 labels=['Low', 'Medium-Low', 'Medium', 'High', 'Very High']
#             )
            
#             # Flag non-compliant
#             results['non_compliant'] = (results['fraud_risk_score'] >= 0.7).astype(int)
            
#             st.text("✓ Predictions complete!")
#             return results, X_scaled, list(trained_features)
            
#     except Exception as e:
#         st.error(f"Error during fraud detection: {str(e)}")
#         st.exception(e)
#         return None, None, None

# def calculate_shap_values(detector, X_scaled):
#     """Calculate SHAP values"""
#     with st.spinner("📊 Calculating SHAP explanations... This may take a moment."):
#         explainer = shap.TreeExplainer(detector.models['random_forest'])
#         shap_values = explainer.shap_values(X_scaled)
        
#         if isinstance(shap_values, list):
#             shap_values = shap_values[1]
        
#         return shap_values, explainer

# def get_red_flags(row, feature_names, shap_vals):
#     """Identify specific red flags for a case"""
#     red_flags = []
    
#     # Charitable deductions
#     charitable_ratio = row.get('e19800', 0) / max(row.get('c00100', 1), 1)
#     if charitable_ratio > 0.1:
#         red_flags.append(f"🚩 High charitable deductions ({charitable_ratio*100:.1f}% of income)")
    
#     # Medical deductions
#     medical_ratio = row.get('e17500', 0) / max(row.get('c00100', 1), 1)
#     if medical_ratio > 0.075:
#         red_flags.append(f"🚩 High medical deductions ({medical_ratio*100:.1f}% of income)")
    
#     # Rounded numbers
#     rounded_count = 0
#     for col in ['e19800', 'e17500', 'e19200', 'e00200']:
#         if col in row and row[col] > 0 and row[col] % 100 == 0:
#             rounded_count += 1
#     if rounded_count >= 2:
#         red_flags.append(f"🚩 Multiple rounded amounts detected ({rounded_count} fields)")
    
#     # Business losses
#     if row.get('e00900', 0) < 0:
#         red_flags.append(f"🚩 Business losses reported (${abs(row.get('e00900', 0)):,.0f})")
    
#     # EITC
#     eitc_ratio = row.get('eitc', 0) / max(row.get('c00100', 1), 1)
#     if eitc_ratio > 0.3:
#         red_flags.append(f"🚩 High EITC claim relative to income ({eitc_ratio*100:.1f}%)")
    
#     # Income per exemption
#     income_per_exemption = row.get('c00100', 0) / max(row.get('XTOT', 1), 1)
#     if income_per_exemption < 15000 and row.get('c00100', 0) > 0:
#         red_flags.append(f"🚩 Low income per household member (${income_per_exemption:,.0f})")
    
#     # Large refund
#     refund_ratio = row.get('refund', 0) / max(row.get('c00100', 1), 1)
#     if refund_ratio > 0.3:
#         red_flags.append(f"🚩 Large refund relative to income ({refund_ratio*100:.1f}%)")
    
#     return red_flags

# def generate_shap_explanation(row, shap_vals, feature_names, top_n=5):
#     """Generate SHAP-based explanation"""
#     feature_contributions = list(zip(feature_names, shap_vals, [row.get(f, 0) for f in feature_names]))
#     feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
#     explanations = []
#     for i, (feature, shap_val, value) in enumerate(feature_contributions[:top_n], 1):
#         impact = "increases" if shap_val > 0 else "decreases"
#         explanations.append({
#             'rank': i,
#             'feature': feature,
#             'shap_value': shap_val,
#             'impact': impact,
#             'value': value
#         })
    
#     return explanations

# # ========== PAGE: UPLOAD ==========
# def page_upload():
#     st.title("🔍 AURA Tax Fraud Detection System")
#     st.markdown("### Upload Tax Return Data for Analysis")
    
#     # Load model
#     detector, model_loaded = load_trained_model()
    
#     if not model_loaded:
#         st.error("❌ Model not found! Please train the model first.")
#         st.code("python3 corrected_aura_detector.py", language="bash")
#         return
    
#     st.session_state.detector = detector
#     st.success("✅ Model loaded successfully")
    
#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Upload Excel or CSV file with tax return data",
#         type=['csv', 'xlsx', 'xls'],
#         help="File should contain standard tax return fields (c00100, e19800, e17500, etc.)"
#     )
    
#     if uploaded_file is None:
#         st.info("👆 Upload a file to begin analysis")
        
#         with st.expander("📋 Required Data Format"):
#             st.markdown("""
#             Your file should contain tax return data with these key columns:
#             - **c00100**: Adjusted Gross Income
#             - **e19800**: Charitable contributions
#             - **e17500**: Medical deductions
#             - **e19200**: Mortgage interest
#             - **e00900**: Business income
#             - **e00200**: Wages and salaries
#             - **XTOT**: Total exemptions
#             - **eitc**: Earned Income Tax Credit
#             - **MARS**: Filing status
#             - **age_head**: Primary taxpayer age
#             - And other standard tax form fields...
#             """)
#         return
    
#     # Process file
#     df = process_uploaded_file(uploaded_file)
    
#     if df is None:
#         return
    
#     st.success(f"✅ Loaded {len(df):,} tax returns")
    
#     # Show preview
#     with st.expander("👀 Preview Data"):
#         st.dataframe(df.head(10), use_container_width=True)
    
#     # Fast mode option
#     use_fast_mode = st.checkbox(
#         "⚡ Fast Mode (Skip SHAP - recommended for >1000 returns)",
#         value=len(df) > 1000,
#         help="SHAP explanations will be calculated on-demand when viewing individual cases"
#     )
    
#     # Run analysis
#     if st.button("🚀 Run Fraud Detection Analysis", type="primary", use_container_width=True):
#         try:
#             results, X_scaled, feature_names = run_fraud_detection(df, detector)
            
#             if results is None or len(results) == 0:
#                 st.error("❌ Analysis failed to produce results")
#                 return
            
#             st.success("✅ Model predictions complete!")
            
#             # Save to session state FIRST
#             st.session_state.results_df = results
#             st.session_state.X_scaled = X_scaled
#             st.session_state.feature_names = feature_names
            
#             # Calculate SHAP (optional)
#             if not use_fast_mode:
#                 try:
#                     with st.spinner("Calculating SHAP explanations... (this may take a few minutes)"):
#                         shap_values, explainer = calculate_shap_values(detector, X_scaled)
#                         st.session_state.shap_values = shap_values
#                         st.session_state.shap_explainer = explainer
#                         st.success("✅ SHAP calculations complete!")
#                 except Exception as shap_error:
#                     st.warning(f"⚠️ SHAP calculation skipped: {shap_error}")
#                     st.session_state.shap_values = None
#                     st.session_state.shap_explainer = None
#             else:
#                 st.info("⚡ Fast mode enabled - SHAP will be calculated on-demand")
#                 st.session_state.shap_values = None
#                 st.session_state.shap_explainer = None
            
#             st.success("✅ Analysis Complete! Navigating to summary...")
            
#             # Manual navigation option
#             st.markdown("---")
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("📊 View Summary", type="primary", use_container_width=True):
#                     st.session_state.page = 'summary'
#                     st.rerun()
#             with col2:
#                 if st.button("📋 View Non-Compliant Cases", use_container_width=True):
#                     st.session_state.page = 'non_compliant_list'
#                     st.rerun()
            
#             # Also try auto-navigation
#             import time
#             time.sleep(0.5)
#             st.session_state.page = 'summary'
#             st.rerun()
            
#         except Exception as e:
#             st.error(f"❌ Analysis failed with error: {str(e)}")
#             st.exception(e)
#             return

# # ========== PAGE: SUMMARY ==========
# def page_summary():
#     st.title("📊 Analysis Summary & Results")
    
#     if st.session_state.results_df is None:
#         st.warning("No results available. Please upload data first.")
#         if st.button("← Back to Upload"):
#             st.session_state.page = 'upload'
#             st.rerun()
#         return
    
#     results = st.session_state.results_df
    
#     # Navigation
#     col1, col2 = st.columns([3, 1])
#     with col2:
#         if st.button("← Back to Upload"):
#             st.session_state.page = 'upload'
#             st.rerun()
    
#     # Key Metrics
#     st.header("📈 Key Metrics")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     total_returns = len(results)
#     non_compliant_count = results['non_compliant'].sum()
#     non_compliant_pct = (non_compliant_count / total_returns * 100) if total_returns > 0 else 0
#     avg_risk = results['fraud_risk_score'].mean()
#     very_high_risk = (results['fraud_risk_score'] >= 0.9).sum()
    
#     with col1:
#         st.metric("Total Returns Analyzed", f"{total_returns:,}")
    
#     with col2:
#         st.metric("Non-Compliant Filings", f"{non_compliant_count:,}", 
#                  delta=f"{non_compliant_pct:.1f}% of total", delta_color="inverse")
    
#     with col3:
#         st.metric("Average Risk Score", f"{avg_risk:.3f}")
    
#     with col4:
#         st.metric("Very High Risk (>0.9)", f"{very_high_risk:,}")
    
#     st.markdown("---")
    
#     # Executive Summary with SHAP Insights
#     st.header("💡 Executive Summary")
    
#     summary_col1, summary_col2 = st.columns([2, 1])
    
#     with summary_col1:
#         st.markdown(f"""
#         ### Analysis Overview
        
#         Out of **{total_returns:,}** tax returns analyzed:
#         - **{non_compliant_count:,} returns ({non_compliant_pct:.1f}%)** were flagged as non-compliant (risk score ≥ 0.70)
#         - **{very_high_risk:,} returns** are considered very high risk (score ≥ 0.90)
#         - The average fraud risk score across all returns is **{avg_risk:.3f}**
        
#         ### Risk Distribution
#         """)
        
#         risk_dist = results['risk_level'].value_counts().sort_index()
#         for level, count in risk_dist.items():
#             pct = count / total_returns * 100
#             st.markdown(f"- **{level}**: {count:,} returns ({pct:.1f}%)")
    
#     with summary_col2:
#         # Risk level pie chart
#         fig_pie = px.pie(
#             values=risk_dist.values,
#             names=risk_dist.index,
#             title="Risk Distribution",
#             color_discrete_map={
#                 'Low': '#4caf50',
#                 'Medium-Low': '#8bc34a',
#                 'Medium': '#ffc107',
#                 'High': '#ff9800',
#                 'Very High': '#f44336'
#             }
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)
    
#     st.markdown("---")
    
#     # SHAP Global Insights
#     st.header("🎯 Key Risk Factors (SHAP Analysis)")
    
#     if st.session_state.shap_values is not None:
#         st.markdown("""
#         The AI model identified the following factors as most important in determining fraud risk across all returns:
#         """)
        
#         # Feature importance from SHAP
#         shap_abs_mean = np.abs(st.session_state.shap_values).mean(axis=0)
#         feature_importance = pd.DataFrame({
#             'Feature': st.session_state.feature_names,
#             'Importance': shap_abs_mean
#         }).sort_values('Importance', ascending=False).head(10)
        
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.dataframe(
#                 feature_importance.style.format({'Importance': '{:.4f}'}),
#                 use_container_width=True,
#                 height=400
#             )
        
#         with col2:
#             # SHAP summary plot
#             fig, ax = plt.subplots(figsize=(10, 6))
#             shap.summary_plot(
#                 st.session_state.shap_values[:1000],  # Sample for speed
#                 features=results[st.session_state.feature_names].iloc[:1000].values,
#                 feature_names=st.session_state.feature_names,
#                 show=False,
#                 max_display=10
#             )
#             st.pyplot(fig)
#             plt.close()
    
#     st.markdown("---")
    
#     # Risk Distribution Chart
#     st.header("📊 Risk Score Distribution")
    
#     fig_hist = px.histogram(
#         results,
#         x='fraud_risk_score',
#         nbins=50,
#         title="Distribution of Fraud Risk Scores",
#         labels={'fraud_risk_score': 'Fraud Risk Score', 'count': 'Number of Returns'}
#     )
#     fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", 
#                       annotation_text="Non-Compliant Threshold (0.7)")
#     fig_hist.add_vline(x=0.9, line_dash="dash", line_color="darkred",
#                       annotation_text="Very High Risk (0.9)")
#     st.plotly_chart(fig_hist, use_container_width=True)
    
#     st.markdown("---")
    
#     # Navigation to non-compliant list
#     st.header("🔍 Next Steps")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("📋 View All Non-Compliant Filings", type="primary", use_container_width=True):
#             st.session_state.page = 'non_compliant_list'
#             st.rerun()
    
#     with col2:
#         # Download button
#         csv = results.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="💾 Download Full Results (CSV)",
#             data=csv,
#             file_name="fraud_detection_results.csv",
#             mime="text/csv",
#             use_container_width=True
#         )

# # ========== PAGE: NON-COMPLIANT LIST ==========
# def page_non_compliant_list():
#     st.title("📋 Non-Compliant Tax Filings")
    
#     if st.session_state.results_df is None:
#         st.warning("No results available.")
#         if st.button("← Back to Upload"):
#             st.session_state.page = 'upload'
#             st.rerun()
#         return
    
#     # Navigation
#     col1, col2 = st.columns([3, 1])
#     with col2:
#         if st.button("← Back to Summary"):
#             st.session_state.page = 'summary'
#             st.rerun()
    
#     results = st.session_state.results_df
#     non_compliant = results[results['non_compliant'] == 1].sort_values('fraud_risk_score', ascending=False)
    
#     st.markdown(f"### Found **{len(non_compliant):,}** non-compliant tax filings (Risk Score ≥ 0.70)")
    
#     if len(non_compliant) == 0:
#         st.success("🎉 No non-compliant filings detected!")
#         return
    
#     # Filters
#     st.subheader("🔍 Filters")
    
#     filter_col1, filter_col2, filter_col3 = st.columns(3)
    
#     with filter_col1:
#         min_risk = st.slider("Minimum Risk Score", 0.0, 1.0, 0.7, 0.05)
    
#     with filter_col2:
#         risk_levels = st.multiselect(
#             "Risk Levels",
#             options=['High', 'Very High'],
#             default=['High', 'Very High']
#         )
    
#     with filter_col3:
#         if 'c00100' in non_compliant.columns:
#             income_range = st.select_slider(
#                 "Income Range",
#                 options=['All', '<$25k', '$25k-50k', '$50k-100k', '$100k-250k', '>$250k'],
#                 value='All'
#             )
    
#     # Apply filters
#     filtered = non_compliant[
#         (non_compliant['fraud_risk_score'] >= min_risk) &
#         (non_compliant['risk_level'].isin(risk_levels))
#     ]
    
#     st.markdown(f"**Showing {len(filtered):,} of {len(non_compliant):,} non-compliant filings**")
    
#     st.markdown("---")
    
#     # Display list
#     st.subheader("📊 Non-Compliant Filings List")
#     st.markdown("*Click on a case to view detailed analysis*")
    
#     # Create display dataframe
#     display_cols = ['fraud_risk_score', 'risk_level', 'dif_score']
    
#     # Add financial columns if available
#     for col in ['c00100', 'e19800', 'e17500', 'e19200', 'eitc', 'XTOT', 'age_head']:
#         if col in filtered.columns:
#             display_cols.append(col)
    
#     # Column name mapping
#     col_names = {
#         'fraud_risk_score': 'Risk Score',
#         'risk_level': 'Risk Level',
#         'dif_score': 'DIF Score',
#         'c00100': 'AGI ($)',
#         'e19800': 'Charitable ($)',
#         'e17500': 'Medical ($)',
#         'e19200': 'Mortgage ($)',
#         'eitc': 'EITC ($)',
#         'XTOT': 'Exemptions',
#         'age_head': 'Age'
#     }
    
#     display_df = filtered[display_cols].copy()
#     display_df = display_df.rename(columns=col_names)
    
#     # Display each case as a clickable card
#     for idx, row in filtered.iterrows():
#         risk_score = row['fraud_risk_score']
#         risk_level = row['risk_level']
        
#         # Color coding
#         if risk_score >= 0.9:
#             card_class = "risk-very-high"
#             emoji = "🔴"
#         else:
#             card_class = "risk-high"
#             emoji = "🟠"
        
#         with st.container():
#             st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
#             col1, col2, col3 = st.columns([2, 2, 1])
            
#             with col1:
#                 st.markdown(f"### {emoji} Case #{idx}")
#                 st.markdown(f"**Risk Score:** {risk_score:.3f}")
#                 st.markdown(f"**Risk Level:** {risk_level}")
            
#             with col2:
#                 if 'c00100' in row:
#                     st.markdown(f"**AGI:** ${row.get('c00100', 0):,.0f}")
#                 if 'e19800' in row:
#                     st.markdown(f"**Charitable:** ${row.get('e19800', 0):,.0f}")
#                 if 'e17500' in row:
#                     st.markdown(f"**Medical:** ${row.get('e17500', 0):,.0f}")
            
#             with col3:
#                 if st.button("View Details →", key=f"btn_{idx}", use_container_width=True):
#                     st.session_state.selected_case = idx
#                     st.session_state.page = 'case_detail'
#                     st.rerun()
            
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown("")  # Spacing
    
#     # Download non-compliant only
#     st.markdown("---")
#     csv_non_compliant = filtered.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="💾 Download Non-Compliant Cases (CSV)",
#         data=csv_non_compliant,
#         file_name="non_compliant_cases.csv",
#         mime="text/csv"
#     )

# # ========== PAGE: CASE DETAIL ==========
# def page_case_detail():
#     st.title("🔬 Detailed Case Analysis")
    
#     if st.session_state.selected_case is None or st.session_state.results_df is None:
#         st.warning("No case selected.")
#         if st.button("← Back to List"):
#             st.session_state.page = 'non_compliant_list'
#             st.rerun()
#         return
    
#     # Navigation
#     if st.button("← Back to Non-Compliant List"):
#         st.session_state.page = 'non_compliant_list'
#         st.rerun()
    
#     idx = st.session_state.selected_case
#     results = st.session_state.results_df
#     case = results.loc[idx]
    
#     # Get SHAP values for this case (calculate on-demand if needed)
#     shap_idx = results.index.get_loc(idx)
    
#     if st.session_state.shap_values is None:
#         # Calculate SHAP on-demand for just this case
#         with st.spinner("Calculating SHAP explanation for this case..."):
#             try:
#                 if st.session_state.X_scaled is not None:
#                     explainer = shap.TreeExplainer(st.session_state.detector.models['random_forest'])
#                     # Only calculate for this one case
#                     shap_vals = explainer.shap_values(st.session_state.X_scaled[shap_idx:shap_idx+1])
#                     if isinstance(shap_vals, list):
#                         shap_vals = shap_vals[1][0]
#                     else:
#                         shap_vals = shap_vals[0]
#                     base_value = explainer.expected_value
#                     if isinstance(base_value, (list, np.ndarray)):
#                         base_value = base_value[1]
#                 else:
#                     shap_vals = None
#                     base_value = 0.5
#             except Exception as e:
#                 st.warning(f"Could not calculate SHAP: {str(e)}")
#                 shap_vals = None
#                 base_value = 0.5
#     else:
#         # Use pre-calculated SHAP values
#         shap_vals = st.session_state.shap_values[shap_idx]
#         base_value = st.session_state.shap_explainer.expected_value
#         if isinstance(base_value, (list, np.ndarray)):
#             base_value = base_value[1]
    
#     # Header
#     risk_score = case['fraud_risk_score']
#     if risk_score >= 0.9:
#         st.error(f"🔴 **VERY HIGH RISK CASE** - Risk Score: {risk_score:.3f}")
#     else:
#         st.warning(f"🟠 **HIGH RISK CASE** - Risk Score: {risk_score:.3f}")
    
#     st.markdown(f"### Case #{idx} - Detailed Analysis")
    
#     st.markdown("---")
    
#     # Case Overview
#     st.header("📋 Case Overview")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Fraud Risk Score", f"{risk_score:.3f}")
#         st.metric("Risk Level", case['risk_level'])
    
#     with col2:
#         st.metric("DIF Score", f"{case.get('dif_score', 0):.1f}")
#         st.metric("RF Probability", f"{case.get('rf_probability', 0):.3f}")
    
#     with col3:
#         if 'c00100' in case:
#             st.metric("Adjusted Gross Income", f"${case['c00100']:,.0f}")
#         if 'XTOT' in case:
#             st.metric("Exemptions", f"{int(case['XTOT'])}")
    
#     with col4:
#         if 'age_head' in case:
#             st.metric("Age", f"{int(case['age_head'])}")
#         if 'MARS' in case:
#             filing_status = {1: 'Single', 2: 'Married Joint', 3: 'Married Separate', 4: 'Head of Household'}
#             st.metric("Filing Status", filing_status.get(int(case['MARS']), 'Unknown'))
    
#     st.markdown("---")
    
#     # Red Flags
#     st.header("🚩 Identified Red Flags")
    
#     red_flags = get_red_flags(case, st.session_state.feature_names, shap_vals if shap_vals is not None else [])
    
#     if red_flags:
#         for flag in red_flags:
#             st.markdown(f'<div class="explanation-box">{flag}</div>', unsafe_allow_html=True)
#     else:
#         st.info("No specific red flags identified - flagged based on statistical patterns")
    
#     st.markdown("---")
    
#     # Financial Details
#     st.header("💰 Financial Details")
    
#     financial_data = {}
#     financial_fields = {
#         'c00100': 'Adjusted Gross Income',
#         'e00200': 'Wages & Salaries',
#         'e00900': 'Business Income',
#         'e19800': 'Charitable Contributions',
#         'e17500': 'Medical Deductions',
#         'e19200': 'Mortgage Interest',
#         'c04470': 'Itemized Deductions',
#         'eitc': 'Earned Income Tax Credit',
#         'refund': 'Tax Refund'
#     }
    
#     for field, label in financial_fields.items():
#         if field in case:
#             financial_data[label] = f"${case[field]:,.0f}"
    
#     col1, col2 = st.columns(2)
    
#     items = list(financial_data.items())
#     mid = len(items) // 2
    
#     with col1:
#         for label, value in items[:mid]:
#             st.markdown(f"**{label}:** {value}")
    
#     with col2:
#         for label, value in items[mid:]:
#             st.markdown(f"**{label}:** {value}")
    
#     st.markdown("---")
    
#     # SHAP Explanation (only if available)
#     if shap_vals is not None and st.session_state.feature_names is not None:
#         st.header("🤖 AI Explanation (SHAP Analysis)")
        
#         st.markdown("""
#         The model identified these factors as contributing most to the risk score for this case.
#         Positive values increase risk, negative values decrease risk.
#         """)
        
#         explanations = generate_shap_explanation(case, shap_vals, st.session_state.feature_names, top_n=10)
    
#     # Display explanations as a table
#     exp_df = pd.DataFrame(explanations)
#     exp_df['shap_value'] = exp_df['shap_value'].apply(lambda x: f"{x:+.4f}")
#     exp_df['feature'] = exp_df['feature'].apply(lambda x: x.replace('_', ' ').title())
    
#     st.dataframe(
#         exp_df[['rank', 'feature', 'shap_value', 'impact']],
#         column_config={
#             'rank': 'Rank',
#             'feature': 'Factor',
#             'shap_value': 'SHAP Value',
#             'impact': 'Impact on Risk'
#         },
#         use_container_width=True,
#         hide_index=True
#     )
    
#     st.markdown("---")
    
#     # SHAP Waterfall Plot
#     st.header("📊 SHAP Waterfall Plot")
#     st.markdown("*Visual breakdown of how each factor contributed to the final risk score*")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         fig, ax = plt.subplots(figsize=(10, 8))
        
#         # Get base value
#         base_value = st.session_state.shap_explainer.expected_value
#         if isinstance(base_value, (list, np.ndarray)):
#             base_value = base_value[1]
        
#         shap.waterfall_plot(
#             shap.Explanation(
#                 values=shap_vals,
#                 base_values=base_value,
#                 data=case[st.session_state.feature_names].values,
#                 feature_names=st.session_state.feature_names
#             ),
#             show=False
#         )
#         st.pyplot(fig)
#         plt.close()
    
#     with col2:
#         st.markdown("### How to Read This Chart")
#         st.markdown("""
#         - **Starting point (E[f(x)])**: Average risk score across all returns
#         - **Red bars**: Factors that INCREASE risk
#         - **Blue bars**: Factors that DECREASE risk
#         - **Ending point (f(x))**: Final risk score for this case
        
#         Each bar shows how much that specific factor pushed the risk score up or down.
#         """)
    
#     st.markdown("---")
    
#     # Comparison to Peers
#     st.header("👥 Comparison to Peer Group")
    
#     # Determine income bracket
#     income = case.get('c00100', 0)
#     if income < 25000:
#         bracket = "Under $25k"
#     elif income < 50000:
#         bracket = "$25k-$50k"
#     elif income < 100000:
#         bracket = "$50k-$100k"
#     elif income < 250000:
#         bracket = "$100k-$250k"
#     elif income < 500000:
#         bracket = "$250k-$500k"
#     else:
#         bracket = "Over $500k"
    
#     st.markdown(f"**Income Bracket:** {bracket}")
    
#     # Calculate peer statistics
#     peer_group = results[
#         (results['c00100'] >= income * 0.8) & 
#         (results['c00100'] <= income * 1.2)
#     ]
    
#     if len(peer_group) > 1:
#         comp_col1, comp_col2 = st.columns(2)
        
#         with comp_col1:
#             st.markdown("### This Case vs. Peers")
            
#             metrics = {
#                 'Charitable Ratio': (case.get('e19800', 0) / max(case.get('c00100', 1), 1), 
#                                     (peer_group['e19800'] / peer_group['c00100'].replace(0, 1)).mean()),
#                 'Medical Ratio': (case.get('e17500', 0) / max(case.get('c00100', 1), 1),
#                                  (peer_group['e17500'] / peer_group['c00100'].replace(0, 1)).mean()),
#                 'EITC Ratio': (case.get('eitc', 0) / max(case.get('c00100', 1), 1),
#                               (peer_group['eitc'] / peer_group['c00100'].replace(0, 1)).mean())
#             }
            
#             for metric_name, (case_val, peer_avg) in metrics.items():
#                 if peer_avg > 0:
#                     diff_pct = ((case_val - peer_avg) / peer_avg * 100) if peer_avg > 0 else 0
#                     if abs(diff_pct) > 50:
#                         color = "🔴" if case_val > peer_avg else "🟢"
#                     else:
#                         color = "⚪"
#                     st.markdown(f"{color} **{metric_name}**: {case_val:.2%} (Peer avg: {peer_avg:.2%}, {diff_pct:+.0f}% diff)")
        
#         with comp_col2:
#             # Risk score distribution for peer group
#             fig_peer = go.Figure()
            
#             fig_peer.add_trace(go.Histogram(
#                 x=peer_group['fraud_risk_score'],
#                 name='Peer Group',
#                 nbinsx=30,
#                 marker_color='lightblue'
#             ))
            
#             fig_peer.add_vline(
#                 x=risk_score,
#                 line_dash="dash",
#                 line_color="red",
#                 annotation_text="This Case"
#             )
            
#             fig_peer.update_layout(
#                 title=f"Risk Scores in {bracket} Bracket",
#                 xaxis_title="Risk Score",
#                 yaxis_title="Count",
#                 showlegend=False
#             )
            
#             st.plotly_chart(fig_peer, use_container_width=True)
    
#     st.markdown("---")
    
#     # Audit Recommendation
#     st.header("✅ Audit Recommendation")
    
#     if risk_score >= 0.9:
#         st.error("""
#         ### 🚨 PRIORITY AUDIT RECOMMENDED
        
#         This case exhibits **very high risk** indicators and should be prioritized for immediate audit.
        
#         **Recommended Actions:**
#         1. Full examination of all deductions and credits claimed
#         2. Verification of income sources and amounts
#         3. Review of supporting documentation for all claims
#         4. Potential for significant revenue recovery
#         """)
#     elif risk_score >= 0.7:
#         st.warning("""
#         ### ⚠️ AUDIT RECOMMENDED
        
#         This case shows **high risk** patterns and warrants audit examination.
        
#         **Recommended Actions:**
#         1. Review specific red flags identified above
#         2. Request documentation for unusual deductions
#         3. Verify income reporting accuracy
#         4. Consider for routine audit queue
#         """)
#     else:
#         st.info("""
#         ### ℹ️ MONITORING RECOMMENDED
        
#         While below the standard audit threshold, this case shows some concerning patterns.
        
#         **Recommended Actions:**
#         1. Monitor for patterns in future filings
#         2. Flag for review if similar patterns continue
#         """)
    
#     st.markdown("---")
    
#     # Export Case Report
#     st.header("💾 Export Case Report")
    
#     # Create detailed case report
#     report_data = {
#         'Case_ID': [idx],
#         'Risk_Score': [risk_score],
#         'Risk_Level': [case['risk_level']],
#         'DIF_Score': [case.get('dif_score', 0)],
#         'AGI': [case.get('c00100', 0)],
#         'Red_Flags': ['; '.join(red_flags) if red_flags else 'None specific'],
#         'Audit_Recommendation': ['Priority' if risk_score >= 0.9 else 'Standard' if risk_score >= 0.7 else 'Monitor']
#     }
    
#     # Add top 5 SHAP factors
#     for i, exp in enumerate(explanations[:5], 1):
#         report_data[f'Factor_{i}'] = [exp['feature']]
#         report_data[f'Factor_{i}_Impact'] = [exp['shap_value']]
    
#     report_df = pd.DataFrame(report_data)
    
#     csv_report = report_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="📥 Download Case Report (CSV)",
#         data=csv_report,
#         file_name=f"case_{idx}_detailed_report.csv",
#         mime="text/csv"
#     )

# # ========== MAIN APP ROUTER ==========
# def main():
#     # Sidebar navigation
#     with st.sidebar:
#         st.image("https://via.placeholder.com/150x50/1976d2/ffffff?text=AURA+System", use_column_width=True)
        
#         st.markdown("---")
#         st.header("Navigation")
        
#         # Page buttons
#         pages = {
#             'upload': '📤 Upload Data',
#             'summary': '📊 Summary',
#             'non_compliant_list': '📋 Non-Compliant',
#             'case_detail': '🔬 Case Detail'
#         }
        
#         for page_key, page_name in pages.items():
#             if st.button(page_name, key=f"nav_{page_key}", use_container_width=True,
#                         type="primary" if st.session_state.page == page_key else "secondary"):
#                 if page_key == 'upload' or st.session_state.results_df is not None:
#                     st.session_state.page = page_key
#                     st.rerun()
        
#         st.markdown("---")
        
#         # Stats in sidebar
#         if st.session_state.results_df is not None:
#             st.markdown("### Quick Stats")
#             results = st.session_state.results_df
#             st.metric("Total Returns", f"{len(results):,}")
#             st.metric("Non-Compliant", f"{results['non_compliant'].sum():,}")
#             st.metric("Avg Risk", f"{results['fraud_risk_score'].mean():.3f}")
        
#         st.markdown("---")
#         st.markdown("### About")
#         st.markdown("""
#         **AURA Tax Fraud Detection**
        
#         AI-powered system using:
#         - Machine Learning (Random Forest)
#         - Anomaly Detection (Isolation Forest)
#         - Explainable AI (SHAP)
#         - DIF-style risk scoring
#         """)
    
#     # Route to appropriate page
#     if st.session_state.page == 'upload':
#         page_upload()
#     elif st.session_state.page == 'summary':
#         page_summary()
#     elif st.session_state.page == 'non_compliant_list':
#         page_non_compliant_list()
#     elif st.session_state.page == 'case_detail':
#         page_case_detail()

# if __name__ == "__main__":
#     main()


# beta_app.py — UI-preserving app updated for CorrectedAURAStyleDetectorV2

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from joblib import load
# import shap
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# # ---------- Import updated detector class (needed so joblib can unpickle) ----------
# from corrected_aura_detector_v2 import CorrectedAURAStyleDetectorV2

# # ---------- Page configuration ----------
# st.set_page_config(
#     page_title="AURA Tax Fraud Detection (v2)",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ---------- Custom CSS ----------
# st.markdown("""
# <style>
#     .big-metric { font-size: 28px; font-weight: bold; color: #1f77b4; }
#     .risk-very-high { background-color: #ffebee; padding: 10px; border-left: 4px solid #d32f2f; margin: 10px 0; }
#     .risk-high { background-color: #fff3e0; padding: 10px; border-left: 4px solid #f57c00; margin: 10px 0; }
#     .case-card { background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1976d2; }
#     .explanation-box { background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin: 10px 0; }
#     .red-flag { color: #d32f2f; font-weight: bold; }
#     .stButton>button { width: 100%; }
#     .badge { padding: 2px 8px; border-radius: 10px; font-size: 12px; }
#     .badge-fast { background:#fff3cd; color:#8a6d3b; }
#     .badge-full { background:#e0f7fa; color:#006064; }
# </style>
# """, unsafe_allow_html=True)

# # ---------- Session state ----------
# if 'page' not in st.session_state:
#     st.session_state.page = 'upload'
# if 'results_df' not in st.session_state:
#     st.session_state.results_df = None
# if 'detector' not in st.session_state:
#     st.session_state.detector = None
# if 'shap_values' not in st.session_state:
#     st.session_state.shap_values = None
# if 'shap_explainer' not in st.session_state:
#     st.session_state.shap_explainer = None
# if 'feature_names' not in st.session_state:
#     st.session_state.feature_names = None
# if 'X_scaled' not in st.session_state:
#     st.session_state.X_scaled = None
# if 'selected_case' not in st.session_state:
#     st.session_state.selected_case = None

# # ---------- Model loading ----------
# def load_trained_model():
#     """Load the trained AURA detector v2 (joblib)."""
#     try:
#         detector = load("corrected_aura_detector_v2.pkl")
#         return detector, True
#     except FileNotFoundError:
#         return None, False
#     except Exception as e:
#         st.error(f"Model load error: {e}")
#         return None, False

# # ---------- File handling ----------
# def process_uploaded_file(uploaded_file):
#     """Process uploaded Excel/CSV file (same behavior as original)."""
#     try:
#         if uploaded_file.name.endswith('.csv'):
#             return pd.read_csv(uploaded_file)
#         elif uploaded_file.name.endswith(('.xls', '.xlsx')):
#             return pd.read_excel(uploaded_file)
#         else:
#             st.error("Unsupported file format. Please upload CSV or Excel")
#             return None
#     except Exception as e:
#         st.error(f"Error reading file: {str(e)}")
#         return None

# # ---------- Feature alignment helper (prevents scaler feature count mismatch) ----------
# def align_features_like_training(detector, X):
#     """
#     Align columns and order to exactly match training.
#     Priority for feature list:
#       1) detector.feature_weights.keys() (from RF fit)
#       2) detector.models['random_forest'].feature_names_in_
#       3) detector.scalers['main'].feature_names_in_
#     """
#     trained_features = None
#     if hasattr(detector, "feature_weights") and detector.feature_weights:
#         trained_features = list(detector.feature_weights.keys())
#     elif "random_forest" in detector.models and hasattr(detector.models["random_forest"], "feature_names_in_"):
#         trained_features = list(detector.models["random_forest"].feature_names_in_)
#     elif "main" in detector.scalers and hasattr(detector.scalers["main"], "feature_names_in_"):
#         trained_features = list(detector.scalers["main"].feature_names_in_)
#     else:
#         # Fallback: use current X columns (last resort)
#         trained_features = list(X.columns)

#     # Add missing with zeros
#     missing = [f for f in trained_features if f not in X.columns]
#     for f in missing:
#         st.warning(f"Adding missing feature: {f}")
#         X[f] = 0

#     # Drop extras not seen in training
#     extras = [c for c in X.columns if c not in trained_features]
#     if extras:
#         X = X.drop(columns=extras)

#     # Order to match training
#     X = X[trained_features]
#     return X, trained_features

# # ---------- Fraud detection core ----------
# def run_fraud_detection(df, detector):
#     """Run fraud detection on uploaded data (UI flow preserved)."""
#     try:
#         with st.spinner("🔍 Analyzing tax returns... This may take a few minutes."):
#             # Remove any ground-truth labels if present
#             drop_cols = ["is_fraud", "fraud_type", "fraud_severity"]
#             df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

#             progress_bar = st.progress(0)
#             st.text("Engineering features...")
#             df_enhanced = detector.engineer_available_features(df_clean)
#             progress_bar.progress(33)

#             st.text("Creating income brackets...")
#             if 'c00100' not in df_enhanced.columns:
#                 st.error("c00100 (AGI) column missing after feature engineering.")
#                 return None, None, None
#             df_enhanced['income_bracket'] = pd.cut(
#                 df_enhanced['c00100'],
#                 bins=[-np.inf, 25000, 50000, 100000, 250000, 500000, np.inf],
#                 labels=['under_25k', '25k_50k', '50k_100k', '100k_250k', '250k_500k', 'over_500k']
#             )

#             if not detector.income_bracket_profiles:
#                 st.text("Creating income bracket profiles...")
#                 profiles = detector.create_income_bracket_profiles(df_enhanced)
#                 detector.income_bracket_profiles = profiles
#             progress_bar.progress(50)

#             st.text("Calculating DIF-style risk scores...")
#             df_scored = detector.calculate_dif_style_risk_score(df_enhanced, detector.income_bracket_profiles)
#             progress_bar.progress(66)

#             exclude_cols = ["income_bracket"]
#             feature_cols = [col for col in df_scored.columns if col not in exclude_cols]
#             X = df_scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

#             # Align features to trained set (prevents scaler mismatch)
#             X, trained_features = align_features_like_training(detector, X)

#             st.text("Generating predictions...")
#             scaler = detector.scalers.get('main', None)
#             if scaler is None:
#                 st.warning("No scaler found in model; using raw features.")
#                 X_scaled = X.values
#             else:
#                 X_scaled = scaler.transform(X.values)

#             # v2 returns 6 values, v1 returns 4 — handle both
#             try:
#                 final_scores, ensemble_scores, rf_proba, lr_proba, iso_scores, exp_adj_norm = detector.predict_audit_risk(X_scaled)
#             except ValueError:
#                 ensemble_scores, rf_proba, lr_proba, iso_scores = detector.predict_audit_risk(X_scaled)
#                 final_scores = ensemble_scores
#                 exp_adj_norm = None

#             # Build results like original UI expects
#             results = df_clean.copy()
#             results['fraud_risk_score'] = final_scores
#             results['rf_probability'] = rf_proba
#             results['lr_probability'] = lr_proba
#             results['anomaly_score'] = iso_scores
#             results['dif_score'] = df_scored['dif_risk_score'].values
#             if exp_adj_norm is not None:
#                 results['predicted_adjustment_severity'] = exp_adj_norm

#             results['risk_level'] = pd.cut(
#                 results['fraud_risk_score'],
#                 bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
#                 labels=['Low', 'Medium-Low', 'Medium', 'High', 'Very High']
#             )
#             results['non_compliant'] = (results['fraud_risk_score'] >= 0.7).astype(int)

#             progress_bar.progress(100)
#             return results, X_scaled, trained_features

#     except Exception as e:
#         st.error(f"Error during fraud detection: {str(e)}")
#         st.exception(e)
#         return None, None, None

# # ---------- SHAP ----------
# def calculate_shap_values(detector, X_scaled):
#     """Calculate SHAP values for global explainability (Random Forest)."""
#     with st.spinner("📊 Calculating SHAP explanations... This may take a moment."):
#         explainer = shap.TreeExplainer(detector.models['random_forest'])
#         shap_values = explainer.shap_values(X_scaled)
#         if isinstance(shap_values, list):
#             shap_values = shap_values[1]
#         return shap_values, explainer

# def get_red_flags(row, feature_names, shap_vals):
#     """Identify specific red flags for a case (same logic as original)."""
#     red_flags = []

#     charitable_ratio = row.get('e19800', 0) / max(row.get('c00100', 1), 1)
#     if charitable_ratio > 0.1:
#         red_flags.append(f"🚩 High charitable deductions ({charitable_ratio*100:.1f}% of income)")

#     medical_ratio = row.get('e17500', 0) / max(row.get('c00100', 1), 1)
#     if medical_ratio > 0.075:
#         red_flags.append(f"🚩 High medical deductions ({medical_ratio*100:.1f}% of income)")

#     rounded_count = 0
#     for col in ['e19800', 'e17500', 'e19200', 'e00200']:
#         if col in row and row[col] > 0 and row[col] % 100 == 0:
#             rounded_count += 1
#     if rounded_count >= 2:
#         red_flags.append(f"🚩 Multiple rounded amounts detected ({rounded_count} fields)")

#     if row.get('e00900', 0) < 0:
#         red_flags.append(f"🚩 Business losses reported (${abs(row.get('e00900', 0)):,.0f})")

#     eitc_ratio = row.get('eitc', 0) / max(row.get('c00100', 1), 1)
#     if eitc_ratio > 0.3:
#         red_flags.append(f"🚩 High EITC claim relative to income ({eitc_ratio*100:.1f}%)")

#     income_per_exemption = row.get('c00100', 0) / max(row.get('XTOT', 1), 1)
#     if income_per_exemption < 15000 and row.get('c00100', 0) > 0:
#         red_flags.append(f"🚩 Low income per household member (${income_per_exemption:,.0f})")

#     refund_ratio = row.get('refund', 0) / max(row.get('c00100', 1), 1)
#     if refund_ratio > 0.3:
#         red_flags.append(f"🚩 Large refund relative to income ({refund_ratio*100:.1f}%)")

#     return red_flags

# def generate_shap_explanation(row, shap_vals, feature_names, top_n=5):
#     """Generate SHAP-based explanation table (safe for array-like SHAP outputs)."""
#     # flatten each shap value in case it’s an array
#     shap_flat = []
#     for s in shap_vals:
#         if isinstance(s, (list, np.ndarray)):
#             # take the first element if it’s 1D or 0 if it’s all zeros
#             s = float(np.ravel(s)[0]) if np.size(s) > 0 else 0.0
#         shap_flat.append(s)

#     feature_contributions = list(zip(feature_names, shap_flat, [row.get(f, 0) for f in feature_names]))
#     feature_contributions.sort(key=lambda x: abs(float(x[1])), reverse=True)

#     explanations = []
#     for i, (feature, shap_val, value) in enumerate(feature_contributions[:top_n], 1):
#         impact = "increases" if shap_val > 0 else "decreases"
#         explanations.append({
#             'rank': i,
#             'feature': feature,
#             'shap_value': shap_val,
#             'impact': impact,
#             'value': value
#         })
#     return explanations


# # ========================== PAGE: UPLOAD ==========================
# def page_upload():
#     st.title("🔍 AURA Tax Fraud Detection System (v2)")
#     st.markdown("### Upload Tax Return Data for Analysis")

#     # Load model
#     detector, model_loaded = load_trained_model()
#     if not model_loaded:
#         st.error("❌ Model not found! Please train the model first.")
#         st.code("python corrected_aura_detector_v2.py your_training_data.csv(.gz)", language="bash")
#         return

#     st.session_state.detector = detector
#     st.success("✅ Model loaded successfully")

#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Upload Excel or CSV file with tax return data",
#         type=['csv', 'xlsx', 'xls'],
#         help="File should contain standard tax return fields (c00100, e19800, e17500, etc.)"
#     )

#     if uploaded_file is None:
#         st.info("👆 Upload a file to begin analysis")
#         with st.expander("📋 Required Data Format"):
#             st.markdown("""
#             Your file should contain tax return data with these key columns:
#             - **c00100**: Adjusted Gross Income
#             - **e19800**: Charitable contributions
#             - **e17500**: Medical deductions
#             - **e19200**: Mortgage interest
#             - **e00900**: Business income
#             - **e00200**: Wages and salaries
#             - **XTOT**: Total exemptions
#             - **eitc**: Earned Income Tax Credit
#             - **MARS**: Filing status
#             - **age_head**: Primary taxpayer age
#             """)
#         return

#     # Process file
#     df = process_uploaded_file(uploaded_file)
#     if df is None:
#         return

#     st.success(f"✅ Loaded {len(df):,} tax returns")

#     # Preview
#     with st.expander("👀 Preview Data"):
#         st.dataframe(df.head(10), use_container_width=True)

#     # Fast Mode toggle (kept exactly as your original design intent)
#     use_fast_mode = st.checkbox(
#         "⚡ Fast Mode (Skip SHAP - recommended for >1000 returns)",
#         value=len(df) > 1000,
#         help="SHAP explanations will be calculated on-demand when viewing individual cases"
#     )

#     # Sidebar badge for mode
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### Mode")
#     if use_fast_mode:
#         st.sidebar.markdown('<span class="badge badge-fast">⚡ Fast Mode Active</span>', unsafe_allow_html=True)
#     else:
#         st.sidebar.markdown('<span class="badge badge-full">🧮 SHAP Active</span>', unsafe_allow_html=True)

#     # Run analysis
#     if st.button("🚀 Run Fraud Detection Analysis", type="primary", use_container_width=True):
#         try:
#             results, X_scaled, feature_names = run_fraud_detection(df, detector)
#             if results is None or len(results) == 0:
#                 st.error("❌ Analysis failed to produce results")
#                 return

#             st.success("✅ Model predictions complete!")

#             # Save to session state
#             st.session_state.results_df = results
#             st.session_state.X_scaled = X_scaled
#             st.session_state.feature_names = feature_names

#             # SHAP (global) if not in fast mode
#             if not use_fast_mode:
#                 try:
#                     with st.spinner("Calculating SHAP explanations... (this may take a few minutes)"):
#                         shap_values, explainer = calculate_shap_values(detector, X_scaled)
#                         st.session_state.shap_values = shap_values
#                         st.session_state.shap_explainer = explainer
#                         st.success("✅ SHAP calculations complete!")
#                 except Exception as shap_error:
#                     st.warning(f"⚠️ SHAP calculation skipped: {shap_error}")
#                     st.session_state.shap_values = None
#                     st.session_state.shap_explainer = None
#             else:
#                 st.info("⚡ Fast mode enabled — SHAP will be calculated on-demand for individual cases.")
#                 st.session_state.shap_values = None
#                 st.session_state.shap_explainer = None

#             # Navigation controls (same as original)
#             st.success("✅ Analysis Complete! Navigating to summary...")
#             st.markdown("---")
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("📊 View Summary", type="primary", use_container_width=True):
#                     st.session_state.page = 'summary'
#                     st.rerun()
#             with col2:
#                 if st.button("📋 View Non-Compliant Cases", use_container_width=True):
#                     st.session_state.page = 'non_compliant_list'
#                     st.rerun()

#             import time
#             time.sleep(0.5)
#             st.session_state.page = 'summary'
#             st.rerun()

#         except Exception as e:
#             st.error(f"❌ Analysis failed with error: {str(e)}")
#             st.exception(e)
#             return

# # ========================== PAGE: SUMMARY ==========================
# def page_summary():
#     st.title("📊 Analysis Summary & Results")

#     if st.session_state.results_df is None:
#         st.warning("No results available. Please upload data first.")
#         if st.button("← Back to Upload"):
#             st.session_state.page = 'upload'
#             st.rerun()
#         return

#     results = st.session_state.results_df

#     # Navigation
#     col1, col2 = st.columns([3, 1])
#     with col2:
#         if st.button("← Back to Upload"):
#             st.session_state.page = 'upload'
#             st.rerun()

#     # Key Metrics
#     st.header("📈 Key Metrics")

#     col1, col2, col3, col4 = st.columns(4)

#     total_returns = len(results)
#     non_compliant_count = results['non_compliant'].sum()
#     non_compliant_pct = (non_compliant_count / total_returns * 100) if total_returns > 0 else 0
#     avg_risk = results['fraud_risk_score'].mean()
#     very_high_risk = (results['fraud_risk_score'] >= 0.9).sum()

#     with col1:
#         st.metric("Total Returns Analyzed", f"{total_returns:,}")

#     with col2:
#         st.metric("Non-Compliant Filings", f"{non_compliant_count:,}",
#                   delta=f"{non_compliant_pct:.1f}% of total", delta_color="inverse")

#     with col3:
#         st.metric("Average Risk Score", f"{avg_risk:.3f}")

#     with col4:
#         st.metric("Very High Risk (>0.9)", f"{very_high_risk:,}")

#     st.markdown("---")

#     # Executive Summary
#     st.header("💡 Executive Summary")

#     summary_col1, summary_col2 = st.columns([2, 1])

#     with summary_col1:
#         st.markdown(f"""
#         ### Analysis Overview

#         Out of **{total_returns:,}** tax returns analyzed:
#         - **{non_compliant_count:,} returns ({non_compliant_pct:.1f}%)** were flagged as non-compliant (risk score ≥ 0.70)
#         - **{very_high_risk:,} returns** are considered very high risk (score ≥ 0.90)
#         - The average fraud risk score across all returns is **{avg_risk:.3f}**

#         ### Risk Distribution
#         """)
#         risk_dist = results['risk_level'].value_counts().sort_index()
#         for level, count in risk_dist.items():
#             pct = count / total_returns * 100
#             st.markdown(f"- **{level}**: {count:,} returns ({pct:.1f}%)")

#     with summary_col2:
#         fig_pie = px.pie(
#             values=risk_dist.values,
#             names=risk_dist.index,
#             title="Risk Distribution",
#             color_discrete_map={
#                 'Low': '#4caf50',
#                 'Medium-Low': '#8bc34a',
#                 'Medium': '#ffc107',
#                 'High': '#ff9800',
#                 'Very High': '#f44336'
#             }
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)

#     st.markdown("---")

#     # SHAP Global Insights (only if computed)
#     st.header("🎯 Key Risk Factors (SHAP Analysis)")

#     if st.session_state.shap_values is not None:
#         st.markdown("The AI model identified the following factors as most important in determining fraud risk:")

#         shap_abs_mean = np.abs(st.session_state.shap_values).mean(axis=0)
#         feature_importance = pd.DataFrame({
#             'Feature': st.session_state.feature_names,
#             'Importance': shap_abs_mean
#         }).sort_values('Importance', ascending=False).head(10)

#         col1, col2 = st.columns([1, 2])
#         with col1:
#             st.dataframe(
#                 feature_importance.style.format({'Importance': '{:.4f}'}),
#                 use_container_width=True,
#                 height=400
#             )
#         with col2:
#             fig, ax = plt.subplots(figsize=(10, 6))
#             shap.summary_plot(
#                 st.session_state.shap_values[:1000],
#                 features=results[st.session_state.feature_names].iloc[:1000].values,
#                 feature_names=st.session_state.feature_names,
#                 show=False,
#                 max_display=10
#             )
#             st.pyplot(fig)
#             plt.close()

#     st.markdown("---")

#     # Risk Distribution Chart
#     st.header("📊 Risk Score Distribution")
#     fig_hist = px.histogram(
#         results,
#         x='fraud_risk_score',
#         nbins=50,
#         title="Distribution of Fraud Risk Scores",
#         labels={'fraud_risk_score': 'Fraud Risk Score', 'count': 'Number of Returns'}
#     )
#     fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red",
#                       annotation_text="Non-Compliant Threshold (0.7)")
#     fig_hist.add_vline(x=0.9, line_dash="dash", line_color="darkred",
#                       annotation_text="Very High Risk (0.9)")
#     st.plotly_chart(fig_hist, use_container_width=True)

#     st.markdown("---")

#     # Next Steps
#     st.header("🔍 Next Steps")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("📋 View All Non-Compliant Filings", type="primary", use_container_width=True):
#             st.session_state.page = 'non_compliant_list'
#             st.rerun()
#     with col2:
#         csv = results.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="💾 Download Full Results (CSV)",
#             data=csv,
#             file_name="fraud_detection_results.csv",
#             mime="text/csv",
#             use_container_width=True
#         )

# # ========================== PAGE: NON-COMPLIANT LIST ==========================
# def page_non_compliant_list():
#     st.title("📋 Non-Compliant Tax Filings")

#     if st.session_state.results_df is None:
#         st.warning("No results available.")
#         if st.button("← Back to Upload"):
#             st.session_state.page = 'upload'
#             st.rerun()
#         return

#     # Navigation
#     col1, col2 = st.columns([3, 1])
#     with col2:
#         if st.button("← Back to Summary"):
#             st.session_state.page = 'summary'
#             st.rerun()

#     results = st.session_state.results_df
#     non_compliant = results[results['non_compliant'] == 1].sort_values('fraud_risk_score', ascending=False)

#     st.markdown(f"### Found **{len(non_compliant):,}** non-compliant tax filings (Risk Score ≥ 0.70)")

#     if len(non_compliant) == 0:
#         st.success("🎉 No non-compliant filings detected!")
#         return

#     # Filters
#     st.subheader("🔍 Filters")
#     filter_col1, filter_col2, filter_col3 = st.columns(3)
#     with filter_col1:
#         min_risk = st.slider("Minimum Risk Score", 0.0, 1.0, 0.7, 0.05)
#     with filter_col2:
#         risk_levels = st.multiselect(
#             "Risk Levels",
#             options=['High', 'Very High'],
#             default=['High', 'Very High']
#         )
#     with filter_col3:
#         if 'c00100' in non_compliant.columns:
#             st.select_slider(
#                 "Income Range",
#                 options=['All', '<$25k', '$25k-50k', '$50k-100k', '$100k-250k', '>$250k'],
#                 value='All'
#             )

#     # Apply filters
#     filtered = non_compliant[
#         (non_compliant['fraud_risk_score'] >= min_risk) &
#         (non_compliant['risk_level'].isin(risk_levels))
#     ]

#     st.markdown(f"**Showing {len(filtered):,} of {len(non_compliant):,} non-compliant filings**")
#     st.markdown("---")

#     # Display list (cards)
#     st.subheader("📊 Non-Compliant Filings List")
#     st.markdown("*Click on a case to view detailed analysis*")

#     for idx, row in filtered.iterrows():
#         risk_score = row['fraud_risk_score']
#         risk_level = row['risk_level']

#         if risk_score >= 0.9:
#             card_class = "risk-very-high"; emoji = "🔴"
#         else:
#             card_class = "risk-high"; emoji = "🟠"

#         with st.container():
#             st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
#             col1, col2, col3 = st.columns([2, 2, 1])

#             with col1:
#                 st.markdown(f"### {emoji} Case #{idx}")
#                 st.markdown(f"**Risk Score:** {risk_score:.3f}")
#                 st.markdown(f"**Risk Level:** {risk_level}")

#             with col2:
#                 if 'c00100' in row:
#                     st.markdown(f"**AGI:** ${row.get('c00100', 0):,.0f}")
#                 if 'e19800' in row:
#                     st.markdown(f"**Charitable:** ${row.get('e19800', 0):,.0f}")
#                 if 'e17500' in row:
#                     st.markdown(f"**Medical:** ${row.get('e17500', 0):,.0f}")

#             with col3:
#                 if st.button("View Details →", key=f"btn_{idx}", use_container_width=True):
#                     st.session_state.selected_case = idx
#                     st.session_state.page = 'case_detail'
#                     st.rerun()

#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown("")

#     st.markdown("---")
#     csv_non_compliant = filtered.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="💾 Download Non-Compliant Cases (CSV)",
#         data=csv_non_compliant,
#         file_name="non_compliant_cases.csv",
#         mime="text/csv"
#     )

# # ========================== PAGE: CASE DETAIL ==========================
# def page_case_detail():
#     st.title("🔬 Detailed Case Analysis")

#     if st.session_state.selected_case is None or st.session_state.results_df is None:
#         st.warning("No case selected.")
#         if st.button("← Back to List"):
#             st.session_state.page = 'non_compliant_list'
#             st.rerun()
#         return

#     if st.button("← Back to Non-Compliant List"):
#         st.session_state.page = 'non_compliant_list'
#         st.rerun()

#     idx = st.session_state.selected_case
#     results = st.session_state.results_df
#     case = results.loc[idx]

#     # Determine SHAP values for this case
#     shap_idx = results.index.get_loc(idx)
#     if st.session_state.shap_values is None:
#         with st.spinner("Calculating SHAP explanation for this case..."):
#             try:
#                 if st.session_state.X_scaled is not None:
#                     explainer = shap.TreeExplainer(st.session_state.detector.models['random_forest'])
#                     shap_vals = explainer.shap_values(st.session_state.X_scaled[shap_idx:shap_idx+1])
#                     if isinstance(shap_vals, list):
#                         shap_vals = shap_vals[1][0]
#                     else:
#                         shap_vals = shap_vals[0]
#                     base_value = explainer.expected_value
#                     if isinstance(base_value, (list, np.ndarray)):
#                         base_value = base_value[1]
#                 else:
#                     shap_vals = None
#                     base_value = 0.5
#             except Exception as e:
#                 st.warning(f"Could not calculate SHAP: {str(e)}")
#                 shap_vals = None
#                 base_value = 0.5
#     else:
#         shap_vals = st.session_state.shap_values[shap_idx]
#         base_value = st.session_state.shap_explainer.expected_value
#         if isinstance(base_value, (list, np.ndarray)):
#             base_value = base_value[1]

#     # Header
#     risk_score = case['fraud_risk_score']
#     if risk_score >= 0.9:
#         st.error(f"🔴 **VERY HIGH RISK CASE** - Risk Score: {risk_score:.3f}")
#     else:
#         st.warning(f"🟠 **HIGH RISK CASE** - Risk Score: {risk_score:.3f}")

#     st.markdown(f"### Case #{idx} - Detailed Analysis")
#     st.markdown("---")

#     # Case Overview
#     st.header("📋 Case Overview")
#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         st.metric("Fraud Risk Score", f"{risk_score:.3f}")
#         st.metric("Risk Level", case['risk_level'])
#     with col2:
#         st.metric("DIF Score", f"{case.get('dif_score', 0):.1f}")
#         st.metric("RF Probability", f"{case.get('rf_probability', 0):.3f}")
#     with col3:
#         if 'c00100' in case:
#             st.metric("Adjusted Gross Income", f"${case['c00100']:,.0f}")
#         if 'XTOT' in case:
#             st.metric("Exemptions", f"{int(case['XTOT'])}")
#     with col4:
#         if 'age_head' in case:
#             st.metric("Age", f"{int(case['age_head'])}")
#         if 'MARS' in case:
#             filing_status = {1: 'Single', 2: 'Married Joint', 3: 'Married Separate', 4: 'Head of Household'}
#             st.metric("Filing Status", filing_status.get(int(case['MARS']), 'Unknown'))

#     st.markdown("---")

#     # Red Flags
#     st.header("🚩 Identified Red Flags")
#     red_flags = get_red_flags(case, st.session_state.feature_names, shap_vals if shap_vals is not None else [])
#     if red_flags:
#         for flag in red_flags:
#             st.markdown(f'<div class="explanation-box">{flag}</div>', unsafe_allow_html=True)
#     else:
#         st.info("No specific red flags identified - flagged based on statistical patterns")

#     st.markdown("---")

#     # Financial Details
#     st.header("💰 Financial Details")
#     financial_data = {}
#     financial_fields = {
#         'c00100': 'Adjusted Gross Income',
#         'e00200': 'Wages & Salaries',
#         'e00900': 'Business Income',
#         'e19800': 'Charitable Contributions',
#         'e17500': 'Medical Deductions',
#         'e19200': 'Mortgage Interest',
#         'c04470': 'Itemized Deductions',
#         'eitc': 'Earned Income Tax Credit',
#         'refund': 'Tax Refund'
#     }
#     for field, label in financial_fields.items():
#         if field in case:
#             financial_data[label] = f"${case[field]:,.0f}"

#     col1, col2 = st.columns(2)
#     items = list(financial_data.items())
#     mid = len(items) // 2
#     with col1:
#         for label, value in items[:mid]:
#             st.markdown(f"**{label}:** {value}")
#     with col2:
#         for label, value in items[mid:]:
#             st.markdown(f"**{label}:** {value}")

#     st.markdown("---")

#     # SHAP Explanation (table + waterfall) if available
#     if shap_vals is not None and st.session_state.feature_names is not None:
#         st.header("🤖 AI Explanation (SHAP Analysis)")
#         st.markdown("Top factors contributing to this case's risk score:")

#         explanations = generate_shap_explanation(case, shap_vals, st.session_state.feature_names, top_n=10)
#         exp_df = pd.DataFrame(explanations)
#         exp_df['shap_value'] = exp_df['shap_value'].apply(lambda x: f"{x:+.4f}")
#         exp_df['feature'] = exp_df['feature'].apply(lambda x: x.replace('_', ' ').title())

#         st.dataframe(
#             exp_df[['rank', 'feature', 'shap_value', 'impact']],
#             use_container_width=True,
#             hide_index=True
#         )

#         st.markdown("---")
#         st.header("📊 SHAP Waterfall Plot")
#         st.markdown("*Visual breakdown of how each factor contributed to the final risk score*")
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             fig, _ = plt.subplots(figsize=(10, 8))
#             shap.waterfall_plot(
#                 shap.Explanation(
#                     values=shap_vals,
#                     base_values=st.session_state.shap_explainer.expected_value[1]
#                         if isinstance(st.session_state.shap_explainer.expected_value, (list, np.ndarray))
#                         else st.session_state.shap_explainer.expected_value,
#                     data=case[st.session_state.feature_names].values,
#                     feature_names=st.session_state.feature_names
#                 ),
#                 show=False
#             )
#             st.pyplot(fig)
#             plt.close()
#         with col2:
#             st.markdown("""
#             ### How to Read This Chart
#             - **Starting point (E[f(x)])**: Average risk score across all returns  
#             - **Red bars**: Factors that INCREASE risk  
#             - **Blue bars**: Factors that DECREASE risk  
#             - **Ending point (f(x))**: Final risk score for this case  
#             """)

#     st.markdown("---")

#     # Peer Comparison
#     st.header("👥 Comparison to Peer Group")
#     income = case.get('c00100', 0)
#     if income < 25000: bracket = "Under $25k"
#     elif income < 50000: bracket = "$25k-$50k"
#     elif income < 100000: bracket = "$50k-$100k"
#     elif income < 250000: bracket = "$100k-$250k"
#     elif income < 500000: bracket = "$250k-$500k"
#     else: bracket = "Over $500k"
#     st.markdown(f"**Income Bracket:** {bracket}")

#     peer_group = results[(results['c00100'] >= income * 0.8) & (results['c00100'] <= income * 1.2)]
#     if len(peer_group) > 1:
#         comp_col1, comp_col2 = st.columns(2)
#         with comp_col1:
#             st.markdown("### This Case vs. Peers")
#             metrics = {
#                 'Charitable Ratio': (case.get('e19800', 0) / max(case.get('c00100', 1), 1),
#                                     (peer_group['e19800'] / peer_group['c00100'].replace(0, 1)).mean()),
#                 'Medical Ratio': (case.get('e17500', 0) / max(case.get('c00100', 1), 1),
#                                  (peer_group['e17500'] / peer_group['c00100'].replace(0, 1)).mean()),
#                 'EITC Ratio': (case.get('eitc', 0) / max(case.get('c00100', 1), 1),
#                               (peer_group['eitc'] / peer_group['c00100'].replace(0, 1)).mean())
#             }
#             for metric_name, (case_val, peer_avg) in metrics.items():
#                 if peer_avg > 0:
#                     diff_pct = ((case_val - peer_avg) / peer_avg * 100) if peer_avg > 0 else 0
#                     color = "🔴" if case_val > peer_avg and abs(diff_pct) > 50 else ("🟢" if case_val < peer_avg and abs(diff_pct) > 50 else "⚪")
#                     st.markdown(f"{color} **{metric_name}**: {case_val:.2%} (Peer avg: {peer_avg:.2%}, {diff_pct:+.0f}% diff)")
#         with comp_col2:
#             fig_peer = go.Figure()
#             fig_peer.add_trace(go.Histogram(x=peer_group['fraud_risk_score'], name='Peer Group', nbinsx=30, marker_color='lightblue'))
#             fig_peer.add_vline(x=risk_score, line_dash="dash", line_color="red", annotation_text="This Case")
#             fig_peer.update_layout(title=f"Risk Scores in {bracket} Bracket", xaxis_title="Risk Score", yaxis_title="Count", showlegend=False)
#             st.plotly_chart(fig_peer, use_container_width=True)

#     st.markdown("---")

#     # Audit Recommendation
#     st.header("✅ Audit Recommendation")
#     if risk_score >= 0.9:
#         st.error("""
#         ### 🚨 PRIORITY AUDIT RECOMMENDED
#         This case exhibits **very high risk** indicators and should be prioritized for immediate audit.
#         **Recommended Actions:**
#         1. Full examination of all deductions and credits claimed
#         2. Verification of income sources and amounts
#         3. Review of supporting documentation for all claims
#         4. Potential for significant revenue recovery
#         """)
#     elif risk_score >= 0.7:
#         st.warning("""
#         ### ⚠️ AUDIT RECOMMENDED
#         This case shows **high risk** patterns and warrants audit examination.
#         **Recommended Actions:**
#         1. Review specific red flags identified above
#         2. Request documentation for unusual deductions
#         3. Verify income reporting accuracy
#         4. Consider for routine audit queue
#         """)
#     else:
#         st.info("""
#         ### ℹ️ MONITORING RECOMMENDED
#         While below the standard audit threshold, this case shows some concerning patterns.
#         **Recommended Actions:**
#         1. Monitor for patterns in future filings
#         2. Flag for review if similar patterns continue
#         """)

#     st.markdown("---")

#     # Export Case Report
#     st.header("💾 Export Case Report")
#     red_flags = get_red_flags(case, st.session_state.feature_names, shap_vals if shap_vals is not None else [])
#     report_data = {
#         'Case_ID': [idx],
#         'Risk_Score': [risk_score],
#         'Risk_Level': [case['risk_level']],
#         'DIF_Score': [case.get('dif_score', 0)],
#         'AGI': [case.get('c00100', 0)],
#         'Red_Flags': ['; '.join(red_flags) if red_flags else 'None specific'],
#         'Audit_Recommendation': ['Priority' if risk_score >= 0.9 else 'Standard' if risk_score >= 0.7 else 'Monitor']
#     }
#     # Add top 5 SHAP factors if available
#     if shap_vals is not None and st.session_state.feature_names is not None:
#         explanations = generate_shap_explanation(case, shap_vals, st.session_state.feature_names, top_n=5)
#         for i, exp in enumerate(explanations[:5], 1):
#             report_data[f'Factor_{i}'] = [exp['feature']]
#             report_data[f'Factor_{i}_Impact'] = [f"{exp['shap_value']:+.4f}"]

#     report_df = pd.DataFrame(report_data)
#     csv_report = report_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="📥 Download Case Report (CSV)",
#         data=csv_report,
#         file_name=f"case_{idx}_detailed_report.csv",
#         mime="text/csv"
#     )

# # ========================== MAIN ROUTER ==========================
# def main():
#     with st.sidebar:
#         st.image("https://via.placeholder.com/150x50/1976d2/ffffff?text=AURA+System", use_column_width=True)
#         st.markdown("---")
#         st.header("Navigation")

#         pages = {
#             'upload': '📤 Upload Data',
#             'summary': '📊 Summary',
#             'non_compliant_list': '📋 Non-Compliant',
#             'case_detail': '🔬 Case Detail'
#         }
#         for page_key, page_name in pages.items():
#             if st.button(page_name, key=f"nav_{page_key}", use_container_width=True,
#                          type="primary" if st.session_state.page == page_key else "secondary"):
#                 if page_key == 'upload' or st.session_state.results_df is not None:
#                     st.session_state.page = page_key
#                     st.rerun()

#         st.markdown("---")
#         if st.session_state.results_df is not None:
#             st.markdown("### Quick Stats")
#             results = st.session_state.results_df
#             st.metric("Total Returns", f"{len(results):,}")
#             st.metric("Non-Compliant", f"{results['non_compliant'].sum():,}")
#             st.metric("Avg Risk", f"{results['fraud_risk_score'].mean():.3f}")
#         st.markdown("---")
#         st.markdown("### About")
#         st.markdown("""
#         **AURA Tax Fraud Detection**
#         - Machine Learning (Random Forest)
#         - Anomaly Detection (Isolation Forest)
#         - Explainable AI (SHAP)
#         - DIF-style risk scoring
#         """)

#     # Routing
#     if st.session_state.page == 'upload':
#         page_upload()
#     elif st.session_state.page == 'summary':
#         page_summary()
#     elif st.session_state.page == 'non_compliant_list':
#         page_non_compliant_list()
#     elif st.session_state.page == 'case_detail':
#         page_case_detail()

# if __name__ == "__main__":
#     main()
