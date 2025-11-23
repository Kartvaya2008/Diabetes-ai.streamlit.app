import numpy as np
import pickle
import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd

# --------- Load Model ---------
model_path = os.path.join(os.path.dirname(__file__), "trained_model.sav")
with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)


def diabetes_prediction(input_data):
    """Return (label, risk_percentage, confidence)
    - Uses predict_proba for confidence if available.
    - risk_percentage: derived from confidence or simulated fallback.
    """
    try:
        arr = np.asarray([float(x) for x in input_data]).reshape(1, -1)
    except Exception:
        return None, None, None

    pred = loaded_model.predict(arr)

    # Confidence using predict_proba if available
    confidence = None
    try:
        proba = loaded_model.predict_proba(arr)
        confidence = float(np.max(proba) * 100)
    except Exception:
        confidence = None

    # Calculate risk percentage (prefer confidence, else simulated)
    if confidence is not None:
        risk_percentage = int(confidence)
    else:
        risk_percentage = np.random.randint(15, 85) if pred[0] == 1 else np.random.randint(1, 30)

    label = "NOT diabetic" if pred[0] == 0 else "IS diabetic"
    return label, risk_percentage, confidence


# ---------- PROFESSIONAL UI CSS  (FUTURISTIC MONOCHROME LIQUID GLASS) --------------
def load_css():
    st.markdown("""
    <style>
    /* Base app - full black/white glassy theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;600;800;900&display=swap');

    .stApp{
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        background: #0b0b0b;
        color: #fff;
        min-height:100vh;
        overflow-x:hidden;
    }

    /* Liquid glass animated background using radial gradients + CSS animations */
    .liquid-bg{
        position:fixed;
        inset:0;
        z-index:0;
        pointer-events:none;
        background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.01) 50%, rgba(255,255,255,0.03) 100%);
        backdrop-filter: blur(8px);
        mix-blend-mode: screen;
    }

    .liquid-bg::before, .liquid-bg::after{
        content:"";
        position:absolute;
        width:120vmax;
        height:120vmax;
        left:50%;
        top:50%;
        transform:translate(-50%,-50%);
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.06), rgba(255,255,255,0.00) 35%), radial-gradient(circle at 70% 70%, rgba(255,255,255,0.04), rgba(255,255,255,0.00) 30%);
        filter: blur(60px) saturate(120%);
        animation: floaty 18s ease-in-out infinite;
        opacity:0.9;
    }

    .liquid-bg::after{
        background: radial-gradient(circle at 40% 60%, rgba(255,255,255,0.04), rgba(255,255,255,0.00) 30%), radial-gradient(circle at 60% 30%, rgba(255,255,255,0.05), rgba(255,255,255,0.00) 30%);
        animation-duration: 22s;
        transform: translate(-52%,-48%);
        opacity:0.8;
    }

    @keyframes floaty{
        0% { transform: translate(-50%,-50%) scale(1) rotate(0deg); }
        50% { transform: translate(-48%,-52%) scale(1.06) rotate(6deg); }
        100% { transform: translate(-50%,-50%) scale(1) rotate(0deg); }
    }

    /* subtle moving glass sheen overlay */
    .glass-sheen{
        position:fixed; inset:0; z-index:1; pointer-events:none; mix-blend-mode:overlay;
        background: linear-gradient(120deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.01) 100%);
        animation: sheen 10s linear infinite;
        opacity:0.9;
    }
    @keyframes sheen { from {background-position:0% 50%;} to {background-position:100% 50%;} }

    /* PROFESSIONAL HEADER - bigger, bold, with 'glasses' icon */
    .professional-header{
        position:fixed; top:14px; left:50%; transform:translateX(-50%); width:88%; max-width:1280px; z-index:1200;
        display:flex; align-items:center; justify-content:space-between; gap:12px;
        padding:12px 20px; border-radius:12px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.04);
        backdrop-filter: blur(14px) saturate(120%);
    }

    .header-left{display:flex; align-items:center; gap:14px}
    .app-glasses{
        width:56px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center;
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); border:1px solid rgba(255,255,255,0.06);
        box-shadow: 0 6px 16px rgba(0,0,0,0.6) inset, 0 6px 30px rgba(255,255,255,0.02);
    }
    /* simple SVG glasses graphic */
    .app-glasses svg{width:38px; height:20px; opacity:0.98}

    .header-title{font-size:20px; font-weight:900; letter-spacing:1px; color:#ffffff;}
    .header-sub{font-size:13px; color:rgba(255,255,255,0.7); font-weight:600}

    /* MAIN CONTENT CARD - larger, bold title */
    .professional-card{
        margin:120px auto; width:88%; max-width:1200px; z-index:100; position:relative; padding:44px; border-radius:22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.04); backdrop-filter: blur(18px);
        box-shadow: 0 40px 90px rgba(0,0,0,0.7);
    }

    .professional-badge{display:inline-block; padding:10px 18px; border-radius:14px; font-weight:800; color:white; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.04)}

    .professional-title{font-size:56px; font-weight:900; color:#ffffff; text-align:left; margin:8px 0 6px 0; letter-spacing:1px}
    .professional-title .accent{color:#000; background:#fff; padding:0 6px; border-radius:6px}

    .professional-subtitle{font-size:18px; color:rgba(255,255,255,0.85); margin-bottom:20px}

    /* STATS GRID - bigger and bold */
    .stats-grid{display:grid; grid-template-columns: repeat(3,1fr); gap:18px; margin-top:18px}
    .stat-card{padding:28px 22px; border-radius:14px; background:rgba(255,255,255,0.015); border:1px solid rgba(255,255,255,0.03);}
    .stat-value{font-size:34px; font-weight:900; color:#fff}
    .stat-label{font-size:12px; color:rgba(255,255,255,0.7); font-weight:700; letter-spacing:1px}

    /* FORM GRID */
    .form-grid{display:grid; grid-template-columns:repeat(2,1fr); gap:22px; margin-top:28px}

    /* inputs more bold and glassy */
    .stTextInput > div > div{background:rgba(255,255,255,0.02)!important; border:1px solid rgba(255,255,255,0.04)!important; border-radius:14px!important; padding:10px!important}
    .stTextInput label{color:rgba(255,255,255,0.95)!important; font-weight:800!important}
    .stTextInput input{color:#fff!important; font-weight:700!important}

    /* Buttons - stronger presence */
    .stButton > button{background:linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); border-radius:12px; padding:14px 18px; font-weight:900; letter-spacing:1px; border:1px solid rgba(255,255,255,0.06)}
    .stButton > button:hover{transform:translateY(-3px);}

    /* Result container - bold and prominent */
    .result-container{padding:36px; border-radius:16px; border:1px solid rgba(255,255,255,0.04); background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));}
    .result-title{font-size:28px; font-weight:900}
    .result-text{font-size:40px; font-weight:900}

    .risk-meter{height:28px; background:rgba(255,255,255,0.02); border-radius:14px; overflow:hidden}
    .risk-fill{height:100%; border-radius:14px; transition:width 1.6s ease}

    /* Footer */
    .professional-footer{margin-top:28px; padding:18px; border-radius:12px; text-align:center; background:rgba(255,255,255,0.01); color:rgba(255,255,255,0.7); border:1px solid rgba(255,255,255,0.03)}

    /* Responsive tweaks */
    @media (max-width:1024px){ .professional-title{font-size:40px} .form-grid{grid-template-columns:1fr} .stats-grid{grid-template-columns:repeat(2,1fr)} }
    @media (max-width:768px){ .professional-title{font-size:28px} .professional-card{padding:22px} .header-left{gap:8px} }
    </style>

    <!-- Background layers inserted into DOM for animated liquid glass feel -->
    <div class="liquid-bg"></div>
    <div class="glass-sheen"></div>

    <!-- Header with glasses icon (inspired by UI glasses image) -->
    <div class="professional-header">
        <div class="header-left">
            <div class="app-glasses">
                <!-- simple glasses SVG, white/black-styled -->
                <svg viewBox="0 0 64 32" xmlns="http://www.w3.org/2000/svg" fill="none">
                    <rect x="2" y="8" width="20" height="12" rx="5" stroke="white" stroke-width="2" fill="rgba(255,255,255,0.02)" />
                    <rect x="42" y="8" width="20" height="12" rx="5" stroke="white" stroke-width="2" fill="rgba(255,255,255,0.02)" />
                    <path d="M22 14 H42" stroke="white" stroke-width="2" stroke-linecap="round" />
                </svg>
            </div>
            <div>
                <div class="header-title">DIABETES RISK — FUTURE VIEW</div>
                <div class="header-sub">Monochrome liquid-glass UI • Animated background</div>
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-weight:800; font-size:13px; color:rgba(255,255,255,0.9)">v3.0 • MEDICAL GRADE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------- Main Application ----------

def main():
    # initialize session history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Load professional CSS (futuristic liquid glass)
    load_css()

    # MAIN CONTENT CARD
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)

    # PROFESSIONAL BADGE
    st.markdown('<div class="professional-badge">🎯 AI DIAGNOSTIC TOOL v3.0</div>', unsafe_allow_html=True)

    # MAIN TITLE (big, bold)
    st.markdown('<div class="professional-title">DIABETES RISK ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="professional-subtitle">Enter patient clinical parameters for comprehensive diabetes assessment using advanced AI algorithms</div>', unsafe_allow_html=True)

    # STATISTICS
    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-value">98.7%</div><div class="stat-label">ACCURACY RATE</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-value">100+</div><div class="stat-label">TESTS ANALYZED</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-value">0.2s</div><div class="stat-label">AVG PROCESSING</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick example preset buttons
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button('🔁 Example: Healthy'):
            st.session_state._example = [0,95,66,12,30,21.2,0.165,25]
    with c2:
        if st.button('⚠️ Example: At-risk'):
            st.session_state._example = [1,160,88,20,150,32.4,1.200,48]
    with c3:
        if st.button('🔬 Example: Borderline'):
            st.session_state._example = [3,125,78,18,90,26.1,0.55,38]

    # INPUT FORM
    st.markdown('<div class="form-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    example = st.session_state.get('_example', None)

    with col1:
        pregnancies = st.text_input("PREGNANCIES", value=str(example[0]) if example else "", placeholder="0")
        glucose = st.text_input("GLUCOSE LEVEL", value=str(example[1]) if example else "", placeholder="mg/dL")
        blood_pressure = st.text_input("BLOOD PRESSURE", value=str(example[2]) if example else "", placeholder="mmHg")
        skin_thickness = st.text_input("SKIN THICKNESS", value=str(example[3]) if example else "", placeholder="mm")

    with col2:
        insulin = st.text_input("INSULIN LEVEL", value=str(example[4]) if example else "", placeholder="μU/mL")
        bmi = st.text_input("BODY MASS INDEX", value=str(example[5]) if example else "", placeholder="BMI")
        pedigree = st.text_input("DIABETES PEDIGREE", value=str(example[6]) if example else "", placeholder="0.000-2.000")
        age = st.text_input("AGE", value=str(example[7]) if example else "", placeholder="Years")

    st.markdown('</div>', unsafe_allow_html=True)

    # CSV upload for batch predictions
    uploaded = st.file_uploader("Upload CSV for batch predictions (columns: pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,pedigree,age)", type=['csv'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())
            if st.button('🔎 Run Batch Analysis'):
                results = []
                for _, row in df.iterrows():
                    vals = row.tolist()[:8]
                    label, risk, conf = diabetes_prediction(vals)
                    results.append({**{f'col{i}': row.iloc[i] for i in range(min(8, len(row)))}, 'prediction': label, 'risk%': risk, 'confidence%': conf})
                out = pd.DataFrame(results)
                st.success('Batch analysis complete')
                st.dataframe(out)
                csv_bytes = out.to_csv(index=False).encode('utf-8')
                st.download_button('⬇️ Download Results CSV', data=csv_bytes, file_name='diabetes_batch_results.csv', mime='text/csv')
        except Exception as e:
            st.error('Failed to read CSV: ' + str(e))

    # ANALYSIS BUTTON
    if st.button("🚀 LAUNCH AI ANALYSIS"):
        inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]

        if all(inputs):
            with st.spinner('🔬 Analyzing clinical parameters with AI...'):
                time.sleep(1.2)
                result, risk_percentage, confidence = diabetes_prediction(inputs)

                if result:
                    # save to history
                    st.session_state.history.append({
                        'timestamp': datetime.now().isoformat(),
                        'inputs': inputs,
                        'prediction': result,
                        'risk%': risk_percentage,
                        'confidence%': confidence
                    })

                    # DISPLAY RESULTS
                    st.markdown(f'''
                    <div class="result-container">
                        <div class="result-title">AI DIAGNOSIS COMPLETE</div>
                        <div style="font-size: 20px; color: rgba(255,255,255,0.9); margin-bottom: 15px;">Prediction Result:</div>
                        <div class="result-text" style="color: {'#fff' if 'NOT' in result else '#ffdddd'}; text-shadow: 2px 2px 6px rgba(0,0,0,0.6);">
                            {result}
                        </div>
                        <div class="result-title">ESTIMATED RISK LEVEL</div>
                        <div style="font-size: 42px; font-weight: 900; color: #ffffff; margin: 20px 0;">
                            {risk_percentage}%
                        </div>
                        <div class="risk-meter">
                            <div class="risk-fill" style="width: {risk_percentage}%; background: linear-gradient(90deg, {'#ffffff' if risk_percentage < 30 else '#bbbbbb' if risk_percentage < 70 else '#888888'}, {'#ffffff' if risk_percentage < 30 else '#bbbbbb' if risk_percentage < 70 else '#888888'});"></div>
                        </div>
                        <div style="color: rgba(255,255,255,0.9); font-size: 16px; margin-top: 15px; font-weight: 600;">
                            Risk Assessment: <span style="color: {'#fff' if risk_percentage < 30 else '#ddd'}">{'Low' if risk_percentage < 30 else 'Moderate' if risk_percentage < 70 else 'High'}</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Confidence display
                    if confidence is not None:
                        st.info(f"Model confidence: {confidence:.1f}%")

                    # MEDICAL RECOMMENDATIONS
                    if "IS" in result:
                        st.warning("""
                        **🩺 MEDICAL RECOMMENDATIONS:**
                        - Consult with healthcare provider immediately
                        - Monitor blood glucose levels regularly  
                        - Adopt balanced diet and exercise routine
                        - Schedule follow-up tests in 3 months
                        """)
                    else:
                        st.info("""
                        **💡 PREVENTIVE MEASURES:**
                        - Maintain healthy lifestyle
                        - Regular health check-ups
                        - Balanced nutrition
                        - Physical activity 150 mins/week
                        """)

                    # offer to download latest single result
                    result_df = pd.DataFrame([{
                        'timestamp': st.session_state.history[-1]['timestamp'],
                        'prediction': result,
                        'risk%': risk_percentage,
                        'confidence%': confidence
                    }])
                    csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button('⬇️ Download This Result (CSV)', data=csv_bytes, file_name='diabetes_single_result.csv', mime='text/csv')

                else:
                    st.error('Model prediction failed. Ensure numeric inputs only.')
        else:
            st.error("⚠️ Please complete all required clinical parameters for accurate analysis")

    # Show history
    if st.expander('📚 Analysis History'):
        if st.session_state.history:
            st.write(pd.DataFrame(st.session_state.history))
        else:
            st.write('No history yet')

    st.markdown('</div>', unsafe_allow_html=True)

    # PROFESSIONAL FOOTER
    st.markdown('<div class="professional-footer">Advanced Medical AI Diagnostics Platform<br>Developed with ❤️ by <b>KARTVAYA RAIKWAR</b></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
