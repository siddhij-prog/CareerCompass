import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from xgboost import XGBClassifier

# ── Page Config ──
st.set_page_config(
    page_title="CareerCompass",
    page_icon="🧭",
    layout="wide"
)

# ── Loading Screen ──
if 'loaded' not in st.session_state:
    st.session_state.loaded = False

if not st.session_state.loaded:
    st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important; 
    }
    
    @keyframes launch {
        0%   { transform: translateY(0px) rotate(-45deg); opacity: 1; }
        20%  { transform: translateY(-30px) rotate(-45deg); opacity: 1; }
        100% { transform: translateY(-600px) rotate(-45deg); opacity: 0; }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); opacity: 0.5; }
        50%       { transform: translateY(-10px); opacity: 1; }
    }

    .loading-wrapper {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 85vh;
        text-align: center;
    }

    .rocket {
        font-size: 5rem;
        animation: launch 1.5s ease-in 0.8s forwards;
        display: inline-block;
        filter: drop-shadow(0 0 20px rgba(240,147,251,0.8));
    }

    .loading-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f093fb, #f5576c, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0 0.5rem 0;
        animation: fadeInUp 0.8s ease forwards;
    }

    .loading-sub {
        color: #a0aec0;
        font-size: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease 0.3s forwards;
        opacity: 0;
    }

    .dot1 {
        width: 12px; height: 12px;
        border-radius: 50%;
        background: #f093fb;
        display: inline-block;
        margin: 0 5px;
        animation: bounce 0.6s ease infinite;
    }
    .dot2 {
        width: 12px; height: 12px;
        border-radius: 50%;
        background: #f5576c;
        display: inline-block;
        margin: 0 5px;
        animation: bounce 0.6s ease 0.2s infinite;
    }
    .dot3 {
        width: 12px; height: 12px;
        border-radius: 50%;
        background: #4facfe;
        display: inline-block;
        margin: 0 5px;
        animation: bounce 0.6s ease 0.4s infinite;
    }
    </style>

    <div class="loading-wrapper">
        <div class="rocket">🚀</div>
        <div class="loading-title"> CareerCompass</div>
        <div class="loading-sub">Finding your path...</div>
        <div>
            <div class="dot1"></div>
            <div class="dot2"></div>
            <div class="dot3"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(3)
    st.session_state.loaded = True
    st.rerun()

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f093fb, #f5576c, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f093fb;
        border-left: 4px solid #f5576c;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }
    .input-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, rgba(240,147,251,0.2), rgba(245,87,108,0.2));
        border: 2px solid rgba(240,147,251,0.5);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin: 1rem 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(240,147,251,0.3); }
        to   { box-shadow: 0 0 40px rgba(240,147,251,0.7), 0 0 80px rgba(245,87,108,0.3); }
    }
    .career-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .course-card {
        background: rgba(79,172,254,0.1);
        border: 1px solid rgba(79,172,254,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
        font-size: 1rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #f093fb, #f5576c) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 2rem !important;
        border-radius: 50px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(240,147,251,0.4) !important;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #4facfe;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.3rem;
    }
    .career-bar {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        display: flex;
        justify-content: space-between;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .profile-card {
        background: rgba(79,172,254,0.08);
        border: 1px solid rgba(79,172,254,0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .profile-item {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .profile-value {
        color: #4facfe;
        font-weight: 600;
    }
    .about-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(240,147,251,0.15);
        border: 1px solid rgba(240,147,251,0.4);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        margin: 0.3rem;
        color: #f093fb;
        font-size: 0.9rem;
        font-weight: 600;
    }
    hr { border-color: rgba(255,255,255,0.1) !important; }
            /* Nav buttons — smaller and subtle */
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    background: rgba(255,255,255,0.05) !important;
    color: #a0aec0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.5rem !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: none !important;
}

div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
    background: rgba(240,147,251,0.1) !important;
    color: #f093fb !important;
    border: 1px solid rgba(240,147,251,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ──

import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')
    
    with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    return model, encoders, feature_columns

model, encoders, feature_columns = load_model()

# ── Data ──
career_icons = {
    'Software Engineer':                         '💻',
    'Web Developer':                             '🌐',
    'Mobile Applications Developer':             '📱',
    'Network Security Engineer':                 '🔐',
    'Systems Security Administrator':            '🛡️',
    'Software Quality Assurance (QA) / Testing': '🧪',
    'Database Developer':                        '🗄️',
    'Applications Developer':                    '⚙️',
    'CRM Technical Developer':                   '📊',
    'Technical Support':                         '🎧',
    'Software Developer':                        '🖥️',
}

course_recommendations = {
    'Software Engineer': [
        '📘 Data Structures & Algorithms — LeetCode / GeeksforGeeks',
        '📘 System Design — Gaurav Sen (YouTube)',
        '📘 Full Stack Development — The Odin Project (free)',
    ],
    'Web Developer': [
        '🌐 HTML/CSS/JS — freeCodeCamp.org',
        '🌐 React.js — official docs + Scrimba',
        '🌐 Node.js & Express — The Net Ninja (YouTube)',
    ],
    'Mobile Applications Developer': [
        '📱 Flutter Development — Flutter.dev official docs',
        '📱 React Native — Expo documentation',
        '📱 Android Development — Google Codelabs (free)',
    ],
    'Network Security Engineer': [
        '🔐 CompTIA Network+ — Professor Messer (free)',
        '🔐 Ethical Hacking — TCM Security (YouTube)',
        '🔐 Cisco CCNA — Jeremy IT Lab (YouTube)',
    ],
    'Systems Security Administrator': [
        '🛡️ CompTIA Security+ — Professor Messer (free)',
        '🛡️ Linux Administration — Linux Journey (free)',
        '🛡️ Cybersecurity — Google Certificate (Coursera)',
    ],
    'Software Quality Assurance (QA) / Testing': [
        '🧪 Software Testing — ISTQB Foundation',
        '🧪 Selenium Automation — Automation Step by Step (YouTube)',
        '🧪 API Testing — Postman Learning Center (free)',
    ],
    'Database Developer': [
        '🗄️ SQL — SQLZoo.net (free, interactive)',
        '🗄️ PostgreSQL — Official docs + freeCodeCamp',
        '🗄️ MongoDB — MongoDB University (free)',
    ],
    'Applications Developer': [
        '⚙️ Python — CS50P Harvard (free)',
        '⚙️ Machine Learning — Andrew Ng Coursera (audit free)',
        '⚙️ Data Engineering — DataTalks.Club (free)',
    ],
    'CRM Technical Developer': [
        '📊 Salesforce — Salesforce Trailhead (free)',
        '📊 SAP Basics — SAP Learning Hub (free tier)',
        '📊 Business Analysis — IBM BA Certificate (Coursera)',
    ],
    'Technical Support': [
        '🎧 Google IT Support — Coursera (audit free)',
        '🎧 CompTIA A+ — Professor Messer (free)',
        '🎧 ITIL Foundation — Axelos free resources',
    ],
    'Software Developer': [
        '🖥️ Clean Code — Robert C. Martin (book)',
        '🖥️ Git & GitHub — GitHub Skills (free)',
        '🖥️ Python/Java — Hyperskill (free tier)',
    ],
}

# NAVIGATION

st.markdown('<div class="main-title"> CareerCompass</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:#a0aec0; font-size:0.95rem; letter-spacing:2px; text-transform:uppercase; margin-top:-0.8rem; margin-bottom:0.5rem">Career Path Prediction & Guidance System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover your ideal tech career based on your skills, interests & personality</div>', unsafe_allow_html=True)

# ── Session state for navigation ──
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Predict My Career"

# ── Bottom Nav CSS ──
st.markdown("""
<style>
    .nav-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(15, 12, 41, 0.95);
        border-top: 1px solid rgba(240,147,251,0.3);
        display: flex;
        justify-content: center;
        gap: 0;
        z-index: 999;
        backdrop-filter: blur(20px);
        padding: 0;
    }
    .nav-btn {
        flex: 1;
        max-width: 200px;
        padding: 1rem 0;
        text-align: center;
        cursor: pointer;
        color: #a0aec0;
        font-size: 0.85rem;
        font-weight: 600;
        border: none;
        background: transparent;
        transition: all 0.3s ease;
        border-top: 3px solid transparent;
        letter-spacing: 0.5px;
    }
    .nav-btn:hover {
        color: #f093fb;
        background: rgba(240,147,251,0.05);
    }
    .nav-btn.active {
        color: #f093fb;
        border-top: 3px solid #f093fb;
        background: rgba(240,147,251,0.08);
    }
    .nav-icon {
        font-size: 1.3rem;
        display: block;
        margin-bottom: 0.2rem;
    }
    /* Add padding so content doesn't hide behind nav */
    .main .block-container {
        padding-bottom: 100px !important;
    }
</style>
""", unsafe_allow_html=True)

page = st.session_state.page

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Predict", key="nav_home", use_container_width=True):
        st.session_state.page = "🏠 Predict My Career"
        st.rerun()
with col2:
    if st.button("📊 Insights", key="nav_insights", use_container_width=True):
        st.session_state.page = "📊 Model Insights"
        st.rerun()
with col3:
    if st.button("ℹ️ About", key="nav_about", use_container_width=True):
        st.session_state.page = "ℹ️ About"
        st.rerun()

page = st.session_state.page
st.divider()

# PAGE 1 — PREDICT

if page == "🏠 Predict My Career":

    # Stats bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">6901</div><div class="metric-label">Students Analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">11</div><div class="metric-label">Career Paths</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">99.28%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">XGBoost</div><div class="metric-label">Best ML Model</div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">📋 Tell Us About Yourself</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**⚡ Skills & Ratings**")
        logical_quotient = st.slider("🧠 Logical Quotient", 1, 9, 5)
        hackathons       = st.slider("🏆 Hackathons Attended", 0, 6, 2)
        coding_skills    = st.slider("💻 Coding Skills", 1, 9, 5)
        public_speaking  = st.slider("🎤 Public Speaking", 0, 9, 3)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**🎯 Work Style**")
        self_learning = st.selectbox("📚 Self Learning?", ['yes', 'no'])
        extra_courses = st.selectbox("➕ Extra Courses?", ['yes', 'no'])
        hard_smart    = st.selectbox("⚡ Work Style", ['hard worker', 'smart worker'])
        team_work     = st.selectbox("👥 Team Player?", ['yes', 'no'])
        introvert     = st.selectbox("🧘 Introvert?", ['yes', 'no'])
        senior_inputs = st.selectbox("👴 Takes Senior Inputs?", ['yes', 'no'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**📜 Education & Skills**")
        certifications = st.selectbox("🏅 Certifications", [
            'app development', 'distro making', 'full stack', 'hadoop',
            'information security', 'machine learning', 'python',
            'r programming', 'shell programming'
        ])
        workshops = st.selectbox("🔧 Workshops", [
            'cloud computing', 'data science', 'database security',
            'game development', 'hacking', 'system designing',
            'testing', 'web technologies'
        ])
        reading_writing = st.selectbox("✍️ Reading & Writing", ['poor', 'medium', 'excellent'])
        memory_score    = st.selectbox("🧩 Memory Score", ['poor', 'medium', 'excellent'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**📖 Interests**")
        interested_subjects = st.selectbox("🔬 Interested Subjects", [
            'Computer Architecture', 'IOT', 'Management',
            'Software Engineering', 'cloud computing', 'data engineering',
            'hacking', 'networks', 'parallel computing', 'programming'
        ])
        book_type = st.selectbox("📚 Favourite Books", [
            'Action and Adventure', 'Anthology', 'Art',
            'Autobiographies', 'Biographies', 'Children',
            'Comics', 'Cookbooks', 'Diaries', 'Dictionary',
            'Drama', 'Encyclopedias', 'Fantasy', 'Guide',
            'Health', 'History', 'Horror', 'Humor and Comedy',
            'Journal', 'Math', 'Mystery', 'Poetry',
            'Prayer books', 'Religion-Spirituality', 'Romance',
            'Science', 'Science fiction', 'Self help', 'Series',
            'Short Stories', 'Suspense', 'Travel', 'Trilogy'
        ])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**🚀 Career Preferences**")
        career_area  = st.selectbox("🎯 Career Area", [
            'Business process analyst', 'cloud computing', 'developer',
            'security', 'system developer', 'testing'
        ])
        company_type = st.selectbox("🏢 Company Type", [
            'BPA', 'Cloud Services', 'Finance', 'Product based',
            'SAaS services', 'Sales and Marketing', 'Service Based',
            'Testing and Maintainance Services', 'Web Services',
            'product development'
        ])
        mgmt_or_tech = st.selectbox("⚙️ Management or Technical?", ['Management', 'Technical'])
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Profile Summary ──
    st.markdown('<div class="section-header">👤 Your Profile Summary</div>', unsafe_allow_html=True)
    st.markdown("*Review your inputs before predicting:*")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f'''<div class="profile-card">
            <div style="color:#f093fb; font-weight:700; margin-bottom:0.8rem">⚡ Skills</div>
            <div class="profile-item"><span>Logical Quotient</span><span class="profile-value">{logical_quotient}/9</span></div>
            <div class="profile-item"><span>Coding Skills</span><span class="profile-value">{coding_skills}/9</span></div>
            <div class="profile-item"><span>Public Speaking</span><span class="profile-value">{public_speaking}/9</span></div>
            <div class="profile-item"><span>Hackathons</span><span class="profile-value">{hackathons}</span></div>
            <div class="profile-item"><span>Reading & Writing</span><span class="profile-value">{reading_writing}</span></div>
            <div class="profile-item"><span>Memory Score</span><span class="profile-value">{memory_score}</span></div>
        </div>''', unsafe_allow_html=True)

    with p2:
        st.markdown(f'''<div class="profile-card">
            <div style="color:#f093fb; font-weight:700; margin-bottom:0.8rem">📜 Background</div>
            <div class="profile-item"><span>Certification</span><span class="profile-value">{certifications}</span></div>
            <div class="profile-item"><span>Workshop</span><span class="profile-value">{workshops}</span></div>
            <div class="profile-item"><span>Subject Interest</span><span class="profile-value">{interested_subjects}</span></div>
            <div class="profile-item"><span>Self Learning</span><span class="profile-value">{self_learning}</span></div>
            <div class="profile-item"><span>Extra Courses</span><span class="profile-value">{extra_courses}</span></div>
            <div class="profile-item"><span>Favourite Books</span><span class="profile-value">{book_type}</span></div>
        </div>''', unsafe_allow_html=True)

    with p3:
        st.markdown(f'''<div class="profile-card">
            <div style="color:#f093fb; font-weight:700; margin-bottom:0.8rem">🚀 Preferences</div>
            <div class="profile-item"><span>Career Area</span><span class="profile-value">{career_area}</span></div>
            <div class="profile-item"><span>Company Type</span><span class="profile-value">{company_type}</span></div>
            <div class="profile-item"><span>Work Style</span><span class="profile-value">{mgmt_or_tech}</span></div>
            <div class="profile-item"><span>Hard/Smart</span><span class="profile-value">{hard_smart}</span></div>
            <div class="profile-item"><span>Team Player</span><span class="profile-value">{team_work}</span></div>
            <div class="profile-item"><span>Introvert</span><span class="profile-value">{introvert}</span></div>
        </div>''', unsafe_allow_html=True)

    st.divider()

    # ── Predict Button ──
    predict_col = st.columns([1, 2, 1])[1]
    with predict_col:
        predict_clicked = st.button("🔍 Predict My Career Path", use_container_width=True)

    # ── Results ──
    if predict_clicked:

        # Loading animation
        with st.spinner("🤖 Analyzing your profile..."):
            time.sleep(1.5)

        input_dict = {
            'Logical quotient rating':             logical_quotient,
            'hackathons':                          hackathons,
            'coding skills rating':                coding_skills,
            'public speaking points':              public_speaking,
            'self-learning capability?':           self_learning,
            'Extra-courses did':                   extra_courses,
            'certifications':                      certifications,
            'workshops':                           workshops,
            'reading and writing skills':          reading_writing,
            'memory capability score':             memory_score,
            'Interested subjects':                 interested_subjects,
            'interested career area ':             career_area,
            'Type of company want to settle in?':  company_type,
            'Taken inputs from seniors or elders': senior_inputs,
            'Interested Type of Books':            book_type,
            'Management or Technical':             mgmt_or_tech,
            'hard/smart worker':                   hard_smart,
            'worked in teams ever?':               team_work,
            'Introvert':                           introvert,
        }

        encoded      = {}
        ordinal_map  = {'poor': 0, 'medium': 1, 'excellent': 2}
        ordinal_cols = ['reading and writing skills', 'memory capability score']

        for col, val in input_dict.items():
            if col in ordinal_cols:
                encoded[col] = ordinal_map[val]
            elif col in encoders:
                le = encoders[col]
                encoded[col] = int(le.transform([val])[0])
            else:
                encoded[col] = val

        input_df           = pd.DataFrame([encoded])[feature_columns]
        prediction_encoded = model.predict(input_df)[0]
        predicted_career   = encoders['Suggested Job Role'].inverse_transform([prediction_encoded])[0]
        probabilities      = model.predict_proba(input_df)[0]
        top_indices        = probabilities.argsort()[::-1][:3]
        top_careers        = encoders['Suggested Job Role'].inverse_transform(top_indices)
        top_probs          = probabilities[top_indices]
        icon               = career_icons.get(predicted_career, '🎯')

        # Animated result card
        st.markdown(f'''
        <div class="result-card">
            <div style="font-size:5rem">{icon}</div>
            <div style="color:#a0aec0; font-size:1rem; margin:0.5rem 0; letter-spacing:2px; text-transform:uppercase">
                Your Recommended Career
            </div>
            <div class="career-title">{predicted_career}</div>
            <div style="margin-top:1rem">
                <span style="background:rgba(79,172,254,0.2); border:1px solid rgba(79,172,254,0.4);
                border-radius:20px; padding:0.4rem 1.2rem; color:#4facfe; font-weight:700; font-size:1.1rem">
                    ✨ {top_probs[0]*100:.1f}% Confidence
                </span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown('<div class="section-header">📊 Top Career Matches</div>', unsafe_allow_html=True)
            for i, (career, prob) in enumerate(zip(top_careers, top_probs)):
                c_icon = career_icons.get(career, '🎯')
                medal  = ['🥇', '🥈', '🥉'][i]
                st.markdown(f'''
                <div class="career-bar">
                    <span>{medal} {c_icon} {career}</span>
                    <span style="color:#4facfe; font-weight:700">{prob*100:.1f}%</span>
                </div>
                ''', unsafe_allow_html=True)
                st.progress(float(prob))

        with right:
            st.markdown('<div class="section-header">📚 Recommended Courses</div>', unsafe_allow_html=True)
            courses = course_recommendations.get(predicted_career, ['🎓 Explore Coursera and edX'])
            for course in courses:
                st.markdown(f'<div class="course-card">{course}</div>', unsafe_allow_html=True)

        st.divider()

        # Action plan
        st.markdown('<div class="section-header">🗺️ Your 6-Month Action Plan</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown('''<div class="metric-card">
                <div style="font-size:2rem">📅</div>
                <div class="metric-value" style="font-size:1.2rem">Month 1-2</div>
                <div class="metric-label">Complete recommended courses above</div>
            </div>''', unsafe_allow_html=True)
        with a2:
            st.markdown('''<div class="metric-card">
                <div style="font-size:2rem">🛠️</div>
                <div class="metric-value" style="font-size:1.2rem">Month 3-4</div>
                <div class="metric-label">Build 2-3 portfolio projects</div>
            </div>''', unsafe_allow_html=True)
        with a3:
            st.markdown('''<div class="metric-card">
                <div style="font-size:2rem">💼</div>
                <div class="metric-value" style="font-size:1.2rem">Month 5-6</div>
                <div class="metric-label">Apply for internships and entry roles</div>
            </div>''', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div style="text-align:center; color:#a0aec0">Built with ❤️ using Streamlit + XGBoost</div>', unsafe_allow_html=True)


# PAGE 2 — MODEL INSIGHTS

elif page == "📊 Model Insights":

    st.markdown('<div class="section-header">📊 Model Performance Overview</div>', unsafe_allow_html=True)

    # Accuracy comparison
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-card"><div class="metric-value">94.06%</div><div class="metric-label">Decision Tree</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><div class="metric-value">75.31%</div><div class="metric-label">Random Forest</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color:#f093fb">99.28%</div><div class="metric-label">🏆 XGBoost</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><div class="metric-value">73.43%</div><div class="metric-label">Neural Network</div></div>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🏆 Model Accuracy Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        models  = ['Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']
        accs    = [94.06, 75.31, 99.28, 73.43]
        colors  = ['#4facfe', '#4facfe', '#f093fb', '#4facfe']

        bars = ax.bar(models, accs, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc}%', ha='center', color='white', fontsize=10, fontweight='bold')

        ax.set_ylim(0, 110)
        ax.set_ylabel('Accuracy (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=15, color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-header">🔍 Feature Importance (XGBoost)</div>', unsafe_allow_html=True)

        feature_names = [
            'interested career area', 'workshops', 'Management or Technical',
            'Interested subjects', 'certifications', 'Type of company',
            'coding skills rating', 'Logical quotient', 'Interested Books',
            'hackathons', 'hard/smart worker', 'public speaking',
            'reading & writing', 'memory score', 'Extra courses',
            'Senior inputs', 'Introvert', 'self learning', 'worked in teams'
        ]
        importances = [
            0.2825, 0.1677, 0.1554, 0.1065, 0.0910, 0.0500,
            0.0450, 0.0380, 0.0290, 0.0250, 0.0200, 0.0180,
            0.0150, 0.0140, 0.0130, 0.0120, 0.0100, 0.0090, 0.0085
        ]

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#1a1a2e')

        colors2 = ['#f093fb' if i < 3 else '#4facfe' for i in range(len(feature_names))]
        ax2.barh(feature_names[::-1], importances[::-1], color=colors2[::-1])
        ax2.tick_params(colors='white', labelsize=8)
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlabel('Importance Score', color='white')
        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()

    # Key findings
    st.markdown('<div class="section-header">💡 Key Findings</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('''<div class="about-card">
            <div style="font-size:2rem">🎯</div>
            <div style="color:#f093fb; font-weight:700; margin:0.5rem 0">Dataset Issue Found</div>
            <div style="color:#a0aec0; font-size:0.9rem">Original dataset had randomly assigned job roles —
            no ML model could learn from it. We rebuilt the target using logical domain rules.</div>
        </div>''', unsafe_allow_html=True)
    with f2:
        st.markdown('''<div class="about-card">
            <div style="font-size:2rem">🏆</div>
            <div style="color:#f093fb; font-weight:700; margin:0.5rem 0">XGBoost Wins</div>
            <div style="color:#a0aec0; font-size:0.9rem">XGBoost achieved 99.28% accuracy —
            outperforming even the Neural Network because it is specifically optimized
            for tabular datasets like ours.</div>
        </div>''', unsafe_allow_html=True)
    with f3:
        st.markdown('''<div class="about-card">
            <div style="font-size:2rem">🔍</div>
            <div style="color:#f093fb; font-weight:700; margin:0.5rem 0">Top Features</div>
            <div style="color:#a0aec0; font-size:0.9rem">Career area interest, workshops attended,
            and management preference are the 3 most decisive factors in career prediction —
            accounting for over 53% of model decisions.</div>
        </div>''', unsafe_allow_html=True)


# PAGE 3 — ABOUT

elif page == "ℹ️ About":

    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown('''<div class="about-card">
        <div style="color:#f093fb; font-weight:700; font-size:1.2rem; margin-bottom:0.8rem">🎯 What is this?</div>
        <div style="color:#e2e8f0; line-height:1.8">
        The Career Path Prediction & Guidance System is an AI-powered web application
        that helps students discover their ideal tech career based on their skills,
        interests, certifications, and personality traits. It uses Machine Learning
        to analyze 19 different student attributes and recommend the most suitable
        career path from 11 possible tech roles.
        </div>
    </div>''', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''<div class="about-card">
            <div style="color:#f093fb; font-weight:700; font-size:1.1rem; margin-bottom:0.8rem">📊 Dataset</div>
            <div class="profile-item"><span style="color:#a0aec0">Total Records</span><span class="profile-value">6,901 students</span></div>
            <div class="profile-item"><span style="color:#a0aec0">Input Features</span><span class="profile-value">19 attributes</span></div>
            <div class="profile-item"><span style="color:#a0aec0">Career Classes</span><span class="profile-value">11 tech roles</span></div>
            <div class="profile-item"><span style="color:#a0aec0">Missing Values</span><span class="profile-value">None ✅</span></div>
            <div class="profile-item"><span style="color:#a0aec0">Train Split</span><span class="profile-value">80% (5,520 rows)</span></div>
            <div class="profile-item"><span style="color:#a0aec0">Test Split</span><span class="profile-value">20% (1,381 rows)</span></div>
        </div>''', unsafe_allow_html=True)

        st.markdown('''<div class="about-card">
            <div style="color:#f093fb; font-weight:700; font-size:1.1rem; margin-bottom:0.8rem">🏆 Model Results</div>
            <div class="profile-item"><span style="color:#a0aec0">Decision Tree</span><span class="profile-value">94.06%</span></div>
            <div class="profile-item"><span style="color:#a0aec0">Random Forest</span><span class="profile-value">75.31%</span></div>
            <div class="profile-item"><span style="color:#a0aec0">XGBoost</span><span class="profile-value" style="color:#f093fb">99.28% 🏆</span></div>
            <div class="profile-item"><span style="color:#a0aec0">Neural Network</span><span class="profile-value">73.43%</span></div>
        </div>''', unsafe_allow_html=True)

    with col2:
        st.markdown('''<div class="about-card">
            <div style="color:#f093fb; font-weight:700; font-size:1.1rem; margin-bottom:1rem">🛠️ Tech Stack</div>
            <div style="margin-bottom:0.5rem; color:#a0aec0; font-size:0.85rem">MACHINE LEARNING</div>
            <span class="tech-badge">Python</span>
            <span class="tech-badge">XGBoost</span>
            <span class="tech-badge">Scikit-learn</span>
            <span class="tech-badge">TensorFlow</span>
            <div style="margin:0.8rem 0 0.5rem; color:#a0aec0; font-size:0.85rem">DATA PROCESSING</div>
            <span class="tech-badge">Pandas</span>
            <span class="tech-badge">NumPy</span>
            <span class="tech-badge">Matplotlib</span>
            <span class="tech-badge">Seaborn</span>
            <div style="margin:0.8rem 0 0.5rem; color:#a0aec0; font-size:0.85rem">WEB APP</div>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Pickle</span>
            <div style="margin:0.8rem 0 0.5rem; color:#a0aec0; font-size:0.85rem">DEVELOPMENT</div>
            <span class="tech-badge">VS Code</span>
            <span class="tech-badge">Jupyter Notebook</span>
            <span class="tech-badge">Git</span>
        </div>''', unsafe_allow_html=True)

        st.markdown('''<div class="about-card">
            <div style="color:#f093fb; font-weight:700; font-size:1.1rem; margin-bottom:0.8rem">📁 Project Structure</div>
            <div style="color:#4facfe; font-family:monospace; font-size:0.85rem; line-height:2">
            📁 career_path_prediction/<br>
            ├── 📁 data/<br>
            ├── 📁 notebooks/<br>
            │&nbsp;&nbsp;&nbsp;├── 01_EDA.ipynb<br>
            │&nbsp;&nbsp;&nbsp;├── 02_Preprocessing.ipynb<br>
            │&nbsp;&nbsp;&nbsp;├── 03_ML_Models.ipynb<br>
            │&nbsp;&nbsp;&nbsp;└── 04_DL_Model.ipynb<br>
            ├── 📁 models/<br>
            ├── 📁 app/<br>
            │&nbsp;&nbsp;&nbsp;└── app.py<br>
            └── requirements.txt
            </div>
        </div>''', unsafe_allow_html=True)

    st.divider()
    st.markdown('''<div style="text-align:center; color:#a0aec0; padding:1rem">
        Built with ❤️ as part of Global Professional Internship (GPI) — Cloud Counselage<br>
        <span style="color:#f093fb">Machine Learning Domain | Career Path Prediction System</span>
    </div>''', unsafe_allow_html=True)