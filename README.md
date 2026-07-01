#  CareerCompass
### AI-Powered Career Path Prediction & Guidance System

> Predict your ideal tech career based on your skills, interests, certifications and personality — powered by Machine Learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://careercompass-ml.streamlit.app)



##  Live Demo
**[careercompass-ml.streamlit.app](https://careercompass-ml.streamlit.app)**



##  About

CareerCompass is an end-to-end Machine Learning web application that analyzes 19 student attributes — including coding skills, certifications, workshops, and interests — to recommend the most suitable tech career from 11 possible roles.

Built as part of the **Global Professional Internship (GPI)** at **Cloud Counselage** in the Machine Learning domain.



##  Models & Results

   Model          Accuracy 
 Decision Tree    94.06% 
 Random Forest    75.31% 
 **XGBoost**     **99.28%** 
 Neural Network   73.43%
 (TensorFlow)   

###  Key Finding
The original dataset had randomly assigned career labels with no learnable pattern — all models scored ~8% accuracy (random chance for 12 classes). The target variable was rebuilt using domain-specific logical rules mapping career interests, certifications, and workshops to appropriate roles, improving accuracy to **99.28%**.



##  App Features

-  Career prediction from 19 student inputs
-  Confidence score and top 3 career matches
-  Student profile summary before prediction
-  Personalized course recommendations per career
-  6-month career action plan
-  Model Insights — accuracy comparison + feature importance chart
-  About page with full project documentation



##  Tech Stack

   Category                 Tools 
| Machine Learning | Python, Scikit-learn, XGBoost |
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Deployment | GitHub, Streamlit Cloud |
| Development | VS Code, Jupyter Notebook |



##  Project Structure


CareerCompass/
├── app/
│   └── app.py                  → Streamlit web application
├── data/
│   └── PS2_Dataset.csv         → Dataset (6,901 student records)
├── models/
│   ├── best_model.pkl          → Trained XGBoost model
│   ├── encoders.pkl            → Feature encoders
│   ├── feature_columns.pkl     → Column order for predictions
│   ├── dl_model.keras          → Neural Network model
│   └── scaler.pkl              → StandardScaler for DL input
├── notebooks/
│   ├── 01_EDA.ipynb            → Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  → Data encoding & preprocessing
│   ├── 03_ML_Models.ipynb      → ML model training & evaluation
│   └── 04_DL_Model.ipynb       → Deep Learning model
└── requirements.txt            → Project dependencies


##  Dataset

| Property | Value |
|---|---|
| Total Records | 6,901 students |
| Input Features | 19 attributes |
| Career Classes | 11 tech roles |
| Missing Values | None |
| Train Split | 80% — 5,520 rows |
| Test Split | 20% — 1,381 rows |

**Career Roles Predicted:**
Applications Developer, CRM Technical Developer, Database Developer, Mobile Applications Developer, Network Security Engineer, Software Developer, Software Engineer, Software Quality Assurance (QA) / Testing, Systems Security Administrator, Technical Support, Web Developer



##  Run Locally

bash
# Clone the repo
git clone https://github.com/siddhij-prog/CareerCompass.git
cd CareerCompass

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
cd app
streamlit run app.py


>  **Note:** Python 3.10 or 3.11 recommended.
> TensorFlow is not supported on Python 3.13+.
> Pre-trained models are already saved in `models/` — no retraining needed.



##  Notebooks Guide

| Notebook | Purpose | Run Required? |
|---|---|---|
| 01_EDA.ipynb | Data exploration & visualizations | Optional |
| 02_Preprocessing.ipynb | Feature encoding | Optional |
| 03_ML_Models.ipynb | Train XGBoost & compare models | Only to retrain |
| 04_DL_Model.ipynb | Train Neural Network | Optional |



##  Author

**Siddhi Jadhav**
Machine Learning Intern — Intern ID: IP-11161
Global Professional Internship (GPI) | Cloud Counselage



 If you found this project useful, consider giving it a star!
