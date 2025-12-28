
Explainable loan approval system using XGBoost and GenAI 


# Explainable Loan Approval System

This project demonstrates a practical approach to combining traditional machine learning with generative AI for decision-support systems.

Loan approval decisions are made using an XGBoost classifier trained on a structured loan dataset. A lightweight language model is used strictly as an explanation layer to translate model outputs into human-readable explanations and counterfactual insights.

## Key Features

- Binary loan approval prediction using XGBoost
- Stratified k-fold cross-validation to assess model stability
- Robust handling of missing values and categorical features
- Feature-importance-based explanations
- Counterfactual explanations describing what could improve approval chances
- Interactive Streamlit interface

## Design Approach

The system is intentionally designed so that:

- The machine learning model is fully responsible for prediction
- The generative model does not influence the decision logic
- Explanations are grounded in model-derived signals

This separation keeps the system deterministic, auditable, and aligned with real-world ML practices, particularly in decision-heavy or regulated domains.

## Workflow Overview

1. Upload a loan dataset through the Streamlit interface  
2. Preprocess numeric and categorical features  
3. Evaluate model stability using stratified k-fold cross-validation  
4. Train a final XGBoost classifier  
5. Evaluate new loan applications  
6. Generate natural language explanations and counterfactual insights using GenAI  




## HOW TO RUN THIS APPLICATION

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Streamlit app:
   streamlit run app.py

3. Open the browser at:
   http://localhost:8501


## DATASET REQUIREMENTS

The application is designed to work with structured loan datasets containing
demographic, financial, and credit-related features, with a binary loan
approval target column.


## FUTURE ENHANCEMENTS

- SHAP-based local explanations for individual predictions
- Fairness and bias evaluation across demographic groups
- Probability calibration for more reliable confidence estimates
- Model persistence and reuse for deployment scenarios




