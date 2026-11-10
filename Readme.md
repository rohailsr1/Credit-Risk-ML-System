# 🛡️ Credit Risk Assessment Engine | Machine Learning & FastAPI

🚀 **Live Interactive Web Application:** <https://credit-risk-ml-system-1g3x.onrender.com/>

# 🎯 Business Problem

In retail banking and credit underwriting, accurately assessing loan default risk is critical to maintaining portfolio stability, mitigating non-performing loans (NPLs), and maximizing net interest margins. Credit risk evaluations rely on complex interactions between applicant demographics, financial ratios, loan terms, and historical defaults:

* Applicant age & income
* Employment duration & home ownership
* Requested loan amount & interest rate
* Loan-to-income debt burden ratio
* Loan intent & assigned risk grade
* Prior historical defaults on record
* Credit bureau history length

The objective of this project was to develop an end-to-end **Automated Credit Risk Intelligence & Classification System**. The machine learning pipeline evaluates applicant profiles in real-time, calculates loan default probabilities via an optimized **XGBoost Classifier**, and applies custom probability thresholding to automate underwriting approvals.

### Objectives

* Analyze historical loan applicant datasets and identify key financial default indicators.
* Clean and impute missing numerical and categorical parameters.
* Standardize categorical features (`loan_intent`, `loan_grade`, `person_home_ownership`, `cb_person_default_on_file`).
* Compute real-time engineered risk metrics (such as `loan_percent_income`).
* Train and evaluate multiple binary classification models (Logistic Regression, Random Forest, LightGBM, XGBoost).
* Fine-tune probability decision thresholds to optimize the **F1-Score / Recall balance** for high-risk detection.
* Build a lightweight **FastAPI** backend with automated JSON/NumPy serialization and Static File mounting.
* Deploy a glassmorphic interactive web dashboard for real-time underwriting risk scoring on **Render**.

# 📊 Exploratory Data Analysis & Risk Metrics

## 🎯 Default Risk vs. Loan-to-Income Ratio

The ratio of requested loan amount to annual income (`loan_percent_income`) proved to be one of the strongest individual predictors of applicant default risk.

* Low Risk Applicants: Average Loan-to-Income Ratio $\approx 0.14$ ($14\%$).
* Defaulted Applicants: Average Loan-to-Income Ratio $> 0.38$ ($38\%+$).

### Insight

Applicants requesting loan amounts exceeding $35\%-40\%$ of their annual salary exhibited exponentially higher default probabilities, regardless of employment length.

## 💳 Prior Default History vs Risk Classification

The relationship between historical credit bureau default flags (`cb_person_default_on_file`) and actual loan default outcomes was evaluated.

| Prior Default Flag (`cb_person_default_on_file`) | Observed Default Rate | Default Risk Category |
| ----- | ----- | ----- |
| **No Flag (`N`)** | $14.2\%$ | Low / Moderate Risk |
| **Default Flagged (`Y`)** | $71.8\%$ | High Risk Flagged |

### Insight

A prior default on file is a primary signal for future defaults. Applicants with an existing default record showed more than **5x higher likelihood** of defaulting on new obligations.

## 📈 Interest Rates & Loan Grades

Interest rates and assigned credit grades ($A$ through $F$) directly mirror underlying risk tiers assigned during credit scoring.

| Loan Grade | Avg Interest Rate (%) | Default Rate (%) | Risk Assessment |
| ----- | ----- | ----- | ----- |
| **Grade A** | $7.2\%$ - $8.5\%$ | $< 8\%$ | Prime / Lowest Risk |
| **Grade B** | $10.0\%$ - $11.5\%$ | $12\%$ - $15\%$ | Low-Moderate Risk |
| **Grade C** | $13.0\%$ - $14.5\%$ | $22\%$ - $28\%$ | Moderate Risk |
| **Grade D** | $16.0\%$ - $18.5\%$ | $45\%$ - $55\%$ | Subprime / High Risk |
| **Grade E / F** | $> 19.0\%$ | $> 65\%$ | Extreme High Risk |

### Business Takeaway

Higher interest rates increase total repayment obligations, creating a feedback loop that elevates default risk for lower-graded applicants.

# 🧹 Data Preparation & Feature Engineering

The pipeline processes 11 key feature attributes passed to the classification pipeline.

### Input Schema Definition

```python
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int
```

### Feature Engineering & Transformation Steps

1. **Auto-Calculated Ratios:**

   $$ \text{Loan Percent Income} = \frac{\text{loan\_amnt}}{\text{person\_income}} $$

2. **Categorical Encoding:** One-Hot and Ordinal encoding applied to `person_home_ownership` (`RENT`, `MORTGAGE`, `OWN`, `OTHER`), `loan_intent` (`PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`), `loan_grade` (`A` to `F`), and `cb_person_default_on_file` (`Y`/`N`).

3. **Data Type Serialization Fix:** Explicit casting of NumPy data types (`numpy.float32`, `numpy.int64`) to Python primitives (`float`, `int`) to prevent FastAPI JSON response serialization exceptions.

# 🤖 Machine Learning Modeling & Comparison

Multiple binary classification algorithms were evaluated to determine the best model for predicting default risk ($1 = \text{Default/High Risk}, 0 = \text{Non-Default/Low Risk}$).

| Algorithm | Precision | Recall | F1-Score | ROC-AUC |
| ----- | ----- | ----- | ----- | ----- |
| Logistic Regression | $0.72$ | $0.54$ | $0.62$ | $0.76$ |
| Random Forest Classifier | $0.88$ | $0.71$ | $0.79$ | $0.91$ |
| LightGBM Classifier | $0.90$ | $0.75$ | $0.82$ | $0.93$ |
| **Tuned XGBoost Classifier** | $0.92$ | $0.79$ | $0.85$ | $0.948$ |

### Decision Threshold Optimization

In credit underwriting, **false negatives** (approving a high-risk borrower who defaults) are significantly costlier than **false positives** (rejecting a safe borrower).

Rather than using a default binary threshold of $0.50$, an optimal decision threshold ($\approx 0.35 - 0.42$) was calculated using the Precision-Recall curve to optimize the F1-Score and maximize default detection.

$$ \text{Prediction Status} = \begin{cases} 1 \quad (\text{High Risk}), & \text{if } P(\text{Default}) \ge \tau \\ 0 \quad (\text{Low Risk}), & \text{if } P(\text{Default}) < \tau \end{cases} $$

*where $\tau$ represents the optimized classification threshold ($\text{threshold} \approx 0.38$).*

# 💻 Web Application & Architecture

The system is delivered as a lightweight, single-service web application built with **FastAPI**, **Uvicorn**, and a custom **Tailwind CSS Glassmorphism Frontend**.

```text
├── app.py                      # FastAPI App (API Endpoints + Static File Server)
├── credit_risk_model.pkl       # Trained XGBoost Machine Learning Pipeline
├── best_threshold.pkl          # Optimal Probability Cutoff Threshold
├── requirements.txt            # Python Dependencies
├── Procfile                    # Render Deployment Entrypoint
├── runtime.txt                 # Python Runtime Version Specification
└── static/
    └── index.html              # Single-Page Interactive Web Dashboard
```

### Application Features

* **Interactive Underwriting Dashboard:** Real-time form evaluation with background gauge animation and risk status indicators.
* **Preset Buttons:** Instant load buttons for "Low Risk" and "High Risk" test profiles.
* **Auto-Calculated Ratios:** Dynamically updates `loan_percent_income` in real time as income or loan values change.
* **Seamless Local & Cloud Serve:** Configured with `app.mount('/', StaticFiles(directory='static', html=True))` to serve both backend REST endpoints and the UI from a single port.

# 🚀 How to Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/credit-risk-ml-system.git
cd credit-risk-ml-system
```

## 2. Set Up Virtual Environment & Dependencies

```bash
# Create environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate
# Activate environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Launch the Server

```bash
uvicorn app:app --reload
```

## 4. Access the App

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000/`**

To view auto-generated Swagger API documentation, visit:
👉 **`http://127.0.0.1:8000/docs`**

# 🌐 Cloud Deployment (Render)

The project is deployed on **Render**:

🔗 **Live URL:** <https://credit-risk-ml-system-1g3x.onrender.com/>

1. **Procfile:** Contains the production web server start command:
   ```text
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
2. **runtime.txt:** Pins Python execution version (`python-3.11.9`).

# 👨‍💻 Author

**Muhammad Rohail**  
*Data Analyst | Machine Learning & Business Intelligence*