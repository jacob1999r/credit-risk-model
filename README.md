# IFRS 9 Credit Risk Model

An end-to-end IFRS 9 expected credit loss model built in Python on 100,000 real loan observations from the LendingClub dataset (2007–2018). The model computes probability of default, loss given default, and exposure at default at loan level, aggregating to a total portfolio provision of **£38.7m** against total exposure of **£299.6m** (12.9% provision rate).

---

## Notebooks

### 01 Data Exploration
Loads and samples the LendingClub dataset. Defines the default flag from `loan_status`, classifying Charged Off, Late (31-120 days), Late (16-30 days) and non-credit-policy charge-offs as defaults. Observed default rate: **13.2%**.

**Limitations:** Sample of 100,000 rows from a larger dataset. Default flag classification involves judgement calls on borderline statuses (e.g. Late 16-30 days).

---

### 02 Feature Engineering
Selects 22 origination-point features from 151 available columns, removing all post-origination data to prevent data leakage. Encodes categorical variables, handles missing values, and introduces a purpose risk tier (1–3) based on empirically observed default rates by loan purpose.

**Limitations:** `int_rate` retained despite multicollinearity with `sub_grade` — identified in notebook 03 and flagged as a model improvement. Regional risk not modelled (`addr_state` excluded).

---

### 03 Probability of Default Model
Trains a logistic regression model on 80% of the feature set, with StandardScaler normalisation and `class_weight='balanced'` to address the 87/13 class imbalance. Validates on the held-out 20%. Applies Platt scaling to recalibrate inflated probabilities back to the true default rate. Includes full mathematical derivation of logistic regression from first principles, including log-likelihood maximisation and Newton-Raphson convergence.

**Key results:** Gini coefficient **0.424**, Area Under the Curve **0.712**. Mean predicted probability post-calibration: **0.132** (matching observed default rate exactly).

**Limitations:** `int_rate` should be dropped in future due to multicollinearity with `sub_grade`. `purpose_risk` had negligible model impact — risk tiering methodology could be revisited. Model is point-in-time rather than through-the-cycle.

---

### 04 Loss Given Default
Analyses recovery rates on 13,221 defaulted loans. Finds that the majority of defaulted loans are fully written off with a mean recovery rate of 7.0%. Due to the heavily boundary-concentrated distribution of loss given default (spike at 1.0), a fixed loss given default of **0.930** is adopted rather than a regression model.

**Limitations:** Fixed loss given default does not capture variation across borrower segments or economic conditions. A two-stage model (predict whether recovery occurs, then model recovery amount) would improve precision. Downturn loss given default adjustment not applied.

---

### 05 Expected Credit Loss Calculation
Multiplies probability of default × loss given default × exposure at default at loan level. Exposure at default taken as full funded amount (origination-point assumption). Aggregates to total portfolio provision.

**Key results:** Total provision **£38.7m**, total exposure **£299.6m**, provision rate **12.9%**.

**Limitations:** Exposure at default uses full funded amount rather than outstanding balance — a known simplification consistent with the origination-point modelling framework. 12-month expected credit loss used throughout rather than lifetime expected credit loss for Stage 2 and 3 loans (staging implemented separately).

---

### 06 Macroeconomic Scenarios
Attempts to establish a statistical relationship between macroeconomic variables (US unemployment, GDP growth) and observed default rates using Federal Reserve data. Three methodological approaches explored: vintage-year regression, lagged regression, and estimated default year methodology. No robust statistical relationship identified.

**Finding:** The dataset is insufficient for reliable macro modelling — 10 annual observations, a dominant vintage effect (lending criteria tighten during recessions, suppressing defaults in high-unemployment years), and uncertainty about the timing of default relative to economic conditions all constrain the analysis. In a production environment this overlay would be estimated using monthly default experience over a 15–20 year history.

**Limitation:** The model does not incorporate forward-looking macroeconomic information, which is a requirement of IFRS 9. This is the primary gap between this demonstration model and a production-grade IFRS 9 framework.

---

## Data
LendingClub accepted loan data (2007–2018), available on Kaggle under CC0 Public Domain licence. Raw data file not included in repository — download from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

## Requirements
- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib
- pandas-datareader

## Author
Jacob Russell — [LinkedIn](https://linkedin.com/in/jacob-russell99) | [GitHub](https://github.com/jacob1999r)
