Telco Customer Churn Analysis
=============================

How to create the cleaned-up database
-------------------------------------

Create a customers table in the TelcoCustomerchurn database in your SQL environment by using TelcoCreateCustomerBase.sql . 
Use TelcoAanpassingen.sql to make the first clean up changes to the telco_customer_churn.csv (original Telco dataset) which will be saved in the customers table.
Use TelcoAanpassingenCustomers2.sql for the final clean up to create the customers_ml table. Save the customers_ml table as TelcoSemiCleanedup.csv and load this file into python file.

Instead, directly download TelcoSemiCleandup.csv from the Data branch.

What this folder contains
-------------------------
This portfolio package documents an exploratory and predictive analysis
of telecom customer churn.

  README.txt          - this file (report + Python code overview)
  report.tex          - LaTeX source for the PDF findings report (Overleaf)
  figures/            - PNG plots embedded by report.tex

  Upstream analysis script and data (project root, one level above
  portfolio/):
  TelcoAnalysis.py
  TelcoSemiCleanedup.csv

What the analysis does
----------------------
1. Load 7,043 customers from TelcoSemiCleanedup.csv (overall churn
   26.5%: 1,869 churned / 5,174 stayed).
2. Keep Churn Reason aside; one-hot encode remaining categoricals for
   modelling (Online Backup, Device Protection, Tech Support,
   Streaming TV/Movies, Contract, Payment Method).
3. Exploratory churn rates by contract, internet service, payment
   method, monthly/total charges (quintiles), and tenure bins.
4. Rank manager-friendly traits by observed churn rate (min 50
   customers per trait).
5. Fit logistic regression on an 80/20 stratified split (seed 42) with
   standardized continuous columns; report ROC-AUC, classification
   metrics, and confusion matrix.
6. Validate with 5-fold stratified cross-validation.
7. Compare Logistic Regression vs Random Forest vs XGBoost (or
   HistGradientBoosting if xgboost is not installed).
8. Compare normalized feature importances across models and publish a
   consensus ranking.

Plots in figures/
-----------------
  churn_rate_by_segment.png          Churn % by key segments (grid)
  trait_churn_ranking.png            Top traits by observed churn rate
  cv_roc_auc.png                     Logistic regression CV ROC-AUC
  model_comparison.png               ROC-AUC / accuracy / recall by model
  feature_importance_agreement.png   Top 12 features across models
  confusion_matrix.png               Logistic regression test CM
  coefficient_comparison.png         Most vs least important coefficients
  consensus_feature_importance.png   Averaged importance ranking

How to regenerate the plots
---------------------------
From the project root (where TelcoAnalysis.py lives):

  set MPLBACKEND=Agg
  py -3 TelcoAnalysis.py

On PowerShell:

  $env:MPLBACKEND='Agg'
  py -3 TelcoAnalysis.py

Figures are written to this folder under figures/. Set SAVE_FIGURES=0 to
skip saving PNGs.

How to build the PDF report (Overleaf)
--------------------------------------
1. Create a new Overleaf project.
2. Upload report.tex and the figures/ folder (keep the same relative
   paths: figures/*.png).
3. Compile. Download the PDF for your portfolio / applications.

Requirements
------------
Python 3.x with: pandas, numpy, matplotlib, scikit-learn
Optional: xgboost (otherwise HistGradientBoostingClassifier is used)

Key result snapshot (from last run)
-----------------------------------
  Customers: 7,043  |  Churn rate: 26.5%
  Logistic Regression test ROC-AUC: 0.856
  5-fold CV mean ROC-AUC: 0.857 (+/- 0.006)
  Model ROC-AUC (same split): LR 0.856 | XGBoost 0.852 | RF 0.843
  Highest-churn segments: tenure 0-12 mo (47.4%), electronic check
  (45.3%), month-to-month (42.7%), fibre optics (41.9%)
  Consensus top drivers: month-to-month contract, tenure, total
  charges, dependents, fibre optics
