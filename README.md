# House Prices Prediction – Kaggle Competition  

This project is based on the **House Prices: Advanced Regression Techniques** Kaggle competition. The goal is to predict housing prices using regression models, with a focus on data preprocessing, feature engineering, and regularization.  

---

## Data Cleaning and Preprocessing  
- **Missing values**: handled through imputation strategies (mean/median for numerical values, most frequent or "None" for categorical values).  
- **Inconsistencies**: categorical levels were standardized, and outliers were analyzed.  
- **Feature transformations**: log-transformations were applied to skewed numerical variables to reduce heteroscedasticity.  

---

## Feature Engineering  
- **Combinations of variables**: created new features by combining existing ones (ratios and products) to capture nonlinear relationships.  
- **Feature selection with Lasso**: used L1 regularization to identify the most relevant predictors. 
- **Scaling and encoding**:  
  - Standardization for numerical variables.  
  - One-hot encoding for categorical variables.  

---

## Models Implemented  
Several linear models were tested and compared:  
- Linear Regression  
- Ridge Regression (L2 regularization)  
- Lasso Regression (L1 regularization, also used for feature selection)  
- Elastic Net (combination of L1 and L2)  

---

## Results and Insights  
- Regularized models (Ridge, Lasso, Elastic Net) provided better generalization than plain Linear Regression.  
- Lasso was particularly useful for selecting variables and reducing overfitting.  
- Engineered features (products, sums, ratios) contributed to performance improvements.  
- The chosen evaluation metric was **RMSLE**, following the competition guidelines.  

---

## Future Work  
- Explore ensemble methods and tree-based models (XGBoost, LightGBM) for further performance improvements.  
- Experiment with advanced feature selection strategies (mutual information, model-based importance).  
- Refine hyperparameter tuning with cross-validation.  


*Current Kaggle position*: 677

*Current Kaggle score*: 0.12521
