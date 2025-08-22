import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,  Lasso, Ridge, ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations, permutations



# Function to label encode the categorical features of a df
def label_encode(X):
    X_label = X.copy()
    for colname in X_label.select_dtypes(["category", "object"]):
        X_label[colname], _ = pd.factorize(X_label[colname])
    return X_label

# Function to evaluate a model with cross-validation
def evaluate_model(X, y, model, return_trained_model=True):
    """
    Evaluate a model with cross-validation and return both MAE and RMSLE in original scale (USD).
    Optionally returns a model trained on the full dataset.
    Detects if model is a pipeline and avoids double encoding.
    """
    log_y = np.log1p(y)  # Transform target to log scale

    # Cross-validation predictions
    preds_log = cross_val_predict(model, X, log_y, cv=5)
    preds = np.expm1(preds_log)  # Convert predictions back to original scale

    mae = mean_absolute_error(y, preds)
    rmsle = np.sqrt(mean_squared_error(log_y, preds_log))

    if return_trained_model:
        model.fit(X, log_y)
        return {"mae" : mae, "rmsle": rmsle}, model
    else:
        return {"mae" : mae, "rmsle": rmsle}


##############################
# Functions to create pipelines including a class to create products and ratios between features
##############################






##############################################
# 1. Preprocessor builder (numeric + categorical)
##############################################
def create_preprocessor(numeric_columns, categorical_columns=None, encoding="label", log_columns=None):
    numeric_transformers = []

    if log_columns and len(log_columns) > 0:
        log_transformer = Pipeline([
            ('log', FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)),
            ('scaler', StandardScaler())
        ])
        numeric_transformers.append(('log', log_transformer, log_columns))

        remaining_cols = [col for col in numeric_columns if col not in log_columns]
        if remaining_cols:
            numeric_transformers.append(('num', Pipeline([('scaler', StandardScaler())]), remaining_cols))
    else:
        numeric_transformers.append(('num', Pipeline([('scaler', StandardScaler())]), numeric_columns))

    transformers = numeric_transformers

    if categorical_columns and len(categorical_columns) > 0:
        if encoding == "label":
            categorical_transformer = Pipeline([('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        elif encoding == "onehot":
            categorical_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        else:
            raise ValueError("encoding must be 'label' or 'onehot'")
        transformers.append(('cat', categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


##########################################
# 2. PIPELINE CREATORS FOR EACH MODEL   #
##########################################
def create_linear_pipeline(numeric_columns, categorical_columns=None, encoding="onehot", log_columns=None):
    preprocessor = create_preprocessor(numeric_columns, categorical_columns, encoding, log_columns)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

def create_lasso_pipeline(numeric_columns, categorical_columns=None, encoding="onehot", log_columns=None,
                          alpha=0.01):
    preprocessor = create_preprocessor(numeric_columns, categorical_columns, encoding, log_columns)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(alpha=alpha, random_state=0))
    ])

def create_ridge_pipeline(numeric_columns, categorical_columns=None, encoding="onehot", log_columns=None,
                          alpha=0.01):
    preprocessor = create_preprocessor(numeric_columns, categorical_columns, encoding, log_columns)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha, random_state=0))
    ])

def create_elasticnet_pipeline(numeric_columns, categorical_columns=None, encoding="onehot", log_columns=None,
                               alpha=0.01, l1_ratio=0.5):
    preprocessor = create_preprocessor(numeric_columns, categorical_columns, encoding, log_columns)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0))
    ])


################ Evaluate all linear-based models ################
def evaluate_all_linear(X, y, numeric_columns, categorical_columns=None, log_columns=None,
                        alpha=0.001, l1_ratio=0.5, print_results=True):
    """
    Evaluates Linear, Lasso, Ridge, and ElasticNet models using onehot encoding.
    Return a dictionary with keys 'linear', 'lasso', ridge, 'lasteicnet'
    ex: {'linear': {'mae': mae, 'rmsle':rmsle }, ...}
    """
    # ---------------- Linear Regression ----------------

    lreg = create_linear_pipeline(numeric_columns, categorical_columns, encoding='onehot',
                                           log_columns=log_columns)
    lreg_dict = evaluate_model(X, y, lreg, return_trained_model=False)

    # ---------------- Lasso Regression ----------------

    lasso= create_lasso_pipeline(numeric_columns, categorical_columns, encoding='onehot',
                                         log_columns=log_columns, alpha=alpha)
    lasso_dict = evaluate_model(X, y, lasso, return_trained_model=False)

    # ---------------- Ridge Regression ----------------

    ridge = create_ridge_pipeline(numeric_columns, categorical_columns, encoding='onehot',
                                         log_columns=log_columns, alpha=alpha)
    ridge_dict = evaluate_model(X, y, ridge, return_trained_model=False)

    # ---------------- ElasticNet Regression ----------------

    elastic = create_elasticnet_pipeline(numeric_columns, categorical_columns, encoding='onehot',
                                          log_columns=log_columns, alpha=alpha, l1_ratio=l1_ratio)
    elastic_dict = evaluate_model(X, y, elastic, return_trained_model=False)

    # ---------------- Print results -------------------------
    if print_results:
      print(f'Linear:           MAE: {lreg_dict['mae']} | RMSLE: {lreg_dict['rmsle']}')
      print(f'Lasso:            MAE: {lasso_dict['mae']} | RMSLE: {lasso_dict['rmsle']}')
      print(f'Ridge:            MAE: {ridge_dict['mae']} | RMSLE: {lreg_dict['rmsle']}')
      print(f'elastic_net:      MAE: {elastic_dict['mae']} | RMSLE: {lreg_dict['rmsle']}')

    return { 'linear' : lreg_dict, 'lasso' : lasso_dict, 'ridge' : ridge_dict, 'elasticnet' : elastic_dict}


###################### Extract and plot coefficients for Linear Regression model ###########

### Extract coeficients by order of absolute value
def get_coef_signed(pipeline):
    """
    Extract feature names and coefficients from a linear pipeline
    (LinearRegression, Lasso, Ridge, ElasticNet).
    """
    # Extract the preprocessor and regressor from the pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']

    feature_names = []

    # Iterate over transformers in the ColumnTransformer
    for name, transformer, cols in preprocessor.transformers_:
        # If the transformer is a pipeline, take the last step
        if hasattr(transformer, 'named_steps'):
            trans = list(transformer.named_steps.values())[-1]
        else:
            trans = transformer

        # If the transformer provides feature names (e.g., OneHotEncoder)
        if hasattr(trans, "get_feature_names_out"):
            try:
                # Get feature names using the columns
                fn = trans.get_feature_names_out(cols)
            except TypeError:
                # Some transformers (like OneHotEncoder without input_features) may not accept cols
                fn = trans.get_feature_names_out()
            feature_names.extend(fn)
        else:
            # Transformers like log or scaler do not change feature names, keep original column names
            feature_names.extend(cols)

    # Create a DataFrame with features and their coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': regressor.coef_.ravel()
    })

    # Sort the DataFrame by absolute value of coefficients
    coef_df = coef_df.sort_values(by='coefficient', key=abs, ascending=False).reset_index(drop=True)

    return coef_df




# Plot top coeficients
def plot_coeficients_importance(coef_df, top_n=20, title="Feature Importances"):
    """
    Plot the coefficients from get_coef_simple output.

    Parameters:
    -----------
    coef_df : pd.DataFrame
        Output from get_coef_simple (must have 'feature' and 'coefficient').
    top_n : int, optional
        Number of top features to plot (default=20).
    title : str, optional
        Plot title.
    """
    # Select top_n by absolute value
    coef_df_sorted = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=False).index)
    coef_df_top = coef_df_sorted.head(top_n)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(coef_df_top['feature'], coef_df_top['coefficient'], color="skyblue")
    plt.axvline(0, color='k', linestyle='--')
    plt.title(title)
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()  # biggest on top

    # Color positive/negative
    for bar, val in zip(bars, coef_df_top['coefficient']):
        if val < 0:
            bar.set_color('salmon')

    plt.show()


# Function to get XGBoost feature importance as DataFrame
def get_xgb_feature_importance_df(XGB_model):
    """
    Return XGBoost feature importance as a DataFrame sorted by gain.

    Parameters:
    -----------
    XGB_model : trained XGBRegressor or XGBClassifier
        Trained XGBoost model
    top_n : int
        Number of top features to return

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['feature', 'gain'] sorted by gain descending
    """
    importance_dict = XGB_model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'gain': list(importance_dict.values())
    }).sort_values(by='gain', ascending=False).reset_index(drop=True)

    return importance_df

# Function to plot XGBoost feature importance from a DataFrame
def plot_xgb_feature_importance(importance_df):
    """
    Plot XGBoost feature importance from a DataFrame.

    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with columns ['feature', 'gain']
    """
    plt.figure(figsize=(10, max(6, 0.3*len(importance_df))))
    plt.barh(importance_df['feature'], importance_df['gain'], color='skyblue')
    plt.gca().invert_yaxis()  # largest gain at top
    plt.xlabel('Gain')
    plt.title(f'Top {len(importance_df)} XGBoost Features by Gain')
    plt.show()



# --- Function to compute MI scores ---
def get_mi_scores(X, y, discrete_features=None, top_n=None):
    """
    Compute mutual information (MI) scores for features in X against y.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or array-like
        Target vector
    discrete_features : list or boolean array, optional
        Specify which features are discrete/categorical
    top_n : int, optional
        If specified, return only the top_n features with highest MI

    Returns:
    --------
    pd.Series
        MI scores sorted descending
    """
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, index=X.columns, name='MI Scores')
    mi_scores = mi_scores.sort_values(ascending=False)
    if top_n is not None:
        mi_scores = mi_scores.iloc[:top_n]
    return mi_scores


# --- Function to plot MI scores ---
def plot_mi_scores(mi_scores, top_n=None):
    """
    Plot mutual information (MI) scores from a Series.

    Parameters:
    -----------
    mi_scores : pd.Series
        Series with MI scores indexed by feature names
    top_n : int, optional
        If specified, plot only the top_n features
    """
    if top_n is not None:
        mi_scores = mi_scores.iloc[:top_n]

    plt.figure(figsize=(10, max(4, 0.3*len(mi_scores))))
    mi_scores.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.xlabel('Mutual Information')
    plt.title('Feature Importance via Mutual Information')
    plt.show()

####### Function to add interactions products and ratios


def add_interactions(X, product_tuples=None, ratio_tuples=None, eps=1e-6, return_numeric_features=True):
    """
    Add product and ratio features efficiently to X based on given tuples.

    Parameters
    ----------
    X : pd.DataFrame
        Original DataFrame.
    product_tuples : list of tuples
        Tuples of columns to generate products (col1*col2).
    ratio_tuples : list of tuples
        Tuples of columns to generate ratios (col1/col2).
    eps : float
        Small value to avoid division by zero in ratios.
    return_numeric_features : bool
        If True, returns the list of numeric columns including the new interactions.

    Returns
    -------
    X_new : pd.DataFrame
        DataFrame with added interaction features.
    numeric_features (optional) : list
        List of numeric columns including newly created interactions (if return_numeric_features=True).
    """
    X_new = X.copy()
    new_features = {}

    # Create product features
    if product_tuples:
        for col1, col2 in product_tuples:
            new_features[f"{col1}*{col2}"] = X_new[col1] * X_new[col2]

    # Create ratio features
    if ratio_tuples:
        for col1, col2 in ratio_tuples:
            new_features[f"{col1}/{col2}"] = X_new[col1] / (X_new[col2] + eps)

    # Add new features to the DataFrame efficiently
    if new_features:
        X_new = pd.concat([X_new, pd.DataFrame(new_features, index=X.index)], axis=1)

    if return_numeric_features:
        # Combine original numeric columns with new interactions
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        numeric_cols += list(new_features.keys())
        return X_new, numeric_cols

    return X_new


