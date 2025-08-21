from rapidfuzz import process, fuzz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Check for the percentage of missing values in the first 10 columns with more missin values
def missing_percentage(df, end=5):
  """
  Check for the percentage of missing values ordered by decreasing order
  df= the data frame
  end= number of columns default 5
  """
  missing_percent = df.isnull().mean() * 100
  print(missing_percent.sort_values(ascending=False)[:end].to_frame())


# Check for negative values in the numerical columns
def check_negative_values(df):
  num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
  neg_values = (df[num_cols] < 0).sum()
  neg_values = neg_values[neg_values > 0]
  return neg_values

  ########### Functions to preprocess the data #######################

### Handle typos function

def handle_typos(df, valid_dict, threshold=90):
    """
    Correct typos in a DataFrame using a dictionary of valid values.
    Print unmatched non-NaN values for the user to review.

    df         : DataFrame to process
    valid_dict : dictionary {column: valid_values_list}
    threshold  : minimum match percentage for replacement
    """

    # Select columns that are either categorical (object) or numeric (int)
    columns = df.select_dtypes(include=['object', 'int']).columns.tolist()

    # Iterate through each column
    for col in columns:
        if col in valid_dict:  # Only process columns present in the dictionary
            valid_vals = valid_dict[col]  # Get list of valid values for this column
            corrected = []  # Temporary list to store corrected values
            unmatched_values = set()  # Set to store values that could not be corrected

            # Process string columns using fuzzy matching
            if df[col].dtype == 'object':
                # Normalize valid values for comparison (remove spaces and lowercase)
                valid_vals_norm = {str(v).replace(" ", "").lower(): v for v in valid_vals}

                # Iterate through each value in the column
                for val in df[col]:
                    if pd.isna(val) or val in valid_vals:
                        # If value is NaN or already valid, keep it
                        corrected.append(val)
                    else:
                        # Normalize the value
                        val_norm = str(val).replace(" ", "").lower()
                        # Find closest match in valid values
                        match_norm, score, _ = process.extractOne(val_norm, valid_vals_norm.keys(), scorer=fuzz.ratio)
                        if score >= threshold:
                            # If match is good enough, use the matched valid value
                            corrected.append(valid_vals_norm[match_norm])
                        else:
                            # If no good match, keep original and store it for review
                            corrected.append(val)
                            unmatched_values.add(val)

            # Process numeric columns (e.g., MSSubClass)
            else:
                for val in df[col]:
                    if pd.isna(val) or val in valid_vals:
                        # Keep NaN or valid values as is
                        corrected.append(val)
                    else:
                        # Keep original value and store it as unmatched
                        corrected.append(val)
                        unmatched_values.add(val)

            # Print any unmatched values to alert the user
            if unmatched_values:
                print(f"Column '{col}' has unmatched values: {unmatched_values}")

            # Replace original column with corrected values
            df[col] = corrected

    # Return the DataFrame with corrections applied
    return df






### Handle nan values function
def handle_nan(df):

    # Divide columns in numerical and categorical
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

    #

    if 'SalePrice' in num_cols:
        num_cols.remove('SalePrice')  # exclude target

    # Fill numerical features with the median
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical: 'None', 'Missing' or mode depending the feature
    fill_with_mode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                      'Electrical', 'SaleType', 'SaleCondition']

    fill_with_none = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                      'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                      'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
                      'MasVnrType']



    for column in cat_cols:
        # Fill with 'None'
        if column in fill_with_none:
          df[column] = df[column].fillna('None')
        # Fill with mode
        elif column in fill_with_mode:
          df[column] = df[column].fillna(df[column].mode()[0])
       # Fill with missing
        elif column not in fill_with_none:
            df[column] = df[column].fillna('Missing')


### Function to pre process the data
def preprocess(df, valid_values_dict, threshole=90):
  handle_typos(df, valid_values_dict, threshole)
  handle_nan(df)


### Function to plot boxplots
def plot_boxplots(df, chunk_size=9):
    """
    Plots boxplots for all numerical columns in a DataFrame.
    If there are many columns, splits them into multiple figures for readability.

    Parameters:
    df : pd.DataFrame
        DataFrame with numerical columns to plot.
    chunk_size : int
        Number of columns per figure.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Loop through columns in chunks
    for i in range(0, len(num_cols), chunk_size):
        chunk = num_cols[i:i+chunk_size]
        n_rows = len(chunk) // 3 + (1 if len(chunk) % 3 != 0 else 0)
        plt.figure(figsize=(15, 5 * n_rows))

        for j, col in enumerate(chunk, 1):
            plt.subplot(n_rows, 3, j)
            sns.boxplot(y=df[col])
            plt.title(col)

        plt.tight_layout()
        plt.show()


def plot_target_scatters(X, y):
    """
    Plot scatter plots of all numeric columns in X vs y (target).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Target vector.
    """
    import matplotlib.pyplot as plt

    # Select only numeric columns
    numeric_cols = X.select_dtypes(include='number').columns

    # Determine subplot grid size
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    # Plot each numeric column vs y
    for i, col in enumerate(numeric_cols):
        axes[i].scatter(X[col], y, alpha=0.5)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("SalePrice")
        axes[i].set_title(f"{col} vs SalePrice")

    # Remove empty axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Group rare categories function
def group_rare_categories(df, min_freq=0.04):
    """
    Replace rare categories (less frequent than min_freq) with 'Other'
    in all categorical columns of the DataFrame.

    Parameters:
    df : pd.DataFrame
        DataFrame to process.
    min_freq : float
        Minimum frequency (proportion) a category must have to not be considered rare.
    """
    # Select all categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        # Compute the frequency of each category
        freq = df[col].value_counts(normalize=True)
        # Identify rare categories
        rare_values = freq[freq < min_freq].index
        # Replace rare categories with 'Other'
        df[col] = df[col].replace(rare_values, 'Other')

      # Show counts of the categories in categorical features
def count_categories(df):
  cat_cols = df.select_dtypes(include=['object']).columns

  # Show frecuency of each column
  for col in cat_cols:
      print(f"\nColumn: {col}")
      print(df[col].value_counts(dropna=False))


# Function to merge 'Other' with most common category
def merge_other_with_most_frequent(df, threshold=0.05):
  cat_cols = df.select_dtypes(include=['object']).columns
  for column in cat_cols:
    value_counts = df[column].value_counts(normalize=True)
    if 'Other' in value_counts and value_counts['Other'] < threshold:
        most_common = value_counts.drop('Other').idxmax()
        df[column] = df[column].replace('Other', most_common)
