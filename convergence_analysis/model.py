import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels as sm
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

import scipy.stats as stats
import sys
import os
import math


def plot(x_col, y_col, df, x_label=None, y_label=None, title=None):
    plt.scatter(df[x_col], df[y_col])
    if x_label:
        plt.xlabel(x_label)
    else:
        plt.xlabel(x_col)
    if y_label:
        plt.ylabel(y_label)
    else:
        plt.ylabel(y_col)
    if title:
        plt.title(title)
    else:
        plt.title(f"{y_col} vs {x_col}")
    plt.show()

def linear_model(x_cols, y_col, df_in, df_out):
    # check that cols are in df
    for col in x_cols + [y_col]:
        if col not in df_in.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Create a linear regression model
    X = df_in[x_cols]
    y = df_out[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Print the model summary
    print(model.summary())
    # Number of subplots based on the number of x_cols
    num_plots = len(x_cols) + 3  # Adding 3 for the diagnostic plots
    cols = 2  # Number of columns in subplot
    rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows

    # Creating a grid of subplots
    fig, ax = plt.subplots(rows, cols, figsize=(15, rows * 5))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Plotting scatter plots for each x_col
    for i, col in enumerate(x_cols):
        r, c = divmod(i, cols)
        ax[r, c].scatter(df_in[col], y)
        ax[r, c].set_xlabel(col)
        ax[r, c].set_ylabel(y_col)
        ax[r, c].set_title(f'{col} vs {y_col}')

    # Q-Q plot of the residuals
    sm.graphics.qqplot(model.resid, line='45', fit=True, ax=ax[len(x_cols) // cols, len(x_cols) % cols])
    ax[len(x_cols) // cols, len(x_cols) % cols].set_title('Q-Q Plot of the Residuals')

    # Residuals vs. Fitted plot
    r, c = divmod(len(x_cols) + 1, cols)
    ax[r, c].scatter(model.fittedvalues, model.resid)
    ax[r, c].axhline(y=0, color='red', linestyle='dashed')
    ax[r, c].set_xlabel('Fitted values')
    ax[r, c].set_ylabel('Residuals')
    ax[r, c].set_title('Residuals vs. Fitted')

    # Leverage vs. Residuals squared plot
    r, c = divmod(len(x_cols) + 2, cols)
    sm.graphics.plot_leverage_resid2(model, ax=ax[r, c])
    ax[r, c].set_title('Leverage vs. Residuals squared')

    plt.show()
    return model

def polynomial_model(x_cols, y_col, df_in, degree=2):
    # Ensure columns exist in DataFrame
    for col in x_cols + [y_col]:
        if col not in df_in.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Generating polynomial features
    X = df_in[x_cols]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Create DataFrame from polynomial features for easier plotting
    col_names = poly.get_feature_names_out(x_cols)
    X_poly_df = pd.DataFrame(X_poly, columns=col_names)

    # Create a linear regression model using polynomial features
    y = df_in[y_col]
    X_poly_df = sm.add_constant(X_poly_df)
    model = sm.OLS(y, X_poly_df).fit()

    # Print the model summary
    print(model.summary())

    # Number of subplots based on the number of polynomial columns
    num_plots = len(col_names) + 3  # Adding 3 for the diagnostic plots
    cols = 2  # Number of columns in subplot
    rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows

    # Creating a grid of subplots
    fig, ax = plt.subplots(rows, cols, figsize=(15, rows * 5))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Plotting scatter plots for each polynomial feature
    for i, col in enumerate(col_names):
        r, c = divmod(i, cols)
        ax[r, c].scatter(X_poly_df[col], y)
        ax[r, c].set_xlabel(col)
        ax[r, c].set_ylabel(y_col)
        ax[r, c].set_title(f'{col} vs {y_col}')

    # Q-Q plot of the residuals
    sm.graphics.qqplot(model.resid, line='45', fit=True, ax=ax[len(col_names) // cols, len(col_names) % cols])
    ax[len(col_names) // cols, len(col_names) % cols].set_title('Q-Q Plot of the Residuals')

    # Residuals vs. Fitted plot
    r, c = divmod(len(col_names) + 1, cols)
    ax[r, c].scatter(model.fittedvalues, model.resid)
    ax[r, c].axhline(y=0, color='red', linestyle='dashed')
    ax[r, c].set_xlabel('Fitted values')
    ax[r, c].set_ylabel('Residuals')
    ax[r, c].set_title('Residuals vs. Fitted')

    # Leverage vs. Residuals squared plot
    r, c = divmod(len(col_names) + 2, cols)
    sm.graphics.plot_leverage_resid2(model, ax=ax[r, c])
    ax[r, c].set_title('Leverage vs. Residuals squared')

    plt.show()
    return model


def plot_residuals(model, x_cols, y_col, df_in, df_out):
    """
    Plots the residuals of a linear regression model.

    Parameters:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): The linear regression model.
    x_col (str): The name of the independent variable column.
    y_col (str): The name of the dependent variable column.
    df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    ax.scatter(df_in[x_cols], model.resid)
    ax.axhline(y=0, color='r', linestyle='-')
    ax.set_xlabel(x_cols)
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs {x_cols}")
    err = np.std(model.resid)
    ax.errorbar(df_in[x_cols], model.resid, yerr=err, fmt='o', color='blue', ecolor='green', capsize=5, capthick=2)
    for i, val in enumerate(model.resid):
        ax.annotate(round(val, 2), (df_in[x_cols][i], val), textcoords="offset points", xytext=(0,10), ha='center')
    plt.show()
    return None


# shows slight linear relationship between FP and Volume to Unit Cell Volume Ratio
# linear_model(['VolumeToUnitCellVolRatio'], 'Mean FP', results_df, results_df)

# conditition number is large, indicates stable model
# lm = linear_model(['VolumeToUnitCellVolRatio'], 'Mean FP', results_df, results_df)

# plot_residuals(lm, 'VolumeToUnitCellVolRatio', 'Mean FP', results_df, results_df)