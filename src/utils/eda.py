"""
EDA utility functions for Customer Churn Prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional




# Helper Function 1

def display_basic_info(df, column):
    """
    Display basic information about a feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Feature name.
    """

    print("=" * 60)
    print(f"Feature : {column}")
    print("=" * 60)

    print(f"Data Type          : {df[column].dtype}")
    print(f"Missing Values     : {df[column].isnull().sum()}")

    missing_percentage = (
        df[column].isnull().sum() / len(df)
    ) * 100

    print(f"Missing Percentage : {missing_percentage:.2f}%")

    print(f"Unique Values      : {df[column].nunique()}")





# Helper Function 2

def display_summary_statistics(df, column):
    """
    Display descriptive statistics for a numerical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Numerical feature name.
    """

    summary = pd.DataFrame({
        "Statistic": [
            "Count",
            "Mean",
            "Median",
            "Mode",
            "Standard Deviation",
            "Minimum",
            "25th Percentile (Q1)",
            "50th Percentile (Median)",
            "75th Percentile (Q3)",
            "Maximum",
            "Variance"
        ],
        "Value": [
            df[column].count(),
            df[column].mean(),
            df[column].median(),
            df[column].mode()[0],
            df[column].std(),
            df[column].min(),
            df[column].quantile(0.25),
            df[column].quantile(0.50),
            df[column].quantile(0.75),
            df[column].max(),
            df[column].var()
        ]
    })

    # display(summary)




# Helper Function 3 – Distribution Plot

def plot_distribution(df, column):
    """
    Plot the distribution of a numerical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Numerical feature name.
    """

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=30,
        kde=True,
        color="steelblue",
        edgecolor="black",
        alpha=0.7
    )

    mean_value = df[column].mean()
    median_value = df[column].median()

    plt.axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean : {mean_value:.2f}"
    )

    plt.axvline(
        median_value,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"Median : {median_value:.2f}"
    )

    plt.title(f"Distribution of {column}", fontsize=14, fontweight="bold")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()






# Helper Function 4 – Boxplot
def plot_boxplot(df, column):
    """
    Plot a boxplot for a numerical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Numerical feature name.
    """

    plt.figure(figsize=(10, 2))

    # sns.boxplot(
    #     data=df,
    #     x=column,
    #     color="skyblue",
    #     linewidth=1.5,
    #     fliersize=5
    # )

    sns.boxplot(
        data=df,
        x=column,
        color="skyblue",
        linewidth=1.5,
        fliersize=5,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markersize": 7
        }
    )

    plt.title(
        f"Boxplot of {column}",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(column)

    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    plt.show()







# Helper Function 5 – Skewness Analysis
def analyze_skewness(df, column):
    """
    Calculate and interpret the skewness of a numerical feature.
    """

    skewness = df[column].skew()

    print("=" * 60)
    print("Skewness Analysis")
    print("=" * 60)

    print(f"Skewness Value : {skewness:.3f}")

    if abs(skewness) < 0.5:
        interpretation = "Approximately Symmetric"
        implication = "No transformation is generally required."

    elif 0.5 <= skewness < 1:
        interpretation = "Moderately Positively Skewed"
        implication = "Feature transformation may improve some models."

    elif skewness >= 1:
        interpretation = "Highly Positively Skewed"
        implication = "Consider log, Box-Cox, or Yeo-Johnson transformation."

    elif -1 < skewness <= -0.5:
        interpretation = "Moderately Negatively Skewed"
        implication = "Transformation may be beneficial depending on the model."

    else:
        interpretation = "Highly Negatively Skewed"
        implication = "Consider suitable transformation techniques."

    print(f"Distribution : {interpretation}")
    print(f"ML Insight   : {implication}")





# Master Fuction for Numerical Feature Analysis

def analyze_numerical_feature(df, column):
    """
    Perform a complete univariate analysis
    for a numerical feature.
    """

    if column not in df.columns:
        raise ValueError(f"'{column}' is not a valid column in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"'{column}' is not a numerical feature.")

    print("\n")
    print("=" * 80)
    print(f"NUMERICAL FEATURE ANALYSIS : {column.upper()}")
    print("=" * 80)

    display_basic_info(df, column)
    print("\n")

    display_summary_statistics(df, column)
    print("\n")

    plot_distribution(df, column)

    plot_boxplot(df, column)

    analyze_skewness(df, column)

    print("\n")
    print("=" * 80)
    print("Analysis Completed")
    print("=" * 80)





# Helper Function 1

def display_basic_info(df, column):
    """
    Display basic information about a feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Feature name.
    """

    print("=" * 60)
    print(f"Feature : {column}")
    print("=" * 60)

    print(f"Data Type          : {df[column].dtype}")
    print(f"Missing Values     : {df[column].isnull().sum()}")

    missing_percentage = (
        df[column].isnull().sum() / len(df)
    ) * 100

    print(f"Missing Percentage : {missing_percentage:.2f}%")

    print(f"Unique Values      : {df[column].nunique()}")





# Helper Function 2

def display_frequency_table(df, column):
    """
    Display the frequency and percentage distribution
    of a categorical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Categorical feature name.
    """

    frequency_table = (
        df[column]
        .value_counts(dropna=False)
        .rename_axis("Category")
        .reset_index(name="Count")
    )

    frequency_table["Percentage"] = (
        frequency_table["Count"] / len(df) * 100
    ).round(2)

    # display(frequency_table)




# Helper Function 3

def plot_countplot(df, column):
    """
    Plot the frequency distribution of a categorical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Categorical feature name.
    """

    plt.figure(figsize=(10, 5))

    ax = sns.countplot(
        data=df,
        x=column,
        color="steelblue"
    )

    plt.title(
        f"Count Plot of {column}",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(column)
    plt.ylabel("Count")

    plt.xticks(rotation=45)

    # Display count on top of each bar
    for container in ax.containers:
        ax.bar_label(container)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.show()





# Helper Function 4  

def analyze_category_balance(df, column):
    """
    Analyze the balance of categories in a categorical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Categorical feature name.
    """

    category_percentage = (
        df[column]
        .value_counts(normalize=True, dropna=False) * 100
    )

    largest_category = category_percentage.idxmax()
    largest_percentage = category_percentage.max()

    smallest_category = category_percentage.idxmin()
    smallest_percentage = category_percentage.min()

    print("=" * 60)
    print("Category Balance Analysis")
    print("=" * 60)

    print(f"Largest Category      : {largest_category} ({largest_percentage:.2f}%)")
    print(f"Smallest Category     : {smallest_category} ({smallest_percentage:.2f}%)")
    print(f"Number of Categories  : {df[column].nunique()}")

    if largest_percentage < 60:
        balance = "Balanced"

    elif largest_percentage < 80:
        balance = "Moderately Imbalanced"

    else:
        balance = "Highly Imbalanced"

    print(f"Balance Status        : {balance}")






# Master Function for Categorical Feature Analysis

def analyze_categorical_feature(df, column):
    """
    Perform a complete univariate analysis
    for a categorical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    column : str
        Categorical feature name.
    """

    if column not in df.columns:
        raise ValueError(f"'{column}' is not a valid column in the DataFrame.")

    if pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"'{column}' is not a categorical feature.")

    print("\n")
    print("=" * 80)
    print(f"CATEGORICAL FEATURE ANALYSIS : {column.upper()}")
    print("=" * 80)

    display_basic_info(df, column)
    print("\n")

    display_frequency_table(df, column)
    print("\n")

    plot_countplot(df, column)
    print("\n")

    analyze_category_balance(df, column)

    print("\n")
    print("=" * 80)
    print("Analysis Completed")
    print("=" * 80)