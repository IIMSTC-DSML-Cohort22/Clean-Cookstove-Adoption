import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(input_path="Data/india_cookstove_survey_2000.csv", output_dir="Data"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(_file_), '..'))
    os.chdir(project_root)
    
    # Step 1 - Load and Inspect
    print("Loading data...")
    df = pd.read_csv(input_path)

    print("Initial Shape:", df.shape)
    
    # Drop household_id
    df.drop(columns=["household_id"], inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    assert set(df["adoption_label"].unique()) == {0, 1}, "adoption_label must only contain 0 and 1"

    # Step 2 - Handle Missing Values
    print("Handling missing values...")
    continuous_cols = [
        "income_inr_month", "monthly_fuel_cost_inr",
        "distance_to_market_km", "cook_hours_per_day",
        "fuel_access_score", "awareness_score", "health_concern_score"
    ]
    integer_cols = ["household_size", "education_level", "number_of_children_under5"]
    binary_cols = [
        "has_electricity", "women_decision_maker",
        "bpl_card_holder", "prior_subsidy_received"
    ]
    categorical_cols = ["state", "zone", "region_type", "current_fuel_type"]

    for col in continuous_cols + integer_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in binary_cols + categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Step 3 - Cap Outliers (Winsorization)
    print("Capping outliers...")
    def cap_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return df

    for col in ["income_inr_month", "monthly_fuel_cost_inr",
                "distance_to_market_km", "cook_hours_per_day"]:
        df = cap_outliers(df, col)

    # Step 4 - Encode Categorical Variables
    print("Encoding categorical variables...")
    df = pd.get_dummies(
        df,
        columns=["zone", "region_type", "current_fuel_type", "state"],
        drop_first=True,
        dtype=int
    )