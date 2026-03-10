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
    # Step 6 - Check Multicollinearity (Optional Output here)
    # Note: the prompt just showed seaborn heatmap, but we'll print highly correlated pairs for CLI usage.
    numeric_features = [
        "income_inr_month", "household_size", "education_level",
        "monthly_fuel_cost_inr", "fuel_access_score", "distance_to_market_km",
        "awareness_score", "cook_hours_per_day", "health_concern_score",
        "number_of_children_under5"
    ]
    corr_matrix = df[numeric_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(upper.columns[x], upper.index[y]) for x, y in zip(*np.where(upper > 0.85))]
    if high_corr:
        print("WARNING: Highly correlated features found (r > 0.85):", high_corr)

    # Step 7 - Train / Test Split (Before Scaling)
    print("Splitting and Scaling data...")
    X = df.drop(columns=["adoption_label"])
    y = df["adoption_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y      
    )

    # Step 5 & 7 continued - Feature Scaling with StandardScaler
    scale_cols = [
        "income_inr_month",
        "household_size",
        "education_level",
        "monthly_fuel_cost_inr",
        "fuel_access_score",
        "distance_to_market_km",
        "awareness_score",
        "cook_hours_per_day",
        "health_concern_score",
        "number_of_children_under5"
    ]

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print(f"Train size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}")
    
    # Step 8 - Handle Class Imbalance (Check balance)
    print("Training set class balance:\n", y_train.value_counts(normalize=True))

    # Step 9 - Final Validation Checks
    assert X_train.isnull().sum().sum() == 0,  "Nulls in train set"
    assert X_test.isnull().sum().sum() == 0,   "Nulls in test set"
    assert np.isinf(X_train.values).sum() == 0, "Infinite values in train set"
    assert all(X_train.dtypes != "object"),    "Non-numeric columns remain"
    assert set(y.unique()) == {0, 1},          "Target is not binary"

    print("All checks passed. Ready for logistic regression.")

    # Step 10 - Save Cleaned Data
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train_clean.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test_clean.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Cleaned datasets saved in '{output_dir}'.")

if __name__ == "__main__":
    clean_data()