import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path="Data/india_cookstove_survey_2000.csv", output_dir="plots/eda"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"File {input_path} not found. Trying root directory...")
        df = pd.read_csv("india_cookstove_survey_2000.csv")

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Target Variable Distribution
    print("Plotting Target Variable Distribution...")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="adoption_label", palette="Set2")
    plt.title("Distribution of Adoption Label")
    plt.xticks(ticks=[0, 1], labels=["Unlikely (0)", "Likely (1)"])
    plt.savefig(os.path.join(output_dir, "01_target_distribution.png"), bbox_inches="tight")
    plt.close()

    # 2. Income vs Adoption
    print("Plotting Income vs Adoption...")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="adoption_label", y="income_inr_month", palette="Set2")
    plt.title("Monthly Income (INR) vs Adoption Label")
    plt.xticks(ticks=[0, 1], labels=["Unlikely", "Likely"])
    plt.savefig(os.path.join(output_dir, "02_income_vs_adoption.png"), bbox_inches="tight")
    plt.close()

    # 3. Zone-wise Adoption Rate
    print("Plotting Zone-wise Adoption...")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="zone", y="adoption_label", palette="coolwarm")
    plt.title("Adoption Rate by Zone")
    plt.ylabel("Adoption Rate")
    plt.savefig(os.path.join(output_dir, "03_zone_adoption_rate.png"), bbox_inches="tight")
    plt.close()

    # 4. Region Type vs Adoption
    print("Plotting Region Type vs Adoption...")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="region_type", y="adoption_label", palette="muted")
    plt.title("Adoption Rate by Region Type")
    plt.ylabel("Adoption Rate")
    plt.savefig(os.path.join(output_dir, "04_region_type_adoption.png"), bbox_inches="tight")
    plt.close()
    
    # 5. Current Fuel Type vs Adoption
    print("Plotting Fuel Type vs Adoption...")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="current_fuel_type", hue="adoption_label", palette="Set2")
    plt.title("Current Fuel Type by Adoption Status")
    plt.xticks(rotation=30, ha="right")
    plt.savefig(os.path.join(output_dir, "05_fuel_type_vs_adoption.png"), bbox_inches="tight")
    plt.close()

    # 6. Awareness Score vs Adoption
    print("Plotting Awareness Score vs Adoption...")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="adoption_label", y="awareness_score", palette="Set2")
    plt.title("Awareness Score by Adoption Label")
    plt.xticks(ticks=[0, 1], labels=["Unlikely", "Likely"])
    plt.savefig(os.path.join(output_dir, "06_awareness_vs_adoption.png"), bbox_inches="tight")
    plt.close()

    # 7. Distance to Market vs Adoption
    print("Plotting Distance to Market vs Adoption...")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="distance_to_market_km", hue="adoption_label",
                 kde=True, palette="Set2", element="step")
    plt.title("Distance to Market by Adoption Status")
    plt.savefig(os.path.join(output_dir, "07_distance_vs_adoption.png"), bbox_inches="tight")
    plt.close()

    # 8. Correlation Heatmap — fix: removed fmt when annot=False
    print("Plotting Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.savefig(os.path.join(output_dir, "08_correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    # 9. Prior Subsidy vs Adoption
    print("Plotting Prior Subsidy Received...")
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="prior_subsidy_received", y="adoption_label", palette="Set3")
    plt.title("Adoption Rate by Prior Subsidy Received")
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
    plt.ylabel("Adoption Rate")
    plt.savefig(os.path.join(output_dir, "09_prior_subsidy_received.png"), bbox_inches="tight")
    plt.close()

    print(f"EDA complete. All plots saved to '{output_dir}'.")

if __name__ == "__main__":
    run_eda()