import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_results():
    # 1. Load the JSON results
    try:
        with open("stress_test_dl_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: stress_test_dl_results.json not found. Run the stress test first!")
        return

    # 2. Convert to DataFrame for easier plotting
    data = []
    for scenario, metrics in results.items():
        data.append({
            "Scenario": scenario,
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0)
        })
    
    df = pd.DataFrame(data)

    # 3. Plotting
    # Set a professional style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Create Bar Plot
    # We focus on Accuracy for the main impact, but you could plot others
    ax = sns.barplot(
        data=df, 
        x="Accuracy", 
        y="Scenario", 
        palette="viridis", 
        hue="Scenario", 
        legend=False
    )

    # Add labels
    plt.title("Impact of Signal Noise on Hybrid IDS Performance", fontsize=16, weight='bold')
    plt.xlabel("Model Accuracy", fontsize=12)
    plt.ylabel("")  # Hide y-label as scenario names are descriptive
    plt.xlim(0, 1.0)

    # Add value numbers to the bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=5)

    # 4. Save
    plt.tight_layout()
    plt.savefig("presentation_stress_test.png", dpi=300)
    print("Success! Created 'presentation_stress_test.png' for your slides.")
    plt.show()

if __name__ == "__main__":
    plot_results()