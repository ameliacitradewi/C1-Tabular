import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

data_df = pd.read_csv("/Users/ameliacitra/Documents/AI:ML/C1 - Tabular/insurance.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Insurance Dataset Overview", fontsize=18, fontweight="bold", y=0.98)

colors_two = ["#4C72B0", "#DD8452"]
colors_four = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def add_bar_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 8,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )


# 1 — Smoker vs Non-Smoker
ax = axes[0, 0]
smoker_counts = data_df["smoker"].value_counts()
bars = ax.bar(
    ["Non-Smoker (No)", "Smoker (Yes)"],
    [smoker_counts.get("no", 0), smoker_counts.get("yes", 0)],
    color=colors_two,
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
)
add_bar_labels(ax, bars)
ax.set_title("Smoker vs Non-Smoker", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Jumlah Orang")

# 2 — Male vs Female
ax = axes[0, 1]
sex_counts = data_df["sex"].value_counts()
bars = ax.bar(
    ["Pria (Male)", "Wanita (Female)"],
    [sex_counts.get("male", 0), sex_counts.get("female", 0)],
    color=colors_two,
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
)
add_bar_labels(ax, bars)
ax.set_title("Pria vs Wanita", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Jumlah Orang")

# 3 — Punya Anak vs Tidak
ax = axes[1, 0]
has_children = (data_df["children"] > 0).sum()
no_children = (data_df["children"] == 0).sum()
bars = ax.bar(
    ["Tidak Punya Anak", "Punya Anak"],
    [no_children, has_children],
    color=colors_two,
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
)
add_bar_labels(ax, bars)
ax.set_title("Punya Anak vs Tidak Punya Anak", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Jumlah Orang")

# 4 — Jumlah per Region
ax = axes[1, 1]
region_counts = data_df["region"].value_counts().sort_index()
bars = ax.bar(
    region_counts.index.str.title(),
    region_counts.values,
    color=colors_four,
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
)
add_bar_labels(ax, bars)
ax.set_title("Jumlah Orang per Region", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Jumlah Orang")

for ax in axes.flat:
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(
    "/Users/ameliacitra/Documents/AI:ML/C1 - Tabular/insurance_charts.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
print("Grafik tersimpan sebagai insurance_charts.png")
