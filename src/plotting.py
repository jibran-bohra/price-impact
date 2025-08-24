import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_exploratory_data_analysis(analysis_data, data, output_dir):
    """
    Generates and saves exploratory data analysis plots.
    - Distribution of Signed Volume
    - Distribution of Price Changes
    - Scatter plot of Signed Volume vs Price Change
    - Time series of mid price
    """
    print("Generating exploratory data analysis plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Exploratory Data Analysis", fontsize=16)

    # Distribution of Signed Volume
    ax1 = axes[0, 0]
    signed_vol_clipped = np.clip(
        analysis_data["signed_volume"],
        analysis_data["signed_volume"].quantile(0.01),
        analysis_data["signed_volume"].quantile(0.99),
    )
    ax1.hist(signed_vol_clipped, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Signed Volume (clipped at 1% and 99% quantiles)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Signed Volume")

    # Price changes distribution
    ax2 = axes[0, 1]
    price_changes_clipped = np.clip(
        analysis_data["price_change"],
        analysis_data["price_change"].quantile(0.01),
        analysis_data["price_change"].quantile(0.99),
    )
    ax2.hist(
        price_changes_clipped, bins=50, edgecolor="black", alpha=0.7, color="orange"
    )
    ax2.set_xlabel("Price Change (clipped at 1% and 99% quantiles)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Price Changes")

    # Scatter plot: Signed Volume vs Price Change
    ax3 = axes[1, 0]
    sample_data = analysis_data.sample(min(5000, len(analysis_data)))
    ax3.scatter(
        sample_data["signed_volume"], sample_data["price_change"], alpha=0.3, s=10
    )
    ax3.set_xlabel("signed_volume")
    ax3.set_ylabel("Price Change")
    ax3.set_title("Signed Volume vs Price Change (sample)")
    ax3.set_xlim(
        analysis_data["signed_volume"].quantile(0.01),
        analysis_data["signed_volume"].quantile(0.99),
    )
    ax3.set_ylim(
        analysis_data["price_change"].quantile(0.01),
        analysis_data["price_change"].quantile(0.99),
    )

    # Time series of mid price
    ax4 = axes[1, 1]
    ax4.plot(data["ts_event"], data["mid_price"], linewidth=0.5)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Mid Price")
    ax4.set_title("Mid Price Over Time")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_exploratory_analysis.png")
    plt.close()


def plot_model_comparison(ow_model, afs_model, data, output_dir):
    """
    Generates and saves a comparison plot of the Linear OW and Nonlinear AFS models.
    """
    print("Generating main comparison plot...")
    volume_min = data["signed_volume"].quantile(0.05)
    volume_max = data["signed_volume"].quantile(0.95)
    volumes = np.linspace(volume_min, volume_max, 1000)

    # Calculate impacts
    ow_impacts_bps = ow_model.calculate_impact(volumes) * 10000
    afs_impacts_bps = afs_model.calculate_impact(volumes) * 10000

    plt.figure(figsize=(14, 8))
    plt.plot(
        volumes, ow_impacts_bps, "b-", linewidth=2, label="Linear OW Model", alpha=0.8
    )
    plt.plot(
        volumes,
        afs_impacts_bps,
        "r-",
        linewidth=2,
        label="Nonlinear AFS Model",
        alpha=0.8,
    )
    plt.fill_between(
        volumes,
        ow_impacts_bps,
        afs_impacts_bps,
        alpha=0.2,
        color="gray",
        label="Model Difference",
    )
    plt.xlabel("signed_volume", fontsize=12)
    plt.ylabel("Price Impact (basis points)", fontsize=12)
    plt.title(
        "Price Impact Models Comparison: Linear OW vs Nonlinear AFS",
        fontsize=14,
        pad=20,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_model_comparison.png")
    plt.close()


def plot_detailed_analysis(ow_model, afs_model, analysis_data, output_dir):
    """
    Generates and saves detailed analysis plots for price impact models.
    - Impact curves with actual data overlay
    - Nonlinearity visualization
    - Impact distribution for typical volumes
    - Model comparison metrics
    """
    print("Generating detailed analysis plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Detailed Price Impact Analysis", fontsize=16)

    volume_min = analysis_data["signed_volume"].quantile(0.05)
    volume_max = analysis_data["signed_volume"].quantile(0.95)
    volumes = np.linspace(volume_min, volume_max, 1000)
    ow_impacts_bps = ow_model.calculate_impact(volumes) * 10000
    afs_impacts_bps = afs_model.calculate_impact(volumes) * 10000

    # Impact curves with actual data overlay
    ax1 = axes[0, 0]
    ax1.plot(volumes, ow_impacts_bps, "b-", linewidth=2, label="Linear OW Model")
    ax1.plot(volumes, afs_impacts_bps, "r-", linewidth=2, label="Nonlinear AFS Model")
    sample = analysis_data.sample(min(1000, len(analysis_data)))
    ax1.scatter(
        sample["signed_volume"],
        sample["price_change"] * 10000,
        alpha=0.2,
        s=5,
        c="gray",
        label="Actual Data",
    )
    ax1.set_xlabel("signed_volume")
    ax1.set_ylabel("Price Impact (bps)")
    ax1.set_title("Model Predictions vs Actual Data")
    ax1.legend()
    ax1.set_xlim(volume_min, volume_max)
    ax1.set_ylim(np.percentile(sample["price_change"] * 10000, [1, 99]))

    # Nonlinearity visualization
    ax2 = axes[0, 1]
    nonlinearity = afs_impacts_bps - ow_impacts_bps
    ax2.plot(volumes, nonlinearity, "g-", linewidth=2)
    ax2.fill_between(volumes, 0, nonlinearity, alpha=0.3, color="green")
    ax2.set_xlabel("signed_volume")
    ax2.set_ylabel("Nonlinearity (AFS - OW) in bps")
    ax2.set_title("Nonlinear Component of Price Impact")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Impact distribution for typical volumes
    ax3 = axes[1, 0]
    typical_volumes = analysis_data["signed_volume"].values
    typical_volumes_filtered = typical_volumes[
        (typical_volumes > np.percentile(typical_volumes, 5))
        & (typical_volumes < np.percentile(typical_volumes, 95))
    ]
    ow_typical_impacts = ow_model.calculate_impact(typical_volumes_filtered) * 10000
    afs_typical_impacts = afs_model.calculate_impact(typical_volumes_filtered) * 10000

    ax3.hist(ow_typical_impacts, bins=50, alpha=0.5, label="OW Model", density=True)
    ax3.hist(afs_typical_impacts, bins=50, alpha=0.5, label="AFS Model", density=True)
    ax3.set_xlabel("Price Impact (bps)")
    ax3.set_ylabel("Density")
    ax3.set_title("Distribution of Price Impact for Typical Volumes")
    ax3.legend()

    # Model comparison metrics
    ax4 = axes[1, 1]
    ax4.axis("off")
    metrics_text = f"""Model Comparison Metrics:

Linear OW Model:
  - Lambda: {ow_model.parameters["lambda"]:.2e}
  - R-squared: {ow_model.parameters["r_squared"]:.4f}
  - N samples: {ow_model.parameters["n_samples"]:,}

Nonlinear AFS Model:
  - Lambda: {afs_model.parameters["lambda"]:.2e}
  - Eta: {afs_model.parameters["eta"]:.2e}
  - Delta: {afs_model.parameters["delta"]}
  - R-squared: {afs_model.parameters["r_squared"]:.4f}
  - N samples: {afs_model.parameters["n_samples"]:,}

Improvement in R²: {((afs_model.parameters["r_squared"] - ow_model.parameters["r_squared"]) / ow_model.parameters["r_squared"] * 100):.1f}%
"""
    ax4.text(
        0.1,
        0.5,
        metrics_text,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_detailed_analysis.png")
    plt.close()


def plot_3d_impact_surfaces(ow_model, afs_model, data, output_dir):
    """
    Generates and saves 3D surface plots of the price impact for both models.
    """
    print("Generating 3D impact surface plots...")

    volume_min = data["signed_volume"].quantile(0.05)
    volume_max = data["signed_volume"].quantile(0.95)

    # Create meshgrid
    volume_range = np.linspace(volume_min, volume_max, 50)
    time_range = np.linspace(1, 60, 50)
    V, T = np.meshgrid(volume_range, time_range)

    # Calculate impacts
    OW_impact_surface = np.zeros_like(V)
    AFS_impact_surface = np.zeros_like(V)

    for i in range(len(time_range)):
        OW_impact_surface[i, :] = (
            ow_model.calculate_impact(volume_range, time_range[i]) * 10000
        )
        AFS_impact_surface[i, :] = (
            afs_model.calculate_impact(volume_range, time_range[i]) * 10000
        )

    fig = plt.figure(figsize=(16, 6))

    # OW Model surface
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(V, T, OW_impact_surface, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("signed_volume")
    ax1.set_ylabel("Time Interval (seconds)")
    ax1.set_zlabel("Price Impact (bps)")
    ax1.set_title("Linear OW Model: Impact Surface")
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # AFS Model surface
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(V, T, AFS_impact_surface, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("signed_volume")
    ax2.set_ylabel("Time Interval (seconds)")
    ax2.set_zlabel("Price Impact (bps)")
    ax2.set_title("Nonlinear AFS Model: Impact Surface")
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_3d_impact_surfaces.png")
    plt.close()


def plot_volume_bucket_analysis(ow_model, afs_model, analysis_data, output_dir):
    """
    Generates and saves a bar plot comparing model impacts across volume buckets.
    """
    print("Generating volume bucket analysis...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define volume buckets
    volume_buckets = np.percentile(
        analysis_data["signed_volume"], np.linspace(10, 90, 9)
    )
    bucket_centers = (volume_buckets[:-1] + volume_buckets[1:]) / 2

    # Calculate average impacts
    ow_bucket_impacts = ow_model.calculate_impact(bucket_centers) * 10000
    afs_bucket_impacts = afs_model.calculate_impact(bucket_centers) * 10000

    # Create bar plot
    x = np.arange(len(bucket_centers))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, ow_bucket_impacts, width, label="Linear OW", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, afs_bucket_impacts, width, label="Nonlinear AFS", alpha=0.8
    )

    ax.set_xlabel("Volume Percentile Buckets", fontsize=12)
    ax.set_ylabel("Average Price Impact (bps)", fontsize=12)
    ax.set_title("Price Impact by Volume Bucket: Model Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)}th" for p in np.linspace(10, 90, 9)[:-1]])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(bars1, ax)
    autolabel(bars2, ax)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_volume_bucket_analysis.png")
    plt.close()
