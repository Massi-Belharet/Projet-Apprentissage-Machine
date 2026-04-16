import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_histbox(df):
    """
    Affiche histogramme + boxplot pour chaque variable.

    Params:
        df (pd.DataFrame): données brutes

    Returns:
        None
    """
    fig, axes = plt.subplots(7, 2, figsize=(14, 30))
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], bins=6, kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f"Histogramme — {col}")
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("Fréquence")

        axes[i, 1].boxplot(df[col], vert=True, patch_artist=True,
                           boxprops=dict(facecolor="steelblue", alpha=0.6))
        axes[i, 1].set_title(f"Boxplot — {col}")
        axes[i, 1].set_ylabel(col)

    plt.tight_layout()
    plt.show()


def show_outliers(df):
    """
    Affiche les outliers de chaque variable via la méthode IQR.

    Params:
        df (pd.DataFrame): données brutes

    Returns:
        None
    """
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            print(f"{col}: {outliers.index.tolist()} - valeurs: {outliers[col].tolist()}")


def plot_correlation(df):
    """
    Calcule et affiche la heatmap de corrélation.

    Params:
        df (pd.DataFrame): données brutes

    Returns:
        pd.DataFrame: matrice de corrélation
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()
    return corr_matrix


def plot_pairplot(df):
    """
    Affiche le pairplot des variables.

    Params:
        df (pd.DataFrame): données brutes

    Returns:
        None
    """
    sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.6, "color": "steelblue"})
    plt.suptitle("Pairplot — relations entre variables", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_screeplot(variance_ratio):
    """
    Affiche le scree plot (variance expliquée par composante).

    Params:
        variance_ratio (np.ndarray): variance expliquée par composante (%)

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.bar([f"PC{i+1}" for i in range(len(variance_ratio))], variance_ratio,
            color="steelblue", alpha=0.8)
    plt.plot([f"PC{i+1}" for i in range(len(variance_ratio))], variance_ratio,
             marker="o", color="black", linewidth=2)
    for i, v in enumerate(variance_ratio):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    plt.title("Scree Plot")
    plt.xlabel("Composantes principales")
    plt.ylabel("Pourcentage de variance expliquée")
    plt.tight_layout()
    plt.show()


def plot_contributions(contributions):
    """
    Affiche les contributions des variables à PC1 et PC2.

    Params:
        contributions (pd.DataFrame): contributions PC1 (%) et PC2 (%) par variable

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pc in zip(axes, ["Contribution PC1 (%)", "Contribution PC2 (%)"]):
        contrib = contributions[pc].sort_values(ascending=False)
        ax.bar(contrib.index, contrib.values, color="steelblue", alpha=0.8)
        ax.axhline(y=100/7, color="red", linestyle="--",
                   label=f"Contribution moyenne ({100/7:.1f}%)")
        ax.set_title(pc)
        ax.set_ylabel("Contribution (%)")
        ax.set_xlabel("Variables")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cercle(loadings_corr, columns, variance_ratio):
    """
    Affiche le cercle des corrélations (ACP).

    Params:
        loadings_corr (np.ndarray): corrélations variables/axes
        columns (pd.Index): noms des variables
        variance_ratio (np.ndarray): variance expliquée par composante (%)

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--"))
    for i, var in enumerate(columns):
        ax.arrow(0, 0, loadings_corr[i, 0], loadings_corr[i, 1],
                 head_width=0.03, head_length=0.03, fc="steelblue", ec="steelblue")
        ax.text(loadings_corr[i, 0] * 1.1, loadings_corr[i, 1] * 1.1,
                var, fontsize=11, ha="center", color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.1f}%)")
    ax.set_title("Cercle des corrélations")
    plt.tight_layout()
    plt.show()


def plot_plan_factoriel(components, df_index, variance_ratio):
    """
    Affiche le plan factoriel des individus (sans clusters).

    Params:
        components (np.ndarray): coordonnées ACP des individus
        df_index (pd.Index): noms des pays
        variance_ratio (np.ndarray): variance expliquée par composante (%)

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(components[:, 0], components[:, 1], color="steelblue", s=50)
    for i, pays in enumerate(df_index):
        ax.annotate(pays, (components[i, 0], components[i, 1]), fontsize=9,
                    ha="center", va="bottom")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.1f}%)")
    ax.set_title("Plan factoriel des individus")
    plt.tight_layout()
    plt.show()


def plot_clusters(components, df, variance_ratio, labels_name):
    """
    Affiche le plan factoriel avec les clusters colorés.

    Params:
        components (np.ndarray): coordonnées ACP des individus
        df (pd.DataFrame): données avec colonne 'cluster'
        variance_ratio (np.ndarray): variance expliquée par composante (%)
        labels_name (list): noms des clusters

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    for i in range(len(labels_name)):
        mask = df["cluster"] == i
        ax.scatter(components[mask, 0], components[mask, 1],
                   label=labels_name[i], s=80)
        for pays in df[mask].index:
            idx = df.index.get_loc(pays)
            ax.annotate(pays, (components[idx, 0], components[idx, 1]), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.1f}%)")
    ax.set_title("Clusters K-means sur le plan factoriel")
    ax.legend()
    plt.tight_layout()
    plt.show()
