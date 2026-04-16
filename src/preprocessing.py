import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def normalize(df):
    """
    Centre et réduit les variables (StandardScaler).

    Params:
        df (pd.DataFrame): données brutes

    Returns:
        pd.DataFrame: données normalisées (moyenne=0, écart-type=1)
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


def run_pca(df_scaled, n_components=2):
    """
    Applique une ACP sur les données normalisées.

    Params:
        df_scaled (pd.DataFrame): données normalisées
        n_components (int): nombre de composantes à retenir (défaut=2)

    Returns:
        pca: ACP complète (toutes les composantes)
        pca2: ACP réduite (n_components)
        components (np.ndarray): coordonnées des individus sur les axes retenus
        variance_ratio (np.ndarray): variance expliquée par composante (%)
        eigen_df (pd.DataFrame): tableau valeurs propres et variance
        loadings_corr (np.ndarray): corrélations variables/axes (cercle)
    """
    pca = PCA()
    pca.fit(df_scaled)
    variance_ratio = pca.explained_variance_ratio_ * 100
    eigenvalues = pca.explained_variance_
    cumulative_variance = np.cumsum(variance_ratio)

    eigen_df = pd.DataFrame({
        "Valeur propre": eigenvalues,
        "Variance expliquée (%)": variance_ratio,
        "Variance cumulée (%)": cumulative_variance
    }, index=[f"PC{i+1}" for i in range(len(eigenvalues))])

    pca2 = PCA(n_components=n_components)
    components = pca2.fit_transform(df_scaled)
    loadings_corr = pca2.components_.T * np.sqrt(pca2.explained_variance_)

    return pca, pca2, components, variance_ratio, eigen_df, loadings_corr


def compute_contributions(pca2, df_scaled):
    """
    Calcule la contribution (%) de chaque variable à PC1 et PC2.

    Params:
        pca2: ACP réduite (fitted)
        df_scaled (pd.DataFrame): données normalisées (pour récupérer les noms de colonnes)

    Returns:
        pd.DataFrame: contributions PC1 (%) et PC2 (%) par variable
    """
    loadings = pca2.components_
    contrib_pc1 = (loadings[0] ** 2) / (loadings[0] ** 2).sum() * 100
    contrib_pc2 = (loadings[1] ** 2) / (loadings[1] ** 2).sum() * 100
    return pd.DataFrame({
        "Contribution PC1 (%)": contrib_pc1,
        "Contribution PC2 (%)": contrib_pc2
    }, index=df_scaled.columns)
