import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


def run_cah(components, df_index):
    """
    Applique la CAH et affiche le dendrogramme.

    Params:
        components (np.ndarray): coordonnées ACP des individus
        df_index (pd.Index): noms des pays

    Returns:
        np.ndarray: matrice de linkage
    """
    linked = linkage(components, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(linked, labels=df_index.tolist(), orientation="top",
               distance_sort="descending", leaf_rotation=90)
    plt.title("Dendrogramme — CAH (critère de Ward)")
    plt.xlabel("Pays")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    return linked


def compute_silhouette(components, k_range=range(2, 6)):
    """
    Calcule le score de silhouette pour chaque valeur de k.

    Params:
        components (np.ndarray): coordonnées ACP des individus
        k_range (range): valeurs de k à tester (défaut: 2 à 5)

    Returns:
        dict: {k: score_silhouette}
    """
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(components)
        scores[k] = silhouette_score(components, labels)
        print(f"k={k} : silhouette = {scores[k]:.3f}")
    return scores


def plot_silhouette(scores):
    """
    Affiche le graphique des scores de silhouette selon k.

    Params:
        scores (dict): {k: score_silhouette}

    Returns:
        None
    """
    plt.figure(figsize=(8, 4))
    plt.plot(scores.keys(), scores.values(), marker="o")
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Score de silhouette")
    plt.title("Score de silhouette selon k")
    plt.xticks(list(scores.keys()))
    plt.tight_layout()
    plt.show()


def run_kmeans(components, k=3):
    """
    Applique le K-means avec k clusters.

    Params:
        components (np.ndarray): coordonnées ACP des individus
        k (int): nombre de clusters (défaut=3)

    Returns:
        np.ndarray: labels des clusters pour chaque individu
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans.fit_predict(components)
