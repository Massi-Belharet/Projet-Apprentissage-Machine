import re
import pandas as pd


def load_data(filepath):
    """
    Charge et parse le fichier pays.txt.

    Params:
        filepath (str): chemin vers le fichier texte

    Returns:
        pd.DataFrame: DataFrame avec les pays en index et 7 variables en colonnes
    """
    rows = []
    with open(filepath, encoding="latin1") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(
                r"^(.+?)\s+([\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+)$",
                line
            )
            if match:
                pays = match.group(1).strip()
                valeurs = list(map(float, match.group(2).split()))
                rows.append([pays] + valeurs)

    df = pd.DataFrame(
        rows,
        columns=["pays", "esp_vie_F", "mort_inf", "activ_F", "chom", "pnb_hb", "education", "sante"]
    )
    return df.set_index("pays")
