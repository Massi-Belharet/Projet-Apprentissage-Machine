# Projet Apprentissage Machine — Analyse des pays européens

Analyse socio-économique de 27 pays européens à partir de 7 indicateurs.  

## Données

Le fichier `data/pays.txt` contient 27 pays européens avec 7 variables :
- `esp_vie_F` : espérance de vie des femmes
- `mort_inf` : mortalité infantile
- `activ_F` : taux d'activité féminine
- `chom` : taux de chômage
- `pnb_hb` : PNB par habitant
- `education` : dépenses en éducation (% du PNB)
- `sante` : dépenses en santé (% du PNB)

## Structure

```
Projet IA/
├── data/               # données brutes
├── notebooks/          # notebook principal (analyse.ipynb)
├── src/                # fonctions Python
│   ├── load_data.py    # chargement des données
│   ├── preprocessing.py # normalisation et ACP
│   ├── eda.py          # visualisations
│   └── clustering.py   # CAH et K-means
└── requirements.txt    # dépendances
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer le notebook

```bash
jupyter notebook notebooks/analyse.ipynb
```
