# Nettoyage des données : températures en Afrique de l'Ouest

Projet de nettoyage de données (TP) réalisé avec Python et pandas, sur les températures mensuelles de cinq pays d'Afrique de l'Ouest.

**Auteurs :** LUBUNGU Pacifique et DJOSSOU Benoît

## Objectif

Préparer un jeu de données propre pour étudier l'**évolution des températures** en Afrique de l'Ouest, en documentant chaque transformation : ce qui a été fait, et pourquoi ce choix plutôt qu'un autre.

> L'objectif n'est pas d'obtenir un fichier sans aucun vide, mais un fichier dont chaque transformation est expliquée.

## Données

- **Source :** [Climate Change: Earth Surface Temperature Data (Berkeley Earth, Kaggle)](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
- **Fichier utilisé :** `GlobalLandTemperaturesByCountry.csv` uniquement
- **Colonnes :**

| Colonne | Description |
|---|---|
| `dt` | Date (premier jour du mois) |
| `AverageTemperature` | Température moyenne mensuelle (°C) |
| `AverageTemperatureUncertainty` | Incertitude de la mesure (°C) |
| `Country` | Pays |

- **Pays retenus :** Nigeria, Ghana, Sénégal (`Senegal`), Mali, Côte d'Ivoire

Le fichier d'origine n'est pas inclus dans ce dépôt (taille et licence). Il faut le télécharger depuis Kaggle, voir la section « Utilisation ».

## Problèmes traités

| Problème | Traitement | Justification |
|---|---|---|
| Dates stockées en texte | Conversion avec `pd.to_datetime`, ajout des colonnes `annee` et `mois` | Conversion sans perte, nécessaire pour trier et agréger dans le temps |
| Un pays sous deux noms (`Ivory Coast` / `Côte D'Ivoire`) | Harmonisation par dictionnaire d'alias | On unifie sans supprimer, pour ne pas perdre d'historique |
| Doublons | Pour chaque couple (pays, date), on garde la ligne la mieux renseignée | Évite de perdre une vraie mesure au profit d'un `NaN` |
| Mesures anciennes manquantes | Interpolation linéaire des trous de 2 mois au plus, par pays, puis suppression du reste | Ces `NaN` correspondent à une absence réelle de mesure : inventer des décennies entières fabriquerait de fausses tendances |
| Incertitudes élevées sur les premières années | Suppression des mesures dont l'incertitude dépasse un seuil (1,5 °C par défaut) | Critère objectif, propre à chaque pays, plutôt qu'une année arbitraire |
| Valeurs aberrantes | Suppression des valeurs physiquement impossibles (hors de -10 à 50 °C) ; valeurs extrêmes détectées par IQR signalées mais conservées | Les extrêmes réels sont de l'information pour une étude du climat |

La détection des valeurs aberrantes par la méthode de l'écart interquartile (IQR) est calculée **par pays et par mois** : un calcul global n'aurait pas de sens à cause du cycle saisonnier.

## Contenu du notebook

1. **État des lieux** avant toute correction : dimensions, types, valeurs manquantes (par pays et par décennie), doublons, valeurs aberrantes
2. **Corrections justifiées**, une par problème, avec mesure de l'impact avant chaque suppression
3. **État final** : les mêmes indicateurs recalculés, tableau avant/après et graphiques
4. **Export** du fichier nettoyé

Un tableau `bilan` récapitule les lignes perdues à chaque étape.

## Structure du dépôt

```
.
├── nettoyage_temperatures_afrique_ouest.ipynb   # notebook commenté en français
├── README.md
├── requirements.txt                             # dépendances (voir ci-dessous)
├── .gitignore                                   # ignore les CSV volumineux
├── output
    └── output/donnees_nettoyees.csv
└── data/
    ├── GlobalLandTemperaturesByCountry.csv      # à télécharger (non versionné)
    └── donnees_nettoyees.csv                    # produit par le notebook

```
## Utilisation

1. Cloner le dépôt :
   ```bash
   git clone <url-du-depot>
   cd <nom-du-depot>
   ```
2. Installer les dépendances :
   ```bash
   pip install pandas numpy matplotlib jupyter
   ```
3. Télécharger `GlobalLandTemperaturesByCountry.csv` depuis [Kaggle](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data) et le placer dans le dossier `data/`.
4. Adapter les chemins en haut du notebook (section « Paramètres ») :
   ```python
   CHEMIN_CSV = "data/GlobalLandTemperaturesByCountry.csv"
   CHEMIN_SORTIE = "output/donnees_nettoyees.csv"
   ```
5. Lancer le notebook :
   ```bash
   jupyter notebook nettoyage_temperatures_afrique_ouest.ipynb
   ```
   puis « Run All ». Le fichier `donnees_nettoyees.csv` est généré à la fin.

## Paramètres modifiables

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `PAYS_CIBLES` | 5 pays d'Afrique de l'Ouest | Périmètre de l'étude |
| `ALIAS` | variantes de la Côte d'Ivoire | Harmonisation des noms de pays |
| `LIMITE_INTERPOLATION` | 2 mois | Longueur maximale d'un trou comblé |
| `SEUIL_INCERTITUDE` | 1,5 °C | Seuil au-delà duquel une mesure est retirée |
| `TEMP_MIN`, `TEMP_MAX` | -10 °C, 50 °C | Bornes physiquement plausibles |

Le seuil d'incertitude est la décision qui a le plus d'impact sur la quantité de données conservée. Le notebook affiche la perte pour plusieurs seuils afin d'aider à le choisir.

## Fichier nettoyé

`donnees_nettoyees.csv` (encodage `utf-8-sig`, lisible avec Excel) contient :

| Colonne | Description |
|---|---|
| `Country` | Pays (noms harmonisés) |
| `dt` | Date (type datetime) |
| `annee`, `mois` | Année et mois extraits de `dt` |
| `AverageTemperature` | Température moyenne mensuelle (°C) |
| `AverageTemperatureUncertainty` | Incertitude (°C) |
| `temp_interpolee` | `True` si la valeur a été reconstituée par interpolation |
| `aberrante_iqr` | `True` si la valeur est extrême selon la méthode IQR (conservée) |

## Résultats

> À compléter après exécution sur le vrai fichier : nombre de lignes avant/après, part de données perdue par étape, première année exploitable pour chaque pays.

| Indicateur | Avant | Après |
|---|---|---|
| Nombre de lignes | | |
| Noms de pays distincts | | |
| Valeurs manquantes (`AverageTemperature`) | | |
| Doublons (pays, date) | | |
| Incertitude moyenne (°C) | | |

## Limites

- L'interpolation reste une estimation, même limitée à de courts trous.
- Le choix du seuil d'incertitude influence directement la période étudiée.
- Les valeurs extrêmes détectées par IQR sont conservées et simplement signalées.

## Licence et crédits

Les données appartiennent à Berkeley Earth et sont distribuées via Kaggle : consulter la page du jeu de données pour les conditions d'utilisation. Le code de ce dépôt est fourni à titre pédagogique.

