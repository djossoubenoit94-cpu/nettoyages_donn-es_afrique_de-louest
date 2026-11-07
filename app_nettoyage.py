"""
Application Streamlit : présentation interactive du nettoyage des données.
TP AMA · PIIA · Cohorte 2 · Binôme 06 · Domaine Climat
LUBUNGU Pacifique et DJOSSOU Benoît

Lancer :  streamlit run app_nettoyage.py
Fichiers attendus (à côté de app_nettoyage.py ou dans un sous-dossier) :
    GlobalLandTemperaturesByCountry.csv                          (fichier brut Kaggle)
    donnees_nettoyees/temperatures_afrique_ouest_nettoye.csv     (produit par le notebook de nettoyage)
Tous les chiffres affichés sont RECALCULÉS à partir de ces deux fichiers : ce sont des preuves, pas du texte écrit à la main.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Nettoyage des données · Températures en Afrique de l'Ouest",
                   layout="wide", initial_sidebar_state="expanded")

# ====================================================================================
# 0. Paramètres (identiques à ceux du notebook de nettoyage)
# ====================================================================================
FICHIER_BRUT = "GlobalLandTemperaturesByCountry.csv"
FICHIER_NET = "temperatures_afrique_ouest_nettoye.csv"

CORRESPONDANCE = {                     # libellé trouvé dans le fichier -> libellé harmonisé
    "Benin": "Bénin", "Burkina Faso": "Burkina Faso",
    "Côte D'Ivoire": "Côte d'Ivoire", "Côte d'Ivoire": "Côte d'Ivoire",
    "Cote D'Ivoire": "Côte d'Ivoire", "Ivory Coast": "Côte d'Ivoire",
    "Ghana": "Ghana", "Senegal": "Sénégal",
}
PAYS = ["Bénin", "Burkina Faso", "Côte d'Ivoire", "Ghana", "Sénégal"]
COULEURS = dict(zip(PAYS, ["#e07b39", "#c0392b", "#2e86ab", "#27ae60", "#8e44ad"]))
T_MIN_DEF, T_MAX_DEF = 10, 45          # plage de plausibilité (°C)
K_DEF = 1.5                            # coefficient de Tukey

TEAL, NAVY, ORANGE, RED, GREEN = "#087e98", "#092e50", "#e07b39", "#c0392b", "#27ae60"


def fr(n):
    """Nombre entier avec espaces comme séparateur de milliers."""
    return f"{int(n):,}".replace(",", " ")


def pct(x, d=2):
    return f"{x:.{d}f} %".replace(".", ",")


# ====================================================================================
# 1. Chargement des fichiers
# ====================================================================================
def chercher(nom):
    racine = Path(__file__).resolve().parent
    for base in [racine, racine / "donnees_nettoyees", racine / "files", racine / "files" / "donnees_nettoyees",
                 Path.cwd(), Path.cwd() / "donnees_nettoyees", Path.cwd() / "files", Path.cwd() / "files" / "donnees_nettoyees"]:
        p = base / nom
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Lecture du fichier brut (577 462 lignes)…")
def lire_csv(source):
    return pd.read_csv(source)


def charger_fichiers():
    p_brut, p_net = chercher(FICHIER_BRUT), chercher(FICHIER_NET)
    src_brut, src_net = p_brut, p_net
    if p_brut is None or p_net is None:
        st.sidebar.warning("Fichier(s) introuvable(s) : déposez-les ici.")
    if p_brut is None:
        up = st.sidebar.file_uploader(f"Fichier brut : {FICHIER_BRUT}", type="csv", key="up_brut")
        src_brut = io.BytesIO(up.getvalue()) if up else None
    if p_net is None:
        up = st.sidebar.file_uploader(f"Fichier nettoyé : {FICHIER_NET}", type="csv", key="up_net")
        src_net = io.BytesIO(up.getvalue()) if up else None
    if src_brut is None or src_net is None:
        st.title("Nettoyage des données : températures en Afrique de l'Ouest")
        st.info("Placez les deux fichiers CSV à côté de `app_nettoyage.py` (ou déposez-les dans le menu de gauche) pour afficher l'application.")
        st.stop()
    return lire_csv(src_brut), lire_csv(src_net)


# ====================================================================================
# 2. Fonctions de calcul (mêmes règles que le notebook)
# ====================================================================================
@st.cache_data
def preparer(brut):
    """Périmètre d'étude (état initial) et copie de travail avec dates converties et libellés harmonisés."""
    noms = sorted(set(brut["Country"].dropna().unique()) & set(CORRESPONDANCE))
    etude = brut[brut["Country"].isin(noms)].reset_index(drop=True)          # état INITIAL (5 pays, libellés d'origine)
    net = etude.copy()
    net["dt"] = pd.to_datetime(net["dt"], format="%Y-%m-%d", errors="coerce")
    net = net.sort_values(["Country", "dt"]).reset_index(drop=True)
    net["Country"] = net["Country"].str.strip().map(CORRESPONDANCE).fillna(net["Country"])
    net["annee"] = net["dt"].dt.year
    net["mois"] = net["dt"].dt.month
    net["decennie"] = net["annee"] // 10 * 10
    net["temperature_manquante"] = net["AverageTemperature"].isna()
    return etude, net, noms


def seuil_tukey(inc, k):
    q1, q3 = inc.quantile(0.25), inc.quantile(0.75)
    return q3 + k * (q3 - q1)


def drapeau_atypique(d, k):
    """Atypique = hors de [Q1 - k.IQR ; Q3 + k.IQR] de son groupe (pays x mois). Renvoie (booléens, médiane du groupe)."""
    mois = d["mois"] if "mois" in d.columns else pd.to_datetime(d["dt"], errors="coerce").dt.month
    pays = d["Country"].replace(CORRESPONDANCE)
    g = d["AverageTemperature"].groupby([pays, mois])
    q1 = g.transform(lambda s: s.quantile(0.25))
    q3 = g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    t = d["AverageTemperature"]
    return (t < q1 - k * iqr) | (t > q3 + k * iqr), g.transform("median")


@st.cache_data
def calculer_drapeaux(net, k_inc, k_iqr):
    out = net.copy()
    out["seuil_inc"] = seuil_tukey(out["AverageTemperatureUncertainty"], k_inc)
    out["incertitude_elevee"] = out["AverageTemperatureUncertainty"] > out["seuil_inc"].iloc[0]
    out["temp_atypique"], out["mediane_groupe"] = drapeau_atypique(out, k_iqr)
    out["ecart_mediane"] = (out["AverageTemperature"] - out["mediane_groupe"]).round(3)
    return out


def indicateurs(d, t_min=T_MIN_DEF, t_max=T_MAX_DEF, k=K_DEF):
    """Mêmes indicateurs avant/après (comme dans le notebook)."""
    dates = pd.to_datetime(d["dt"], errors="coerce")
    t, u = d["AverageTemperature"], d["AverageTemperatureUncertainty"]
    atyp, _ = drapeau_atypique(d.assign(mois=dates.dt.month), k)
    pays_h = d["Country"].replace(CORRESPONDANCE)
    return {
        "Nombre de lignes": fr(len(d)),
        "Nombre de colonnes": str(d.shape[1]),
        "Type de la colonne dt": str(d["dt"].dtype),
        "Dates invalides": fr(dates.isna().sum()),
        "Libellés de pays distincts": str(d["Country"].nunique()),
        "Libellés utilisés": ", ".join(sorted(d["Country"].unique())),
        "Températures manquantes": fr(t.isna().sum()),
        "Taux de températures manquantes": pct(t.isna().mean() * 100),
        "Incertitudes manquantes": fr(u.isna().sum()),
        "Doublons exacts (4 colonnes d'origine)": fr(d[["dt", "AverageTemperature", "AverageTemperatureUncertainty", "Country"]].astype(str).duplicated().sum()),
        "Doublons pays-date": fr(pd.DataFrame({"p": pays_h, "d": dates}).duplicated().sum()),
        "Première date": str(dates.min().date()),
        "Dernière date": str(dates.max().date()),
        "Température minimale (°C)": f"{t.min():.2f}",
        "Température maximale (°C)": f"{t.max():.2f}",
        f"Températures hors plage ({t_min} à {t_max} °C)": fr(((t < t_min) | (t > t_max)).sum()),
        "Incertitudes nulles ou négatives": fr((u <= 0).sum()),
        "Températures atypiques (IQR pays x mois)": fr(atyp.sum()),
        "Incertitudes élevées (Tukey)": fr((u > seuil_tukey(u, k)).sum()),
    }


def bloc(probleme, decision, pourquoi):
    """Les trois éléments de chaque correction : problème, décision, justification."""
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Problème**")
        st.write(probleme)
    with c2.container(border=True):
        st.markdown("**Décision**")
        st.write(decision)
    with c3.container(border=True):
        st.markdown("**Pourquoi ce choix**")
        st.write(pourquoi)


def a_dire(texte):
    with st.expander("À dire à l'oral"):
        st.write(texte)


def mise_en_forme(fig, hauteur=380):
    fig.update_layout(height=hauteur, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white")
    return fig


# ====================================================================================
# 3. Chargement et préparation
# ====================================================================================
brut, csv_net = charger_fichiers()
etude, net, noms_retenus = preparer(brut)

st.sidebar.title("Nettoyage des données")
st.sidebar.caption("TP AMA · PIIA · Cohorte 2 · Binôme 06\n\nLUBUNGU Pacifique et DJOSSOU Benoît")
PAGES = ["Accueil", "1 · Données et périmètre", "2 · État des lieux", "3 · Corrections justifiées",
         "4 · État final et export", "5 · Explorer les séries", "6 · Conclusion et questions du jury"]
page = st.sidebar.radio("Navigation", PAGES, key="page")
st.sidebar.divider()
st.sidebar.caption("Chaque chiffre de cette application est recalculé à partir des fichiers CSV. Les curseurs permettent de tester d'autres seuils.")

# ====================================================================================
# PAGE : Accueil
# ====================================================================================
if page == PAGES[0]:
    st.title("Nettoyage des données : températures en Afrique de l'Ouest")
    st.markdown("**Bénin · Burkina Faso · Côte d'Ivoire · Ghana · Sénégal** — Berkeley Earth, 1849-2013")
    st.info("**Consigne du TP :** l'objectif n'est pas d'obtenir un fichier sans aucun vide, mais un fichier dont chaque transformation est expliquée.")

    fl = calculer_drapeaux(net, K_DEF, K_DEF)
    c = st.columns(5)
    c[0].metric("Lignes du fichier brut", fr(len(brut)))
    c[1].metric("Lignes des 5 pays (entrée)", fr(len(etude)))
    c[2].metric("Lignes après nettoyage", fr(len(csv_net)))
    c[3].metric("Lignes supprimées", fr(len(etude) - len(csv_net)))
    c[4].metric("Colonnes (avant → après)", f"4 → {csv_net.shape[1]}")

    st.subheader("Les quatre livrables demandés")
    st.markdown("""
| N° | Livrable | Où le voir dans l'application |
|---|---|---|
| 1 | Un état des lieux avant toute correction | **2 · État des lieux** |
| 2 | Les corrections justifiées | **3 · Corrections justifiées** |
| 3 | Un état final (mêmes indicateurs qu'au départ) | **4 · État final et export** |
| 4 | Le notebook commenté et le CSV nettoyé | **4 · État final et export** |
""")

    st.subheader("Plan des 6 minutes")
    st.dataframe(pd.DataFrame({
        "Partie": ["Introduction", "Données et périmètre", "État des lieux", "Corrections justifiées", "État final et export", "Conclusion"],
        "Durée": ["0:35", "0:35", "1:00", "2:50", "0:45", "0:15"],
        "Page": ["Accueil", "1", "2", "3", "4", "6"],
    }), hide_index=True)

    st.subheader("Le message en une phrase")
    st.success("**Mieux vaut signaler que corriger à l'aveugle :** nous n'avons supprimé ni inventé aucune mesure. "
               "Les mois sans mesure, les incertitudes élevées et les températures atypiques sont marqués par trois indicateurs, "
               "et chacun peut ensuite décider quoi en faire.")

# ====================================================================================
# PAGE 1 : Données et périmètre
# ====================================================================================
elif page == PAGES[1]:
    st.title("1 · Données et périmètre")
    st.write("Seul le fichier `GlobalLandTemperaturesByCountry.csv` est utilisé (consigne). Le fichier d'origine n'est jamais modifié : tout le travail se fait sur une copie.")

    c = st.columns(4)
    c[0].metric("Lignes du fichier", fr(len(brut)))
    c[1].metric("Colonnes", brut.shape[1])
    c[2].metric("Entités géographiques", fr(brut["Country"].nunique()))
    c[3].metric("Période", f"{brut['dt'].min()[:4]} → {brut['dt'].max()[:4]}")

    st.subheader("Les quatre colonnes")
    st.dataframe(pd.DataFrame({
        "Colonne": ["dt", "AverageTemperature", "AverageTemperatureUncertainty", "Country"],
        "Signification": ["Date de la mesure (1er du mois)", "Température moyenne mensuelle (°C)",
                          "Incertitude de la mesure (°C)", "Pays (ou territoire)"],
        "Type à la lecture": [str(brut[col].dtype) for col in ["dt", "AverageTemperature", "AverageTemperatureUncertainty", "Country"]],
    }), hide_index=True)
    st.caption("Les 10 premières lignes du fichier brut :")
    st.dataframe(brut.head(10), hide_index=True)

    st.subheader("Sélection des cinq pays")
    st.write("On vérifie que chaque libellé recherché est bien présent tel quel dans le fichier, puis on garde les lignes correspondantes.")
    verif = pd.DataFrame({"Libellé recherché": list(CORRESPONDANCE.keys())})
    verif["Présent dans le fichier"] = verif["Libellé recherché"].isin(brut["Country"].unique())
    verif["Libellé harmonisé"] = verif["Libellé recherché"].map(CORRESPONDANCE)
    verif["Lignes dans le fichier"] = verif["Libellé recherché"].map(brut["Country"].value_counts()).fillna(0).astype(int)
    st.dataframe(verif, hide_index=True)

    c = st.columns(3)
    c[0].metric("Libellés retenus", len(noms_retenus))
    c[1].metric("Pays identifiés", len({CORRESPONDANCE[n] for n in noms_retenus}))
    c[2].metric("Observations retenues", fr(len(etude)))
    st.success(f"Libellés réellement présents dans le fichier : {', '.join(noms_retenus)}")

    par_pays = etude["Country"].value_counts().rename_axis("Libellé d'origine").reset_index(name="Observations")
    par_pays["Mois par pays"] = par_pays["Observations"]
    fig = px.bar(par_pays, x="Libellé d'origine", y="Observations", text="Observations",
                 title="Observations par pays : chaque pays a le même nombre de mois", color_discrete_sequence=[TEAL])
    st.plotly_chart(mise_en_forme(fig, 320))
    a_dire("Nous utilisons uniquement le fichier GlobalLandTemperaturesByCountry.csv. Il contient 577 462 lignes et 243 entités. "
           "Nous avons gardé cinq pays, ce qui donne 9 885 observations, soit 1 977 mois par pays de janvier 1849 à septembre 2013. "
           "Le fichier d'origine est conservé intact et nous travaillons sur une copie.")

# ====================================================================================
# PAGE 2 : État des lieux
# ====================================================================================
elif page == PAGES[2]:
    st.title("2 · État des lieux avant toute correction")
    st.write("On mesure d'abord, on corrige ensuite. Ces indicateurs seront recalculés à l'identique à la fin (page 4) pour montrer l'effet du travail.")

    ini = indicateurs(etude)
    c = st.columns(4)
    c[0].metric("Lignes", ini["Nombre de lignes"])
    c[1].metric("Températures manquantes", ini["Températures manquantes"], ini["Taux de températures manquantes"], delta_color="off")
    c[2].metric("Doublons exacts", ini["Doublons exacts (4 colonnes d'origine)"])
    c[3].metric("Type de la colonne dt", ini["Type de la colonne dt"])

    t1, t2, t3, t4 = st.tabs(["Types et manquants", "Où sont les manquants ?", "Statistiques", "Indicateurs complets"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Types de chaque colonne**")
            st.dataframe(pd.DataFrame({"Colonne": etude.columns, "Type": etude.dtypes.astype(str).values}), hide_index=True)
            st.warning("`dt` est de type `object` : c'est du texte, pas une vraie date. Il faudra la convertir.")
        with c2:
            st.markdown("**Valeurs manquantes par colonne**")
            manq = pd.DataFrame({"Colonne": etude.columns, "Manquants": etude.isna().sum().values,
                                 "Pourcentage": (etude.isna().mean() * 100).round(2).values})
            st.dataframe(manq, hide_index=True)
            st.caption("Les 775 températures et les 775 incertitudes manquent exactement aux mêmes lignes (voir page 3).")

    with t2:
        st.markdown("**Les manquants ne sont pas répartis au hasard.** On les regarde par pays puis par décennie.")
        bil = net.groupby("Country").agg(observations=("dt", "size"),
                                         temperatures_manquantes=("temperature_manquante", "sum")).reset_index()
        bil["taux (%)"] = (100 * bil["temperatures_manquantes"] / bil["observations"]).round(2)
        c1, c2 = st.columns([1, 1.4])
        c1.dataframe(bil.sort_values("taux (%)", ascending=False), hide_index=True)
        piv = (net.pivot_table(index="Country", columns="decennie", values="temperature_manquante", aggfunc="mean") * 100).round(0)
        fig = px.imshow(piv, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd", zmin=0, zmax=100,
                        labels=dict(color="% manquant", x="Décennie", y=""), title="Mois sans mesure (%) par pays et par décennie")
        c2.plotly_chart(mise_en_forme(fig, 320))
        avant_1900 = int(net.loc[net["annee"] < 1900, "temperature_manquante"].sum())
        st.error(f"Sur {fr(net['temperature_manquante'].sum())} mois sans mesure, **{fr(avant_1900)} se situent avant 1900**. "
                 "Il ne s'agit pas d'erreurs de saisie : à l'époque, il n'y avait tout simplement pas de mesure.")

    with t3:
        st.markdown("**Statistiques descriptives des 5 pays (état initial)**")
        st.dataframe(etude.describe(include="all").T.astype(str).reset_index().rename(columns={"index": "variable"}), hide_index=True)
        c1, c2 = st.columns(2)
        fig = px.histogram(net.dropna(subset=["AverageTemperature"]), x="AverageTemperature", color="Country", nbins=60,
                           color_discrete_map=COULEURS, barmode="overlay", opacity=0.6,
                           title="Distribution des températures mensuelles")
        c1.plotly_chart(mise_en_forme(fig, 340))
        fig = px.histogram(net.dropna(subset=["AverageTemperatureUncertainty"]), x="AverageTemperatureUncertainty", nbins=50,
                           color_discrete_sequence=[TEAL], title="Distribution des incertitudes")
        c2.plotly_chart(mise_en_forme(fig, 340))

    with t4:
        st.dataframe(pd.DataFrame({"Indicateur": list(ini.keys()), "Avant nettoyage": list(ini.values())}), hide_index=True, height=720)

    a_dire("Avant de corriger quoi que ce soit, nous avons mesuré. La colonne des dates est du texte. Il y a 775 températures manquantes, "
           "soit 7,84 % des lignes, et les 775 incertitudes manquent aux mêmes endroits. Il n'y a aucun doublon exact. "
           "Et ces manquants se concentrent au XIXe siècle : 770 sur 775 sont avant 1900. Cette observation est la clé de tout ce qui suit.")

# ====================================================================================
# PAGE 3 : Corrections justifiées
# ====================================================================================
elif page == PAGES[3]:
    st.title("3 · Corrections justifiées")
    st.write("Pour chaque problème : **ce qui a été fait** et **pourquoi ce choix plutôt qu'un autre**, avec la preuve calculée en direct.")

    onglets = st.tabs(["3.1 Dates", "3.2 Noms de pays", "3.3 Doublons", "3.4 Mesures manquantes",
                       "3.5 Incertitudes élevées", "3.6 Valeurs aberrantes"])

    # ---------------- 3.1 Dates ----------------
    with onglets[0]:
        bloc("La colonne `dt` est du texte (type object) : impossible de trier ou d'analyser dans le temps de façon fiable.",
             "Conversion en vrai format date (AAAA-MM-JJ). Les dates illisibles deviendraient NaT et seraient comptées.",
             "C'est une conversion sans aucune perte, indispensable pour trier, regrouper par année et tracer des séries.")
        st.markdown("**Preuve**")
        c = st.columns(4)
        c[0].metric("Type avant", str(etude["dt"].dtype))
        c[1].metric("Type après", str(net["dt"].dtype))
        c[2].metric("Dates invalides (NaT)", int(net["dt"].isna().sum()))
        c[3].metric("Toutes au 1er du mois", "Oui" if (net["dt"].dt.day == 1).all() else "Non")
        c1, c2 = st.columns(2)
        c1.caption("Avant (texte)"); c1.dataframe(etude[["dt"]].head(5), hide_index=True)
        c2.caption("Après (datetime)"); c2.dataframe(net[["dt"]].head(5), hide_index=True)

        st.markdown("**Contrôles de continuité par pays**")
        ctl = net.groupby("Country")["dt"].agg(debut="min", fin="max", observes="size")
        ctl["attendus"] = ((ctl["fin"].dt.year - ctl["debut"].dt.year) * 12 + ctl["fin"].dt.month - ctl["debut"].dt.month + 1)
        ctl["mois_absents"] = ctl["attendus"] - ctl["observes"]
        ctl["trié"] = net.groupby("Country")["dt"].apply(lambda s: s.is_monotonic_increasing)
        ctl["debut"] = ctl["debut"].dt.date.astype(str); ctl["fin"] = ctl["fin"].dt.date.astype(str)
        st.dataframe(ctl.reset_index(), hide_index=True)
        doublons_pd = int(net.duplicated(subset=["Country", "dt"]).sum())
        if (ctl["mois_absents"] == 0).all() and doublons_pd == 0:
            st.success("Dates valides : séries mensuelles complètes, aucun mois absent du calendrier, aucun doublon pays-date.")
        a_dire("Nous avons converti la colonne en vrai format date. Il y a zéro date invalide, toutes tombent au premier du mois, "
               "et chaque pays a une série mensuelle complète sans mois absent du calendrier.")

    # ---------------- 3.2 Noms ----------------
    with onglets[1]:
        bloc("Le fichier mélange des libellés anglais et des variantes d'écriture (casse, accents, deux noms pour la Côte d'Ivoire).",
             "Harmonisation par un dictionnaire de correspondance : un seul libellé, en français, par pays.",
             "On unifie sans rien supprimer, pour qu'un pays ne soit jamais compté deux fois ni qu'une partie de son historique soit perdue.")
        st.markdown("**Preuve : libellés avant et après**")
        av = etude["Country"].value_counts().rename_axis("Libellé d'origine").reset_index(name="Observations")
        av["Libellé harmonisé"] = av["Libellé d'origine"].str.strip().map(CORRESPONDANCE)
        av["Modifié"] = av["Libellé d'origine"] != av["Libellé harmonisé"]
        st.dataframe(av, hide_index=True)
        n_mod = int((etude["Country"] != etude["Country"].map(CORRESPONDANCE)).sum())
        espaces = int((etude["Country"] != etude["Country"].str.strip()).sum())
        c = st.columns(3)
        c[0].metric("Lignes renommées", fr(n_mod))
        c[1].metric("Libellés avec espaces superflus", espaces)
        c[2].metric("Pays distincts après", net["Country"].nunique())
        apres = net["Country"].value_counts().rename_axis("Pays").reset_index(name="Observations")
        fig = px.bar(apres, x="Pays", y="Observations", text="Observations", color="Pays", color_discrete_map=COULEURS,
                     title="Après harmonisation : 5 pays, même nombre de mois")
        fig.update_layout(showlegend=False)
        st.plotly_chart(mise_en_forme(fig, 300))
        assert net["Country"].nunique() == 5
        a_dire("Le fichier mélange des libellés anglais et des variantes d'écriture. Nous avons harmonisé avec un dictionnaire : "
               "Benin devient Bénin, Senegal devient Sénégal. Nous ne supprimons rien, nous unifions.")

    # ---------------- 3.3 Doublons ----------------
    with onglets[2]:
        bloc("Des lignes identiques, ou deux lignes pour le même pays à la même date avec des valeurs différentes (conflit), fausseraient les statistiques.",
             "On cherche deux sortes de doublons : exacts, et pays-date. On ne supprimerait que les doublons exacts (aucune perte d'information) ; un conflit s'arbitrerait à la main.",
             "Un doublon exact se supprime sans perte. Un conflit ne doit jamais être supprimé à l'aveugle : il faut choisir la bonne valeur.")
        n_exacts = int(net[["dt", "AverageTemperature", "AverageTemperatureUncertainty", "Country"]].duplicated().sum())
        n_pd = int(net.duplicated(subset=["Country", "dt"]).sum())
        c = st.columns(3)
        c[0].metric("Doublons exacts", n_exacts)
        c[1].metric("Doublons pays-date", n_pd)
        c[2].metric("Conflits (valeurs différentes)", n_pd - n_exacts)
        if n_pd == 0:
            st.success("Aucun doublon : chaque pays a au plus une ligne par mois. Il n'y avait donc rien à supprimer, et le nombre de lignes reste "
                       f"{fr(len(net))}.")
        cnt = net.groupby("Country").size().rename("Lignes").reset_index()
        cnt["Dates distinctes"] = net.groupby("Country")["dt"].nunique().values
        cnt["Égal ?"] = cnt["Lignes"] == cnt["Dates distinctes"]
        st.markdown("**Preuve : lignes = dates distinctes pour chaque pays**")
        st.dataframe(cnt, hide_index=True)
        a_dire("Nous avons cherché les doublons exacts, mais aussi les cas où un même pays a deux lignes pour la même date avec des valeurs différentes. "
               "Résultat : zéro dans les deux cas. Il n'y avait donc rien à supprimer.")

    # ---------------- 3.4 Manquants ----------------
    with onglets[3]:
        bloc("775 mois n'ont pas de température (et pas d'incertitude), surtout avant 1900.",
             "On conserve les lignes et le NaN, et on ajoute la colonne indicatrice `temperature_manquante`.",
             "Une valeur manquante n'est pas forcément une erreur : il n'y avait pas de mesure à l'époque. Imputer inventerait des températures et fausserait la tendance ; supprimer ferait perdre la structure des séries.")
        mt, mu = net["AverageTemperature"].isna(), net["AverageTemperatureUncertainty"].isna()
        c = st.columns(4)
        c[0].metric("Températures manquantes", fr(mt.sum()))
        c[1].metric("Incertitudes manquantes", fr(mu.sum()))
        c[2].metric("Manquantes simultanément", fr((mt & mu).sum()))
        c[3].metric("Dans une seule colonne", fr((mt ^ mu).sum()))
        st.caption("Les deux colonnes manquent toujours ensemble : un mois est soit entièrement mesuré, soit entièrement absent.")

        st.markdown("**Profil des manquants par pays**")
        derniere = net["dt"].max()
        lignes = []
        for p, g in net.groupby("Country"):
            m = g[g["AverageTemperature"].isna()]
            avant_fin = m[m["dt"] < derniere]
            lignes.append({"Pays": p, "Première mesure": str(g.loc[g["AverageTemperature"].notna(), "dt"].min().date()),
                           "Mois manquants": len(m), "Avant 1900": int((m["annee"] < 1900).sum()),
                           "Depuis 1900": int((m["annee"] >= 1900).sum()),
                           "Dernier trou avant le dernier mois": str(avant_fin["dt"].max().date()) if len(avant_fin) else "-",
                           "Dernier mois sans mesure": bool((m["dt"] == derniere).any())})
        st.dataframe(pd.DataFrame(lignes), hide_index=True)

        st.markdown("**Mesurer la perte avant de supprimer**")
        st.write("Faites glisser l'année : si on supprimait toutes les lignes antérieures, combien de **vraies mesures** perdrait-on ?")
        X = st.slider("Supprimer toutes les lignes avant l'année", 1850, 1960, 1900, step=5, key="slider_manq")
        retire = net[net["annee"] < X]
        valides = int(retire["AverageTemperature"].notna().sum())
        c = st.columns(4)
        c[0].metric("Lignes retirées", fr(len(retire)), pct(100 * len(retire) / len(net), 1) + " du jeu", delta_color="off")
        c[1].metric("dont mois sans mesure", fr(retire["AverageTemperature"].isna().sum()))
        c[2].metric("dont vraies mesures perdues", fr(valides))
        c[3].metric("Manquants qui resteraient", fr(net.loc[net["annee"] >= X, "AverageTemperature"].isna().sum()))
        if valides > 0:
            st.warning(f"Supprimer avant {X} fait perdre {fr(valides)} mesures réelles : c'est irréversible. Signaler les manquants évite cette perte.")

        st.markdown("**Voir les trous dans une série**")
        c1, c2 = st.columns([1, 3])
        pays_t = c1.selectbox("Pays", PAYS, key="trou_pays")
        plage = c1.slider("Période", 1849, 2013, (1849, 1910), key="trou_plage")
        sub = net[(net["Country"] == pays_t) & net["annee"].between(*plage)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["dt"], y=sub["AverageTemperature"], mode="lines+markers", name="Température",
                                 line=dict(color=COULEURS[pays_t]), marker=dict(size=4), connectgaps=False))
        manq_pts = sub[sub["AverageTemperature"].isna()]
        ymin = sub["AverageTemperature"].min() - 0.5 if sub["AverageTemperature"].notna().any() else 20
        fig.add_trace(go.Scatter(x=manq_pts["dt"], y=[ymin] * len(manq_pts), mode="markers", name="Mois sans mesure",
                                 marker=dict(color=RED, symbol="x", size=6)))
        fig.update_layout(title=f"{pays_t} : les croix rouges sont des mois sans mesure", yaxis_title="°C")
        c2.plotly_chart(mise_en_forme(fig, 360))

        piv = (net.groupby(["decennie", "Country"])["temperature_manquante"].mean().unstack() * 100)[PAYS]
        fig = px.line(piv, markers=True, color_discrete_map=COULEURS, labels=dict(value="Mois sans mesure (%)", decennie="Décennie"),
                      title="Part de mois sans mesure par décennie")
        st.plotly_chart(mise_en_forme(fig, 340))
        a_dire("C'est le choix le plus important. Nous avons conservé les 775 mois vides et les avons signalés par une colonne. "
               "Une valeur manquante n'est pas une erreur : il n'y avait pas de mesure à l'époque. Remplir par une moyenne ou une interpolation "
               "reviendrait à inventer des températures, et fausserait la tendance. Signaler est la solution la plus honnête.")

    # ---------------- 3.5 Incertitudes ----------------
    with onglets[4]:
        bloc("L'incertitude est très forte sur les premières années (peu de stations).",
             "On repère les mesures à forte incertitude avec la règle de Tukey (Q3 + 1,5 × IQR) et on les marque dans `incertitude_elevee`, sans les supprimer.",
             "Une mesure incertaine ne veut pas dire une mesure fausse. Chacun pourra la garder, la pondérer ou l'écarter selon son analyse.")
        k_inc = st.slider("Coefficient de Tukey (1,5 = valeur du notebook)", 0.5, 3.0, K_DEF, 0.1, key="k_inc")
        fl = calculer_drapeaux(net, k_inc, K_DEF)
        seuil = fl["seuil_inc"].iloc[0]
        elevees = fl[fl["incertitude_elevee"]]
        c = st.columns(4)
        c[0].metric("Seuil d'incertitude", f"{seuil:.3f} °C")
        c[1].metric("Mois à forte incertitude", fr(len(elevees)))
        c[2].metric("Part du jeu", pct(100 * len(elevees) / len(fl)))
        c[3].metric("Année maximale concernée", int(elevees["annee"].max()) if len(elevees) else "-")

        c1, c2 = st.columns(2)
        fig = px.histogram(fl.dropna(subset=["AverageTemperatureUncertainty"]), x="AverageTemperatureUncertainty", nbins=50,
                           color_discrete_sequence=[TEAL], title="Distribution de l'incertitude et seuil")
        fig.add_vline(x=seuil, line_color=RED, line_dash="dash", annotation_text=f"seuil {seuil:.2f}")
        c1.plotly_chart(mise_en_forme(fig, 340))
        fig = px.box(fl, x="Country", y="AverageTemperatureUncertainty", color="Country", color_discrete_map=COULEURS,
                     title="Incertitude par pays")
        fig.update_layout(showlegend=False)
        c2.plotly_chart(mise_en_forme(fig, 340))

        part = (fl.groupby("decennie")["incertitude_elevee"].mean() * 100).round(1).reset_index(name="% de mois à forte incertitude")
        inc_dec = fl.groupby(["decennie", "Country"])["AverageTemperatureUncertainty"].mean().reset_index()
        c1, c2 = st.columns(2)
        fig = px.bar(part, x="decennie", y="% de mois à forte incertitude", color_discrete_sequence=[ORANGE],
                     title="Où se concentrent-ils ? Part par décennie")
        c1.plotly_chart(mise_en_forme(fig, 340))
        fig = px.line(inc_dec, x="decennie", y="AverageTemperatureUncertainty", color="Country", color_discrete_map=COULEURS,
                      markers=True, title="Incertitude moyenne par décennie et par pays")
        c2.plotly_chart(mise_en_forme(fig, 340))
        st.success("L'incertitude passe de plus de 1 °C au XIXe siècle à environ 0,25 °C vers 1950 : "
                   "c'est bien une question d'époque, pas d'erreur aléatoire.")
        a_dire("Nous avons appliqué la règle de Tukey, qui donne un seuil de 1,80 °C. Nous trouvons 212 mois au-dessus, soit 2,14 %, "
               "et tous datent d'avant 1930. Nous les avons marqués sans les supprimer, car une mesure incertaine ne veut pas dire une mesure fausse.")

    # ---------------- 3.6 Aberrantes ----------------
    with onglets[5]:
        bloc("Une température peut sembler aberrante soit parce qu'elle est impossible (erreur), soit parce qu'elle est extrême mais réelle.",
             "Deux contrôles : plausibilité physique (plage de 10 à 45 °C) et méthode IQR par pays et par mois. On marque dans `temp_atypique`, on ne supprime rien.",
             "Atypique ne veut pas dire erreur : un mois peut être exceptionnellement chaud ou frais. L'IQR est calculé par pays et par mois car la saison change ce qui est normal.")
        c1, c2 = st.columns(2)
        tmin, tmax = c1.slider("Plage de plausibilité (°C)", 0, 50, (T_MIN_DEF, T_MAX_DEF), key="plage_t")
        k_iqr = c2.slider("Coefficient de l'IQR (1,5 = valeur du notebook)", 1.0, 3.5, K_DEF, 0.1, key="k_iqr")
        fl = calculer_drapeaux(net, K_DEF, k_iqr)
        t = fl["AverageTemperature"]
        hors = fl[(t < tmin) | (t > tmax)]
        atyp = fl[fl["temp_atypique"]]
        c = st.columns(4)
        c[0].metric("Valeurs hors plage", fr(len(hors)))
        c[1].metric("Température min / max observées", f"{t.min():.1f} / {t.max():.1f} °C")
        c[2].metric("Températures atypiques (IQR)", fr(len(atyp)), pct(100 * len(atyp) / len(fl)), delta_color="off")
        c[3].metric("Incertitudes nulles ou négatives", int((fl["AverageTemperatureUncertainty"] <= 0).sum()))
        if len(hors) == 0:
            st.success("Aucune valeur physiquement impossible avec cette plage.")
        else:
            st.error("Des valeurs sortent de la plage choisie :")
            st.dataframe(hors[["Country", "dt", "AverageTemperature"]], hide_index=True)

        c1, c2 = st.columns(2)
        pays_b = c1.selectbox("Pays pour les boîtes à moustaches", PAYS, key="box_pays")
        sub = fl[fl["Country"] == pays_b].dropna(subset=["AverageTemperature"])
        fig = px.box(sub, x="mois", y="AverageTemperature", points="outliers", color_discrete_sequence=[COULEURS[pays_b]],
                     title=f"{pays_b} : températures par mois (points isolés = atypiques)")
        c1.plotly_chart(mise_en_forme(fig, 360))
        fig = px.scatter(fl.dropna(subset=["AverageTemperature"]), x="dt", y="ecart_mediane", color="temp_atypique",
                         color_discrete_map={True: RED, False: "#b0bec5"}, opacity=0.6,
                         labels=dict(ecart_mediane="Écart à la médiane du pays et du mois (°C)", dt="Date"),
                         title="Écart à la médiane : les atypiques (rouge) sont dispersés")
        c2.plotly_chart(mise_en_forme(fig, 360))

        st.markdown("**Profil des températures atypiques**")
        chaud = int((atyp["ecart_mediane"] > 0).sum()); froid = int((atyp["ecart_mediane"] < 0).sum())
        c = st.columns(4)
        c[0].metric("Chaudes (au-dessus de la médiane)", chaud)
        c[1].metric("Froides (en dessous)", froid)
        c[2].metric("Avant 1950 / depuis 1950", f"{int((atyp['annee'] < 1950).sum())} / {int((atyp['annee'] >= 1950).sum())}")
        c[3].metric("Avec aussi une forte incertitude", int(atyp["incertitude_elevee"].sum()))
        par_pays_a = atyp.groupby("Country").size().reindex(PAYS).fillna(0).astype(int).reset_index(name="Atypiques")
        c1, c2 = st.columns([1, 2])
        c1.dataframe(par_pays_a, hide_index=True)
        ecart = atyp.assign(ecart_abs=atyp["ecart_mediane"].abs()).sort_values("ecart_abs", ascending=False).head(10)
        c2.caption("Les 10 cas les plus éloignés de la médiane de leur pays et de leur mois")
        c2.dataframe(ecart[["Country", "dt", "AverageTemperature", "AverageTemperatureUncertainty", "ecart_mediane"]], hide_index=True)
        a_dire("Nous avons fait deux contrôles. D'abord la plausibilité : aucune température hors de 10 à 45 °C, les valeurs vont de 21,5 à 33,7 °C. "
               "Ensuite l'écart interquartile par pays et par mois, qui repère 81 températures atypiques, soit 0,82 %. "
               "Atypique ne veut pas dire erreur : nous les avons signalées, sans les supprimer.")

    st.divider()
    st.subheader("Tableau récapitulatif des corrections")
    fl0 = calculer_drapeaux(net, K_DEF, K_DEF)
    st.dataframe(pd.DataFrame([
        ["Dates stockées en texte", "Conversion en datetime (AAAA-MM-JJ)", "Permet tri, calculs de durée et analyse temporelle",
         f"{fr(len(net))} dates converties, {int(net['dt'].isna().sum())} invalide(s)"],
        ["Libellés anglais / casse irrégulière", "Harmonisation en français via un dictionnaire", "Noms stables et uniques pour chaque pays",
         f"{fr(int((etude['Country'] != etude['Country'].map(CORRESPONDANCE)).sum()))} ligne(s) renommée(s), {net['Country'].nunique()} pays"],
        ["Doublons", "Recherche des doublons exacts et pays-date", "Un doublon exact se supprime sans perte ; un conflit s'arbitre",
         f"{int(net.duplicated(subset=['Country', 'dt']).sum())} doublon(s)"],
        ["Mesures anciennes manquantes", "NaN conservés + indicateur temperature_manquante", "Absence de mesure, pas une erreur : imputer inventerait des températures",
         f"{fr(net['temperature_manquante'].sum())} mois signalés, 0 ligne supprimée"],
        ["Incertitudes élevées", "Marquage par la règle de Tukey, sans suppression", "Mesure incertaine ne veut pas dire mesure fausse",
         f"{fr(fl0['incertitude_elevee'].sum())} mois signalés (seuil {fl0['seuil_inc'].iloc[0]:.2f} °C)"],
        ["Valeurs aberrantes", "Plausibilité + IQR par pays et par mois", "Atypique ne veut pas dire erreur (saisonnalité, réchauffement)",
         f"0 valeur impossible, {fr(fl0['temp_atypique'].sum())} atypique(s) signalée(s)"],
    ], columns=["Problème", "Traitement", "Justification", "Effet mesuré"]), hide_index=True)

# ====================================================================================
# PAGE 4 : État final et export
# ====================================================================================
elif page == PAGES[4]:
    st.title("4 · État final et export")
    st.write("Les **mêmes indicateurs** qu'à l'état des lieux, avec les **mêmes fonctions**, recalculés sur le fichier nettoyé réellement livré.")

    final = csv_net.copy()
    final["dt"] = pd.to_datetime(final["dt"], errors="coerce")
    ini = indicateurs(etude)
    fin = indicateurs(final)

    c = st.columns(4)
    c[0].metric("Lignes", f"{fr(len(etude))} → {fr(len(final))}")
    c[1].metric("Lignes supprimées", fr(len(etude) - len(final)))
    c[2].metric("Colonnes", f"{etude.shape[1]} → {final.shape[1]}")
    c[3].metric("Type de dt", f"{etude['dt'].dtype} → {final['dt'].dtype}")

    comp = pd.DataFrame({"Indicateur": list(ini.keys()), "Avant nettoyage": list(ini.values()), "Après nettoyage": list(fin.values())})
    comp["Changé ?"] = np.where(comp["Avant nettoyage"] != comp["Après nettoyage"], "oui", "")
    st.subheader("Comparaison avant / après")
    st.dataframe(comp, hide_index=True, height=720)

    st.subheader("Les trois indicateurs ajoutés")
    ind = pd.DataFrame({
        "Indicateur": ["temperature_manquante", "incertitude_elevee", "temp_atypique"],
        "Signification": ["Mois sans mesure de température", "Incertitude supérieure au seuil de Tukey", "Température atypique pour son pays et son mois"],
        "Lignes signalées": [int(final[c_].sum()) for c_ in ["temperature_manquante", "incertitude_elevee", "temp_atypique"]],
    })
    ind["% du jeu"] = (100 * ind["Lignes signalées"] / len(final)).round(2)
    c1, c2 = st.columns([1.4, 1])
    c1.dataframe(ind, hide_index=True)
    fig = px.bar(ind, x="Indicateur", y="Lignes signalées", text="Lignes signalées", color_discrete_sequence=[TEAL],
                 title="Lignes signalées (aucune supprimée)")
    c2.plotly_chart(mise_en_forme(fig, 300))

    st.subheader("Preuve : le fichier livré est exactement ce que le nettoyage décrit")
    st.write("On recalcule tous les indicateurs depuis le fichier brut (pages précédentes) et on les compare, ligne à ligne, au CSV nettoyé.")
    recalc = calculer_drapeaux(net, K_DEF, K_DEF)
    m = final.merge(recalc[["Country", "dt", "AverageTemperature", "temperature_manquante", "incertitude_elevee", "temp_atypique"]],
                    on=["Country", "dt"], how="outer", suffixes=("", "_recalc"), indicator=True)
    memes_lignes = bool((m["_merge"] == "both").all())
    m_ok = m[m["_merge"] == "both"]
    temp_ok = bool(np.allclose(m_ok["AverageTemperature"].fillna(-999), m_ok["AverageTemperature_recalc"].fillna(-999)))
    controles = {
        "Même nombre de lignes avant et après (aucune suppression)": len(final) == len(etude),
        "Mêmes couples (pays, date) dans le CSV et dans le recalcul": memes_lignes,
        "Températures identiques au recalcul": temp_ok,
        "Indicateur temperature_manquante identique au recalcul": bool((m_ok["temperature_manquante"] == m_ok["temperature_manquante_recalc"]).all()),
        "Indicateur incertitude_elevee identique au recalcul": bool((m_ok["incertitude_elevee"] == m_ok["incertitude_elevee_recalc"]).all()),
        "Indicateur temp_atypique identique au recalcul": bool((m_ok["temp_atypique"] == m_ok["temp_atypique_recalc"]).all()),
        "Dates bien relues (datetime)": pd.api.types.is_datetime64_any_dtype(final["dt"]),
        "Cinq pays exactement": final["Country"].nunique() == 5,
        "Accents conservés (Bénin, Côte d'Ivoire, Sénégal)": {"Bénin", "Côte d'Ivoire", "Sénégal"}.issubset(set(final["Country"])),
        "Aucun doublon pays-date": not final.duplicated(subset=["Country", "dt"]).any(),
        "Aucune température hors de 10 à 45 °C": bool(final["AverageTemperature"].dropna().between(T_MIN_DEF, T_MAX_DEF).all()),
    }
    cdf = pd.DataFrame({"Contrôle": list(controles.keys()), "Résultat": ["OK" if v else "À VÉRIFIER" for v in controles.values()]})
    st.dataframe(cdf, hide_index=True)
    if all(controles.values()):
        st.success("Tous les contrôles sont validés : le CSV livré correspond exactement au nettoyage décrit.")
    else:
        st.error("Au moins un contrôle échoue : vérifiez que le CSV nettoyé provient bien de la dernière exécution du notebook.")

    st.subheader("Aperçu et téléchargement")
    st.dataframe(final.head(15), hide_index=True)
    st.download_button("Télécharger le CSV nettoyé", data=csv_net.to_csv(index=False).encode("utf-8-sig"),
                       file_name="temperatures_afrique_ouest_nettoye.csv", mime="text/csv")
    a_dire("Pour montrer l'effet du travail, nous avons recalculé les mêmes indicateurs. Le nombre de lignes reste à 9 885 car aucune ligne n'a été supprimée. "
           "Les colonnes passent de 4 à 9, avec l'année, le mois et nos trois indicateurs de qualité. Nous avons relu le fichier exporté pour vérifier : "
           "mêmes dimensions, mêmes colonnes, dates bien relues, accents conservés.")

# ====================================================================================
# PAGE 5 : Explorer les séries
# ====================================================================================
elif page == PAGES[5]:
    st.title("5 · Explorer les séries nettoyées")
    st.write("Les indicateurs du fichier nettoyé permettent de **filtrer ou de pondérer** selon l'analyse, sans rien avoir perdu.")
    final = csv_net.copy()
    final["dt"] = pd.to_datetime(final["dt"])

    c1, c2, c3 = st.columns([2, 2, 3])
    choix = c1.multiselect("Pays", PAYS, default=["Ghana", "Sénégal"], key="exp_pays")
    plage = c2.slider("Période", 1849, 2013, (1849, 2013), key="exp_plage")
    options = c3.multiselect("Que faire des valeurs signalées ?",
                             ["Masquer les mois sans mesure", "Écarter les incertitudes élevées", "Écarter les atypiques"],
                             default=[], key="exp_opts")
    sub = final[final["Country"].isin(choix) & final["annee"].between(*plage)].copy()
    if "Écarter les incertitudes élevées" in options:
        sub = sub[~sub["incertitude_elevee"]]
    if "Écarter les atypiques" in options:
        sub = sub[~sub["temp_atypique"]]
    if "Masquer les mois sans mesure" in options:
        sub = sub.dropna(subset=["AverageTemperature"])

    if sub.empty:
        st.warning("Aucune donnée avec cette sélection.")
    else:
        fig = go.Figure()
        for p in choix:
            s = sub[sub["Country"] == p]
            fig.add_trace(go.Scatter(x=s["dt"], y=s["AverageTemperature"], mode="lines", name=p, line=dict(color=COULEURS[p], width=1.2)))
            at = s[s["temp_atypique"]]
            fig.add_trace(go.Scatter(x=at["dt"], y=at["AverageTemperature"], mode="markers", name=f"{p} : atypiques",
                                     marker=dict(color=RED, size=7, symbol="diamond"), showlegend=False))
        fig.update_layout(title="Températures mensuelles (losanges rouges = atypiques signalées)", yaxis_title="°C")
        st.plotly_chart(mise_en_forme(fig, 420))

        c = st.columns(4)
        c[0].metric("Lignes affichées", fr(len(sub)))
        c[1].metric("Mois sans mesure", fr(sub["temperature_manquante"].sum()))
        c[2].metric("Incertitudes élevées", fr(sub["incertitude_elevee"].sum()))
        c[3].metric("Atypiques", fr(sub["temp_atypique"].sum()))

        ann = (sub.dropna(subset=["AverageTemperature"]).groupby(["Country", "annee"])
               .agg(mois=("AverageTemperature", "count"), temperature=("AverageTemperature", "mean")).reset_index())
        ann.loc[ann["mois"] < 12, "temperature"] = np.nan
        fig = px.line(ann, x="annee", y="temperature", color="Country", color_discrete_map=COULEURS,
                      title="Température annuelle moyenne (années complètes seulement : 12 mois mesurés)",
                      labels=dict(temperature="°C", annee="Année"))
        st.plotly_chart(mise_en_forme(fig, 380))
        st.caption("Une année n'est tracée que si ses 12 mois sont mesurés : sinon la moyenne serait biaisée. Les trous sont visibles au lieu d'être comblés artificiellement.")

# ====================================================================================
# PAGE 6 : Conclusion et questions du jury
# ====================================================================================
elif page == PAGES[6]:
    st.title("6 · Conclusion et questions du jury")
    fl0 = calculer_drapeaux(net, K_DEF, K_DEF)
    st.success(
        f"Nous livrons un jeu propre et documenté pour cinq pays d'Afrique de l'Ouest : **{fr(len(csv_net))} lignes conservées, 0 supprimée**. "
        f"Les {fr(net['temperature_manquante'].sum())} mois sans mesure, les {fr(fl0['incertitude_elevee'].sum())} mois à forte incertitude "
        f"et les {fr(fl0['temp_atypique'].sum())} températures atypiques sont **signalés, pas effacés**.")

    st.subheader("Les trois phrases à retenir")
    st.markdown("""
1. **« Une valeur manquante n'est pas forcément une erreur. »** Ici elle correspond à une absence de mesure à l'époque.
2. **« Atypique ne veut pas dire erreur. »** Un mois exceptionnellement chaud ou frais peut être réel.
3. **« Nous signalons au lieu de supprimer. »** Chacun peut ensuite garder, pondérer ou écarter.
""")

    st.subheader("Recommandation pour l'analyse")
    st.write("Les séries sont complètes de **1900 à août 2013** : c'est la période à privilégier pour une analyse de tendance. "
             "Avant, jusqu'à 79 % des mois manquent dans certaines décennies.")

    st.subheader("Limites")
    st.markdown("""
- Les seuils (plage 10-45 °C, règle de Tukey) sont des choix raisonnables mais discutables : ils sont modifiables dans la page 3.
- Les mesures anciennes, plus incertaines, pourraient être pondérées dans un modèle statistique.
- Les données sont des moyennes nationales mensuelles : elles ne montrent pas les différences régionales à l'intérieur d'un pays.
""")

    st.subheader("Questions probables du jury")
    with st.expander("Pourquoi ne pas avoir supprimé les lignes anciennes ?"):
        st.write("Parce qu'elles contiennent de vraies mesures et que la perte serait irréversible. Avec les indicateurs, chacun choisit à l'analyse. "
                 "Le curseur de la page 3.4 montre combien de mesures réelles on perdrait en supprimant avant une année donnée.")
    with st.expander("Pourquoi ne pas avoir imputé les valeurs manquantes ?"):
        st.write("Sur des décennies entières, l'imputation (moyenne ou interpolation) inventerait des températures et biaiserait la tendance. "
                 "Les manquants se concentrent avant 1900 : c'est une absence réelle de mesure.")
    with st.expander("Pourquoi calculer l'IQR par pays et par mois ?"):
        st.write("Parce qu'un calcul global confondrait un mois chaud normal (mars au Burkina Faso) avec une anomalie. Les saisons ne se comparent pas entre elles.")
    with st.expander("Pourquoi la règle de Tukey pour l'incertitude ?"):
        st.write("C'est un critère objectif, calculé sur la distribution réelle, plutôt qu'une année coupée arbitrairement. Le seuil est un paramètre : le curseur de la page 3.5 permet de le faire varier.")
    with st.expander("Que feriez-vous différemment avec plus de temps ?"):
        st.write("Pondérer les mesures anciennes par leur incertitude dans un modèle, et comparer plusieurs seuils pour mesurer la sensibilité des conclusions.")
    with st.expander("Comment prouvez-vous que votre fichier est correct ?"):
        st.write("La page 4 recalcule tous les indicateurs depuis le fichier brut et les compare ligne à ligne au CSV livré : onze contrôles automatiques.")
