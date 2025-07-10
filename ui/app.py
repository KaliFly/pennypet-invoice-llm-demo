import sys
import os

# Ajouter la racine du projet au PYTHONPATH dès le début
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from pathlib import Path
import pandas as pd
import re
import json

import streamlit as st
from st_supabase_connection import SupabaseConnection

# Initialisation sécurisée de la connexion Supabase
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    conn = st.connection(
        "supabase",
        type=SupabaseConnection,
        url=supabase_url,
        key=supabase_key
    )
except Exception as e:
    st.error(f"Erreur de connexion à Supabase : {e}")
    st.stop()


class PennyPetConfig:
    """
    Chargement et validation robuste de la configuration du prototype PennyPet.
    Correction des chemins, gestion des erreurs, et logs détaillés.
    """

    def __init__(self, base_dir: Path = None):
        # base_dir par défaut => dossier ui/
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        # racine du projet
        project_root = base_dir.parent

        # dossier config placé à la racine du projet
        self.config_dir = project_root / "config"

        # Vérification de l'existence du dossier de configuration
        if not self.config_dir.exists():
            st.error(f"Dossier de configuration introuvable : {self.config_dir}")
            st.stop()

        # Chargement des lexiques et regex
        try:
            st.write("→ Chargement actes_normalises.csv depuis", self.config_dir / "lexiques/actes_normalises.csv")
            self.actes_df = self._load_csv_regex("lexiques/actes_normalises.csv", sep=";")

            st.write("→ Chargement medicaments_normalises.json depuis", self.config_dir / "lexiques/medicaments_normalises.json")
            self.medicaments_df = self._load_json_df("lexiques/medicaments_normalises.json")

            st.write("→ Chargement calculs_codes_int.csv depuis", self.config_dir / "regex/calculs_codes_int.csv")
            self.calculs_codes_df = self._load_csv_regex("regex/calculs_codes_int.csv", sep=";")

            st.write("→ Chargement infos_financieres.csv depuis", self.config_dir / "regex/infos_financieres.csv")
            self.infos_financieres_df = self._load_csv_regex("regex/infos_financieres.csv", sep=";")

            st.write("→ Chargement metadonnees.csv depuis", self.config_dir / "regex/metadonnees.csv")
            self.metadonnees_df = self._load_csv_regex(
                "regex/metadonnees.csv",
                sep=";",
                quotechar='"'
            )

            st.write("→ Chargement parties_benef.csv depuis", self.config_dir / "regex/parties_benef.csv")
            self.parties_benef_df = self._load_csv_regex("regex/parties_benef.csv", sep=";")

            st.write("→ Chargement suivi_SLA.csv depuis", self.config_dir / "regex/suivi_SLA.csv")
            self.suivi_sla_df = self._load_csv_regex("regex/suivi_SLA.csv", sep=";")
        except Exception as e:
            st.error(f"Erreur lors du chargement des lexiques/regex : {e}")
            st.exception(e)
            st.stop()

        # Chargement des règles et formules
        try:
            st.write("→ Chargement regles_prise_en_charge.csv depuis", self.config_dir / "regles_prise_en_charge.csv")
            self.regles_pc_df = self._load_regles("regles_prise_en_charge.csv", sep=";")

            st.write("→ Chargement mapping_amv_pennypet.json depuis", self.config_dir / "mapping_amv_pennypet.json")
            self.mapping_amv = self._load_json("mapping_amv_pennypet.json")

            st.write("→ Chargement formules_pennypet.json depuis", self.config_dir / "formules_pennypet.json")
            self.formules = self._load_json("formules_pennypet.json")
        except Exception as e:
            st.error(f"Erreur lors du chargement des règles ou formules : {e}")
            st.exception(e)
            st.stop()

    def _load_json(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Le fichier JSON {path} est manquant.")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_json_df(self, filename: str) -> pd.DataFrame:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Le fichier JSON {path} est manquant.")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        if not data:
            raise ValueError(f"Aucune donnée trouvée dans {path}.")
        return pd.DataFrame(data)

    def _load_csv(self, relpath: str, **kwargs) -> pd.DataFrame:
        path = self.config_dir / relpath
        if not path.exists():
            raise FileNotFoundError(f"Le fichier CSV {path} est manquant.")
        try:
            return pd.read_csv(path, encoding="utf-8", **kwargs)
        except Exception as e:
            raise ValueError(f"Erreur de lecture du CSV {path} : {e}")

    def _load_csv_regex(self, relpath: str, **kwargs) -> pd.DataFrame:
        df = self._load_csv(relpath, **kwargs)
        df.columns = [c.strip() for c in df.columns]
        mapping_cols = {
            "Terme/Libellé": "field_label",
            "Regex OCR": "regex_pattern",
            "Variantes/Synonymes": "variantes"
        }
        for old, new in mapping_cols.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        if "regex_pattern" in df.columns:
            def compile_or_none(pat: str):
                try:
                    return re.compile(pat, re.IGNORECASE) if pat and str(pat).lower() != "nan" else None
                except re.error:
                    return None
            df["pattern"] = df["regex_pattern"].astype(str).apply(compile_or_none)
        return df

    def _load_regles(self, relpath: str, **kwargs) -> pd.DataFrame:
        df = self._load_csv(relpath, **kwargs)
        for col in ["exclusions", "actes_couverts", "conditions_speciales"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .apply(lambda s: [v.strip() for v in s.split("|") if v.strip()] if s else [])
                )
        for col in ["taux_remboursement", "plafond_annuel"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df


# Instanciation et point de contrôle
config = PennyPetConfig()
st.write("✅ Configuration chargée depuis :", config.config_dir)

# Exemple d’UI Streamlit à placer ici
st.title("PennyPet Invoice Parser")
uploaded_file = st.file_uploader("Téléversez votre facture")
if uploaded_file:
    st.write("Traitement en cours…")
    # … votre logique de parsing …
    st.write("Résultat du parsing :", {})  # Remplacez {} par vos données
