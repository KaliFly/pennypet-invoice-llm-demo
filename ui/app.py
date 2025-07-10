import pandas as pd
import re
import json
from pathlib import Path

class PennyPetConfig:
    """
    Chargement et validation robuste de la configuration du prototype PennyPet.
    Correction des chemins, gestion des erreurs, et logs détaillés.
    """

    def __init__(self, base_dir: Path = None):
        # Détermination robuste de la racine du projet
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.config_dir = self.base_dir / "config"

        # Vérification de l'existence du dossier de configuration
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Dossier de configuration introuvable : {self.config_dir}")

        # Chargement des lexiques et regex
        self.actes_df = self._load_csv_regex("lexiques/actes_normalises.csv", sep=";")
        self.medicaments_df = self._load_json_df("medicaments_normalises.json")
        self.calculs_codes_df = self._load_csv_regex("regex/calculs_codes_int.csv", sep=";")
        self.infos_financieres_df = self._load_csv_regex("regex/infos_financieres.csv", sep=";")
        self.metadonnees_df = self._load_csv_regex("regex/metadonnees.csv", sep=";", quotechar='"')
        self.parties_benef_df = self._load_csv_regex("regex/parties_benef.csv", sep=";")
        self.suivi_sla_df = self._load_csv_regex("regex/suivi_SLA.csv", sep=";")

        # Chargement des règles et formules
        self.regles_pc_df = self._load_regles("regles_prise_en_charge.csv", sep=";")
        self.mapping_amv = self._load_json("mapping_amv_pennypet.json")
        self.formules = self._load_json("formules_pennypet.json")

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
        # Si le JSON est un dict, on prend ses valeurs
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
        # Renommage des colonnes métiers
        for old, new in {
            "Terme/Libellé": "field_label",
            "Regex OCR": "regex_pattern",
            "Variantes/Synonymes": "variantes"
        }.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        # Compilation sécurisée des patterns
        if "regex_pattern" in df.columns:
            def compile_or_none(pat: str):
                try:
                    return re.compile(pat, re.IGNORECASE) if pat and pat.lower() != "nan" else None
                except re.error:
                    return None
            df["pattern"] = df["regex_pattern"].astype(str).apply(compile_or_none)
        return df

    def _load_regles(self, relpath: str, **kwargs) -> pd.DataFrame:
        df = self._load_csv(relpath, **kwargs)
        # Colonnes à splitter en listes
        for col in ["exclusions", "actes_couverts", "conditions_speciales"]:
            if col in df.columns:
                df[col] = (df[col]
                           .fillna("")
                           .apply(lambda s: [v.strip() for v in s.split("|") if v.strip()] if s else []))
        # Conversion des colonnes numériques
        for col in ["taux_remboursement", "plafond_annuel"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
