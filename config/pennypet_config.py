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
        # 1. Détermination robuste de la racine du projet
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.config_dir = self.base_dir / "config"

        # 2. Vérification de l'existence du dossier de configuration
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Dossier de configuration introuvable : {self.config_dir}")

        # 3. Chargement des lexiques et regex
        self.actes_df = self._load_csv_regex("lexiques/actes_normalises.csv", sep=";")
        self.medicaments_df = self._load_json_df("medicaments_normalises.json")
        self.calculs_codes_df = self._load_csv_regex("regex/calculs_codes_int.csv", sep=";")
        self.infos_financieres_df = self._load_csv_regex("regex/infos_financieres.csv", sep=";")
        self.metadonnees_df = self._load_csv_regex("regex/metadonnees.csv", sep=";", quotechar='"')
        self.parties_benef_df = self._load_csv_regex("regex/parties_benef.csv", sep=";")
        self.suivi_sla_df = self._load_csv_regex("regex/suivi_SLA.csv", sep=";")

        # 4. Chargement des règles et formules
        self.regles_pc_df = self._load_regles("regles_prise_en_charge.csv", sep=";")
        self.mapping_amv = self._load_json("mapping_amv_pennypet.json")
        self.formules = self._load_json("formules_pennypet.json")

        # 5. Chargement du glossaire pharmaceutique ultra-exhaustif
        self.glossaire_pharmaceutique = self._load_glossaire_pharmaceutique("glossaire_pharmaceutique.json")

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
            raise ValueError(f"Aucune donnée trouvée dans : {path}")
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
        rename_map = {
            "Terme/Libellé": "field_label",
            "Regex OCR": "regex_pattern",
            "Variantes/Synonymes": "variantes"
        }
        df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns}, inplace=True)
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

    def _load_glossaire_pharmaceutique(self, filename: str) -> set:
        """
        Charge le glossaire pharmaceutique ultra-exhaustif depuis un JSON,
        fusionne toutes les valeurs en un set de termes uniques.
        """
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Le fichier de glossaire {path} est manquant.")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Fusionne toutes les valeurs (listes) en un set unique
        termes = set()
        for v in data.values():
            if isinstance(v, list):
                termes.update([t.strip().lower() for t in v if t.strip()])
        return termes
