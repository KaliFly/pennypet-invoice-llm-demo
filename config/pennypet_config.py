import pandas as pd
import re
import json
from pathlib import Path

class PennyPetConfig:
    def __init__(self, base_dir: Path = Path(__file__).parent.parent):
        self.base_dir      = base_dir
        self.config_dir    = base_dir / "config"

        # Lexiques OCR
        self.actes_df             = self._load_csv_regex("lexiques/actes_normalises.csv", sep=";")
        # Utilisation du JSON pour les médicaments (dans config/, pas lexiques/)
        self.medicaments_df       = self._load_json_df("medicaments_normalises.json")
        self.calculs_codes_df     = self._load_csv_regex("regex/calculs_codes_int.csv", sep=";")
        self.infos_financieres_df = self._load_csv_regex("regex/infos_financieres.csv", sep=";")
        self.metadonnees_df       = self._load_csv_regex("regex/metadonnees.csv", sep=";", quotechar='"')
        self.parties_benef_df     = self._load_csv_regex("regex/parties_benef.csv", sep=";")
        self.suivi_sla_df         = self._load_csv_regex("regex/suivi_SLA.csv", sep=";")

        # Règles et formules
        self.regles_pc_df = self._load_regles("regles_prise_en_charge.csv", sep=";")
        self.mapping_amv  = self._load_json("mapping_amv_pennypet.json")
        self.formules     = self._load_json("formules_pennypet.json")

    def _load_json(self, filename: str) -> dict:
        path = self.config_dir / filename
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_json_df(self, filename: str) -> pd.DataFrame:
        # Charger directement depuis config/, pas config/lexiques/
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Le fichier {path} est manquant.")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _load_csv(self, relpath: str, **kwargs) -> pd.DataFrame:
        path = self.config_dir / relpath
        if not path.exists():
            raise FileNotFoundError(f"Le fichier {path} est manquant.")
        return pd.read_csv(path, encoding="utf-8", **kwargs)

    def _load_csv_regex(self, relpath: str, **kwargs) -> pd.DataFrame:
        df = self._load_csv(relpath, **kwargs)
        df.columns = [c.strip() for c in df.columns]
        for old, new in {
            "Terme/Libellé": "field_label",
            "Regex OCR": "regex_pattern",
            "Variantes/Synonymes": "variantes"
        }.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        if "regex_pattern" in df.columns:
            df["pattern"] = df["regex_pattern"].astype(str) \
                .apply(lambda p: re.compile(p, re.IGNORECASE) if p and p != "nan" else None)
        return df

    def _load_regles(self, relpath: str, **kwargs) -> pd.DataFrame:
        df = self._load_csv(relpath, **kwargs)
        for col in ["exclusions", "actes_couverts", "conditions_speciales"]:
            if col in df.columns:
                df[col] = df[col].fillna("").apply(lambda s: s.split("|") if s else [])
        return df
