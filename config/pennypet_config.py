import pandas as pd
import re
import json
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.warning(f"Dossier de configuration introuvable : {self.config_dir}")
            self._init_empty_config()
            return

        try:
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

            # 5. Chargement du glossaire pharmaceutique ultra-exhaustif (VOTRE VERSION)
            self.glossaire_pharmaceutique = self._load_glossaire_pharmaceutique("glossaire_pharmaceutique.json")
            
            logger.info("Configuration PennyPet chargée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            self._init_empty_config()

    def _init_empty_config(self):
        """Initialise une configuration vide pour éviter les erreurs"""
        logger.warning("Initialisation avec une configuration vide")
        self.actes_df = pd.DataFrame(columns=['field_label', 'regex_pattern', 'pattern', 'code_acte'])
        self.medicaments_df = pd.DataFrame()
        self.calculs_codes_df = pd.DataFrame()
        self.infos_financieres_df = pd.DataFrame()
        self.metadonnees_df = pd.DataFrame()
        self.parties_benef_df = pd.DataFrame()
        self.suivi_sla_df = pd.DataFrame()
        self.regles_pc_df = pd.DataFrame()
        self.mapping_amv = {}
        self.formules = {}
        self.glossaire_pharmaceutique = set()

    def _load_json(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            logger.warning(f"Le fichier JSON {path} est manquant.")
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lecture JSON {path}: {e}")
            return {}

    def _load_json_df(self, filename: str) -> pd.DataFrame:
        path = self.config_dir / filename
        if not path.exists():
            logger.warning(f"Le fichier JSON {path} est manquant.")
            return pd.DataFrame()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())
            if not data:
                logger.warning(f"Aucune donnée trouvée dans : {path}")
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Erreur lecture JSON DataFrame {path}: {e}")
            return pd.DataFrame()

    def _load_csv(self, relpath: str, **kwargs) -> pd.DataFrame:
        path = self.config_dir / relpath
        if not path.exists():
            logger.warning(f"Le fichier CSV {path} est manquant.")
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8", **kwargs)
        except Exception as e:
            logger.error(f"Erreur de lecture du CSV {path} : {e}")
            return pd.DataFrame()

    def _load_csv_regex(self, relpath: str, **kwargs) -> pd.DataFrame:
        df = self._load_csv(relpath, **kwargs)
        
        if df.empty:
            return pd.DataFrame(columns=['field_label', 'regex_pattern', 'pattern', 'code_acte'])
        
        try:
            df.columns = [c.strip() for c in df.columns]
            rename_map = {
                "Terme/Libellé": "field_label",
                "Regex OCR": "regex_pattern",
                "Variantes/Synonymes": "variantes"
            }
            
            # Renommer uniquement les colonnes existantes
            actual_rename = {old: new for old, new in rename_map.items() if old in df.columns}
            if actual_rename:
                df.rename(columns=actual_rename, inplace=True)
            
            # S'assurer que field_label existe
            if "field_label" not in df.columns:
                # Chercher une colonne appropriée
                text_columns = df.select_dtypes(include=['object']).columns
                if len(text_columns) > 0:
                    df["field_label"] = df[text_columns[0]]
                    logger.info(f"Utilisation de '{text_columns[0]}' comme field_label pour {relpath}")
                else:
                    df["field_label"] = ""
                    logger.warning(f"Aucune colonne texte trouvée dans {relpath}")
            
            # S'assurer que code_acte existe
            if "code_acte" not in df.columns:
                if "field_label" in df.columns:
                    df["code_acte"] = df["field_label"].str.upper()
                else:
                    df["code_acte"] = ""
            
            # Compilation des regex
            if "regex_pattern" in df.columns:
                def compile_or_none(pat):
                    try:
                        if pd.isna(pat) or str(pat).lower() in ["nan", ""]:
                            return None
                        return re.compile(str(pat), re.IGNORECASE)
                    except (re.error, TypeError):
                        return None
                
                df["pattern"] = df["regex_pattern"].apply(compile_or_none)
            else:
                df["pattern"] = None
                
        except Exception as e:
            logger.error(f"Erreur traitement CSV regex {relpath}: {e}")
            return pd.DataFrame(columns=['field_label', 'regex_pattern', 'pattern', 'code_acte'])
        
        return df

    def _load_regles(self, relpath: str, **kwargs) -> pd.DataFrame:
        df = self._load_csv(relpath, **kwargs)
        
        if df.empty:
            return pd.DataFrame()
        
        try:
            for col in ["exclusions", "actes_couverts", "conditions_speciales"]:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .fillna("")
                        .apply(lambda s: [v.strip() for v in str(s).split("|") if v.strip()] if s else [])
                    )
            
            for col in ["taux_remboursement", "plafond_annuel"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        except Exception as e:
            logger.error(f"Erreur traitement règles {relpath}: {e}")
        
        return df

    def _load_glossaire_pharmaceutique(self, filename: str) -> set:
        """
        Charge le glossaire pharmaceutique ultra-exhaustif depuis un JSON,
        fusionne toutes les valeurs en un set de termes uniques.
        VERSION AMÉLIORÉE avec gestion d'erreurs
        """
        path = self.config_dir / filename
        if not path.exists():
            logger.warning(f"Le fichier de glossaire {path} est manquant.")
            return set()
        
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Fusionne toutes les valeurs (listes) en un set unique
            termes = set()
            for v in data.values():
                if isinstance(v, list):
                    termes.update([t.strip().lower() for t in v if t and t.strip()])
                elif isinstance(v, str) and v.strip():
                    termes.add(v.strip().lower())
            
            logger.info(f"Glossaire pharmaceutique chargé: {len(termes)} termes")
            return termes
            
        except Exception as e:
            logger.error(f"Erreur chargement glossaire {path}: {e}")
            return set()
