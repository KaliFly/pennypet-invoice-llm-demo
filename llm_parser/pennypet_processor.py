import json
import re
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient
import unicodedata

# Configuration du logging avec fichier de debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pennypet_debug.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
    logger.info("RapidFuzz disponible - Fuzzy matching activé")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("RapidFuzz non disponible, fuzzy matching désactivé")

def pseudojson_to_json_ameliore(text: str) -> str:
    """
    Correction robuste pour JSON mal formé avec gestion d'erreurs étendue.
    """
    try:
        text = text.strip()
        
        # Corrections de base
        text = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
        text = text.replace("'", '"')
        text = re.sub(r',\s*([}\]])', r'\1', text)
        text = re.sub(r'}\s*{', '},{', text)
        text = re.sub(r']\s*\[', '],[', text)
        text = re.sub(r',,+', ',', text)
        text = re.sub(r'::+', ':', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        
        return text
        
    except Exception as e:
        logger.error(f"Erreur nettoyage JSON: {e}")
        return text

def extraire_json_robuste(content: str) -> dict:
    """
    Extraction JSON ultra-robuste avec méthodes de fallback.
    """
    # Méthode 1: Extraction standard
    try:
        start = content.find("{")
        if start < 0:
            raise ValueError("Pas de JSON trouvé")
        
        depth = 0
        json_str = None
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = content[start : i + 1]
                    break
        
        if not json_str:
            raise ValueError("JSON malformé")
        
        json_clean = pseudojson_to_json_ameliore(json_str)
        return json.loads(json_clean)
        
    except json.JSONDecodeError as e:
        logger.warning(f"Parsing JSON échoué: {e}")
        
        # Méthode 2: Reconstruction manuelle
        try:
            return reconstruire_json_manuellement(content)
        except Exception as e2:
            logger.error(f"Reconstruction manuelle échouée: {e2}")
            
            # Fallback: structure minimale
            return {
                "lignes": [{"code_acte": "ERREUR_JSON", "description": "Erreur parsing JSON", "montant_ht": 0.0}],
                "montant_total": 0.0,
                "informations_client": {}
            }

def reconstruire_json_manuellement(content: str) -> dict:
    """
    Reconstruction manuelle du JSON à partir de patterns.
    """
    result = {
        "lignes": [],
        "montant_total": 0.0,
        "informations_client": {}
    }
    
    try:
        # Patterns pour les lignes
        lignes_patterns = [
            r'"code_acte"\s*:\s*"([^"]*)".*?"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([0-9.]+)',
            r'"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([0-9.]+)'
        ]
        
        for pattern in lignes_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    if len(match) == 3:
                        result["lignes"].append({
                            "code_acte": match[0],
                            "description": match[1],
                            "montant_ht": float(match[2])
                        })
                    elif len(match) == 2:
                        result["lignes"].append({
                            "code_acte": match[0],
                            "description": match[0],
                            "montant_ht": float(match[1])
                        })
                break
        
        # Pattern pour montant total
        montant_match = re.search(r'"montant_total"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
        if montant_match:
            result["montant_total"] = float(montant_match.group(1))
        
        # Patterns pour informations client
        client_patterns = {
            "nom_proprietaire": r'"nom_proprietaire"\s*:\s*"([^"]*)"',
            "nom_animal": r'"nom_animal"\s*:\s*"([^"]*)"',
            "identification": r'"identification"\s*:\s*"([^"]*)"'
        }
        
        for key, pattern in client_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result["informations_client"][key] = match.group(1)
        
        logger.info(f"JSON reconstruit: {len(result['lignes'])} lignes")
        return result
        
    except Exception as e:
        logger.error(f"Erreur reconstruction: {e}")
        return {
            "lignes": [{"code_acte": "ERREUR", "description": "Erreur parsing", "montant_ht": 0.0}],
            "montant_total": 0.0,
            "informations_client": {}
        }

def normaliser_accents(texte: str) -> str:
    """
    Normalise les accents et caractères spéciaux.
    """
    if not texte:
        return ""
    
    texte_nfd = unicodedata.normalize('NFD', texte)
    texte_sans_accents = ''.join(c for c in texte_nfd if unicodedata.category(c) != 'Mn')
    texte_clean = re.sub(r'[^\w\s]', ' ', texte_sans_accents.lower())
    
    return ' '.join(texte_clean.split())

class NormaliseurAMVOptimise:
    """
    Normaliseur optimisé utilisant DIRECTEMENT les fichiers de configuration existants.
    Aucune duplication - utilise uniquement ce qui existe dans PennyPetConfig.
    """
    
    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.cache: Dict[str, Optional[str]] = {}
        
        # UTILISATION DIRECTE des DataFrames existants dans config
        self.actes_df = self._validate_dataframe(config.actes_df, "actes_df")
        self.medicaments_df = self._validate_dataframe(config.medicaments_df, "medicaments_df")
        self.calculs_codes_df = self._validate_dataframe(config.calculs_codes_df, "calculs_codes_df")
        self.infos_financieres_df = self._validate_dataframe(config.infos_financieres_df, "infos_financieres_df")
        self.metadonnees_df = self._validate_dataframe(config.metadonnees_df, "metadonnees_df")
        self.parties_benef_df = self._validate_dataframe(config.parties_benef_df, "parties_benef_df")
        self.suivi_sla_df = self._validate_dataframe(config.suivi_sla_df, "suivi_sla_df")
        
        # UTILISATION DIRECTE des autres éléments existants
        self.glossaire_pharmaceutique = config.glossaire_pharmaceutique
        self.mapping_amv = config.mapping_amv
        self.formules = config.formules
        
        # Extraction intelligente des termes d'actes depuis les DataFrames existants
        self.termes_actes = self._extraire_termes_actes()
        
        # Normalisation du glossaire pharmaceutique existant
        self.glossaire_normalise = self._normaliser_glossaire_existant()
        
        # Patterns dynamiques basés sur les données existantes
        self.patterns_medicaments = self._construire_patterns_medicaments()
        self.patterns_actes = self._construire_patterns_actes()
        
        logger.info(f"Normaliseur initialisé: {len(self.termes_actes)} actes, "
                   f"{len(self.glossaire_pharmaceutique)} médicaments, "
                   f"{len(self.glossaire_normalise)} entrées normalisées")

    def _validate_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Valide et nettoie les DataFrames existants"""
        if df is None or df.empty:
            logger.warning(f"{name} est vide ou None")
            return pd.DataFrame()
        
        logger.info(f"{name} chargé: {len(df)} lignes, colonnes: {list(df.columns)}")
        return df

    def _extraire_termes_actes(self) -> set:
        """Extrait intelligemment les termes d'actes depuis TOUS les DataFrames existants"""
        termes = set()
        
        # Extraction depuis actes_df
        if not self.actes_df.empty:
            for col in ['field_label', 'label', 'acte', 'description', 'libelle']:
                if col in self.actes_df.columns:
                    termes.update(self.actes_df[col].dropna().astype(str).str.lower())
                    logger.info(f"Termes d'actes extraits de {col}: {len(termes)} total")
                    break
        
        # Extraction depuis calculs_codes_df (codes d'actes)
        if not self.calculs_codes_df.empty:
            for col in ['field_label', 'code', 'description']:
                if col in self.calculs_codes_df.columns:
                    termes.update(self.calculs_codes_df[col].dropna().astype(str).str.lower())
        
        # Extraction depuis metadonnees_df (métadonnées d'actes)
        if not self.metadonnees_df.empty:
            for col in ['field_label', 'type', 'categorie']:
                if col in self.metadonnees_df.columns:
                    termes.update(self.metadonnees_df[col].dropna().astype(str).str.lower())
        
        return termes

    def _normaliser_glossaire_existant(self) -> Dict[str, str]:
        """Normalise le glossaire pharmaceutique EXISTANT"""
        glossaire_normalise = {}
        
        try:
            for terme in self.glossaire_pharmaceutique:
                if not terme:
                    continue
                
                # Normalisation principale
                terme_norm = normaliser_accents(str(terme))
                if terme_norm:
                    glossaire_normalise[terme_norm] = terme
                
                # Variantes courantes
                variantes = self._generer_variantes_basiques(str(terme))
                for variante in variantes:
                    variante_norm = normaliser_accents(variante)
                    if variante_norm:
                        glossaire_normalise[variante_norm] = terme
        
        except Exception as e:
            logger.error(f"Erreur normalisation glossaire: {e}")
        
        return glossaire_normalise

    def _generer_variantes_basiques(self, terme: str) -> List[str]:
        """Génère des variantes de base pour un terme"""
        variantes = [terme]
        
        # Avec/sans 's' final
        if terme.endswith('s'):
            variantes.append(terme[:-1])
        else:
            variantes.append(terme + 's')
        
        # Abréviations courantes
        abbrevs = {
            'comprimé': 'cp', 'gélule': 'gél', 'solution': 'sol',
            'injection': 'inj', 'milligramme': 'mg', 'millilitre': 'ml'
        }
        
        for complet, abrege in abbrevs.items():
            if complet in terme.lower():
                variantes.append(terme.lower().replace(complet, abrege))
        
        return variantes

    def _construire_patterns_medicaments(self) -> List[str]:
        """Construit des patterns dynamiques basés sur les données existantes"""
        patterns = [
            r'\b\d+\s*(mg|ml|g|l|ui|iu|mcg|µg)\b',
            r'\b(comprimé|gélule|cp|gél|sol|inj|ampoule|flacon|tube|boîte)\b',
            r'\b(vaccin|vaccination|antibiotic|anti-inflammatoire)\b'
        ]
        
        # Ajouter des patterns basés sur mapping_amv si disponible
        if self.mapping_amv:
            for key in self.mapping_amv.keys():
                if key.lower() in ['medicament', 'produit', 'substance']:
                    patterns.append(rf'\b{re.escape(key.lower())}\b')
        
        return patterns

    def _construire_patterns_actes(self) -> List[str]:
        """Construit des patterns pour les actes depuis les données existantes"""
        patterns = [
            r'\b(consultation|examen|visite|contrôle)\b',
            r'\b(chirurgie|opération|intervention)\b',
            r'\b(radio|échographie|scanner)\b'
        ]
        
        # Ajouter des patterns depuis les termes d'actes trouvés
        if self.termes_actes:
            # Prendre les termes les plus fréquents pour créer des patterns
            termes_frequents = list(self.termes_actes)[:50]  # Limite pour performance
            for terme in termes_frequents:
                if len(terme) > 3:  # Éviter les termes trop courts
                    patterns.append(rf'\b{re.escape(terme)}\b')
        
        return patterns

    def normalise_acte(self, libelle_brut: str) -> Optional[str]:
        """Normalise un acte en utilisant les DataFrames existants"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Recherche dans actes_df avec patterns compilés
            if not self.actes_df.empty and 'pattern' in self.actes_df.columns:
                for _, row in self.actes_df.iterrows():
                    pattern = row.get("pattern")
                    if pattern and hasattr(pattern, 'search'):
                        try:
                            if pattern.search(cle):
                                code_acte = row.get("code_acte", row.get("field_label", cle))
                                self.cache[cle] = code_acte
                                return code_acte
                        except Exception:
                            continue
            
            # 2. Recherche sémantique dans termes extraits
            for terme in self.termes_actes:
                try:
                    terme_norm = normaliser_accents(terme)
                    if terme_norm in libelle_norm:
                        code = terme.upper()
                        self.cache[cle] = code
                        return code
                except Exception:
                    continue
            
            # 3. Patterns dynamiques pour actes
            for pattern in self.patterns_actes:
                if re.search(pattern, libelle_norm, re.IGNORECASE):
                    self.cache[cle] = "ACTE_MEDICAL"
                    return "ACTE_MEDICAL"
            
            # 4. Fuzzy matching si disponible
            if RAPIDFUZZ_AVAILABLE and not self.actes_df.empty:
                try:
                    codes_col = 'code_acte' if 'code_acte' in self.actes_df.columns else 'field_label'
                    if codes_col in self.actes_df.columns:
                        codes = self.actes_df[codes_col].dropna().astype(str).tolist()
                        if codes:
                            match, score, _ = process.extractOne(cle, codes, scorer=fuzz.token_sort_ratio)
                            if score >= 80:
                                self.cache[cle] = match
                                return match
                except Exception:
                    pass
            
            self.cache[cle] = None
            return None
            
        except Exception as e:
            logger.error(f"Erreur normalisation acte '{libelle_brut}': {e}")
            return None

    def normalise_medicament(self, libelle_brut: str) -> Optional[str]:
        """Normalise un médicament en utilisant le glossaire existant"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Patterns regex pour médicaments
            for pattern in self.patterns_medicaments:
                if re.search(pattern, libelle_norm, re.IGNORECASE):
                    self.cache[cle] = "MEDICAMENTS"
                    return "MEDICAMENTS"
            
            # 2. Recherche exacte dans glossaire normalisé
            if libelle_norm in self.glossaire_normalise:
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
            
            # 3. Recherche partielle dans glossaire
            for terme_norm in self.glossaire_normalise.keys():
                try:
                    if (terme_norm in libelle_norm or 
                        libelle_norm in terme_norm or
                        any(word in libelle_norm for word in terme_norm.split() if len(word) > 3)):
                        self.cache[cle] = "MEDICAMENTS"
                        return "MEDICAMENTS"
                except Exception:
                    continue
            
            # 4. Recherche dans medicaments_df si disponible
            if not self.medicaments_df.empty:
                for col in ['medicament', 'nom', 'designation', 'libelle']:
                    if col in self.medicaments_df.columns:
                        meds_normalises = [normaliser_accents(str(m)) for m in self.medicaments_df[col].dropna()]
                        if libelle_norm in meds_normalises:
                            self.cache[cle] = "MEDICAMENTS"
                            return "MEDICAMENTS"
                        break
            
            # 5. Fuzzy matching sur glossaire
            if RAPIDFUZZ_AVAILABLE and self.glossaire_normalise:
                try:
                    match, score, _ = process.extractOne(
                        libelle_norm, 
                        list(self.glossaire_normalise.keys()), 
                        scorer=fuzz.partial_ratio
                    )
                    if score >= 85:
                        self.cache[cle] = "MEDICAMENTS"
                        return "MEDICAMENTS"
                except Exception:
                    pass
            
            self.cache[cle] = None
            return None
            
        except Exception as e:
            logger.error(f"Erreur normalisation médicament '{libelle_brut}': {e}")
            return None

    def normalise(self, libelle_brut: str) -> Optional[str]:
        """Normalise un libellé (acte ou médicament)"""
        if not libelle_brut:
            return None
        
        try:
            # Priorité : actes d'abord, puis médicaments
            result = self.normalise_acte(libelle_brut)
            if result:
                return result
            
            result = self.normalise_medicament(libelle_brut)
            if result:
                return result
            
            # Fallback
            return str(libelle_brut).strip().upper()
            
        except Exception as e:
            logger.error(f"Erreur normalisation '{libelle_brut}': {e}")
            return str(libelle_brut).strip().upper() if libelle_brut else None

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de mapping"""
        return {
            "cache_size": len(self.cache),
            "termes_actes": len(self.termes_actes),
            "glossaire_pharmaceutique": len(self.glossaire_pharmaceutique),
            "glossaire_normalise": len(self.glossaire_normalise),
            "patterns_medicaments": len(self.patterns_medicaments),
            "patterns_actes": len(self.patterns_actes),
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
            "dataframes_charges": {
                "actes_df": len(self.actes_df),
                "medicaments_df": len(self.medicaments_df),
                "calculs_codes_df": len(self.calculs_codes_df),
                "infos_financieres_df": len(self.infos_financieres_df),
                "metadonnees_df": len(self.metadonnees_df),
                "parties_benef_df": len(self.parties_benef_df),
                "suivi_sla_df": len(self.suivi_sla_df)
            }
        }

class PennyPetProcessor:
    """
    Processeur principal utilisant DIRECTEMENT la configuration existante.
    """
    
    def __init__(
        self,
        client_qwen: OpenRouterClient = None,
        client_mistral: OpenRouterClient = None,
        config: PennyPetConfig = None,
    ):
        try:
            # Utilisation DIRECTE de la config existante
            self.config = config or PennyPetConfig()
            
            # Clients LLM
            try:
                self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
            except Exception as e:
                logger.warning(f"Client Qwen non disponible: {e}")
                self.client_qwen = None
                
            try:
                self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
            except Exception as e:
                logger.warning(f"Client Mistral non disponible: {e}")
                self.client_mistral = None
            
            # UTILISATION DIRECTE des DataFrames de règles existants
            self.regles_pc_df = self.config.regles_pc_df
            
            # Normaliseur optimisé utilisant tous les fichiers existants
            self.normaliseur = NormaliseurAMVOptimise(self.config)
            
            # Statistiques
            self.stats = {
                'lignes_traitees': 0,
                'medicaments_detectes': 0,
                'actes_detectes': 0,
                'erreurs_normalisation': 0
            }
            
            logger.info("PennyPetProcessor initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation PennyPetProcessor: {e}")
            raise

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """Extrait les lignes avec parsing JSON robuste"""
        try:
            # Sélection du client
            if llm_provider.lower() == "qwen" and self.client_qwen:
                client = self.client_qwen
            elif llm_provider.lower() == "mistral" and self.client_mistral:
                client = self.client_mistral
            else:
                raise ValueError(f"Client {llm_provider} non disponible")
            
            # Appel LLM
            resp = client.analyze_invoice_image(image_bytes, formule)
            content = resp.choices[0].message.content
            
            if not content:
                raise ValueError("Réponse vide du LLM")
            
            # Logs de debug
            logger.info(f"Réponse LLM (150 premiers chars): {content[:150]}")
            logger.info(f"Réponse LLM (150 derniers chars): {content[-150:]}")
            
            # Extraction JSON robuste
            try:
                data = extraire_json_robuste(content)
            except Exception as e:
                logger.error(f"Erreur extraction JSON: {e}")
                # Structure minimale en cas d'échec
                data = {
                    "lignes": [{"code_acte": "ERREUR_JSON", "description": "Erreur parsing", "montant_ht": 0.0}],
                    "montant_total": 0.0,
                    "informations_client": {}
                }
            
            # Validation et nettoyage
            if "lignes" not in data:
                raise ValueError("Pas de lignes dans les données extraites")
            
            # Nettoyage des lignes
            for ligne in data["lignes"]:
                try:
                    ligne["montant_ht"] = float(ligne.get("montant_ht", 0))
                except (ValueError, TypeError):
                    ligne["montant_ht"] = 0.0
                
                for key in ["code_acte", "description"]:
                    if key in ligne:
                        ligne[key] = str(ligne[key]).strip()
            
            return data, content
            
        except Exception as e:
            logger.error(f"Erreur extract_lignes_from_image: {e}")
            raise

    def calculer_remboursement(
        self, montant: float, code_acte: str, formule: str, est_accident: bool
    ) -> Dict[str, Any]:
        """Calcule le remboursement en utilisant regles_pc_df existant"""
        if self.regles_pc_df.empty:
            logger.warning("regles_pc_df vide, calcul simplifié")
            return self._calcul_remboursement_simplifie(montant, code_acte, formule, est_accident)
        
        try:
            df = self.regles_pc_df.copy()
            
            # Masque de filtrage
            mask = (
                (df["formule"] == formule)
                & (
                    df["code_acte"].eq(code_acte)
                    | (
                        df["code_acte"].fillna("ALL").eq("ALL")
                        & df["actes_couverts"].apply(lambda l: code_acte in l if isinstance(l, list) else False)
                    )
                )
                & (
                    (df["type_couverture"] == "ACCIDENT_MALADIE")
                    | ((df["type_couverture"] == "ACCIDENT_SEULEMENT") & est_accident)
                )
            )
            
            reg = df[mask]
            if reg.empty:
                return self._calcul_remboursement_simplifie(montant, code_acte, formule, est_accident)
            
            # Calcul selon règles
            r = reg.iloc[0]
            taux = r["taux_remboursement"] / 100
            plafond = r.get("plafond_annuel", float('inf'))
            
            remb_brut = montant * taux
            remb_final = min(remb_brut, plafond)
            
            return {
                "montant_ht": montant,
                "taux": taux * 100,
                "remb_final": remb_final,
                "reste": montant - remb_final
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul remboursement: {e}")
            return self._calcul_remboursement_simplifie(montant, code_acte, formule, est_accident)

    def _calcul_remboursement_simplifie(self, montant: float, code_acte: str, formule: str, est_accident: bool) -> Dict[str, Any]:
        """Calcul de remboursement simplifié selon les règles PennyPet standard"""
        try:
            if formule == "START":
                taux, plafond = 0, 0
            elif formule == "PREMIUM":
                taux, plafond = (100, 500) if est_accident else (0, 0)
            elif formule == "INTEGRAL":
                taux, plafond = 50, 1000
            elif formule == "INTEGRAL_PLUS":
                taux, plafond = 100, 1000
            else:
                taux, plafond = 0, 0
            
            remb_brut = montant * (taux / 100)
            remb_final = min(remb_brut, plafond)
            
            return {
                "montant_ht": montant,
                "taux": taux,
                "remb_final": remb_final,
                "reste": montant - remb_final
            }
        except Exception as e:
            logger.error(f"Erreur calcul simplifié: {e}")
            return {
                "montant_ht": montant,
                "taux": 0,
                "remb_final": 0,
                "reste": montant
            }

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Traite une facture complète"""
        
        # Reset stats
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0
        }
        
        try:
            # Extraction données
            data, raw_content = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
            
            resultats: List[Dict[str, Any]] = []
            accidents = {"accident", "urgent", "urgence", "fract", "trauma", "traumatisme"}
            
            # Traitement des lignes
            for ligne in data["lignes"]:
                try:
                    libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
                    montant = float(ligne.get("montant_ht", 0) or 0)
                    
                    # Normalisation
                    code_norm = self.normaliseur.normalise(libelle)
                    
                    # Détection accident
                    est_acc = any(mot in libelle.lower() for mot in accidents)
                    
                    # Calcul remboursement
                    remb = self.calculer_remboursement(montant, code_norm, formule_client, est_acc)
                    
                    # Stats
                    self.stats['lignes_traitees'] += 1
                    if code_norm == "MEDICAMENTS":
                        self.stats['medicaments_detectes'] += 1
                    else:
                        self.stats['actes_detectes'] += 1
                    
                    # Résultats
                    ligne.update({
                        "code_norm": code_norm,
                        "est_accident": est_acc,
                        **remb
                    })
                    resultats.append(ligne)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    logger.error(f"Erreur ligne {ligne}: {e}")
                    continue
            
            # Totaux
            total_remb = sum(r.get("remb_final", 0) for r in resultats)
            total_facture = sum(r.get("montant_ht", 0) for r in resultats)
            
            return {
                "success": True,
                "lignes": resultats,
                "total_remb": total_remb,
                "total_facture": total_facture,
                "reste_a_charge": total_facture - total_remb,
                "stats": self.stats,
                "mapping_stats": self.normaliseur.get_mapping_stats(),
                "raw_llm_response": raw_content,
                "informations_client": data.get("informations_client", {})
            }
            
        except Exception as e:
            logger.error(f"Erreur process_facture_pennypet: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }

    def get_processor_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes"""
        return {
            **self.stats,
            **self.normaliseur.get_mapping_stats()
        }

# Instance globale
pennypet_processor = PennyPetProcessor()
