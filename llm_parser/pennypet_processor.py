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
        # Nettoyage initial
        text = text.strip()
        
        # 1. Correction des propriétés non quotées
        text = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
        
        # 2. Remplacement des guillemets simples par doubles
        text = text.replace("'", '"')
        
        # 3. Suppression des virgules avant fermantes
        text = re.sub(r',\s*([}\]])', r'\1', text)
        
        # 4. Correction des virgules manquantes entre objets
        text = re.sub(r'}\s*{', '},{', text)
        text = re.sub(r']\s*\[', '],[', text)
        
        # 5. Correction des virgules manquantes après valeurs
        text = re.sub(r'(".*?")\s*\n\s*(".*?")', r'\1,\n\2', text)
        text = re.sub(r'(\d+\.?\d*)\s*\n\s*(".*?")', r'\1,\n\2', text)
        
        # 6. Suppression des doubles virgules
        text = re.sub(r',,+', ',', text)
        
        # 7. Correction des deux points multiples
        text = re.sub(r'::+', ':', text)
        
        # 8. Suppression des caractères non ASCII problématiques
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        
        # 9. Correction des nombres mal formatés
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        
        return text
        
    except Exception as e:
        logger.error(f"Erreur nettoyage JSON: {e}")
        return text

def extraire_json_robuste(content: str) -> dict:
    """
    Extraction JSON ultra-robuste avec plusieurs méthodes de fallback.
    """
    logger.info(f"Début extraction JSON - Longueur contenu: {len(content)}")
    
    # Méthode 1: Extraction standard avec nettoyage amélioré
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
        
        logger.info(f"JSON extrait - Longueur: {len(json_str)}")
        
        # Nettoyage et tentative de parsing
        json_clean = pseudojson_to_json_ameliore(json_str)
        data = json.loads(json_clean)
        
        logger.info("JSON parsé avec succès (méthode 1)")
        return data
        
    except json.JSONDecodeError as e:
        logger.warning(f"Parsing JSON échoué (méthode 1): {e}")
        logger.warning(f"Position erreur: ligne {getattr(e, 'lineno', 'N/A')}, colonne {getattr(e, 'colno', 'N/A')}")
        
        # Méthode 2: Reconstruction manuelle à partir de patterns
        try:
            logger.info("Tentative de reconstruction manuelle du JSON")
            data = reconstruire_json_manuellement(content)
            logger.info("JSON reconstruit manuellement avec succès")
            return data
        except Exception as e2:
            logger.error(f"Reconstruction manuelle échouée: {e2}")
            
            # Méthode 3: Fallback structure minimale
            logger.warning("Utilisation de la structure de fallback")
            return {
                "lignes": [{"code_acte": "ERREUR_JSON", "description": "Erreur parsing JSON", "montant_ht": 0.0}],
                "montant_total": 0.0,
                "informations_client": {}
            }

def reconstruire_json_manuellement(content: str) -> dict:
    """
    Reconstruction manuelle du JSON à partir de patterns connus.
    """
    result = {
        "lignes": [],
        "montant_total": 0.0,
        "informations_client": {}
    }
    
    try:
        # Extraction des lignes avec regex robuste
        lignes_patterns = [
            r'"code_acte"\s*:\s*"([^"]*)".*?"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([0-9.]+)',
            r'"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([0-9.]+)',
            r'acte.*?:\s*"([^"]*)".*?montant.*?:\s*([0-9.]+)',
            r'"([^"]*)".*?(\d+\.?\d*)\s*€'
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
        
        # Si aucune ligne trouvée, essayer une approche plus simple
        if not result["lignes"]:
            montants = re.findall(r'(\d+\.?\d*)\s*€', content)
            descriptions = re.findall(r'"([^"]{5,})"', content)
            
            for i, montant in enumerate(montants[:5]):  # Limiter à 5 lignes max
                desc = descriptions[i] if i < len(descriptions) else f"Ligne {i+1}"
                result["lignes"].append({
                    "code_acte": desc,
                    "description": desc,
                    "montant_ht": float(montant)
                })
        
        # Calcul du montant total
        if result["lignes"]:
            result["montant_total"] = sum(ligne["montant_ht"] for ligne in result["lignes"])
        
        # Extraction des informations client avec patterns flexibles
        client_patterns = {
            "nom_proprietaire": [
                r'"nom_proprietaire"\s*:\s*"([^"]*)"',
                r'proprietaire.*?:\s*"([^"]*)"',
                r'nom.*?:\s*"([^"]*)"'
            ],
            "nom_animal": [
                r'"nom_animal"\s*:\s*"([^"]*)"',
                r'animal.*?:\s*"([^"]*)"',
                r'pet.*?:\s*"([^"]*)"'
            ],
            "identification": [
                r'"identification"\s*:\s*"([^"]*)"',
                r'identification.*?:\s*"([^"]*)"',
                r'id.*?:\s*"([^"]*)"'
            ]
        }
        
        for key, patterns in client_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    result["informations_client"][key] = match.group(1)
                    break
        
        logger.info(f"JSON reconstruit manuellement avec {len(result['lignes'])} lignes")
        return result
        
    except Exception as e:
        logger.error(f"Erreur reconstruction manuelle: {e}")
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

class NormaliseurAMVAmeliore:
    """
    Normaliseur amélioré utilisant tous les fichiers de configuration PennyPet
    """
    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.cache: Dict[str, Optional[str]] = {}
        
        # Récupération sécurisée de tous les éléments de configuration
        self.termes_actes = self._get_termes_actes_safe(config)
        self.actes_df = self._get_actes_df_safe(config)
        
        # Utilisation du glossaire pharmaceutique existant
        self.termes_medicaments = getattr(config, 'glossaire_pharmaceutique', set())
        self.medicaments_df = getattr(config, 'medicaments_df', pd.DataFrame())
        self.mapping_amv = getattr(config, 'mapping_amv', {})
        
        # Intégration de TOUS les fichiers de configuration
        self.calculs_codes_df = getattr(config, 'calculs_codes_df', pd.DataFrame())
        self.infos_financieres_df = getattr(config, 'infos_financieres_df', pd.DataFrame())
        self.metadonnees_df = getattr(config, 'metadonnees_df', pd.DataFrame())
        self.parties_benef_df = getattr(config, 'parties_benef_df', pd.DataFrame())
        self.suivi_sla_df = getattr(config, 'suivi_sla_df', pd.DataFrame())
        self.formules = getattr(config, 'formules', {})
        
        # Préprocessage du glossaire
        self.glossaire_normalise = self._preprocess_glossaire()
        
        # Patterns regex étendus
        self.patterns_medicaments = [
            r'\b\d+\s*(mg|ml|g|l|ui|iu|mcg|µg|mg/ml|ui/ml)\b',
            r'\b(comprimé|gélule|cp|gél|sol|inj|ampoule|flacon|tube|boîte|sachet|pipette)\.?\s*\d*',
            r'\b(antibiotic|anti-inflammatoire|antiparasitaire|antifongique|antiviral|vermifuge)\b',
            r'\b(vaccin|vaccination|rappel|primo-vaccination|sérum|immunoglobuline)\b',
            r'\b(seringue|pipette|spray|pommade|crème|lotion|collyre|gouttes)\b',
            r'\b\d+\s*x\s*\d+\s*(mg|ml|g|l|cp|gél)\b',
            r'\b(principe|actif|laboratoire|generique|specialite|marque)\b'
        ]
        
        self.patterns_actes = [
            r'\b(consultation|examen|visite|contrôle|bilan)\b',
            r'\b(chirurgie|opération|intervention|anesthésie)\b',
            r'\b(radio|échographie|scanner|irm|endoscopie)\b',
            r'\b(analyse|prélèvement|biopsie|cytologie)\b',
            r'\b(hospitalisation|perfusion|soin|pansement)\b'
        ]
        
        logger.info(f"Normaliseur initialisé: {len(self.termes_actes)} actes, {len(self.termes_medicaments)} médicaments")

    def _get_termes_actes_safe(self, config: PennyPetConfig) -> set:
        """Récupère les termes d'actes depuis tous les fichiers"""
        termes = set()
        
        try:
            # Depuis actes_df
            if hasattr(config, 'actes_df') and not config.actes_df.empty:
                df = config.actes_df
                for col in ['field_label', 'label', 'acte', 'description', 'libelle']:
                    if col in df.columns:
                        termes.update(df[col].dropna().astype(str).str.lower())
                        break
            
            # Depuis calculs_codes_df
            if hasattr(config, 'calculs_codes_df') and not config.calculs_codes_df.empty:
                df = config.calculs_codes_df
                for col in ['field_label', 'description']:
                    if col in df.columns:
                        termes.update(df[col].dropna().astype(str).str.lower())
            
            logger.info(f"Total termes d'actes: {len(termes)}")
            
        except Exception as e:
            logger.error(f"Erreur extraction termes actes: {e}")
        
        return termes

    def _get_actes_df_safe(self, config: PennyPetConfig) -> pd.DataFrame:
        """Récupère le DataFrame des actes"""
        try:
            if hasattr(config, 'actes_df') and not config.actes_df.empty:
                df = config.actes_df
                if 'pattern' in df.columns:
                    return df.dropna(subset=["pattern"])
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erreur extraction DataFrame actes: {e}")
            return pd.DataFrame()

    def _preprocess_glossaire(self) -> Dict[str, str]:
        """Préprocesse le glossaire pharmaceutique"""
        glossaire_normalise = {}
        
        try:
            for terme in self.termes_medicaments:
                if not terme:
                    continue
                terme_norm = normaliser_accents(str(terme))
                if terme_norm:
                    glossaire_normalise[terme_norm] = terme
            
            # Depuis medicaments_df
            if not self.medicaments_df.empty and 'medicament' in self.medicaments_df.columns:
                for medicament in self.medicaments_df['medicament'].dropna():
                    terme_norm = normaliser_accents(str(medicament))
                    if terme_norm:
                        glossaire_normalise[terme_norm] = str(medicament)
                        
        except Exception as e:
            logger.error(f"Erreur préprocessing glossaire: {e}")
        
        logger.info(f"Glossaire normalisé: {len(glossaire_normalise)} entrées")
        return glossaire_normalise

    def _detecter_patterns_medicaments(self, texte: str) -> bool:
        """Détecte les patterns de médicaments"""
        try:
            texte_norm = normaliser_accents(texte)
            return any(re.search(pattern, texte_norm, re.IGNORECASE) for pattern in self.patterns_medicaments)
        except Exception:
            return False

    def _detecter_patterns_actes(self, texte: str) -> bool:
        """Détecte les patterns d'actes"""
        try:
            texte_norm = normaliser_accents(texte)
            return any(re.search(pattern, texte_norm, re.IGNORECASE) for pattern in self.patterns_actes)
        except Exception:
            return False

    def normalise_acte(self, libelle_brut: str) -> Optional[str]:
        """Normalise un acte médical"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Patterns regex dans actes_df
            if not self.actes_df.empty:
                for _, row in self.actes_df.iterrows():
                    pattern = row.get("pattern")
                    if pattern and hasattr(pattern, 'search'):
                        try:
                            if pattern.search(cle):
                                code_acte = row.get("code_acte", cle)
                                self.cache[cle] = code_acte
                                return code_acte
                        except Exception:
                            continue
            
            # 2. Patterns prédéfinis
            if self._detecter_patterns_actes(libelle_brut):
                self.cache[cle] = "ACTE_MEDICAL"
                return "ACTE_MEDICAL"
            
            # 3. Recherche sémantique
            for terme in self.termes_actes:
                try:
                    terme_norm = normaliser_accents(terme)
                    if re.search(rf"(?<!\w){re.escape(terme_norm)}(?!\w)", libelle_norm):
                        code = terme.upper()
                        self.cache[cle] = code
                        return code
                except Exception:
                    continue
            
            # 4. Fuzzy matching
            if RAPIDFUZZ_AVAILABLE and not self.actes_df.empty:
                try:
                    codes = self.actes_df["code_acte"].dropna().astype(str).tolist()
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
        """Normalise un médicament"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Patterns regex
            if self._detecter_patterns_medicaments(libelle_brut):
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
            
            # 2. Glossaire exact
            if libelle_norm in self.glossaire_normalise:
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
            
            # 3. Recherche partielle
            for terme_norm in self.glossaire_normalise.keys():
                try:
                    if (terme_norm in libelle_norm or 
                        libelle_norm in terme_norm or
                        any(word in libelle_norm for word in terme_norm.split() if len(word) > 3)):
                        self.cache[cle] = "MEDICAMENTS"
                        return "MEDICAMENTS"
                except Exception:
                    continue
            
            # 4. Fuzzy matching
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
        """Normalise un libellé avec priorité intelligente"""
        if not libelle_brut:
            return None
            
        try:
            libelle_norm = normaliser_accents(libelle_brut)
            
            # Détection rapide du type
            indicateurs_medicaments = ['mg', 'ml', 'comprimé', 'gélule', 'flacon', 'injection']
            indicateurs_actes = ['consultation', 'examen', 'visite', 'chirurgie', 'radio']
            
            est_medicament = any(ind in libelle_norm for ind in indicateurs_medicaments)
            est_acte = any(ind in libelle_norm for ind in indicateurs_actes)
            
            if est_medicament and not est_acte:
                result = self.normalise_medicament(libelle_brut)
                if result:
                    return result
                result = self.normalise_acte(libelle_brut)
                if result:
                    return result
            else:
                result = self.normalise_acte(libelle_brut)
                if result:
                    return result
                result = self.normalise_medicament(libelle_brut)
                if result:
                    return result
            
            return str(libelle_brut).strip().upper()
            
        except Exception as e:
            logger.error(f"Erreur normalisation '{libelle_brut}': {e}")
            return str(libelle_brut).strip().upper() if libelle_brut else None

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Statistiques complètes"""
        return {
            "cache_size": len(self.cache),
            "actes": len(self.termes_actes),
            "medicaments": len(self.termes_medicaments),
            "glossaire_normalise": len(self.glossaire_normalise),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE,
            "dataframes": {
                "actes_df": len(self.actes_df),
                "medicaments_df": len(self.medicaments_df),
                "calculs_codes_df": len(self.calculs_codes_df),
                "infos_financieres_df": len(self.infos_financieres_df)
            }
        }

class PennyPetProcessor:
    """
    Pipeline complet PennyPet avec gestion robuste des erreurs JSON
    """
    def __init__(
        self,
        client_qwen: OpenRouterClient = None,
        client_mistral: OpenRouterClient = None,
        config: PennyPetConfig = None,
    ):
        try:
            self.config = config or PennyPetConfig()
            
            # Clients LLM
            try:
                self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
                logger.info("Client Qwen initialisé")
            except Exception as e:
                logger.warning(f"Erreur client Qwen: {e}")
                self.client_qwen = None
                
            try:
                self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
                logger.info("Client Mistral initialisé")
            except Exception as e:
                logger.warning(f"Erreur client Mistral: {e}")
                self.client_mistral = None
            
            # Configuration
            self.regles_pc_df = getattr(self.config, 'regles_pc_df', pd.DataFrame())
            self.normaliseur = NormaliseurAMVAmeliore(self.config)
            
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
        """Extraction avec gestion robuste des erreurs JSON"""
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
            
            # Log pour debug
            logger.info(f"Réponse LLM - Longueur: {len(content)}")
            logger.debug(f"Début réponse: {content[:200]}")
            logger.debug(f"Fin réponse: {content[-200:]}")
            
            # Extraction JSON robuste
            data = extraire_json_robuste(content)
            
            # Validation et nettoyage
            if "lignes" not in data:
                raise ValueError("Pas de lignes dans la réponse")
            
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
            logger.error(f"Erreur extraction: {e}")
            raise

    def calculer_remboursement(
        self, montant: float, code_acte: str, formule: str, est_accident: bool
    ) -> Dict[str, Any]:
        """Calcule le remboursement selon les règles PennyPet"""
        try:
            # Règles PennyPet directes (sans DataFrame pour éviter les erreurs)
            if formule == "START":
                return {
                    "montant_ht": montant,
                    "taux": 0.0,
                    "remb_final": 0.0,
                    "reste": montant
                }
            elif formule == "PREMIUM":
                if est_accident:
                    remb = min(montant, 500)  # Plafond 500€
                    return {
                        "montant_ht": montant,
                        "taux": 100.0,
                        "remb_final": remb,
                        "reste": montant - remb
                    }
                else:
                    return {
                        "montant_ht": montant,
                        "taux": 0.0,
                        "remb_final": 0.0,
                        "reste": montant
                    }
            elif formule == "INTEGRAL":
                remb = min(montant * 0.5, 1000)  # 50% jusqu'à 1000€
                return {
                    "montant_ht": montant,
                    "taux": 50.0,
                    "remb_final": remb,
                    "reste": montant - remb
                }
            elif formule == "INTEGRAL_PLUS":
                remb = min(montant, 1000)  # 100% jusqu'à 1000€
                return {
                    "montant_ht": montant,
                    "taux": 100.0,
                    "remb_final": remb,
                    "reste": montant - remb
                }
            else:
                return {
                    "montant_ht": montant,
                    "taux": 0.0,
                    "remb_final": 0.0,
                    "reste": montant
                }
                
        except Exception as e:
            logger.error(f"Erreur calcul remboursement: {e}")
            return {
                "montant_ht": montant,
                "taux": 0.0,
                "remb_final": 0.0,
                "reste": montant
            }

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Traitement complet d'une facture PennyPet"""
        
        # Reset stats
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0
        }
        
        try:
            logger.info(f"Début traitement facture - Formule: {formule_client}")
            
            # Extraction
            data, raw_content = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
            
            resultats = []
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
                    
                    # Remboursement
                    remb = self.calculer_remboursement(montant, code_norm, formule_client, est_acc)
                    
                    # Stats
                    self.stats['lignes_traitees'] += 1
                    if code_norm == "MEDICAMENTS":
                        self.stats['medicaments_detectes'] += 1
                    else:
                        self.stats['actes_detectes'] += 1
                    
                    # Création ligne résultat
                    ligne_result = {
                        "ligne": {
                            "code_acte": libelle,
                            "description": ligne.get("description", libelle),
                            "montant_ht": montant,
                            "est_medicament": (code_norm == "MEDICAMENTS")
                        },
                        "code_norm": code_norm,
                        "est_accident": est_acc,
                        "taux_remboursement": remb["taux"],
                        "montant_rembourse": remb["remb_final"],
                        "montant_reste_charge": remb["reste"]
                    }
                    
                    resultats.append(ligne_result)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    logger.error(f"Erreur ligne {ligne}: {e}")
                    continue
            
            # Totaux
            total_facture = sum(r["ligne"]["montant_ht"] for r in resultats)
            total_rembourse = sum(r["montant_rembourse"] for r in resultats)
            
            return {
                "success": True,
                "lignes": resultats,
                "resume": {
                    "total_facture": total_facture,
                    "total_rembourse": total_rembourse,
                    "reste_a_charge": total_facture - total_rembourse,
                    "taux_remboursement_global": (total_rembourse / total_facture * 100) if total_facture > 0 else 0
                },
                "informations_client": data.get("informations_client", {}),
                "statistiques": self.stats,
                "raw_llm_response": raw_content
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement facture: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistiques": self.stats
            }

    def get_processor_stats(self) -> Dict[str, Any]:
        """Statistiques du processeur"""
        return {
            **self.stats,
            **self.normaliseur.get_mapping_stats()
        }

# Instance globale
pennypet_processor = PennyPetProcessor()
