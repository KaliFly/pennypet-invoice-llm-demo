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
        text = re.sub(r'(\d+)\s*\n\s*(".*?")', r'\1,\n\2', text)
        
        # 6. Suppression des doubles virgules
        text = re.sub(r',,+', ',', text)
        
        # 7. Correction des deux points multiples
        text = re.sub(r'::+', ':', text)
        
        # 8. Suppression des caractères non ASCII problématiques
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        
        return text
        
    except Exception as e:
        logger.error(f"Erreur nettoyage JSON: {e}")
        return text

def extraire_json_robuste(content: str) -> dict:
    """
    Extraction JSON ultra-robuste avec plusieurs méthodes de fallback.
    """
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
        
        # Nettoyage et tentative de parsing
        json_clean = pseudojson_to_json_ameliore(json_str)
        return json.loads(json_clean)
        
    except json.JSONDecodeError as e:
        logger.warning(f"Parsing JSON échoué (méthode 1): {e}")
        
        # Méthode 2: Reconstruction manuelle à partir de patterns
        try:
            return reconstruire_json_manuellement(content)
        except Exception as e2:
            logger.error(f"Toutes les méthodes de parsing ont échoué: {e2}")
            
            # Méthode 3: Fallback structure minimale
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
            r'acte.*?:\s*"([^"]*)".*?montant.*?:\s*([0-9.]+)'
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
        
        # Extraction du montant total
        montant_patterns = [
            r'"montant_total"\s*:\s*([0-9.]+)',
            r'total.*?:\s*([0-9.]+)',
            r'montant.*?total.*?:\s*([0-9.]+)'
        ]
        
        for pattern in montant_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result["montant_total"] = float(match.group(1))
                break
        
        # Extraction des informations client
        client_patterns = {
            "nom_proprietaire": [r'"nom_proprietaire"\s*:\s*"([^"]*)"', r'proprietaire.*?:\s*"([^"]*)"'],
            "nom_animal": [r'"nom_animal"\s*:\s*"([^"]*)"', r'animal.*?:\s*"([^"]*)"'],
            "identification": [r'"identification"\s*:\s*"([^"]*)"', r'identification.*?:\s*"([^"]*)"']
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
    Utilise unicodedata pour une normalisation robuste.
    """
    if not texte:
        return ""
    
    # Normalisation NFD (décomposition) puis suppression des accents
    texte_nfd = unicodedata.normalize('NFD', texte)
    texte_sans_accents = ''.join(c for c in texte_nfd if unicodedata.category(c) != 'Mn')
    
    # Conversion en minuscules et suppression des caractères spéciaux
    texte_clean = re.sub(r'[^\w\s]', ' ', texte_sans_accents.lower())
    
    # Normalisation des espaces
    return ' '.join(texte_clean.split())

class NormaliseurAMVAmeliore:
    """
    Normaliseur amélioré utilisant tous les fichiers de configuration PennyPet
    """
    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.cache: Dict[str, Optional[str]] = {}
        
        # Récupération sécurisée de tous les DataFrames de config
        self.termes_actes = self._get_termes_actes_safe(config)
        self.actes_df = self._get_actes_df_safe(config)
        
        # Utilisation du glossaire pharmaceutique EXISTANT
        self.termes_medicaments = config.glossaire_pharmaceutique
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
        
        # Patterns regex étendus pour médicaments
        self.patterns_medicaments = [
            r'\b\d+\s*(mg|ml|g|l|ui|iu|mcg|µg|mg/ml|ui/ml)\b',
            r'\b(comprimé|gélule|cp|gél|sol|inj|ampoule|flacon|tube|boîte|sachet|pipette)\.?\s*\d*',
            r'\b(antibiotic|anti-inflammatoire|antiparasitaire|antifongique|antiviral|vermifuge)\b',
            r'\b(vaccin|vaccination|rappel|primo-vaccination|sérum|immunoglobuline)\b',
            r'\b(seringue|pipette|spray|pommade|crème|lotion|collyre|gouttes)\b',
            r'\b\d+\s*x\s*\d+\s*(mg|ml|g|l|cp|gél)\b',
            r'\b(principe|actif|laboratoire|generique|specialite|marque)\b',
            r'\b(anesthé|analg|cortico|hormon|vitamin|mineral|complément)\w*\b'
        ]
        
        # Patterns regex pour actes médicaux
        self.patterns_actes = [
            r'\b(consultation|examen|visite|contrôle|bilan)\b',
            r'\b(chirurgie|opération|intervention|anesthésie)\b',
            r'\b(radio|échographie|scanner|irm|imagerie)\b',
            r'\b(analyse|prélèvement|laboratoire|test)\b',
            r'\b(urgence|hospitalisation|perfusion)\b'
        ]
        
        # Variantes orthographiques étendues
        self.variantes = {
            'medicament': ['médicament', 'medicaments', 'médicaments', 'pharma', 'traitement'],
            'gelule': ['gélule', 'gélules', 'gelules', 'capsule'],
            'comprimes': ['comprimé', 'comprimés', 'comprimes', 'tablet'],
            'solution': ['solutions', 'sol', 'liquide'],
            'injection': ['injections', 'inj', 'piqûre'],
            'milligramme': ['mg', 'milligrammes', 'mgr'],
            'millilitre': ['ml', 'millilitres', 'mL'],
            'gramme': ['g', 'grammes', 'gr'],
            'litre': ['l', 'litres', 'L']
        }
        
        logger.info(f"Normaliseur initialisé: {len(self.termes_actes)} actes, {len(self.termes_medicaments)} médicaments")

    def _get_termes_actes_safe(self, config: PennyPetConfig) -> set:
        """Récupère les termes d'actes de tous les fichiers de config"""
        termes = set()
        
        try:
            # Actes du fichier principal
            if hasattr(config, 'actes_df') and not config.actes_df.empty:
                df = config.actes_df
                if "field_label" in df.columns:
                    termes.update(df["field_label"].dropna().astype(str).str.lower())
                else:
                    # Fallback sur d'autres colonnes
                    text_columns = df.select_dtypes(include=['object']).columns
                    if len(text_columns) > 0:
                        termes.update(df[text_columns[0]].dropna().astype(str).str.lower())
            
            # Actes des autres fichiers regex
            for df_name in ['calculs_codes_df', 'infos_financieres_df', 'metadonnees_df', 'parties_benef_df']:
                df = getattr(config, df_name, pd.DataFrame())
                if not df.empty and 'field_label' in df.columns:
                    termes.update(df["field_label"].dropna().astype(str).str.lower())
            
            logger.info(f"Termes d'actes extraits: {len(termes)}")
            return termes
            
        except Exception as e:
            logger.error(f"Erreur extraction termes actes: {e}")
            return set()

    def _get_actes_df_safe(self, config: PennyPetConfig) -> pd.DataFrame:
        """Récupère et consolide tous les DataFrames d'actes"""
        try:
            dfs = []
            
            # DataFrame principal des actes
            if hasattr(config, 'actes_df') and not config.actes_df.empty:
                df = config.actes_df
                if 'pattern' in df.columns:
                    dfs.append(df.dropna(subset=["pattern"]))
            
            # Autres DataFrames avec patterns
            for df_name in ['calculs_codes_df', 'infos_financieres_df', 'metadonnees_df']:
                df = getattr(config, df_name, pd.DataFrame())
                if not df.empty and 'pattern' in df.columns:
                    dfs.append(df.dropna(subset=["pattern"]))
            
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Erreur extraction DataFrame actes: {e}")
            return pd.DataFrame()

    def _preprocess_glossaire(self) -> Dict[str, str]:
        """Préprocesse le glossaire pharmaceutique avec toutes les variantes"""
        glossaire_normalise = {}
        
        try:
            # Glossaire principal
            for terme in self.termes_medicaments:
                if not terme:
                    continue
                    
                terme_norm = normaliser_accents(str(terme))
                if terme_norm:
                    glossaire_normalise[terme_norm] = terme
                
                # Variantes du terme
                for variante in self._generer_variantes(str(terme)):
                    variante_norm = normaliser_accents(variante)
                    if variante_norm:
                        glossaire_normalise[variante_norm] = terme
            
            # Ajout des médicaments du DataFrame si disponible
            if not self.medicaments_df.empty and 'medicament' in self.medicaments_df.columns:
                for med in self.medicaments_df['medicament'].dropna():
                    med_norm = normaliser_accents(str(med))
                    if med_norm:
                        glossaire_normalise[med_norm] = str(med)
                        
        except Exception as e:
            logger.error(f"Erreur préprocessing glossaire: {e}")
        
        logger.info(f"Glossaire normalisé: {len(glossaire_normalise)} entrées")
        return glossaire_normalise

    def _generer_variantes(self, terme: str) -> List[str]:
        """Génère des variantes orthographiques étendues"""
        variantes = [terme]
        
        try:
            # Variantes avec/sans 's' final
            if terme.endswith('s'):
                variantes.append(terme[:-1])
            else:
                variantes.append(terme + 's')
            
            # Variantes avec abréviations
            for base, abbrevs in self.variantes.items():
                if base in terme.lower():
                    for abbrev in abbrevs:
                        variantes.append(terme.lower().replace(base, abbrev))
            
            # Variantes de forme (sing/plur)
            if terme.endswith('aux'):
                variantes.append(terme[:-3] + 'al')
            elif terme.endswith('al'):
                variantes.append(terme[:-2] + 'aux')
                
        except Exception as e:
            logger.debug(f"Erreur génération variantes pour '{terme}': {e}")
        
        return variantes

    def _detecter_patterns_medicaments(self, texte: str) -> bool:
        """Détecte les patterns typiques des médicaments"""
        try:
            texte_norm = normaliser_accents(texte)
            
            for pattern in self.patterns_medicaments:
                if re.search(pattern, texte_norm, re.IGNORECASE):
                    return True
        except Exception as e:
            logger.debug(f"Erreur détection pattern médicament '{texte}': {e}")
        
        return False

    def _detecter_patterns_actes(self, texte: str) -> bool:
        """Détecte les patterns typiques des actes médicaux"""
        try:
            texte_norm = normaliser_accents(texte)
            
            for pattern in self.patterns_actes:
                if re.search(pattern, texte_norm, re.IGNORECASE):
                    return True
        except Exception as e:
            logger.debug(f"Erreur détection pattern acte '{texte}': {e}")
        
        return False

    def normalise_acte(self, libelle_brut: str) -> Optional[str]:
        """Normalise un acte médical avec tous les patterns disponibles"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Détection par patterns actes
            if self._detecter_patterns_actes(libelle_brut):
                self.cache[cle] = "ACTES"
                return "ACTES"
            
            # 2. Pattern CSV exact sur tous les DataFrames
            if not self.actes_df.empty:
                for _, row in self.actes_df.iterrows():
                    pattern = row.get("pattern")
                    if pattern and hasattr(pattern, 'search'):
                        try:
                            if pattern.search(cle):
                                code_acte = row.get("code_acte", "ACTES")
                                self.cache[cle] = code_acte
                                return code_acte
                        except Exception:
                            continue
            
            # 3. Fallback sémantique actes
            for terme in self.termes_actes:
                try:
                    terme_norm = normaliser_accents(terme)
                    if re.search(rf"(?<!\w){re.escape(terme_norm)}(?!\w)", libelle_norm):
                        code = terme.upper()
                        self.cache[cle] = code
                        return code
                except Exception:
                    continue
            
            # 4. Fuzzy matching sur les actes
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
        """Normalise un médicament avec le glossaire complet"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Détection par patterns regex
            if self._detecter_patterns_medicaments(libelle_brut):
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
            
            # 4. Recherche dans mapping AMV
            if self.mapping_amv:
                for cle_amv, valeur in self.mapping_amv.items():
                    if cle_amv.lower() in libelle_norm or libelle_norm in cle_amv.lower():
                        if 'medicament' in str(valeur).lower():
                            self.cache[cle] = "MEDICAMENTS"
                            return "MEDICAMENTS"
            
            # 5. Fuzzy matching intelligent
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
        """Normalise un libellé (acte ou médicament) avec priorité intelligente"""
        if not libelle_brut:
            return None
            
        try:
            # Détection intelligente du type
            libelle_lower = libelle_brut.lower()
            
            # Priorité médicaments si patterns évidents
            if any(pattern in libelle_lower for pattern in ['mg', 'ml', 'comprimé', 'gélule', 'vaccin', 'injection']):
                result = self.normalise_medicament(libelle_brut)
                if result:
                    return result
                # Fallback actes si pas trouvé
                result = self.normalise_acte(libelle_brut)
                if result:
                    return result
            else:
                # Priorité actes sinon
                result = self.normalise_acte(libelle_brut)
                if result:
                    return result
                # Fallback médicaments
                result = self.normalise_medicament(libelle_brut)
                if result:
                    return result
            
            # Fallback final
            return str(libelle_brut).strip().upper()
            
        except Exception as e:
            logger.error(f"Erreur normalisation '{libelle_brut}': {e}")
            return str(libelle_brut).strip().upper() if libelle_brut else None

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes de mapping"""
        return {
            "cache_size": len(self.cache),
            "actes": len(self.termes_actes),
            "medicaments": len(self.termes_medicaments),
            "glossaire_normalise": len(self.glossaire_normalise),
            "patterns_medicaments": len(self.patterns_medicaments),
            "patterns_actes": len(self.patterns_actes),
            "variantes": len(self.variantes),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE,
            "actes_df_size": len(self.actes_df),
            "medicaments_df_size": len(self.medicaments_df),
            "config_files": {
                "calculs_codes": len(self.calculs_codes_df),
                "infos_financieres": len(self.infos_financieres_df),
                "metadonnees": len(self.metadonnees_df),
                "parties_benef": len(self.parties_benef_df),
                "suivi_sla": len(self.suivi_sla_df)
            }
        }

class PennyPetProcessor:
    """
    Pipeline extraction LLM, normalisation complète, calcul remboursement PennyPet.
    """
    def __init__(
        self,
        client_qwen: OpenRouterClient = None,
        client_mistral: OpenRouterClient = None,
        config: PennyPetConfig = None,
    ):
        try:
            self.config = config or PennyPetConfig()
            
            # Initialisation des clients LLM
            try:
                self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
            except Exception as e:
                logger.warning(f"Erreur initialisation client Qwen: {e}")
                self.client_qwen = None
                
            try:
                self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
            except Exception as e:
                logger.warning(f"Erreur initialisation client Mistral: {e}")
                self.client_mistral = None
            
            # Chargement de tous les DataFrames de config
            self.regles_pc_df = getattr(self.config, 'regles_pc_df', pd.DataFrame())
            self.formules = getattr(self.config, 'formules', {})
            
            # Initialisation du normaliseur amélioré
            self.normaliseur = NormaliseurAMVAmeliore(self.config)
            
            # Statistiques de traitement
            self.stats = {
                'lignes_traitees': 0,
                'medicaments_detectes': 0,
                'actes_detectes': 0,
                'erreurs_normalisation': 0,
                'erreurs_json': 0,
                'reconstructions_manuelles': 0
            }
            
            logger.info("PennyPetProcessor initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation PennyPetProcessor: {e}")
            raise

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """Extrait les lignes d'une image avec gestion JSON ultra-robuste"""
        try:
            # Sélection du client
            if llm_provider.lower() == "qwen" and self.client_qwen:
                client = self.client_qwen
            elif llm_provider.lower() == "mistral" and self.client_mistral:
                client = self.client_mistral
            else:
                raise ValueError(f"Client {llm_provider} non disponible")
            
            # Appel au LLM
            resp = client.analyze_invoice_image(image_bytes, formule)
            content = resp.choices[0].message.content
            
            if not content:
                raise ValueError("Réponse vide du LLM")
            
            # Log pour debug
            logger.info(f"Réponse LLM (premiers 200 chars): {content[:200]}")
            logger.info(f"Réponse LLM (derniers 200 chars): {content[-200:]}")
            
            # Extraction JSON robuste
            try:
                data = extraire_json_robuste(content)
                if data.get("lignes") and data["lignes"][0].get("code_acte") == "ERREUR_JSON":
                    self.stats['erreurs_json'] += 1
                elif data.get("lignes") and data["lignes"][0].get("code_acte") == "ERREUR":
                    self.stats['reconstructions_manuelles'] += 1
            except Exception as e:
                logger.error(f"Erreur extraction JSON: {e}")
                self.stats['erreurs_json'] += 1
                
                # Structure de fallback
                data = {
                    "lignes": [{"code_acte": "ERREUR_CRITIQUE", "description": f"Erreur critique: {str(e)}", "montant_ht": 0.0}],
                    "montant_total": 0.0,
                    "informations_client": {}
                }
            
            # Validation et nettoyage des données
            self._nettoyer_donnees_extraites(data)
            
            if not data.get("lignes"):
                raise ValueError("Aucune ligne extraite après nettoyage")
            
            return data, content
            
        except Exception as e:
            logger.error(f"Erreur dans extract_lignes_from_image: {e}")
            raise

    def _nettoyer_donnees_extraites(self, data: Dict[str, Any]) -> None:
        """Nettoie et valide les données extraites"""
        try:
            # Nettoyage des lignes
            if "lignes" in data and isinstance(data["lignes"], list):
                lignes_propres = []
                for ligne in data["lignes"]:
                    if isinstance(ligne, dict):
                        # Nettoyage montant_ht
                        try:
                            ligne["montant_ht"] = float(ligne.get("montant_ht", 0))
                        except (ValueError, TypeError):
                            ligne["montant_ht"] = 0.0
                        
                        # Nettoyage des chaînes
                        for key in ["code_acte", "description"]:
                            if key in ligne:
                                ligne[key] = str(ligne[key]).strip()
                                # Suppression des caractères spéciaux problématiques
                                ligne[key] = re.sub(r'[^\w\s\-\(\)\.,]', ' ', ligne[key])
                                ligne[key] = ' '.join(ligne[key].split())
                        
                        # Validation minimale
                        if ligne.get("montant_ht", 0) >= 0 and ligne.get("code_acte"):
                            lignes_propres.append(ligne)
                
                data["lignes"] = lignes_propres
            
            # Nettoyage montant total
            try:
                data["montant_total"] = float(data.get("montant_total", 0))
            except (ValueError, TypeError):
                data["montant_total"] = 0.0
            
            # Nettoyage informations client
            if "informations_client" in data and isinstance(data["informations_client"], dict):
                for key, value in data["informations_client"].items():
                    if value:
                        data["informations_client"][key] = str(value).strip()
            else:
                data["informations_client"] = {}
                
        except Exception as e:
            logger.error(f"Erreur nettoyage données: {e}")

    def calculer_remboursement(
        self, montant: float, code_acte: str, formule: str, est_accident: bool
    ) -> Dict[str, Any]:
        """Calcule le remboursement selon les règles PennyPet réelles"""
        try:
            # Application directe des règles PennyPet
            if formule == "START":
                return {
                    "montant_ht": montant,
                    "taux": 0.0,
                    "remb_final": 0.0,
                    "reste": montant,
                    "formule": formule,
                    "type_couverture": "aucune"
                }
            elif formule == "PREMIUM":
                if est_accident:
                    remb = min(montant, 500.0)  # 100% jusqu'à 500€
                    return {
                        "montant_ht": montant,
                        "taux": 100.0,
                        "remb_final": remb,
                        "reste": montant - remb,
                        "formule": formule,
                        "type_couverture": "accident_seulement"
                    }
                else:
                    return {
                        "montant_ht": montant,
                        "taux": 0.0,
                        "remb_final": 0.0,
                        "reste": montant,
                        "formule": formule,
                        "type_couverture": "accident_seulement"
                    }
            elif formule == "INTEGRAL":
                remb = min(montant * 0.5, 1000.0)  # 50% jusqu'à 1000€
                return {
                    "montant_ht": montant,
                    "taux": 50.0,
                    "remb_final": remb,
                    "reste": montant - remb,
                    "formule": formule,
                    "type_couverture": "accident_et_maladie"
                }
            elif formule == "INTEGRAL_PLUS":
                remb = min(montant, 1000.0)  # 100% jusqu'à 1000€
                return {
                    "montant_ht": montant,
                    "taux": 100.0,
                    "remb_final": remb,
                    "reste": montant - remb,
                    "formule": formule,
                    "type_couverture": "accident_et_maladie"
                }
            else:
                # Formule inconnue - fallback DataFrame si disponible
                if not self.regles_pc_df.empty:
                    return self._calculer_avec_dataframe(montant, code_acte, formule, est_accident)
                else:
                    return {
                        "erreur": f"Formule inconnue: {formule}",
                        "montant_ht": montant,
                        "taux": 0.0,
                        "remb_final": 0.0,
                        "reste": montant
                    }
                
        except Exception as e:
            logger.error(f"Erreur calcul remboursement: {e}")
            return {
                "erreur": f"Erreur calcul: {e}",
                "montant_ht": montant,
                "taux": 0.0,
                "remb_final": 0.0,
                "reste": montant
            }

    def _calculer_avec_dataframe(self, montant: float, code_acte: str, formule: str, est_accident: bool) -> Dict[str, Any]:
        """Calcul avec DataFrame de règles (fallback)"""
        try:
            df = self.regles_pc_df.copy()
            
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
                return {
                    "erreur": f"Aucune règle DataFrame pour {formule}/{code_acte}",
                    "montant_ht": montant,
                    "taux": 0.0,
                    "remb_final": 0.0,
                    "reste": montant
                }
            
            r = reg.iloc[0]
            taux = r["taux_remboursement"] / 100
            plafond = r.get("plafond_annuel", float('inf'))
            
            remb_brut = montant * taux
            remb_final = min(remb_brut, plafond)
            
            return {
                "montant_ht": montant,
                "taux": taux * 100,
                "remb_final": remb_final,
                "reste": montant - remb_final,
                "formule": formule,
                "regle_source": "dataframe"
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul DataFrame: {e}")
            return {
                "erreur": f"Erreur calcul DataFrame: {e}",
                "montant_ht": montant,
                "taux": 0.0,
                "remb_final": 0.0,
                "reste": montant
            }

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Traite une facture PennyPet complète avec gestion d'erreurs robuste"""
        
        # Réinitialisation des stats
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0,
            'erreurs_json': 0,
            'reconstructions_manuelles': 0
        }
        
        try:
            # Extraction des données
            data, raw_content = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
            
            resultats: List[Dict[str, Any]] = []
            accidents = {"accident", "urgent", "urgence", "fract", "trauma", "traumatisme", "blessure"}
            
            # Traitement de chaque ligne
            for ligne in data["lignes"]:
                try:
                    # Extraction des données de ligne
                    libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
                    montant = float(ligne.get("montant_ht", 0) or 0)
                    
                    if montant <= 0:
                        continue  # Ignorer les lignes sans montant
                    
                    # Normalisation du code acte
                    code_norm = self.normaliseur.normalise(libelle)
                    
                    # Détection d'accident (recherche étendue)
                    est_acc = any(mot in libelle.lower() for mot in accidents)
                    
                    # Calcul du remboursement
                    remb = self.calculer_remboursement(montant, code_norm, formule_client, est_acc)
                    
                    # Détermination du type (médicament/acte)
                    est_medicament = (code_norm == "MEDICAMENTS")
                    
                    # Mise à jour des statistiques
                    self.stats['lignes_traitees'] += 1
                    if est_medicament:
                        self.stats['medicaments_detectes'] += 1
                    else:
                        self.stats['actes_detectes'] += 1
                    
                    # Structure de résultat uniforme
                    resultat = {
                        "ligne": {
                            "code_acte": libelle,
                            "description": ligne.get("description", libelle),
                            "montant_ht": montant,
                            "est_medicament": est_medicament,
                            "code_normalise": code_norm
                        },
                        "est_accident": est_acc,
                        "taux_remboursement": remb.get("taux", 0.0),
                        "montant_rembourse": remb.get("remb_final", 0.0),
                        "montant_reste_charge": remb.get("reste", montant),
                        "formule_appliquee": formule_client,
                        "erreur": remb.get("erreur")
                    }
                    
                    resultats.append(resultat)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    logger.error(f"Erreur traitement ligne {ligne}: {e}")
                    continue
            
            # Calcul des totaux
            total_facture = sum(r["ligne"]["montant_ht"] for r in resultats)
            total_rembourse = sum(r["montant_rembourse"] for r in resultats)
            reste_a_charge = total_facture - total_rembourse
            
            # Calcul du taux global
            taux_global = (total_rembourse / total_facture * 100) if total_facture > 0 else 0
            
            return {
                "success": True,
                "lignes": resultats,
                "resume": {
                    "total_facture": round(total_facture, 2),
                    "total_rembourse": round(total_rembourse, 2),
                    "reste_a_charge": round(reste_a_charge, 2),
                    "taux_remboursement_global": round(taux_global, 2)
                },
                "informations_client": data.get("informations_client", {}),
                "statistiques": self.stats,
                "mapping_stats": self.normaliseur.get_mapping_stats(),
                "formule_utilisee": formule_client,
                "raw_llm_response": raw_content
            }
            
        except Exception as e:
            logger.error(f"Erreur générale process_facture_pennypet: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistiques": self.stats,
                "formule_utilisee": formule_client
            }

    def get_processor_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes du processeur"""
        return {
            **self.stats,
            **self.normaliseur.get_mapping_stats(),
            "clients_llm": {
                "qwen_disponible": self.client_qwen is not None,
                "mistral_disponible": self.client_mistral is not None
            }
        }

# Instance globale pour usage direct
pennypet_processor = PennyPetProcessor()
