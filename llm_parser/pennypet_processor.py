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
        
        # 9. Correction des valeurs null/undefined
        text = re.sub(r':\s*(null|undefined|None)\s*([,}])', r': ""\\2', text)
        
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
        data = json.loads(json_clean)
        logger.info("JSON extrait avec succès (méthode standard)")
        return data
        
    except json.JSONDecodeError as e:
        logger.warning(f"Parsing JSON échoué (méthode 1): {e}")
        
        # Méthode 2: Extraction par regex patterns
        try:
            return reconstruire_json_par_regex(content)
        except Exception as e2:
            logger.warning(f"Parsing JSON échoué (méthode 2): {e2}")
            
            # Méthode 3: Reconstruction manuelle minimale
            return structure_json_fallback()

def reconstruire_json_par_regex(content: str) -> dict:
    """
    Reconstruction du JSON à partir de patterns regex robustes.
    """
    result = {
        "lignes": [],
        "montant_total": 0.0,
        "informations_client": {}
    }
    
    try:
        # Pattern pour lignes multiples avec variations
        lignes_patterns = [
            # Pattern standard complet
            r'"code_acte"\s*:\s*"([^"]*)".*?"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([0-9.,]+)',
            # Pattern sans code_acte
            r'"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([0-9.,]+)',
            # Pattern simplifié
            r'acte[^:]*:\s*"([^"]*)".*?montant[^:]*:\s*([0-9.,]+)',
            # Pattern très permissif
            r'"([^"]*)"[^0-9]*([0-9.,]+)(?=\s*[,}\]])'
        ]
        
        for pattern in lignes_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                logger.info(f"Pattern trouvé: {len(matches)} lignes")
                for match in matches:
                    try:
                        if len(match) == 3:
                            # Pattern complet
                            montant = float(match[2].replace(',', '.'))
                            result["lignes"].append({
                                "code_acte": match[0].strip(),
                                "description": match[1].strip(),
                                "montant_ht": montant
                            })
                        elif len(match) == 2:
                            # Pattern partiel
                            montant = float(match[1].replace(',', '.'))
                            result["lignes"].append({
                                "code_acte": match[0].strip(),
                                "description": match[0].strip(),
                                "montant_ht": montant
                            })
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Erreur parsing ligne: {e}")
                        continue
                break
        
        # Extraction montant total avec patterns multiples
        montant_patterns = [
            r'"montant_total"\s*:\s*([0-9.,]+)',
            r'total[^:]*:\s*([0-9.,]+)',
            r'montant[^:]*total[^:]*:\s*([0-9.,]+)',
            r'total[^0-9]*([0-9.,]+)'
        ]
        
        for pattern in montant_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    result["montant_total"] = float(match.group(1).replace(',', '.'))
                    break
                except ValueError:
                    continue
        
        # Si pas de montant total trouvé, calculer depuis les lignes
        if result["montant_total"] == 0.0 and result["lignes"]:
            result["montant_total"] = sum(ligne["montant_ht"] for ligne in result["lignes"])
        
        # Extraction informations client avec patterns flexibles
        client_patterns = {
            "nom_proprietaire": [
                r'"nom_proprietaire"\s*:\s*"([^"]*)"',
                r'proprietaire[^:]*:\s*"([^"]*)"',
                r'nom[^:]*:\s*"([^"]*)"'
            ],
            "nom_animal": [
                r'"nom_animal"\s*:\s*"([^"]*)"',
                r'animal[^:]*:\s*"([^"]*)"'
            ],
            "identification": [
                r'"identification"\s*:\s*"([^"]*)"',
                r'id[^:]*:\s*"([^"]*)"',
                r'numero[^:]*:\s*"([^"]*)"'
            ]
        }
        
        for key, patterns in client_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    result["informations_client"][key] = match.group(1).strip()
                    break
        
        logger.info(f"JSON reconstruit: {len(result['lignes'])} lignes, total: {result['montant_total']}")
        return result
        
    except Exception as e:
        logger.error(f"Erreur reconstruction regex: {e}")
        return structure_json_fallback()

def structure_json_fallback() -> dict:
    """
    Structure JSON minimale en cas d'échec total.
    """
    logger.warning("Utilisation de la structure JSON fallback")
    return {
        "lignes": [{
            "code_acte": "ERREUR_EXTRACTION",
            "description": "Erreur extraction JSON - Veuillez réessayer",
            "montant_ht": 0.0
        }],
        "montant_total": 0.0,
        "informations_client": {
            "nom_proprietaire": "",
            "nom_animal": "",
            "identification": ""
        }
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
        
        # Récupération sécurisée de tous les DataFrames
        self.termes_actes = self._get_termes_actes_safe(config)
        self.actes_df = self._get_actes_df_safe(config)
        
        # Utilisation du glossaire pharmaceutique existant
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
        
        # Patterns regex étendus
        self.patterns_medicaments = [
            r'\b\d+\s*(mg|ml|g|l|ui|iu|mcg|µg|mg/ml|ui/ml)\b',
            r'\b(comprimé|gélule|cp|gél|sol|inj|ampoule|flacon|tube|boîte|sachet|pipette)\.?\s*\d*',
            r'\b(antibiotic|anti-inflammatoire|antiparasitaire|antifongique|antiviral|vermifuge)\b',
            r'\b(vaccin|vaccination|rappel|primo-vaccination|sérum)\b',
            r'\b(seringue|spray|pommade|crème|lotion|collyre|gouttes)\b',
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
        
        # Variantes orthographiques
        self.variantes = {
            'medicament': ['médicament', 'medicaments', 'médicaments'],
            'gelule': ['gélule', 'gélules', 'gelules', 'capsule'],
            'comprimes': ['comprimé', 'comprimés', 'comprimes', 'cp'],
            'solution': ['solutions', 'sol', 'soluté'],
            'injection': ['injections', 'inj', 'piqûre'],
            'consultation': ['consult', 'visite', 'rdv']
        }
        
        logger.info(f"Normaliseur initialisé: {len(self.termes_actes)} actes, {len(self.termes_medicaments)} médicaments")

    def _get_termes_actes_safe(self, config: PennyPetConfig) -> set:
        """Récupère les termes d'actes depuis tous les fichiers"""
        termes = set()
        
        try:
            # Actes depuis actes_df
            if hasattr(config, 'actes_df') and not config.actes_df.empty:
                df = config.actes_df
                possible_columns = ['field_label', 'label', 'acte', 'description', 'libelle']
                for col in possible_columns:
                    if col in df.columns:
                        termes.update(df[col].dropna().astype(str).str.lower())
                        break
            
            # Autres DataFrames
            for df_name in ['calculs_codes_df', 'infos_financieres_df', 'metadonnees_df']:
                if hasattr(config, df_name):
                    df = getattr(config, df_name)
                    if not df.empty:
                        for col in ['field_label', 'description']:
                            if col in df.columns:
                                termes.update(df[col].dropna().astype(str).str.lower())
            
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
                
                for variante in self._generer_variantes(str(terme)):
                    variante_norm = normaliser_accents(variante)
                    if variante_norm:
                        glossaire_normalise[variante_norm] = terme
                        
        except Exception as e:
            logger.error(f"Erreur préprocessing glossaire: {e}")
        
        return glossaire_normalise

    def _generer_variantes(self, terme: str) -> List[str]:
        """Génère des variantes orthographiques"""
        variantes = [terme]
        
        try:
            if terme.endswith('s'):
                variantes.append(terme[:-1])
            else:
                variantes.append(terme + 's')
            
            for base, abbrevs in self.variantes.items():
                if base in terme.lower():
                    for abbrev in abbrevs:
                        variantes.append(terme.lower().replace(base, abbrev))
                        
        except Exception as e:
            logger.debug(f"Erreur génération variantes: {e}")
        
        return variantes

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
            
            # Recherche par patterns
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
            
            # Détection par patterns prédéfinis
            if self._detecter_patterns_actes(libelle_brut):
                self.cache[cle] = "ACTE_MEDICAL"
                return "ACTE_MEDICAL"
            
            # Recherche sémantique
            for terme in self.termes_actes:
                try:
                    terme_norm = normaliser_accents(terme)
                    if re.search(rf"(?<!\w){re.escape(terme_norm)}(?!\w)", libelle_norm):
                        code = terme.upper()
                        self.cache[cle] = code
                        return code
                except Exception:
                    continue
            
            self.cache[cle] = None
            return None
            
        except Exception as e:
            logger.error(f"Erreur normalisation acte: {e}")
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
            
            # Détection par patterns
            if self._detecter_patterns_medicaments(libelle_brut):
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
            
            # Recherche dans glossaire
            if libelle_norm in self.glossaire_normalise:
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
            
            # Recherche partielle
            for terme_norm in self.glossaire_normalise.keys():
                try:
                    if (terme_norm in libelle_norm or 
                        libelle_norm in terme_norm or
                        any(word in libelle_norm for word in terme_norm.split() if len(word) > 3)):
                        self.cache[cle] = "MEDICAMENTS"
                        return "MEDICAMENTS"
                except Exception:
                    continue
            
            self.cache[cle] = None
            return None
            
        except Exception as e:
            logger.error(f"Erreur normalisation médicament: {e}")
            return None

    def normalise(self, libelle_brut: str) -> Optional[str]:
        """Normalise un libellé avec priorité intelligente"""
        if not libelle_brut:
            return None
        
        try:
            libelle_norm = normaliser_accents(libelle_brut)
            
            # Détection intelligente du type
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
            logger.error(f"Erreur normalisation: {e}")
            return str(libelle_brut).strip().upper() if libelle_brut else None

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Statistiques de mapping"""
        return {
            "cache_size": len(self.cache),
            "actes": len(self.termes_actes),
            "medicaments": len(self.termes_medicaments),
            "glossaire_normalise": len(self.glossaire_normalise),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE,
            "dataframes_loaded": {
                "actes_df": len(self.actes_df),
                "medicaments_df": len(self.medicaments_df),
                "calculs_codes_df": len(self.calculs_codes_df),
                "infos_financieres_df": len(self.infos_financieres_df)
            }
        }

class PennyPetProcessor:
    """
    Pipeline complet d'extraction LLM avec gestion JSON robuste
    """
    def __init__(
        self,
        client_qwen: OpenRouterClient = None,
        client_mistral: OpenRouterClient = None,
        config: PennyPetConfig = None,
    ):
        try:
            self.config = config or PennyPetConfig()
            
            # Initialisation des clients
            try:
                self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
            except Exception as e:
                logger.warning(f"Erreur client Qwen: {e}")
                self.client_qwen = None
                
            try:
                self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
            except Exception as e:
                logger.warning(f"Erreur client Mistral: {e}")
                self.client_mistral = None
            
            self.regles_pc_df = getattr(self.config, 'regles_pc_df', pd.DataFrame())
            self.normaliseur = NormaliseurAMVAmeliore(self.config)
            
            # Statistiques étendues
            self.stats = {
                'lignes_traitees': 0,
                'medicaments_detectes': 0,
                'actes_detectes': 0,
                'erreurs_normalisation': 0,
                'erreurs_json': 0,
                'reconstructions_json': 0
            }
            
            logger.info("PennyPetProcessor initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            raise

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """Extraction avec gestion JSON ultra-robuste"""
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
            
            logger.info(f"Réponse LLM reçue - Longueur: {len(content)}")
            
            # Extraction JSON robuste
            try:
                data = extraire_json_robuste(content)
                logger.info("Extraction JSON réussie")
            except Exception as e:
                logger.error(f"Erreur JSON: {e}")
                self.stats['erreurs_json'] += 1
                self.stats['reconstructions_json'] += 1
                data = structure_json_fallback()
            
            # Validation finale
            if "lignes" not in data or not data["lignes"]:
                logger.warning("Aucune ligne dans les données extraites")
                data["lignes"] = [{
                    "code_acte": "AUCUNE_DONNEE",
                    "description": "Aucune donnée extraite - Vérifiez la qualité de l'image",
                    "montant_ht": 0.0
                }]
            
            # Nettoyage des lignes
            lignes_valides = []
            for ligne in data["lignes"]:
                try:
                    # Validation et nettoyage
                    ligne["montant_ht"] = float(ligne.get("montant_ht", 0))
                    ligne["code_acte"] = str(ligne.get("code_acte", "")).strip()
                    ligne["description"] = str(ligne.get("description", "")).strip()
                    
                    if ligne["code_acte"] or ligne["description"]:
                        lignes_valides.append(ligne)
                except Exception as e:
                    logger.debug(f"Ligne ignorée: {e}")
                    continue
            
            data["lignes"] = lignes_valides
            
            return data, content
            
        except Exception as e:
            logger.error(f"Erreur extraction: {e}")
            raise

    def calculer_remboursement(
        self, montant: float, code_acte: str, formule: str, est_accident: bool
    ) -> Dict[str, Any]:
        """Calcule le remboursement selon les règles PennyPet"""
        if not self.regles_pc_df.empty:
            df = self.regles_pc_df.copy()
            
            mask = (
                (df["formule"] == formule) &
                (df["code_acte"].eq(code_acte) | 
                 (df["code_acte"].fillna("ALL").eq("ALL") &
                  df["actes_couverts"].apply(lambda l: code_acte in l if isinstance(l, list) else False)))
            )
            
            reg = df[mask]
            if not reg.empty:
                r = reg.iloc[0]
                taux = r.get("taux_remboursement", 0) / 100
                plafond = r.get("plafond_annuel", float('inf'))
                
                remb_brut = montant * taux
                remb_final = min(remb_brut, plafond)
                
                return {
                    "montant_ht": montant,
                    "taux": taux * 100,
                    "remb_final": remb_final,
                    "reste": montant - remb_final
                }
        
        # Fallback: règles PennyPet par défaut
        taux = 0.0
        if formule == "PREMIUM" and est_accident:
            taux = 1.0  # 100% pour accidents
        elif formule == "INTEGRAL":
            taux = 0.5  # 50% pour tout
        elif formule == "INTEGRAL_PLUS":
            taux = 1.0  # 100% pour tout
        
        remb = montant * taux
        return {
            "montant_ht": montant,
            "taux": taux * 100,
            "remb_final": remb,
            "reste": montant - remb
        }

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Traite une facture PennyPet complète"""
        
        # Reset stats
        self.stats = {k: 0 for k in self.stats.keys()}
        
        try:
            # Extraction des données
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
                    
                    # Calcul remboursement
                    remb = self.calculer_remboursement(montant, code_norm, formule_client, est_acc)
                    
                    # Stats
                    self.stats['lignes_traitees'] += 1
                    if code_norm == "MEDICAMENTS":
                        self.stats['medicaments_detectes'] += 1
                    else:
                        self.stats['actes_detectes'] += 1
                    
                    # Résultat
                    ligne.update({
                        "code_norm": code_norm,
                        "est_accident": est_acc,
                        **remb
                    })
                    resultats.append(ligne)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    logger.error(f"Erreur ligne: {e}")
                    continue
            
            # Calculs finaux
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
