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

def _strip_accents(txt: str) -> str:
    """Supprime les accents et normalise le texte"""
    if not txt: 
        return ""
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", txt).lower().strip()

def _insert_comma_at_error(text: str, pos: int) -> str:
    """Insère une virgule à la position d'erreur JSON"""
    if pos > 0 and pos < len(text):
        return text[:pos] + "," + text[pos:]
    return text

def parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Parser JSON ultra-robuste avec réparation automatique
    1) Isole le JSON {…}
    2) Nettoie clés non-quotées et guillemets simples  
    3) Boucle de réparation : json.loads → insert comma at pos → retry
    4) Fallback minimal par regex
    """
    # 1. Isolation du JSON
    start, end = raw.find('{'), raw.rfind('}') + 1
    if start < 0 or end <= start:
        return _fallback_regex_parser(raw)
    
    txt = raw[start:end]
    
    # 2. Nettoyage de base
    txt = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', txt)  # Clés non quotées
    txt = txt.replace("'", '"')  # Guillemets simples
    txt = re.sub(r',\s*([}\]])', r'\1', txt)  # Virgules avant fermantes
    txt = re.sub(r'}\s*{', '},{', txt)  # Objects collés
    txt = re.sub(r']\s*\[', '],[', txt)  # Arrays collés
    txt = re.sub(r',,+', ',', txt)  # Doubles virgules
    
    # 3. Boucle de réparation avec insertion de virgules
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            data = json.loads(txt)
            logger.info(f"JSON parsé avec succès (tentative {attempt + 1})")
            return data
        except json.JSONDecodeError as e:
            if "Expecting ',' delimiter" in str(e) and hasattr(e, 'pos'):
                logger.warning(f"Tentative {attempt + 1}: Insertion virgule à position {e.pos}")
                txt = _insert_comma_at_error(txt, e.pos)
            else:
                logger.warning(f"Erreur JSON non réparable: {e}")
                break
    
    # 4. Fallback par regex
    logger.warning("Utilisation du parser de fallback")
    return _fallback_regex_parser(raw)

def _fallback_regex_parser(txt: str) -> Dict[str, Any]:
    """Parser de fallback par regex pour cas désespérés"""
    # Extraction des lignes avec patterns flexibles
    patterns = [
        r'"?(?:code_acte|acte)"?\s*[:=]\s*"([^"]+)"[^}]*"?(?:montant_ht|montant)"?\s*[:=]\s*([\d.]+)',
        r'"?(?:description|desc)"?\s*[:=]\s*"([^"]+)"[^}]*"?(?:montant_ht|montant)"?\s*[:=]\s*([\d.]+)',
        r'([^{}"]+?)\s*[:=]\s*([\d.]+)'
    ]
    
    lines = []
    for pattern in patterns:
        matches = re.findall(pattern, txt, re.IGNORECASE | re.DOTALL)
        if matches:
            for match in matches:
                try:
                    lines.append({
                        "code_acte": match[0].strip(),
                        "description": match[0].strip(),
                        "montant_ht": float(match[1])
                    })
                except ValueError:
                    continue
            break
    
    if not lines:
        lines = [{"code_acte": "ERREUR_JSON", "description": "Parsing impossible", "montant_ht": 0.0}]
    
    # Extraction informations client
    client_info = {}
    client_patterns = {
        "nom_proprietaire": r'"?(?:proprietaire|owner|nom)"?\s*[:=]\s*"([^"]+)"',
        "nom_animal": r'"?(?:animal|pet|nom_animal)"?\s*[:=]\s*"([^"]+)"',
        "identification": r'"?(?:identification|id|puce)"?\s*[:=]\s*"([^"]+)"'
    }
    
    for key, pattern in client_patterns.items():
        match = re.search(pattern, txt, re.IGNORECASE)
        if match:
            client_info[key] = match.group(1).strip()
    
    total = sum(l["montant_ht"] for l in lines)
    
    return {
        "lignes": lines,
        "montant_total": total,
        "informations_client": client_info
    }

def normaliser_accents(texte: str) -> str:
    """Normalise les accents et caractères spéciaux"""
    if not texte:
        return ""
    
    texte_nfd = unicodedata.normalize('NFD', texte)
    texte_sans_accents = ''.join(c for c in texte_nfd if unicodedata.category(c) != 'Mn')
    texte_clean = re.sub(r'[^\w\s]', ' ', texte_sans_accents.lower())
    
    return ' '.join(texte_clean.split())

class NormaliseurAMVAmeliore:
    """Normaliseur amélioré utilisant tous les fichiers de configuration PennyPet"""
    
    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.cache: Dict[str, Optional[str]] = {}
        
        # Récupération sécurisée de tous les DataFrames
        self.termes_actes = self._get_termes_actes_safe(config)
        self.actes_df = self._get_actes_df_safe(config)
        
        # Glossaire pharmaceutique
        self.termes_medicaments = getattr(config, 'glossaire_pharmaceutique', set())
        self.medicaments_df = getattr(config, 'medicaments_df', pd.DataFrame())
        self.mapping_amv = getattr(config, 'mapping_amv', {})
        
        # Tous les autres DataFrames de config
        self.calculs_codes_df = getattr(config, 'calculs_codes_df', pd.DataFrame())
        self.infos_financieres_df = getattr(config, 'infos_financieres_df', pd.DataFrame())
        self.metadonnees_df = getattr(config, 'metadonnees_df', pd.DataFrame())
        self.parties_benef_df = getattr(config, 'parties_benef_df', pd.DataFrame())
        self.suivi_sla_df = getattr(config, 'suivi_sla_df', pd.DataFrame())
        self.formules = getattr(config, 'formules', {})
        
        # Préprocessage
        self.glossaire_normalise = self._preprocess_glossaire()
        
        # Patterns regex étendus
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
        
        self.patterns_actes = [
            r'\b(consultation|examen|visite|contrôle|bilan)\b',
            r'\b(chirurgie|opération|intervention|anesthésie)\b',
            r'\b(radio|échographie|scanner|irm|endoscopie)\b',
            r'\b(analyse|prélèvement|biopsie|cytologie)\b',
            r'\b(hospitalisation|perfusion|soin|pansement)\b'
        ]
        
        # Variantes orthographiques
        self.variantes = {
            'medicament': ['médicament', 'medicaments', 'médicaments', 'medoc', 'produit'],
            'gelule': ['gélule', 'gélules', 'gelules', 'capsule', 'caps'],
            'comprimes': ['comprimé', 'comprimés', 'comprimes', 'cp', 'tablet'],
            'solution': ['solutions', 'sol', 'soluté', 'liquide'],
            'injection': ['injections', 'inj', 'piqûre', 'vaccin'],
            'consultation': ['consult', 'visite', 'rdv'],
            'chirurgie': ['chir', 'operation', 'intervention']
        }
        
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
                for col in ['field_label', 'description', 'code']:
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
            logger.error(f"Erreur DataFrame actes: {e}")
            return pd.DataFrame()

    def _preprocess_glossaire(self) -> Dict[str, str]:
        """Préprocesse tous les termes médicaux"""
        glossaire_normalise = {}
        
        try:
            # Glossaire principal
            for terme in self.termes_medicaments:
                if terme:
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
            logger.error(f"Erreur préprocessing: {e}")
        
        logger.info(f"Glossaire: {len(glossaire_normalise)} entrées")
        return glossaire_normalise

    def _detecter_patterns_medicaments(self, texte: str) -> bool:
        """Détecte les patterns de médicaments"""
        try:
            texte_norm = normaliser_accents(texte)
            return any(re.search(pattern, texte_norm, re.IGNORECASE) for pattern in self.patterns_medicaments)
        except:
            return False

    def _detecter_patterns_actes(self, texte: str) -> bool:
        """Détecte les patterns d'actes"""
        try:
            texte_norm = normaliser_accents(texte)
            return any(re.search(pattern, texte_norm, re.IGNORECASE) for pattern in self.patterns_actes)
        except:
            return False

    def normalise(self, libelle_brut: str) -> str:
        """Normalise un libellé (point d'entrée principal)"""
        if not libelle_brut:
            return "INDÉTERMINÉ"
        
        cle = str(libelle_brut).upper().strip()
        if cle in self.cache:
            return self.cache[cle]
        
        libelle_norm = normaliser_accents(libelle_brut)
        
        # 1. Détection médicaments
        if (self._detecter_patterns_medicaments(libelle_brut) or 
            libelle_norm in self.glossaire_normalise):
            self.cache[cle] = "MEDICAMENTS"
            return "MEDICAMENTS"
        
        # 2. Détection actes
        if (self._detecter_patterns_actes(libelle_brut) or 
            any(terme in libelle_norm for terme in self.termes_actes)):
            self.cache[cle] = "ACTES"
            return "ACTES"
        
        # 3. Recherche fuzzy si disponible
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
            except:
                pass
        
        # 4. Fallback
        self.cache[cle] = cle
        return cle

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Statistiques du normaliseur"""
        return {
            "cache_size": len(self.cache),
            "actes": len(self.termes_actes),
            "medicaments": len(self.termes_medicaments),
            "glossaire_normalise": len(self.glossaire_normalise),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE
        }

class PennyPetProcessor:
    """Pipeline principal PennyPet"""
    
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
            except Exception as e:
                logger.warning(f"Client Qwen indisponible: {e}")
                self.client_qwen = None
                
            try:
                self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
            except Exception as e:
                logger.warning(f"Client Mistral indisponible: {e}")
                self.client_mistral = None
            
            # Normaliseur
            self.normaliseur = NormaliseurAMVAmeliore(self.config)
            
            # Stats
            self.stats = {
                'lignes_traitees': 0,
                'medicaments_detectes': 0,
                'actes_detectes': 0,
                'erreurs_normalisation': 0
            }
            
            logger.info("PennyPetProcessor initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            raise

    def _calculer_remboursement_pennypet(self, montant: float, formule: str, est_accident: bool) -> float:
        """Calcule le remboursement selon les vraies règles PennyPet"""
        if formule == "START":
            return 0
        elif formule == "PREMIUM":
            return min(montant, 500) if est_accident else 0
        elif formule == "INTEGRAL":
            return min(montant * 0.5, 1000)
        elif formule == "INTEGRAL_PLUS":
            return min(montant, 1000)
        return 0

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """Extraction avec parsing JSON robuste"""
        try:
            # Sélection du client
            if llm_provider.lower() == "qwen" and self.client_qwen:
                client = self.client_qwen
            elif llm_provider.lower() == "mistral" and self.client_mistral:
                client = self.client_mistral
            else:
                raise ValueError(f"Client {llm_provider} indisponible")
            
            # Appel LLM
            resp = client.analyze_invoice_image(image_bytes, formule)
            content = resp.choices[0].message.content
            
            if not content:
                raise ValueError("Réponse LLM vide")
            
            # Parsing JSON robuste
            data = parse_llm_json(content)
            
            if "lignes" not in data:
                raise ValueError("Pas de lignes dans le JSON")
            
            # Nettoyage des montants
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

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Traitement complet d'une facture"""
        
        # Reset stats
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0
        }
        
        try:
            # Extraction
            data, raw_content = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
            
            resultats = []
            accidents = {"accident", "urgent", "urgence", "fract", "trauma", "traumatisme"}
            
            # Traitement des lignes
            for ligne in data["lignes"]:
                try:
                    libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
                    montant = float(ligne.get("montant_ht", 0) or 0)
                    
                    if montant <= 0:
                        continue
                    
                    # Normalisation
                    code_norm = self.normaliseur.normalise(libelle)
                    est_medicament = (code_norm == "MEDICAMENTS")
                    
                    # Détection accident
                    est_accident = any(mot in libelle.lower() for mot in accidents)
                    
                    # Calcul remboursement
                    remboursement = self._calculer_remboursement_pennypet(montant, formule_client, est_accident)
                    
                    # Stats
                    self.stats['lignes_traitees'] += 1
                    if est_medicament:
                        self.stats['medicaments_detectes'] += 1
                    else:
                        self.stats['actes_detectes'] += 1
                    
                    # Résultat
                    resultat = {
                        "ligne": {
                            "code_acte": ligne.get("code_acte", ""),
                            "description": ligne.get("description", ""),
                            "montant_ht": montant,
                            "est_medicament": est_medicament
                        },
                        "code_normalise": code_norm,
                        "est_accident": est_accident,
                        "montant_rembourse": remboursement,
                        "montant_reste_charge": montant - remboursement,
                        "taux_remboursement": (remboursement / montant * 100) if montant > 0 else 0
                    }
                    
                    resultats.append(resultat)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    logger.error(f"Erreur traitement ligne: {e}")
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
                "mapping_stats": self.normaliseur.get_mapping_stats(),
                "raw_llm_response": raw_content
            }
            
        except Exception as e:
            logger.error(f"Erreur process_facture_pennypet: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistiques": self.stats
            }

# Instance globale
pennypet_processor = PennyPetProcessor()
