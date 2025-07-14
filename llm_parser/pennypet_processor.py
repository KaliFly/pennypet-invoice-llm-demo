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
    Correction ultra-robuste pour JSON mal formé.
    """
    try:
        # Nettoyage initial agressif
        text = text.strip()
        
        # Suppression des caractères non ASCII problématiques
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        
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
        text = re.sub(r'(".*?")\s*\n\s*(\d+)', r'\1,\n\2', text)
        
        # 6. Suppression des doubles virgules
        text = re.sub(r',,+', ',', text)
        
        # 7. Correction des deux points multiples
        text = re.sub(r'::+', ':', text)
        
        # 8. Correction des accolades non fermées
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        
        # 9. Correction des crochets non fermés
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        if open_brackets > close_brackets:
            text += ']' * (open_brackets - close_brackets)
        
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
            logger.warning("Aucun '{' trouvé, tentative avec patterns alternatifs")
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
            logger.warning("JSON incomplet détecté, reconstruction...")
            # Essayer de prendre tout jusqu'à la fin
            json_str = content[start:] + "}"
        
        logger.info(f"JSON brut extrait: {json_str[:200]}...")
        
        # Nettoyage et tentative de parsing
        json_clean = pseudojson_to_json_ameliore(json_str)
        logger.info(f"JSON nettoyé: {json_clean[:200]}...")
        
        result = json.loads(json_clean)
        logger.info("Méthode 1 réussie - JSON parsé avec succès")
        return result
        
    except json.JSONDecodeError as e:
        logger.warning(f"Parsing JSON échoué (méthode 1) à la position {e.pos}: {e.msg}")
        
        # Méthode 2: Nettoyage plus agressif
        try:
            logger.info("Tentative méthode 2 - Nettoyage agressif")
            
            # Extraction ligne par ligne
            lines = content.split('\n')
            json_lines = []
            in_json = False
            brace_count = 0
            
            for line in lines:
                if '{' in line and not in_json:
                    in_json = True
                
                if in_json:
                    json_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count <= 0 and '}' in line:
                        break
            
            json_str = '\n'.join(json_lines)
            json_clean = pseudojson_to_json_ameliore(json_str)
            
            result = json.loads(json_clean)
            logger.info("Méthode 2 réussie - JSON parsé avec succès")
            return result
            
        except Exception as e2:
            logger.warning(f"Méthode 2 échouée: {e2}")
            
            # Méthode 3: Reconstruction manuelle
            try:
                logger.info("Tentative méthode 3 - Reconstruction manuelle")
                result = reconstruire_json_manuellement(content)
                logger.info("Méthode 3 réussie - JSON reconstruit manuellement")
                return result
                
            except Exception as e3:
                logger.error(f"Toutes les méthodes ont échoué: {e3}")
                
                # Méthode 4: Structure minimale garantie
                logger.warning("Utilisation de la structure minimale de fallback")
                return {
                    "lignes": [{"code_acte": "ERREUR_JSON", "description": "Erreur parsing JSON", "montant_ht": 0.0}],
                    "montant_total": 0.0,
                    "informations_client": {"nom_proprietaire": "", "nom_animal": "", "identification": ""}
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
        logger.info("Reconstruction manuelle en cours...")
        
        # Patterns multiples pour les lignes
        lignes_patterns = [
            # Pattern complet avec code_acte, description et montant
            r'"code_acte"\s*:\s*"([^"]*)"[^}]*"description"\s*:\s*"([^"]*)"[^}]*"montant_ht"\s*:\s*([0-9]+\.?[0-9]*)',
            r'"description"\s*:\s*"([^"]*)"[^}]*"montant_ht"\s*:\s*([0-9]+\.?[0-9]*)[^}]*"code_acte"\s*:\s*"([^"]*)"',
            r'"montant_ht"\s*:\s*([0-9]+\.?[0-9]*)[^}]*"code_acte"\s*:\s*"([^"]*)"[^}]*"description"\s*:\s*"([^"]*)"',
            
            # Patterns simplifiés
            r'"description"\s*:\s*"([^"]*)"[^}]*"montant_ht"\s*:\s*([0-9]+\.?[0-9]*)',
            r'"code_acte"\s*:\s*"([^"]*)"[^}]*"montant_ht"\s*:\s*([0-9]+\.?[0-9]*)',
            r'acte[^:]*:\s*"([^"]*)"[^}]*montant[^:]*:\s*([0-9]+\.?[0-9]*)',
            
            # Pattern très basique
            r'"([^"]*)"[^}]*([0-9]+\.?[0-9]*)'
        ]
        
        lignes_trouvees = []
        
        for i, pattern in enumerate(lignes_patterns):
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                logger.info(f"Pattern {i+1} trouvé: {len(matches)} correspondances")
                
                for match in matches:
                    if len(match) == 3:
                        # Pattern complet
                        if i == 1:  # description, montant, code_acte
                            ligne = {
                                "code_acte": match[2],
                                "description": match[0],
                                "montant_ht": float(match[1])
                            }
                        elif i == 2:  # montant, code_acte, description
                            ligne = {
                                "code_acte": match[1],
                                "description": match[2],
                                "montant_ht": float(match[0])
                            }
                        else:  # code_acte, description, montant
                            ligne = {
                                "code_acte": match[0],
                                "description": match[1],
                                "montant_ht": float(match[2])
                            }
                    elif len(match) == 2:
                        # Pattern simplifié
                        try:
                            montant = float(match[1])
                            ligne = {
                                "code_acte": match[0],
                                "description": match[0],
                                "montant_ht": montant
                            }
                        except ValueError:
                            # Peut-être que l'ordre est inversé
                            try:
                                montant = float(match[0])
                                ligne = {
                                    "code_acte": match[1],
                                    "description": match[1],
                                    "montant_ht": montant
                                }
                            except ValueError:
                                continue
                    else:
                        continue
                    
                    # Validation et nettoyage
                    if ligne["montant_ht"] > 0 and ligne["code_acte"].strip():
                        lignes_trouvees.append(ligne)
                
                if lignes_trouvees:
                    break
        
        result["lignes"] = lignes_trouvees
        
        # Extraction du montant total
        montant_patterns = [
            r'"montant_total"\s*:\s*([0-9]+\.?[0-9]*)',
            r'total[^:]*:\s*([0-9]+\.?[0-9]*)',
            r'montant[^:]*total[^:]*:\s*([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in montant_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result["montant_total"] = float(match.group(1))
                break
        
        # Si pas de montant total trouvé, calculer la somme
        if result["montant_total"] == 0.0 and result["lignes"]:
            result["montant_total"] = sum(ligne["montant_ht"] for ligne in result["lignes"])
        
        # Extraction des informations client
        client_patterns = {
            "nom_proprietaire": [
                r'"nom_proprietaire"\s*:\s*"([^"]*)"',
                r'proprietaire[^:]*:\s*"([^"]*)"',
                r'nom[^:]*proprietaire[^:]*:\s*"([^"]*)"'
            ],
            "nom_animal": [
                r'"nom_animal"\s*:\s*"([^"]*)"',
                r'animal[^:]*:\s*"([^"]*)"',
                r'nom[^:]*animal[^:]*:\s*"([^"]*)"'
            ],
            "identification": [
                r'"identification"\s*:\s*"([^"]*)"',
                r'identification[^:]*:\s*"([^"]*)"',
                r'id[^:]*:\s*"([^"]*)"',
                r'numero[^:]*:\s*"([^"]*)"'
            ]
        }
        
        for key, patterns in client_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match and match.group(1).strip():
                    result["informations_client"][key] = match.group(1).strip()
                    break
        
        logger.info(f"JSON reconstruit: {len(result['lignes'])} lignes, montant total: {result['montant_total']}")
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
    
    try:
        # Normalisation NFD puis suppression des accents
        texte_nfd = unicodedata.normalize('NFD', texte)
        texte_sans_accents = ''.join(c for c in texte_nfd if unicodedata.category(c) != 'Mn')
        
        # Conversion en minuscules et nettoyage
        texte_clean = re.sub(r'[^\w\s]', ' ', texte_sans_accents.lower())
        
        # Normalisation des espaces
        return ' '.join(texte_clean.split())
    except Exception as e:
        logger.warning(f"Erreur normalisation accents pour '{texte}': {e}")
        return str(texte).lower().strip()

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
        
        # Glossaire pharmaceutique existant
        self.termes_medicaments = getattr(config, 'glossaire_pharmaceutique', set())
        self.medicaments_df = getattr(config, 'medicaments_df', pd.DataFrame())
        self.mapping_amv = getattr(config, 'mapping_amv', {})
        
        # TOUS les fichiers de configuration
        self.calculs_codes_df = getattr(config,
