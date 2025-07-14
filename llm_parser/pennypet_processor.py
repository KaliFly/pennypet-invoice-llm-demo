import json
import re
import logging
import traceback
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient
import unicodedata

# Configuration logging ultra-détaillé
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pennypet_debug_complet.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
    logger.info("✅ RapidFuzz disponible")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("⚠️ RapidFuzz non disponible")

class DebugInfo:
    """Classe pour collecter toutes les informations de debug"""
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.steps = []
        self.raw_responses = []
        self.json_attempts = []
    
    def add_error(self, step: str, error: Exception, context: str = ""):
        error_info = {
            "step": step,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc()
        }
        self.errors.append(error_info)
        logger.error(f"❌ {step}: {error} | Context: {context}")
    
    def add_warning(self, step: str, message: str):
        warning_info = {"step": step, "message": message}
        self.warnings.append(warning_info)
        logger.warning(f"⚠️ {step}: {message}")
    
    def add_step(self, step: str, status: str, details: str = ""):
        step_info = {"step": step, "status": status, "details": details}
        self.steps.append(step_info)
        logger.info(f"📋 {step}: {status} | {details}")
    
    def get_debug_report(self) -> Dict[str, Any]:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "steps": self.steps,
            "raw_responses": self.raw_responses,
            "json_attempts": self.json_attempts
        }

def debug_json_extraction(content: str, debug: DebugInfo) -> Dict[str, Any]:
    """Extraction JSON avec debug complet à chaque étape"""
    debug.add_step("JSON_EXTRACTION", "DEBUT", f"Contenu longueur: {len(content)}")
    
    # Sauvegarde du contenu brut
    debug.raw_responses.append(content)
    
    try:
        # Étape 1: Recherche des délimiteurs JSON
        start = content.find("{")
        if start < 0:
            debug.add_error("JSON_EXTRACTION", ValueError("Aucun '{' trouvé"), content[:200])
            return {"lignes": [], "error": "NO_JSON_START"}
        
        debug.add_step("JSON_EXTRACTION", "DELIMITEUR_TROUVE", f"Position de départ: {start}")
        
        # Étape 2: Extraction par comptage de brackets
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
            debug.add_error("JSON_EXTRACTION", ValueError("JSON malformé - brackets non équilibrés"), 
                          f"Dernière position: {len(content)}, depth final: {depth}")
            return {"lignes": [], "error": "UNBALANCED_BRACKETS"}
        
        debug.add_step("JSON_EXTRACTION", "JSON_BRUT_EXTRAIT", f"Longueur: {len(json_str)}")
        debug.json_attempts.append({"raw": json_str[:500]})
        
        # Étape 3: Tentatives de parsing progressives
        parsing_attempts = [
            ("DIRECT", lambda x: json.loads(x)),
            ("NETTOYAGE_LEGER", lambda x: json.loads(clean_json_light(x))),
            ("NETTOYAGE_AGRESSIF", lambda x: json.loads(clean_json_aggressive(x))),
            ("RECONSTRUCTION", lambda x: reconstruct_json_manual(x))
        ]
        
        for attempt_name, parser_func in parsing_attempts:
            try:
                debug.add_step("JSON_PARSING", f"TENTATIVE_{attempt_name}", "En cours...")
                result = parser_func(json_str)
                debug.add_step("JSON_PARSING", f"SUCCES_{attempt_name}", f"Lignes trouvées: {len(result.get('lignes', []))}")
                return result
            except Exception as e:
                debug.add_error("JSON_PARSING", e, f"Méthode: {attempt_name}")
                debug.json_attempts.append({attempt_name: str(e)})
                continue
        
        # Si tout échoue, retourner structure d'erreur
        debug.add_error("JSON_EXTRACTION", Exception("TOUTES_METHODES_ECHOUEES"), "Aucune méthode de parsing n'a fonctionné")
        return {
            "lignes": [{"code_acte": "ERREUR_JSON", "description": "Échec de parsing", "montant_ht": 0.0}],
            "error": "ALL_PARSING_FAILED",
            "debug_info": debug.get_debug_report()
        }
        
    except Exception as e:
        debug.add_error("JSON_EXTRACTION", e, "Erreur générale")
        return {"lignes": [], "error": str(e), "debug_info": debug.get_debug_report()}

def clean_json_light(text: str) -> str:
    """Nettoyage JSON léger"""
    text = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
    text = text.replace("'", '"')
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

def clean_json_aggressive(text: str) -> str:
    """Nettoyage JSON agressif"""
    # Tous les nettoyages légers
    text = clean_json_light(text)
    
    # Suppression caractères non-ASCII
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    # Correction virgules multiples
    text = re.sub(r',,+', ',', text)
    
    # Correction des deux points
    text = re.sub(r'::+', ':', text)
    
    # Correction des objets mal fermés
    text = re.sub(r'}\s*{', '},{', text)
    
    return text

def reconstruct_json_manual(content: str) -> Dict[str, Any]:
    """Reconstruction manuelle avec patterns spécifiques"""
    result = {
        "lignes": [],
        "montant_total": 0.0,
        "informations_client": {}
    }
    
    # Patterns pour extraction manuelle
    ligne_patterns = [
        r'"code_acte"\s*:\s*"([^"]*)"[^}]*"description"\s*:\s*"([^"]*)"[^}]*"montant_ht"\s*:\s*([0-9.,]+)',
        r'code_acte[^:]*:\s*"([^"]*)"[^}]*description[^:]*:\s*"([^"]*)"[^}]*montant[^:]*:\s*([0-9.,]+)',
        r'"([^"]*)"[^,]*,\s*"([^"]*)"[^,]*,\s*([0-9.,]+)'
    ]
    
    for pattern in ligne_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        if matches:
            for match in matches:
                try:
                    montant = float(str(match[2]).replace(',', '.'))
                    result["lignes"].append({
                        "code_acte": match[0],
                        "description": match[1], 
                        "montant_ht": montant
                    })
                except ValueError:
                    continue
            break
    
    # Extraction montant total
    total_pattern = r'(?:montant_total|total)[^:]*:\s*([0-9.,]+)'
    total_match = re.search(total_pattern, content, re.IGNORECASE)
    if total_match:
        try:
            result["montant_total"] = float(total_match.group(1).replace(',', '.'))
        except ValueError:
            pass
    
    return result

class PennyPetProcessorDebug:
    """Version debug complète du PennyPetProcessor"""
    
    def __init__(self, client_qwen=None, client_mistral=None, config=None):
        self.debug = DebugInfo()
        self.debug.add_step("INIT", "DEBUT", "Initialisation du processor")
        
        try:
            # Chargement de la config
            self.config = config or PennyPetConfig()
            self.debug.add_step("INIT", "CONFIG_LOADED", f"Config chargée")
            
            # Clients LLM
            try:
                self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
                self.debug.add_step("INIT", "QWEN_OK", "Client Qwen initialisé")
            except Exception as e:
                self.debug.add_error("INIT", e, "Client Qwen")
                self.client_qwen = None
                
            try:
                self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
                self.debug.add_step("INIT", "MISTRAL_OK", "Client Mistral initialisé")
            except Exception as e:
                self.debug.add_error("INIT", e, "Client Mistral")
                self.client_mistral = None
            
            # Vérification des DataFrames de config
            self._debug_config_status()
            
            self.debug.add_step("INIT", "COMPLETE", "Processor initialisé avec succès")
            
        except Exception as e:
            self.debug.add_error("INIT", e, "Erreur générale d'initialisation")
            raise
    
    def _debug_config_status(self):
        """Debug du statut de la configuration"""
        config_items = [
            'actes_df', 'medicaments_df', 'regles_pc_df', 'glossaire_pharmaceutique',
            'calculs_codes_df', 'infos_financieres_df', 'mapping_amv', 'formules'
        ]
        
        for item in config_items:
            if hasattr(self.config, item):
                value = getattr(self.config, item)
                if isinstance(value, pd.DataFrame):
                    status = f"DataFrame {len(value)} lignes"
                elif isinstance(value, (dict, set)):
                    status = f"{type(value).__name__} {len(value)} éléments"
                else:
                    status = f"{type(value).__name__}"
                self.debug.add_step("CONFIG_CHECK", item.upper(), status)
            else:
                self.debug.add_warning("CONFIG_CHECK", f"{item} manquant")
    
    def extract_lignes_from_image_debug(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Version debug de l'extraction avec TOUS les détails"""
        
        self.debug.add_step("EXTRACTION", "DEBUT", f"Provider: {llm_provider}, Formule: {formule}")
        
        try:
            # Sélection du client
            if llm_provider.lower() == "qwen" and self.client_qwen:
                client = self.client_qwen
                self.debug.add_step("EXTRACTION", "CLIENT_SELECTED", "Qwen sélectionné")
            elif llm_provider.lower() == "mistral" and self.client_mistral:
                client = self.client_mistral
                self.debug.add_step("EXTRACTION", "CLIENT_SELECTED", "Mistral sélectionné")
            else:
                error_msg = f"Client {llm_provider} non disponible"
                self.debug.add_error("EXTRACTION", ValueError(error_msg), "Sélection client")
                return {}, "", self.debug.get_debug_report()
            
            # Appel LLM avec debug
            self.debug.add_step("LLM_CALL", "DEBUT", f"Taille image: {len(image_bytes)} bytes")
            
            try:
                resp = client.analyze_invoice_image(image_bytes, formule)
                self.debug.add_step("LLM_CALL", "SUCCES", "Réponse LLM reçue")
            except Exception as e:
                self.debug.add_error("LLM_CALL", e, "Appel API LLM")
                return {}, "", self.debug.get_debug_report()
            
            # Extraction du contenu
            try:
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("Contenu vide")
                self.debug.add_step("LLM_RESPONSE", "CONTENT_OK", f"Longueur: {len(content)}")
            except Exception as e:
                self.debug.add_error("LLM_RESPONSE", e, "Extraction contenu")
                return {}, "", self.debug.get_debug_report()
            
            # Log du contenu pour debug
            logger.debug(f"CONTENU LLM COMPLET:\n{content}")
            
            # Extraction JSON avec debug complet
            data = debug_json_extraction(content, self.debug)
            
            self.debug.add_step("EXTRACTION", "COMPLETE", f"Données extraites: {len(data.get('lignes', []))} lignes")
            
            return data, content, self.debug.get_debug_report()
            
        except Exception as e:
            self.debug.add_error("EXTRACTION", e, "Erreur générale extraction")
            return {}, "", self.debug.get_debug_report()
    
    def process_facture_pennypet_debug(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Version debug complète du traitement de facture"""
        
        self.debug.add_step("PROCESS", "DEBUT", f"Formule: {formule_client}")
        
        try:
            # Extraction avec debug
            data, raw_content, debug_report = self.extract_lignes_from_image_debug(
                file_bytes, formule_client, llm_provider
            )
            
            # Vérification des données extraites
            if not data or not data.get("lignes"):
                return {
                    "success": False,
                    "error": "EXTRACTION_FAILED",
                    "debug_report": debug_report,
                    "raw_content": raw_content,
                    "data_extracted": data
                }
            
            # Traitement des lignes
            resultats = []
            accidents = {"accident", "urgent", "urgence", "fract", "trauma", "traumatisme"}
            
            for i, ligne in enumerate(data["lignes"]):
                try:
                    self.debug.add_step("PROCESS_LINE", f"LIGNE_{i}", f"Traitement: {ligne.get('code_acte', 'N/A')}")
                    
                    # Extraction des données
                    libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
                    montant = float(ligne.get("montant_ht", 0) or 0)
                    
                    # Détection d'accident
                    est_acc = any(mot in libelle.lower() for mot in accidents)
                    
                    # Simulation du remboursement (règles PennyPet simplifiées)
                    if formule_client == "START":
                        remb = 0.0
                    elif formule_client == "PREMIUM":
                        remb = min(montant, 500) if est_acc else 0.0
                    elif formule_client == "INTEGRAL":
                        remb = min(montant * 0.5, 1000)
                    elif formule_client == "INTEGRAL_PLUS":
                        remb = min(montant, 1000)
                    else:
                        remb = 0.0
                    
                    ligne_result = {
                        **ligne,
                        "est_accident": est_acc,
                        "remboursement": remb,
                        "reste": montant - remb
                    }
                    
                    resultats.append(ligne_result)
                    self.debug.add_step("PROCESS_LINE", f"LIGNE_{i}_OK", f"Remb: {remb}€")
                    
                except Exception as e:
                    self.debug.add_error("PROCESS_LINE", e, f"Ligne {i}: {ligne}")
                    continue
            
            # Calcul des totaux
            total_facture = sum(r.get("montant_ht", 0) for r in resultats)
            total_remb = sum(r.get("remboursement", 0) for r in resultats)
            
            result = {
                "success": True,
                "lignes": resultats,
                "total_facture": total_facture,
                "total_remboursement": total_remb,
                "reste_a_charge": total_facture - total_remb,
                "informations_client": data.get("informations_client", {}),
                "debug_report": debug_report,
                "raw_llm_response": raw_content
            }
            
            self.debug.add_step("PROCESS", "COMPLETE", f"Traitement terminé: {len(resultats)} lignes")
            
            return result
            
        except Exception as e:
            self.debug.add_error("PROCESS", e, "Erreur générale traitement")
            return {
                "success": False,
                "error": str(e),
                "debug_report": self.debug.get_debug_report(),
                "traceback": traceback.format_exc()
            }

# Instance globale pour usage direct
pennypet_processor_debug = PennyPetProcessorDebug()
