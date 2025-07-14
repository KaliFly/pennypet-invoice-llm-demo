import json
import re
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient
import unicodedata

# Configuration du logging détaillé
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

# [Insérer ici toutes les fonctions de parsing JSON définies ci-dessus]

def normaliser_accents(texte: str) -> str:
    """Normalise les accents et caractères spéciaux"""
    if not texte:
        return ""
    
    texte_nfd = unicodedata.normalize('NFD', texte)
    texte_sans_accents = ''.join(c for c in texte_nfd if unicodedata.category(c) != 'Mn')
    texte_clean = re.sub(r'[^\w\s]', ' ', texte_sans_accents.lower())
    return ' '.join(texte_clean.split())

class NormaliseurAMVAmeliore:
    """Normaliseur amélioré utilisant le glossaire JSON existant"""
    
    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.cache: Dict[str, Optional[str]] = {}
        
        # Récupération sécurisée des termes d'actes
        self.termes_actes = self._get_termes_actes_safe(config)
        self.actes_df = self._get_actes_df_safe(config)
        
        # Utilisation du glossaire pharmaceutique EXISTANT
        self.termes_medicaments = config.glossaire_pharmaceutique
        self.medicaments_df = getattr(config, 'medicaments_df', pd.DataFrame())
        self.mapping_amv = getattr(config, 'mapping_amv', {})
        
        # Préprocessage du glossaire
        self.glossaire_normalise = self._preprocess_glossaire()
        
        # Patterns regex pour médicaments
        self.patterns_medicaments = [
            r'\b\d+\s*(mg|ml|g|l|ui|iu|mcg|µg)\b',
            r'\b(comprimé|gélule|cp|gél|sol|inj|ampoule|flacon|tube|boîte)\.?\s*\d*',
            r'\b(antibiotic|anti-inflammatoire|antiparasitaire|antifongique|antiviral)\b',
            r'\b(vaccin|vaccination|rappel|primo-vaccination)\b',
            r'\b(seringue|pipette|spray|pommade|crème|lotion)\b',
            r'\b\d+\s*x\s*\d+\s*(mg|ml|g|l)\b',
            r'\b(principe|actif|laboratoire|generique|specialite)\b'
        ]
        
        logger.info(f"Normaliseur initialisé: {len(self.termes_actes)} actes, {len(self.termes_medicaments)} médicaments")

    def _get_termes_actes_safe(self, config: PennyPetConfig) -> set:
        """Récupère les termes d'actes de manière sécurisée"""
        try:
            if not hasattr(config, 'actes_df') or config.actes_df is None or config.actes_df.empty:
                logger.warning("actes_df non disponible")
                return set()
            
            df = config.actes_df
            
            if "field_label" in df.columns:
                logger.info("Utilisation de la colonne 'field_label' pour les actes")
                return set(df["field_label"].dropna().astype(str).str.lower())
            
            # Fallback sur d'autres colonnes
            possible_columns = ['label', 'acte', 'description', 'libelle', 'terme']
            for col in possible_columns:
                if col in df.columns:
                    logger.info(f"Utilisation de la colonne '{col}' pour les actes")
                    return set(df[col].dropna().astype(str).str.lower())
            
            return set()
            
        except Exception as e:
            logger.error(f"Erreur extraction termes actes: {e}")
            return set()

    def _get_actes_df_safe(self, config: PennyPetConfig) -> pd.DataFrame:
        """Récupère le DataFrame des actes de manière sécurisée"""
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
                        
        except Exception as e:
            logger.error(f"Erreur préprocessing glossaire: {e}")
        
        logger.info(f"Glossaire normalisé: {len(glossaire_normalise)} entrées")
        return glossaire_normalise

    def normalise_medicament(self, libelle_brut: str) -> Optional[str]:
        """Normalise un médicament avec le glossaire JSON existant"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Détection par patterns regex
            for pattern in self.patterns_medicaments:
                if re.search(pattern, libelle_norm, re.IGNORECASE):
                    self.cache[cle] = "MEDICAMENTS"
                    return "MEDICAMENTS"
            
            # 2. Recherche dans glossaire normalisé
            if libelle_norm in self.glossaire_normalise:
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
            
            # 3. Recherche partielle
            for terme_norm in self.glossaire_normalise.keys():
                if terme_norm in libelle_norm or libelle_norm in terme_norm:
                    self.cache[cle] = "MEDICAMENTS"
                    return "MEDICAMENTS"
            
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
            result = self.normalise_medicament(libelle_brut)
            if result:
                return result
            
            return str(libelle_brut).strip().upper()
            
        except Exception as e:
            logger.error(f"Erreur normalisation '{libelle_brut}': {e}")
            return str(libelle_brut).strip().upper() if libelle_brut else None

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de mapping"""
        return {
            "cache_size": len(self.cache),
            "actes": len(self.termes_actes),
            "medicaments": len(self.termes_medicaments),
            "glossaire_normalise": len(self.glossaire_normalise),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE
        }

class PennyPetProcessor:
    """Pipeline extraction LLM, normalisation améliorée, calcul remboursement PennyPet"""
    
    def __init__(self, client_qwen=None, client_mistral=None, config=None):
        try:
            self.config = config or PennyPetConfig()
            
            # Initialisation des clients
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
            
            self.regles_pc_df = getattr(self.config, 'regles_pc_df', pd.DataFrame())
            self.normaliseur = NormaliseurAMVAmeliore(self.config)
            
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

    def extract_lignes_from_image(self, image_bytes: bytes, formule: str, llm_provider: str = "qwen") -> Tuple[Dict[str, Any], str]:
        """Extrait les lignes d'une image avec parsing JSON ultra-robuste"""
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
            
            # Logs détaillés pour debug
            logger.info(f"Réponse LLM (longueur: {len(content)})")
            logger.info(f"Premiers 200 chars: {content[:200]}")
            logger.info(f"Derniers 200 chars: {content[-200:]}")
            logger.info(f"Accolades ouvrantes: {content.count('{')}, fermantes: {content.count('}')}")
            
            # Parsing JSON ultra-robuste
            try:
                data = parser_json_ultra_robuste(content)
                logger.info("✅ JSON parsé avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur parsing JSON: {e}")
                logger.error(f"Contenu problématique sauvegardé dans pennypet_debug.log")
                
                # Fallback ultime
                data = {
                    "lignes": [{"code_acte": "ERREUR_JSON", "description": "Erreur parsing JSON", "montant_ht": 0.0}],
                    "montant_total": 0.0,
                    "informations_client": {}
                }
            
            # Validation et nettoyage des données
            if "lignes" not in data:
                data["lignes"] = []
            
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
            logger.error(f"Erreur dans extract_lignes_from_image: {e}")
            raise

    def calculer_remboursement(self, montant: float, code_acte: str, formule: str, est_accident: bool) -> Dict[str, Any]:
        """Calcule le remboursement selon les règles PennyPet"""
        # Application directe des règles PennyPet
        if formule == "START":
            return {"montant_ht": montant, "taux": 0.0, "remb_final": 0.0, "reste": montant}
        elif formule == "PREMIUM":
            if est_accident:
                remb = min(montant, 500)  # 100% jusqu'à 500€
                return {"montant_ht": montant, "taux": 100.0, "remb_final": remb, "reste": montant - remb}
            else:
                return {"montant_ht": montant, "taux": 0.0, "remb_final": 0.0, "reste": montant}
        elif formule == "INTEGRAL":
            remb = min(montant * 0.5, 1000)  # 50% jusqu'à 1000€
            return {"montant_ht": montant, "taux": 50.0, "remb_final": remb, "reste": montant - remb}
        elif formule == "INTEGRAL_PLUS":
            remb = min(montant, 1000)  # 100% jusqu'à 1000€
            return {"montant_ht": montant, "taux": 100.0, "remb_final": remb, "reste": montant - remb}
        else:
            return {"montant_ht": montant, "taux": 0.0, "remb_final": 0.0, "reste": montant}

    def process_facture_pennypet(self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen") -> Dict[str, Any]:
        """Traite une facture PennyPet complète"""
        
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0
        }
        
        try:
            # Extraction des données avec parsing JSON robuste
            data, raw_content = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
            
            resultats: List[Dict[str, Any]] = []
            accidents = {"accident", "urgent", "urgence", "fract", "trauma", "traumatisme"}
            
            # Traitement de chaque ligne
            for ligne in data["lignes"]:
                try:
                    libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
                    montant = float(ligne.get("montant_ht", 0) or 0)
                    
                    # Normalisation
                    code_norm = self.normaliseur.normalise(libelle)
                    
                    # Détection d'accident
                    est_acc = any(mot in libelle.lower() for mot in accidents)
                    
                    # Calcul du remboursement
                    remb = self.calculer_remboursement(montant, code_norm, formule_client, est_acc)
                    
                    # Mise à jour des statistiques
                    self.stats['lignes_traitees'] += 1
                    if code_norm == "MEDICAMENTS":
                        self.stats['medicaments_detectes'] += 1
                    else:
                        self.stats['actes_detectes'] += 1
                    
                    # Structure de retour pour compatibilité avec l'interface
                    ligne_result = {
                        "ligne": {
                            "code_acte": ligne.get("code_acte", ""),
                            "description": ligne.get("description", ""),
                            "montant_ht": montant,
                            "est_medicament": (code_norm == "MEDICAMENTS")
                        },
                        "code_norm": code_norm,
                        "est_accident": est_acc,
                        "taux_remboursement": remb.get("taux", 0),
                        "montant_rembourse": remb.get("remb_final", 0),
                        "montant_reste_charge": remb.get("reste", montant)
                    }
                    resultats.append(ligne_result)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    logger.error(f"Erreur traitement ligne {ligne}: {e}")
                    continue
            
            # Calcul des totaux
            total_remb = sum(r.get("montant_rembourse", 0) for r in resultats)
            total_facture = sum(r.get("ligne", {}).get("montant_ht", 0) for r in resultats)
            
            return {
                "success": True,
                "lignes": resultats,
                "resume": {
                    "total_facture": total_facture,
                    "total_rembourse": total_remb,
                    "reste_a_charge": total_facture - total_remb,
                    "taux_remboursement_global": (total_remb / total_facture * 100) if total_facture > 0 else 0
                },
                "informations_client": data.get("informations_client", {}),
                "statistiques": self.stats,
                "mapping_stats": self.normaliseur.get_mapping_stats(),
                "raw_llm_response": raw_content
            }
            
        except Exception as e:
            logger.error(f"Erreur dans process_facture_pennypet: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistiques": self.stats
            }

    def get_processor_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes du processeur"""
        return {
            **self.stats,
            **self.normaliseur.get_mapping_stats()
        }

# Instance globale pour usage direct
pennypet_processor = PennyPetProcessor()
