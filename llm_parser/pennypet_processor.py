import json
import re
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient
import unicodedata

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("RapidFuzz non disponible, fuzzy matching désactivé")

def pseudojson_to_json(text: str) -> str:
    """
    Correction minimale pour JSON mal formé.
    """
    text = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
    text = text.replace("'", '"')
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

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
    Normaliseur amélioré utilisant le glossaire JSON existant
    """
    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.cache: Dict[str, Optional[str]] = {}
        
        # Récupération sécurisée des termes d'actes
        self.termes_actes = self._get_termes_actes_safe(config)
        self.actes_df = self._get_actes_df_safe(config)
        
        # Utilisation du glossaire pharmaceutique EXISTANT (set)
        self.termes_medicaments = config.glossaire_pharmaceutique
        self.medicaments_df = getattr(config, 'medicaments_df', pd.DataFrame())
        self.mapping_amv = getattr(config, 'mapping_amv', {})
        
        # Préprocessage du glossaire (le glossaire est déjà un set)
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
        
        # Variantes orthographiques courantes
        self.variantes = {
            'medicament': ['médicament', 'medicaments', 'médicaments'],
            'gelule': ['gélule', 'gélules', 'gelules'],
            'comprimes': ['comprimé', 'comprimés', 'comprimes'],
            'solution': ['solutions', 'sol'],
            'injection': ['injections', 'inj'],
            'milligramme': ['mg', 'milligrammes'],
            'millilitre': ['ml', 'millilitres'],
            'gramme': ['g', 'grammes'],
            'litre': ['l', 'litres']
        }
        
        logger.info(f"Normaliseur initialisé: {len(self.termes_actes)} actes, {len(self.termes_medicaments)} médicaments")

    def _get_termes_actes_safe(self, config: PennyPetConfig) -> set:
        """Récupère les termes d'actes de manière sécurisée"""
        try:
            if not hasattr(config, 'actes_df') or config.actes_df is None or config.actes_df.empty:
                logger.warning("actes_df non disponible")
                return set()
            
            df = config.actes_df
            
            # Vérifier field_label existe
            if "field_label" in df.columns:
                logger.info("Utilisation de la colonne 'field_label' pour les actes")
                return set(df["field_label"].dropna().astype(str).str.lower())
            
            # Fallback sur d'autres colonnes possibles
            possible_columns = ['label', 'acte', 'description', 'libelle', 'terme']
            for col in possible_columns:
                if col in df.columns:
                    logger.info(f"Utilisation de la colonne '{col}' pour les actes")
                    return set(df[col].dropna().astype(str).str.lower())
            
            # Dernière tentative avec la première colonne texte
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                col = text_columns[0]
                logger.warning(f"Utilisation de '{col}' par défaut pour les actes")
                return set(df[col].dropna().astype(str).str.lower())
            
            return set()
            
        except Exception as e:
            logger.error(f"Erreur extraction termes actes: {e}")
            return set()

    def _get_actes_df_safe(self, config: PennyPetConfig) -> pd.DataFrame:
        """Récupère le DataFrame des actes de manière sécurisée"""
        try:
            if not hasattr(config, 'actes_df') or config.actes_df is None or config.actes_df.empty:
                return pd.DataFrame()
            
            df = config.actes_df
            
            if 'pattern' in df.columns:
                return df.dropna(subset=["pattern"])
            else:
                logger.warning("Colonne 'pattern' manquante dans actes_df")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Erreur extraction DataFrame actes: {e}")
            return pd.DataFrame()

    def _preprocess_glossaire(self) -> Dict[str, str]:
        """Préprocesse le glossaire pharmaceutique (déjà un set)"""
        glossaire_normalise = {}
        
        try:
            # Le glossaire est déjà un set de termes en minuscules
            for terme in self.termes_medicaments:
                if not terme:
                    continue
                    
                # Normalisation principale
                terme_norm = normaliser_accents(str(terme))
                if terme_norm:
                    glossaire_normalise[terme_norm] = terme
                
                # Ajouter les variantes
                for variante in self._generer_variantes(str(terme)):
                    variante_norm = normaliser_accents(variante)
                    if variante_norm:
                        glossaire_normalise[variante_norm] = terme
                        
        except Exception as e:
            logger.error(f"Erreur préprocessing glossaire: {e}")
        
        logger.info(f"Glossaire normalisé: {len(glossaire_normalise)} entrées")
        return glossaire_normalise

    def _generer_variantes(self, terme: str) -> List[str]:
        """Génère des variantes orthographiques d'un terme"""
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

    def normalise_acte(self, libelle_brut: str) -> Optional[str]:
        """Normalise un acte médical"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            # Normalisation pour recherche
            libelle_norm = normaliser_accents(libelle_brut)
            
            # 1. Pattern CSV exact
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
            
            # 2. Fallback sémantique actes
            for terme in self.termes_actes:
                try:
                    terme_norm = normaliser_accents(terme)
                    if re.search(rf"(?<!\w){re.escape(terme_norm)}(?!\w)", libelle_norm):
                        code = terme.upper()
                        self.cache[cle] = code
                        return code
                except Exception:
                    continue
            
            # 3. Fuzzy matching sur les actes
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
        """Normalise un médicament avec le glossaire JSON existant"""
        if not libelle_brut:
            return None
        
        try:
            cle = str(libelle_brut).upper().strip()
            if cle in self.cache:
                return self.cache[cle]
            
            # Normalisation pour recherche
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
            
            # 4. Fuzzy matching intelligent
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
            
            # Fallback : retourner le libellé original normalisé
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
            "patterns_medicaments": len(self.patterns_medicaments),
            "variantes": len(self.variantes),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE,
            "actes_df_size": len(self.actes_df),
            "medicaments_df_size": len(self.medicaments_df)
        }

# Le reste de la classe PennyPetProcessor reste identique...
class PennyPetProcessor:
    """
    Pipeline extraction LLM, normalisation améliorée, calcul remboursement PennyPet.
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
                logger.warning(f"Erreur initialisation client Qwen: {e}")
                self.client_qwen = None
                
            try:
                self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
            except Exception as e:
                logger.warning(f"Erreur initialisation client Mistral: {e}")
                self.client_mistral = None
            
            self.regles_pc_df = getattr(self.config, 'regles_pc_df', pd.DataFrame())
            self.normaliseur = NormaliseurAMVAmeliore(self.config)
            
            # Statistiques de traitement
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

    # Les autres méthodes restent identiques à votre version originale...
    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """Extrait les lignes d'une image avec le LLM"""
        client = self.client_qwen if llm_provider.lower() == "qwen" else self.client_mistral
        resp = client.analyze_invoice_image(image_bytes, formule)
        content = resp.choices[0].message.content
        
        # Extraction JSON robuste
        start = content.find("{")
        if start < 0:
            raise ValueError("JSON non trouvé dans la réponse LLM.")
        
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
            raise ValueError("JSON malformé dans la réponse LLM.")
        
        # Nettoyage et parsing JSON
        json_str = pseudojson_to_json(json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur parsing JSON: {e}")
        
        if "lignes" not in data:
            raise ValueError("Le LLM n'a pas extrait de lignes exploitables.")
        
        return data, content

    def calculer_remboursement(
        self, montant: float, code_acte: str, formule: str, est_accident: bool
    ) -> Dict[str, Any]:
        """Calcule le remboursement selon les règles"""
        df = self.regles_pc_df.copy()
        
        # Construction du masque de filtrage
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
                "erreur": f"Aucune règle trouvée pour {formule}/{code_acte}",
                "montant_ht": montant,
                "taux": 0.0,
                "remb_final": 0.0,
                "reste": montant
            }
        
        # Calcul du remboursement
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

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Traite une facture PennyPet complète"""
        
        # Réinitialisation des stats
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0
        }
        
        try:
            # Extraction des données
            data, raw_content = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
            
            resultats: List[Dict[str, Any]] = []
            accidents = {"accident", "urgent", "urgence", "fract", "trauma", "traumatisme"}
            
            # Traitement de chaque ligne
            for ligne in data["lignes"]:
                try:
                    # Extraction des données de ligne
                    libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
                    montant = float(ligne.get("montant_ht", 0) or 0)
                    
                    # Normalisation du code acte
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
                    
                    # Ajout des résultats
                    ligne.update({
                        "code_norm": code_norm,
                        "est_accident": est_acc,
                        **remb
                    })
                    resultats.append(ligne)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    logger.error(f"Erreur traitement ligne {ligne}: {e}")
                    continue
            
            # Calcul des totaux
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
        """Retourne les statistiques complètes du processeur"""
        return {
            **self.stats,
            **self.normaliseur.get_mapping_stats()
        }

# Instance globale pour usage direct
pennypet_processor = PennyPetProcessor()
