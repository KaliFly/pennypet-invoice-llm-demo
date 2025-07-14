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
    Rend un texte pseudo-JSON parse-able par json.loads().
    → 1. Nettoyage standard
    → 2. Détection d'une virgule manquante juste avant un guillemet ouvrant
    → 3. Validation finale : on boucle tant que json.loads échoue et qu'on
        réussit à corriger un nouveau défaut.
    """
    if not text:
        return "{}"

    # ---------- 1. Nettoyage standard ----------
    def _clean_once(t: str) -> str:
        t = t.strip()

        # Propriétés non quotées
        t = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', t)
        # Guillemets simples
        t = t.replace("'", '"')
        # Doubles virgules
        t = re.sub(r',\s*,', ',', t)
        # Virgule juste avant } ou ]
        t = re.sub(r',\s*([}\]])', r'\1', t)
        # Collage d'objets ou de listes
        t = re.sub(r'}\s*{', '},{', t)
        t = re.sub(r']\s*\[', '],[', t)
        # Nombres mal formés « 1 . 23 »
        t = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', t)
        # Caractères non imprimables
        t = re.sub(r'[^\x20-\x7E\n]', '', t)
        return t

    text = _clean_once(text)

    # ---------- 2. Virgule manquante avant un guillemet ----------
    # Cas le plus fréquent causant l'erreur ligne 2 col 1093 :
    # … "key":"value"  "next_key":...
    missing_comma = re.compile(r'(":[^,{}\[\]]+?"\s+")')
    text = missing_comma.sub(lambda m: m.group(0).replace('" ', '", '), text)

    # ---------- 3. Boucle de validation / correction ----------
    # On tente json.loads ; si ça rate on insère la virgule manquante
    # située juste avant la position de l'exception.
    max_iter, done = 10, False
    while max_iter and not done:
        try:
            json.loads(text)          # ✅ prêt à parser
            done = True
        except json.JSONDecodeError as e:
            pos = e.pos
            # Recherche du prochain guillemet ouvrant ; si la
            # position précédente n'est pas une virgule ou { ou [,
            # on injecte la virgule salvatrice.
            if pos < len(text) and pos > 0 and text[pos-1] not in '{[,"':
                text = text[:pos] + ',' + text[pos:]
                logger.debug(f"Virgule ajoutée à la position {pos}")
            else:
                # Plus de correctif réalisable → on abandonne la boucle
                logger.warning(f"Impossible de corriger JSON à la position {pos}")
                break
        max_iter -= 1

    return text

def extraire_json_robuste(content: str) -> dict:
    """
    Extraction JSON avec micro-validateur et fallback robuste.
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
        
        # Nettoyage avec la fonction améliorée
        json_clean = pseudojson_to_json_ameliore(json_str)
        data = json.loads(json_clean)
        
        # Micro-validateur : vérifier que lignes est bien une liste
        if not isinstance(data.get("lignes"), list):
            raise json.JSONDecodeError("Champ 'lignes' non-liste", json_clean, 0)
        
        logger.info("JSON parsé avec succès (méthode 1)")
        return data
        
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
            r'total.*?:\s*([0-9.]+)'
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

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """Extrait les lignes d'une image avec gestion d'erreurs JSON robuste"""
        try:
            # Sélection du client
            if llm_provider.lower() == "qwen" and self.client_qwen:
                client = self.client_qwen
            elif llm_provider.lower() == "mistral" and self.client_mistral:
                client = self.client_mistral
            else:
                raise ValueError(f"Client {llm_provider} non disponible")
            
            resp = client.analyze_invoice_image(image_bytes, formule)
            content = resp.choices[0].message.content
            
            if not content:
                raise ValueError("Réponse vide du LLM")
            
            # Log pour debug
            logger.info(f"Réponse LLM reçue - Longueur: {len(content)} caractères")
            logger.debug(f"Début de réponse: {content[:200]}...")
            logger.debug(f"Fin de réponse: ...{content[-200:]}")
            
            # Extraction JSON robuste avec correctif ciblé
            data = extraire_json_robuste(content)
            
            if "lignes" not in data:
                raise ValueError("Le LLM n'a pas extrait de lignes exploitables.")
            
            # Nettoyage des données
            for ligne in data["lignes"]:
                # S'assurer que montant_ht est un nombre
                try:
                    ligne["montant_ht"] = float(ligne.get("montant_ht", 0))
                except (ValueError, TypeError):
                    ligne["montant_ht"] = 0.0
                
                # S'assurer que les chaînes sont propres
                for key in ["code_acte", "description"]:
                    if key in ligne:
                        ligne[key] = str(ligne[key]).strip()
            
            return data, content
            
        except Exception as e:
            logger.error(f"Erreur dans extract_lignes_from_image: {e}")
            raise

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
