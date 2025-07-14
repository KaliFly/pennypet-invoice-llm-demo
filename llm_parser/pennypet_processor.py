# pennypet_processor.py
import json, re, logging, unicodedata, pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient

# Configuration logging améliorée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('pennypet_debug.log')]
)
log = logging.getLogger("PennyPet")

try:
    from rapidfuzz import process, fuzz
    RAPID = True
    log.info("RapidFuzz activé")
except ImportError:
    RAPID = False
    log.warning("RapidFuzz non disponible")


# ──────────────────────────  UTILITAIRES AMÉLIORÉS  ──────────────────────────
def _strip_accents(txt: str) -> str:
    """Normalisation robuste des accents et caractères spéciaux"""
    if not txt:
        return ""
    txt = unicodedata.normalize("NFD", str(txt))
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    txt = re.sub(r"[^\w\s]", " ", txt).lower().strip()
    return " ".join(txt.split())  # Normalise les espaces multiples

def _ultra_clean_json(j: str) -> str:
    """Nettoyage JSON ultra-robuste avec gestion étendue des erreurs"""
    try:
        # 1. Nettoyage initial
        j = j.strip()
        
        # 2. Suppression des caractères non-ASCII problématiques
        j = re.sub(r'[^\x20-\x7E\n]', '', j)
        
        # 3. Correction des clés non quotées
        j = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', j)
        
        # 4. Remplacement guillemets simples → doubles
        j = j.replace("'", '"')
        
        # 5. Suppression virgules avant fermantes
        j = re.sub(r',\s*([}\]])', r'\1', j)
        
        # 6. NOUVEAU: Gestion des virgules manquantes après valeurs
        j = re.sub(r'(".*?")\s*\n\s*(".*?")', r'\1,\n\2', j)  # "val"\n"clé"
        j = re.sub(r'(\d+\.?\d*)\s*\n\s*(".*?")', r'\1,\n\2', j)  # 123\n"clé"
        j = re.sub(r'(".*?")\s*\n\s*(\{)', r'\1,\n\2', j)  # "val"\n{
        j = re.sub(r'(\})\s*\n\s*(\{)', r'\1,\n\2', j)  # }\n{
        j = re.sub(r'(\])\s*\n\s*(".*?")', r'\1,\n\2', j)  # ]\n"clé"
        
        # 7. Correction des espaces dans les valeurs numériques
        j = re.sub(r':\s*(\d+)\s+(\d+)', r': \1\2', j)  # "val": 12 34 → "val": 1234
        
        # 8. Suppression doubles virgules
        j = re.sub(r',,+', ',', j)
        
        # 9. Correction deux points multiples
        j = re.sub(r'::+', ':', j)
        
        # 10. NOUVEAU: Gestion des objets/arrays mal fermés
        # Compte les { } et [ ] pour équilibrer
        open_braces = j.count('{') - j.count('}')
        open_brackets = j.count('[') - j.count(']')
        
        if open_braces > 0:
            j += '}' * open_braces
        if open_brackets > 0:
            j += ']' * open_brackets
            
        return j
        
    except Exception as e:
        log.error(f"Erreur nettoyage JSON: {e}")
        return j

def _advanced_json_rebuild(txt: str) -> dict:
    """Reconstruction JSON avancée avec patterns multiples pour factures vétérinaires"""
    result = {
        "lignes": [],
        "montant_total": 0.0,
        "informations_client": {}
    }
    
    try:
        # Pattern 1: Structure complète classique
        pattern1 = r'"code_acte"\s*:\s*"([^"]*)".*?"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([\d.,]+)'
        matches1 = re.findall(pattern1, txt, re.DOTALL | re.IGNORECASE)
        
        # Pattern 2: Structure simplifiée
        pattern2 = r'"description"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([\d.,]+)'
        matches2 = re.findall(pattern2, txt, re.DOTALL | re.IGNORECASE)
        
        # Pattern 3: Structure alternative avec libellé
        pattern3 = r'"libelle"\s*:\s*"([^"]*)".*?"montant"\s*:\s*([\d.,]+)'
        matches3 = re.findall(pattern3, txt, re.DOTALL | re.IGNORECASE)
        
        # Pattern 4: Structure encore plus libre
        pattern4 = r'(consultation|medicament|vaccin|examen|chirurgie|traitement)[^"]*"([^"]*)"[^0-9]*([\d.,]+)'
        matches4 = re.findall(pattern4, txt, re.DOTALL | re.IGNORECASE)
        
        # Traitement des matches par priorité
        if matches1:
            for m in matches1:
                montant = float(re.sub(r'[^\d.]', '', m[2]))
                result["lignes"].append({
                    "code_acte": m[0].strip(),
                    "description": m[1].strip(),
                    "montant_ht": montant
                })
        elif matches2:
            for m in matches2:
                montant = float(re.sub(r'[^\d.]', '', m[1]))
                result["lignes"].append({
                    "code_acte": m[0].strip(),
                    "description": m[0].strip(),
                    "montant_ht": montant
                })
        elif matches3:
            for m in matches3:
                montant = float(re.sub(r'[^\d.]', '', m[1]))
                result["lignes"].append({
                    "code_acte": m[0].strip(),
                    "description": m[0].strip(),
                    "montant_ht": montant
                })
        elif matches4:
            for m in matches4:
                montant = float(re.sub(r'[^\d.]', '', m[2]))
                result["lignes"].append({
                    "code_acte": f"{m[0]} - {m[1]}".strip(),
                    "description": m[1].strip(),
                    "montant_ht": montant
                })
        
        # Fallback: Extraction de base
        if not result["lignes"]:
            # Recherche de montants isolés
            montants = re.findall(r'[\d.,]+\s*€?', txt)
            descriptions = re.findall(r'"([^"]{5,50})"', txt)  # Textes entre guillemets
            
            for i, desc in enumerate(descriptions[:len(montants)]):
                try:
                    montant = float(re.sub(r'[^\d.]', '', montants[i]))
                    if montant > 0:
                        result["lignes"].append({
                            "code_acte": desc,
                            "description": desc,
                            "montant_ht": montant
                        })
                except (ValueError, IndexError):
                    continue
        
        # Calcul du montant total
        if result["lignes"]:
            result["montant_total"] = sum(l["montant_ht"] for l in result["lignes"])
        else:
            # Structure d'erreur minimale
            result["lignes"] = [{
                "code_acte": "ERREUR_PARSING",
                "description": "Impossible d'extraire les données de la facture",
                "montant_ht": 0.0
            }]
        
        # Extraction informations client avec patterns multiples
        client_patterns = {
            "nom_proprietaire": [
                r'(?:proprietaire|client|nom)\s*[":]\s*"?([^"{\n,]+)"?',
                r'"nom_proprietaire"\s*:\s*"([^"]*)"'
            ],
            "nom_animal": [
                r'(?:animal|patient|nom_animal)\s*[":]\s*"?([^"{\n,]+)"?',
                r'"nom_animal"\s*:\s*"([^"]*)"'
            ],
            "identification": [
                r'(?:identification|id|numero|tatouage|puce)\s*[":]\s*"?([^"{\n,]+)"?',
                r'"identification"\s*:\s*"([^"]*)"'
            ]
        }
        
        for key, patterns in client_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, txt, re.IGNORECASE)
                if match:
                    result["informations_client"][key] = match.group(1).strip()
                    break
        
        log.info(f"JSON reconstruit: {len(result['lignes'])} lignes, total: {result['montant_total']}€")
        return result
        
    except Exception as e:
        log.error(f"Erreur reconstruction JSON: {e}")
        return {
            "lignes": [{"code_acte": "ERREUR_CRITIQUE", "description": "Échec total du parsing", "montant_ht": 0.0}],
            "montant_total": 0.0,
            "informations_client": {}
        }

def parse_llm_json(raw: str) -> dict:
    """Parser JSON multi-niveaux avec fallbacks robustes"""
    log.info(f"Parsing JSON - taille: {len(raw)} caractères")
    
    # Méthode 1: Parsing standard avec nettoyage
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        
        if start == -1 or end == 0:
            raise ValueError("Pas de structure JSON détectée")
        
        json_part = raw[start:end]
        cleaned = _ultra_clean_json(json_part)
        
        # Log pour debug
        log.debug(f"JSON nettoyé (100 premiers chars): {cleaned[:100]}")
        
        data = json.loads(cleaned)
        log.info("✅ Parsing JSON standard réussi")
        return data
        
    except json.JSONDecodeError as e:
        log.warning(f"Parsing JSON standard échoué: {e}")
        log.debug(f"JSON problématique (autour erreur): {cleaned[max(0, e.pos-50):e.pos+50] if 'cleaned' in locals() else 'N/A'}")
        
    except Exception as e:
        log.warning(f"Erreur parsing JSON: {e}")
    
    # Méthode 2: Reconstruction avancée
    try:
        data = _advanced_json_rebuild(raw)
        log.info("✅ Reconstruction JSON avancée réussie")
        return data
    except Exception as e:
        log.error(f"Reconstruction JSON échouée: {e}")
        
    # Méthode 3: Structure minimale d'urgence
    log.error("❌ Tous les parsings ont échoué, structure d'urgence")
    return {
        "lignes": [{"code_acte": "ECHEC_PARSING", "description": "Erreur parsing JSON", "montant_ht": 0.0}],
        "montant_total": 0.0,
        "informations_client": {}
    }


# ───────────────────────  NORMALISEUR AMÉLIORÉ  ───────────────────────
class SimpleNormalizer:
    def __init__(self, cfg: PennyPetConfig):
        self.cache: Dict[str, str] = {}
        
        # Chargement sécurisé du glossaire pharmaceutique
        try:
            self.meds = {_strip_accents(t) for t in cfg.glossaire_pharmaceutique if t}
            log.info(f"Glossaire médicaments chargé: {len(self.meds)} termes")
        except Exception as e:
            log.error(f"Erreur chargement glossaire médicaments: {e}")
            self.meds = set()
        
        # Chargement sécurisé des actes
        try:
            actes_df = getattr(cfg, 'actes_df', pd.DataFrame())
            if not actes_df.empty and 'field_label' in actes_df.columns:
                self.actes = {_strip_accents(t) for t in actes_df['field_label'].dropna() if t}
            else:
                self.actes = set()
            log.info(f"Termes d'actes chargés: {len(self.actes)} termes")
        except Exception as e:
            log.error(f"Erreur chargement actes: {e}")
            self.actes = set()

        # Patterns regex améliorés
        self.med_patterns = [
            r'\b\d+\s*(mg|ml|g|l|ui|iu|mcg|µg)\b',
            r'\b(vaccin|injection|comprimé|gélule|solution|pommade|spray)\b',
            r'\b(antibiotic|anti-inflammatoire|antiparasitaire|vermifuge)\b',
            r'\b(medicament|médicament|traitement|produit)\b'
        ]
        
        self.acte_patterns = [
            r'\b(consultation|examen|visite|contrôle)\b',
            r'\b(chirurgie|opération|intervention|anesthésie)\b',
            r'\b(radio|échographie|scanner|analyse)\b',
            r'\b(hospitalisation|soin|pansement)\b'
        ]
        
        self.med_rx = re.compile('|'.join(self.med_patterns), re.I)
        self.acte_rx = re.compile('|'.join(self.acte_patterns), re.I)

    def norm(self, label: str) -> str:
        """Normalisation robuste avec cache et fallbacks multiples"""
        if not label or not label.strip():
            return "INDÉTERMINÉ"
            
        key = str(label).upper().strip()
        if key in self.cache:
            return self.cache[key]

        norm_label = _strip_accents(label)
        
        # 1. Détection par patterns regex
        if self.med_rx.search(norm_label):
            result = "MEDICAMENTS"
        elif self.acte_rx.search(norm_label):
            result = "ACTES"
        # 2. Recherche dans glossaires
        elif norm_label in self.meds:
            result = "MEDICAMENTS"
        elif norm_label in self.actes:
            result = "ACTES"
        # 3. Recherche partielle dans glossaires
        elif any(med in norm_label or norm_label in med for med in self.meds):
            result = "MEDICAMENTS"
        elif any(acte in norm_label or norm_label in acte for acte in self.actes):
            result = "ACTES"
        # 4. Fuzzy matching si disponible
        elif RAPID and self.meds:
            try:
                match, score, _ = process.extractOne(norm_label, list(self.meds), scorer=fuzz.partial_ratio)
                if score > 85:
                    result = "MEDICAMENTS"
                elif self.actes:
                    match_acte, score_acte, _ = process.extractOne(norm_label, list(self.actes), scorer=fuzz.partial_ratio)
                    result = "ACTES" if score_acte > 80 else key
                else:
                    result = key
            except Exception as e:
                log.debug(f"Erreur fuzzy matching: {e}")
                result = key
        else:
            result = key

        self.cache[key] = result
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du normaliseur"""
        return {
            "cache_size": len(self.cache),
            "medicaments_count": len(self.meds),
            "actes_count": len(self.actes),
            "rapidfuzz_available": RAPID
        }


# ───────────────────────────  PROCESSOR PRINCIPAL  ────────────────────────────
class PennyPetProcessor:
    def __init__(self, cfg: PennyPetConfig = None):
        try:
            self.cfg = cfg or PennyPetConfig()
            self.norm = SimpleNormalizer(self.cfg)
            
            # Initialisation sécurisée du client
            try:
                self.client = OpenRouterClient(model_key="primary")
                log.info("Client OpenRouter initialisé (primary)")
            except Exception as e:
                log.error(f"Erreur initialisation client: {e}")
                self.client = None
                
            log.info("PennyPetProcessor initialisé avec succès")
        except Exception as e:
            log.error(f"Erreur initialisation PennyPetProcessor: {e}")
            raise

    def _rembourse(self, montant: float, formule: str, accident: bool) -> float:
        """Calcul remboursement selon règles PennyPet exactes"""
        try:
            montant = float(montant)
            if montant <= 0:
                return 0.0
                
            if formule == "START":
                return 0.0
            elif formule == "PREMIUM":
                return min(montant, 500.0) if accident else 0.0
            elif formule == "INTEGRAL":
                return min(montant * 0.5, 1000.0)
            elif formule == "INTEGRAL_PLUS":
                return min(montant, 1000.0)
            else:
                log.warning(f"Formule inconnue: {formule}")
                return 0.0
        except (ValueError, TypeError) as e:
            log.error(f"Erreur calcul remboursement: {e}")
            return 0.0

    def process(self, img: bytes, formule: str = "INTEGRAL") -> Dict[str, Any]:
        """Pipeline principal de traitement avec gestion d'erreurs complète"""
        if not self.client:
            return {"success": False, "error": "Client OpenRouter non initialisé"}
        
        if not img:
            return {"success": False, "error": "Aucune image fournie"}
            
        try:
            log.info(f"Début traitement - Formule: {formule}, Taille image: {len(img)} bytes")
            
            # Appel LLM
            response = self.client.analyze_invoice_image(img, formule)
            raw = response.choices[0].message.content
            
            if not raw:
                return {"success": False, "error": "Réponse vide du LLM"}
            
            log.info(f"Réponse LLM reçue - Taille: {len(raw)} caractères")
            
            # Parsing JSON robuste
            data = parse_llm_json(raw)
            
            if not data.get("lignes"):
                return {"success": False, "error": "Aucune ligne extraite"}

            # Traitement des lignes
            lignes_out: List[Dict[str, Any]] = []
            total_fact, total_remb = 0.0, 0.0
            accident_kw = {"accident", "urgent", "urgence", "trauma", "fracture", "chute"}

            for i, l in enumerate(data.get("lignes", [])):
                try:
                    lib = l.get("code_acte") or l.get("description", "") or f"Ligne {i+1}"
                    mt = float(l.get("montant_ht", 0) or 0)
                    
                    if mt <= 0:
                        log.warning(f"Montant invalide ligne {i+1}: {mt}")
                        continue

                    code = self.norm.norm(lib)
                    is_acc = any(k in lib.lower() for k in accident_kw)
                    remb = self._rembourse(mt, formule, is_acc)

                    lignes_out.append({
                        "libelle": lib,
                        "montant_ht": round(mt, 2),
                        "code_norm": code,
                        "est_accident": is_acc,
                        "est_medicament": code == "MEDICAMENTS",
                        "rembourse": round(remb, 2),
                        "reste": round(mt - remb, 2)
                    })
                    
                    total_fact += mt
                    total_remb += remb
                    
                except Exception as e:
                    log.error(f"Erreur traitement ligne {i+1}: {e}")
                    continue

            # Statistiques
            stats = {
                "lignes_traitees": len(lignes_out),
                "medicaments_detectes": sum(1 for l in lignes_out if l["est_medicament"]),
                "actes_detectes": sum(1 for l in lignes_out if not l["est_medicament"]),
                "normalizer_stats": self.norm.get_stats()
            }

            result = {
                "success": True,
                "lignes": lignes_out,
                "total_facture": round(total_fact, 2),
                "total_rembourse": round(total_remb, 2),
                "reste_a_charge": round(total_fact - total_remb, 2),
                "informations_client": data.get("informations_client", {}),
                "stats": stats,
                "raw_llm_preview": raw[:500] + "..." if len(raw) > 500 else raw
            }
            
            log.info(f"✅ Traitement réussi - {len(lignes_out)} lignes, {total_fact:.2f}€ total")
            return result
            
        except Exception as e:
            log.error(f"❌ Erreur dans le pipeline principal: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": {"lignes_traitees": 0, "medicaments_detectes": 0, "actes_detectes": 0}
            }


# ------------ Instance globale utilisable directement ------------
pennypet_processor = PennyPetProcessor()
