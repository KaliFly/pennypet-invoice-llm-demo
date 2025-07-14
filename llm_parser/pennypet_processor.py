# pennypet_processor.py
import json, re, logging, unicodedata, pandas as pd
from typing import Dict, List, Any
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pennypet_debug.log")]
)
log = logging.getLogger("PennyPet")

try:
    from rapidfuzz import process, fuzz
    RAPID = True
except ImportError:
    RAPID = False


# ───────────────────  UTILITAIRES GÉNÉRIQUES  ───────────────────
def _strip_accents(txt: str) -> str:
    if not txt:
        return ""
    txt = unicodedata.normalize("NFD", str(txt))
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", txt).lower().strip()


# ───────────────────  PARSING JSON ULTRA-ROBUSTE  ───────────────────
def _basic_clean(j: str) -> str:
    """Nettoyage de base : clés non quotées, guillemets simples, virgules avant }}"""
    j = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', j)
    j = j.replace("'", '"')
    j = re.sub(r',\s*([}\]])', r'\1', j)
    return j

def _autofix_json(text: str, max_iter: int = 10) -> str:
    """Boucle : json.loads → si erreur, ajoute la virgule manquante à la position indiquée."""
    for _ in range(max_iter):
        try:
            json.loads(text)
            return text                       # ✅ valide
        except json.JSONDecodeError as e:
            pos = e.pos
            # si la virgule manque avant un guillemet ouvrant OU une accolade/coilhet qui suit directement une valeur
            if 0 < pos < len(text) - 1 and text[pos] in '"{' and text[pos-1] not in '{[,"':
                text = text[:pos] + ',' + text[pos:]
            else:
                break                          # on ne sait pas réparer
    return text

def parse_llm_json(raw: str) -> dict:
    """
    1) isole le bloc { ... } le plus long ;
    2) nettoyage de base ;
    3) boucle d’auto-réparation _autofix_json ;
    4) si échec ⇒ reconstruction regex minimale.
    """
    start, end = raw.find('{'), raw.rfind('}') + 1
    if start == -1 or end == 0:
        log.error("Aucun bloc JSON trouvé.")
        return _json_rebuild(raw)

    candidate = raw[start:end]
    candidate = re.sub(r'[^\x20-\x7E\n]', '', candidate)   # supprime caractères illisibles
    candidate = _basic_clean(candidate)
    candidate = _autofix_json(candidate)

    try:
        return json.loads(candidate)
    except Exception as e:
        log.warning(f"Parsing JSON encore en échec : {e}")
        return _json_rebuild(raw)


# ─────────────  Fallback minimal si parsing impossible  ─────────────
def _json_rebuild(txt: str) -> dict:
    pat = r'"code_acte"\s*:\s*"([^"]+)".*?"montant_ht"\s*:\s*([\d.]+)'
    lignes = [{"code_acte": m, "description": m, "montant_ht": float(p)}
              for m, p in re.findall(pat, txt, re.S)]
    if not lignes:
        lignes = [{"code_acte": "ERREUR_JSON", "description": "Parsing impossible", "montant_ht": 0.0}]
    return {
        "lignes": lignes,
        "montant_total": sum(l["montant_ht"] for l in lignes),
        "informations_client": {}
    }


# ─────────────────────  NORMALISEUR MINIMAL  ─────────────────────
class SimpleNormalizer:
    def __init__(self, cfg: PennyPetConfig):
        self.cache: Dict[str, str] = {}
        self.meds = {_strip_accents(t) for t in cfg.glossaire_pharmaceutique}
        actes_df = getattr(cfg, 'actes_df', pd.DataFrame())
        self.actes = {_strip_accents(t) for t in actes_df.get('field_label', [])}
        self.med_rx = re.compile(r'\b(\d+\s*(mg|ml|g|l|ui)|vaccin|inj|comprim|gélule)\b', re.I)
        self.acte_rx = re.compile(r'\b(consult|examen|chirurg|radio|analyse)\b', re.I)

    def norm(self, label: str) -> str:
        if not label:
            return "INDÉTERMINÉ"
        key = label.upper().strip()
        if key in self.cache:
            return self.cache[key]

        label_norm = _strip_accents(label)
        if self.med_rx.search(label_norm) or label_norm in self.meds:
            res = "MEDICAMENTS"
        elif self.acte_rx.search(label_norm) or label_norm in self.actes:
            res = "ACTES"
        elif RAPID and self.meds:
            match, sc, _ = process.extractOne(label_norm, list(self.meds), scorer=fuzz.partial_ratio)
            res = "MEDICAMENTS" if sc > 85 else key
        else:
            res = key

        self.cache[key] = res
        return res


# ────────────────────────  PROCESSOR  ────────────────────────
class PennyPetProcessor:
    def __init__(self, cfg: PennyPetConfig = None):
        self.cfg = cfg or PennyPetConfig()
        self.norm = SimpleNormalizer(self.cfg)
        self.client = OpenRouterClient(model_key="primary")

    def _rembourse(self, montant: float, formule: str, accident: bool) -> float:
        if formule == "START":          return 0
        if formule == "PREMIUM":        return min(montant, 500) if accident else 0
        if formule == "INTEGRAL":       return min(montant * .5, 1000)
        if formule == "INTEGRAL_PLUS":  return min(montant, 1000)
        return 0

    def process(self, img: bytes, formule: str = "INTEGRAL") -> Dict[str, Any]:
        raw = self.client.analyze_invoice_image(img, formule).choices[0].message.content
        data = parse_llm_json(raw)

        lignes, tf, tr = [], 0.0, 0.0
        accident_kw = {"accident", "urgent", "urgence", "trauma"}

        for l in data.get("lignes", []):
            lib = l.get("code_acte") or l.get("description", "")
            mt = float(l.get("montant_ht", 0) or 0)
            if mt <= 0: continue
            code = self.norm.norm(lib)
            is_acc = any(k in lib.lower() for k in accident_kw)
            remb = self._rembourse(mt, formule, is_acc)

            lignes.append({
                "libelle": lib,
                "montant_ht": mt,
                "code_norm": code,
                "est_accident": is_acc,
                "rembourse": remb,
                "reste": mt - remb
            })
            tf += mt
            tr += remb

        return {
            "success": True,
            "lignes": lignes,
            "total_facture": tf,
            "total_rembourse": tr,
            "reste_a_charge": tf - tr,
            "informations_client": data.get("informations_client", {}),
            "raw_llm_preview": raw[:400] + "…" if len(raw) > 400 else raw
        }


# ----------------- instance prête à l’emploi -----------------
pennypet_processor = PennyPetProcessor()
