# pennypet_processor.py
import json, re, logging, unicodedata, pandas as pd
from typing import Dict, List, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO); log = logging.getLogger("PennyPet")

try:
    from rapidfuzz import process, fuzz
    RAPID = True
except ImportError:
    RAPID = False


# ──────────────────────────  UTILITAIRES  ──────────────────────────
def _strip_accents(txt: str) -> str:
    txt = unicodedata.normalize("NFD", txt or "")
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", txt).lower().strip()

def _quick_clean(j: str) -> str:
    j = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', j)          # clés sans guillemets
    j = j.replace("'", '"')
    j = re.sub(r',\s*([}\]])', r'\1', j)                                 # virgule avant } ]
    j = re.sub(r'(")\s+(")', r'\1, \2', j)                               # …"val"  "clé":
    j = re.sub(r'(\d|\])\s+(")', r'\1, \2', j)                           # …123  "clé":
    j = re.sub(r',,+', ',', j)
    return j

def _json_rebuild(txt: str) -> dict:
    """Fallback : recueille lignes + total via regex."""
    pat = r'"code_acte"\s*:\s*"([^"]*)".*?"montant_ht"\s*:\s*([\d.]+)'
    lignes = [{"code_acte": m[0],
               "description": m[0],
               "montant_ht": float(m[1])} for m in re.findall(pat, txt, re.S)]
    return {
        "lignes": lignes or [{"code_acte": "ERREUR_JSON",
                              "description": "Parsing impossible",
                              "montant_ht": 0.0}],
        "montant_total": sum(l["montant_ht"] for l in lignes),
        "informations_client": {}
    }

def parse_llm_json(raw: str) -> dict:
    """2 passes : clean → json, sinon rebuild."""
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        cleaned = _quick_clean(raw[start:end])
        return json.loads(cleaned)
    except Exception:
        return _json_rebuild(raw)


# ───────────────────────  NORMALISEUR MINIMAL  ───────────────────────
class SimpleNormalizer:
    def __init__(self, cfg: PennyPetConfig):
        self.cache: Dict[str, str] = {}
        self.meds = { _strip_accents(t) for t in cfg.glossaire_pharmaceutique }
        self.actes = { _strip_accents(t) for t in getattr(cfg, 'actes_df', pd.DataFrame()).get('field_label', []) }

        self.med_rx = re.compile(r'\b(\d+\s*(mg|ml|g|l|ui)|vaccin|inj|comprim|gélule)\b', re.I)
        self.acte_rx = re.compile(r'\b(consult|examen|chirurg|radio|analyse)\b', re.I)

    def norm(self, label: str) -> str:
        if not label:
            return "INDÉTERMINÉ"
        key = label.upper().strip()
        if key in self.cache:
            return self.cache[key]

        norm = _strip_accents(label)
        if self.med_rx.search(norm) or norm in self.meds:
            res = "MEDICAMENTS"
        elif self.acte_rx.search(norm) or norm in self.actes:
            res = "ACTES"
        elif RAPID and self.meds:
            match, sc, _ = process.extractOne(norm, list(self.meds), scorer=fuzz.partial_ratio)
            res = "MEDICAMENTS" if sc > 85 else key
        else:
            res = key

        self.cache[key] = res
        return res


# ───────────────────────────  PROCESSOR  ────────────────────────────
class PennyPetProcessor:
    def __init__(self, cfg: PennyPetConfig = None):
        self.cfg = cfg or PennyPetConfig()
        self.norm = SimpleNormalizer(self.cfg)
        self.client = OpenRouterClient(model_key="primary")

    # ------------- règles de remboursement basiques -------------
    def _rembourse(self, montant: float, formule: str, accident: bool) -> float:
        if formule == "START":                 return 0
        if formule == "PREMIUM":               return min(montant, 500) if accident else 0
        if formule == "INTEGRAL":              return min(montant * .5, 1000)
        if formule == "INTEGRAL_PLUS":         return min(montant, 1000)
        return 0

    # ------------------- pipeline principal ---------------------
    def process(self, img: bytes, formule: str = "INTEGRAL") -> Dict[str, Any]:
        raw = self.client.analyze_invoice_image(img, formule).choices[0].message.content
        data = parse_llm_json(raw)

        lignes_out: List[Dict[str, Any]] = []
        total_fact, total_remb = 0.0, 0.0
        accident_kw = {"accident", "urgent", "urgence", "trauma"}

        for l in data.get("lignes", []):
            lib = l.get("code_acte") or l.get("description", "")
            mt  = float(l.get("montant_ht", 0) or 0)
            if mt <= 0: continue

            code = self.norm.norm(lib)
            is_acc = any(k in lib.lower() for k in accident_kw)
            remb   = self._rembourse(mt, formule, is_acc)

            lignes_out.append({
                "libelle": lib,
                "montant_ht": mt,
                "code_norm": code,
                "est_accident": is_acc,
                "rembourse": remb,
                "reste": mt - remb
            })
            total_fact += mt
            total_remb += remb

        return {
            "success": True,
            "lignes": lignes_out,
            "total_facture": total_fact,
            "total_rembourse": total_remb,
            "reste_a_charge": total_fact - total_remb,
            "raw_llm": raw[:500] + "…"   # log résumé
        }


# ------------ instance utilisable directement ------------
pennypet_processor = PennyPetProcessor()
