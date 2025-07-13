import json
import re
from typing import Dict, List, Any, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available. Fuzzy matching will be disabled.")

def pseudojson_to_json(text: str) -> str:
    """
    Correction minimale pour JSON mal formé.
    """
    text = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
    text = text.replace("'", '"')
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

class NormaliseurAMV:
    """
    Normaliseur des actes et médicaments :
      1. Exact / regex patterns
      2. Fallback sémantique (glossaires actes & médicaments)
      3. Fuzzy matching (en dernier)
    """
    def __init__(self, config: PennyPetConfig):
        self.config = config
        # Glossaire actes (mots-clés en minuscules)
        self.termes_actes = set(config.actes_df["field_label"].str.lower())
        self.actes_df = config.actes_df.dropna(subset=["pattern"])
        # Glossaire médicaments
        self.termes_medicaments = config.glossaire_pharmaceutique
        self.medicaments_df = config.medicaments_df
        self.mapping_amv = config.mapping_amv
        self.cache: Dict[str, Optional[str]] = {}

    def normalise_acte(self, libelle_brut: str) -> Optional[str]:
        if not libelle_brut:
            return None
        cle = libelle_brut.upper().strip()
        low = libelle_brut.lower().strip()
        if cle in self.cache:
            return self.cache[cle]
        # 1. Pattern CSV
        for _, row in self.actes_df.iterrows():
            if row["pattern"].search(cle):
                self.cache[cle] = row["code_acte"]
                return row["code_acte"]
        # 2. Fallback sémantique actes
        for t in self.termes_actes:
            if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", low):
                code = t.upper()
                self.cache[cle] = code
                return code
        # 3. Fuzzy matching
        if RAPIDFUZZ_AVAILABLE:
            codes = self.actes_df["code_acte"].dropna().tolist()
            match, score, _ = process.extractOne(cle, codes, scorer=fuzz.token_sort_ratio)
            if score >= 80:
                self.cache[cle] = match
                return match
        self.cache[cle] = None
        return None

    def normalise_medicament(self, libelle_brut: str) -> Optional[str]:
        if not libelle_brut:
            return None
        cle = libelle_brut.upper().strip()
        low = libelle_brut.lower().strip()
        if cle in self.cache:
            return self.cache[cle]
        # 1. Fallback sémantique médicaments (glossaire pharma)
        for t in self.termes_medicaments:
            if re.search(rf"(\d+\s*)?(?<!\w){re.escape(t)}s?(?!\w)", low):
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
        # 2. Recherche exacte dans la base de médicaments
        meds = [m.lower() for m in self.medicaments_df["medicament"].dropna()]
        if low in meds:
            self.cache[cle] = "MEDICAMENTS"
            return "MEDICAMENTS"
        # 3. Fuzzy matching
        if RAPIDFUZZ_AVAILABLE:
            choices = [m.upper() for m in meds]
            match, score, _ = process.extractOne(cle, choices, scorer=fuzz.token_set_ratio)
            if score >= 85:
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
        self.cache[cle] = None
        return None

    def normalise(self, libelle_brut: str) -> Optional[str]:
        return (
            self.normalise_acte(libelle_brut)
            or self.normalise_medicament(libelle_brut)
            or libelle_brut.strip().upper()
        )

    def get_mapping_stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self.cache),
            "actes": len(self.termes_actes),
            "medicaments": len(self.termes_medicaments),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE
        }

class PennyPetProcessor:
    """
    Pipeline extraction LLM, normalisation, calcul remboursement PennyPet.
    """
    def __init__(
        self,
        client_qwen: OpenRouterClient = None,
        client_mistral: OpenRouterClient = None,
        config: PennyPetConfig = None,
    ):
        self.config = config or PennyPetConfig()
        self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
        self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
        self.regles_pc_df = self.config.regles_pc_df
        self.normaliseur = NormaliseurAMV(self.config)

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        client = self.client_qwen if llm_provider.lower() == "qwen" else self.client_mistral
        resp = client.analyze_invoice_image(image_bytes, formule)
        content = resp.choices[0].message.content
        start = content.find("{")
        if start < 0:
            raise ValueError("JSON non trouvé dans la réponse LLM.")
        depth = 0
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = content[start : i + 1]
                    break
        json_str = pseudojson_to_json(json_str)
        data = json.loads(json_str)
        if "lignes" not in data:
            raise ValueError("Le LLM n'a pas extrait de lignes exploitables.")
        return data, content

    def calculer_remboursement(
        self, montant: float, code_acte: str, formule: str, est_accident: bool
    ) -> Dict[str, Any]:
        df = self.regles_pc_df.copy()
        mask = (
            (df["formule"] == formule)
            & (
                df["code_acte"].eq(code_acte)
                | (
                    df["code_acte"].fillna("ALL").eq("ALL")
                    & df["actes_couverts"].apply(lambda l: code_acte in l)
                )
            )
            & (
                (df["type_couverture"] == "ACCIDENT_MALADIE")
                | ((df["type_couverture"] == "ACCIDENT_SEULEMENT") & est_accident)
            )
        )
        reg = df[mask]
        if reg.empty:
            return {"erreur": f"{formule}/{code_acte}", "reste": montant}
        r = reg.iloc[0]
        taux, plafond = r["taux_remboursement"] / 100, r["plafond_annuel"]
        brut = montant * taux
        final = min(brut, plafond)
        return {
            "montant_ht": montant,
            "taux": taux * 100,
            "remb_final": final,
            "reste": montant - final
        }

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        data, _ = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
        resultats: List[Dict[str, Any]] = []
        accidents = {"accident", "urgent", "urgence", "fract", "trauma"}
        for ligne in data["lignes"]:
            libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
            code_norm = self.normaliseur.normalise(libelle)
            montant = float(ligne.get("montant_ht", 0) or 0)
            est_acc = any(m in libelle.lower() for m in accidents)
            remb = self.calculer_remboursement(montant, code_norm, formule_client, est_acc)
            ligne.update({"code_norm": code_norm, **remb})
            resultats.append(ligne)
        total_remb = sum(r["remb_final"] for r in resultats)
        return {"lignes": resultats, "total_remb": total_remb}

# Instance globale pour usage direct
pennypet_processor = PennyPetProcessor()
