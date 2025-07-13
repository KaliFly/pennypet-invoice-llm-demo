import json
import re
from typing import Dict, List, Any, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient

# Ajout pour la normalisation avec fuzzy matching
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available. Fuzzy matching will be disabled.")

class NormaliseurAMV:
    """
    Normaliseur pour mapper les libellés bruts LLM vers codes d'actes/médicaments
    standardisés, avec enrichissement sémantique ultra-robuste.
    """

    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.actes_df = config.actes_df.dropna(subset=["pattern"])
        self.medicaments_df = config.medicaments_df
        self.mapping_amv = config.mapping_amv
        self.cache: Dict[str, Optional[str]] = {}

        # Mots-clés pharmaceutiques – variantes singulier/pluriel/abréviations
        self.termes_medicaments_semantiques = [
            # Formes orales
            "comprimé", "comprime", "comprimés", "comprimee", "comprimes", "comprimess", "comp", "comp.", "cp",
            "cps", "pilule", "pilules", "dragée", "dragee", "dragées", "dragees",
            "cachet", "cachets", "gélule", "gelule", "gélules", "gelules",
            "tablette", "tablettes", "capsule", "capsules",
            # Unités & dosages
            "mg", "mg.", "g", "gr", "µg", "mcg", "ml", "ml.", "l", "iu", "ui",
            "dose", "doses", "unité", "unité(s)", "un.", "dose(s)",
            # Liquides & suspensions
            "sirop", "sirops", "goutte", "gouttes", "suspension", "suspensions",
            "solution", "solutions", "élixir", "elixir", "elixirs", "collyre",
            "collyres",
            # Injectables
            "injection", "injections", "injectable", "injectables", "perf",
            "perfusion", "perfusions", "iv", "im", "sc", "seringue",
            "seringues",
            # Topiques
            "crème", "creme", "crèmes", "cremes", "pommade", "pommades",
            "lotion", "lotions", "gel", "gels", "spray", "sprays", "patch",
            "patchs", "transdermique", "transdermal", "ointment",
            # Suppositoires et inserts
            "suppositoire", "suppositoires", "insert", "inserts", "pessaire",
            "pessaires",
            # Inhalation
            "inhalant", "inhalants", "aérosol", "aerosol", "aérosols",
            "aerosols", "nébuliseur", "nebuliseur", "nébuliseurs",
            "nebuliseurs", "spray nasal", "inhalateur", "inhalateurs",
            # Solides spéciaux
            "sachet", "sachets", "granule", "granules", "poudre", "poudres",
            # Abréviations usuelles
            "tbl", "cap", "inj", "soln", "susp", "drg", "un.", "comp.",
            "mg/ml"
        ]

    def normalise_acte(self, libelle_brut: str) -> Optional[str]:
        if not libelle_brut:
            return None

        libelle_clean = libelle_brut.upper().strip()

        if libelle_clean in self.cache:
            return self.cache[libelle_clean]

        for _, row in self.actes_df.iterrows():
            if row["pattern"].search(libelle_clean):
                self.cache[libelle_clean] = row["code_acte"]
                return row["code_acte"]

        if RAPIDFUZZ_AVAILABLE:
            codes_actes = self.actes_df["code_acte"].dropna().tolist()
            match, score, _ = process.extractOne(
                libelle_clean, codes_actes, scorer=fuzz.token_sort_ratio
            )
            if score >= 80:
                self.cache[libelle_clean] = match
                return match

        self.cache[libelle_clean] = None
        return None

    def normalise_medicament(self, libelle_brut: str) -> Optional[str]:
        if not libelle_brut:
            return None

        libelle_clean = libelle_brut.upper().strip()
        libelle_lower = libelle_brut.lower().strip()

        if libelle_clean in self.cache:
            return self.cache[libelle_clean]

        # 1. Fuzzy matching (si RapidFuzz dispo)
        if RAPIDFUZZ_AVAILABLE:
            medicaments = []
            for _, row in self.medicaments_df.iterrows():
                if "medicament" in row:
                    medicaments.append(row["medicament"])
                if isinstance(row.get("synonymes_ocr"), list):
                    medicaments.extend(row["synonymes_ocr"])

            if medicaments:
                match, score, _ = process.extractOne(
                    libelle_clean, medicaments, scorer=fuzz.token_set_ratio
                )
                if score >= 85:
                    code_amv = self.mapping_amv.get(match)
                    self.cache[libelle_clean] = code_amv or "MEDICAMENTS"
                    return code_amv or "MEDICAMENTS"

        # 2. Fallback sémantique (regex mot entier, tolère le s final)
        for t in self.termes_medicaments_semantiques:
            if re.search(rf"\b{re.escape(t)}s?\b", libelle_lower):
                self.cache[libelle_clean] = "MEDICAMENTS"
                return "MEDICAMENTS"

        # 3. Cas « 10mg / 5ml / 2comp » accolé
        if re.search(r"\d+\s?(mg|mg\.|ml|ml\.|comp|comp\.|cps|tbl|g|ui|iu)\b",
                     libelle_lower):
            self.cache[libelle_clean] = "MEDICAMENTS"
            return "MEDICAMENTS"

        self.cache[libelle_clean] = None
        return None

    def normalise(self, libelle_brut: str) -> Optional[str]:
        if not libelle_brut:
            return None
        return (
            self.normalise_acte(libelle_brut)
            or self.normalise_medicament(libelle_brut)
            or libelle_brut.upper().strip()
        )

    def get_mapping_stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self.cache),
            "actes_disponibles": len(self.actes_df),
            "medicaments_disponibles": len(self.medicaments_df),
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
        }


class PennyPetProcessor:
    """
    Pipeline complet : extraction LLM, normalisation, calcul remboursement.
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

    def calculer_remboursement_pennypet(
        self,
        montant: float,
        code_acte: str,
        formule: str,
        acte_est_accident: bool,
    ) -> Dict[str, Any]:
        if not formule or not code_acte:
            return {
                "erreur": "Formule ou code acte manquant",
                "montant_facture": montant,
                "code_acte": code_acte,
                "formule_utilisee": formule,
                "taux_applique": 0,
                "remboursement_brut": 0,
                "remboursement_final": 0,
                "reste_a_charge": montant,
                "plafond_formule": 0,
            }

        df = self.regles_pc_df.copy()

        mask = (
            (df["formule"] == formule)
            &
            (
                df["code_acte"].eq(code_acte)
                |
                (
                    df["code_acte"].fillna("ALL").eq("ALL")
                    & df["actes_couverts"].apply(
                        lambda lst: code_acte in lst
                        if isinstance(lst, list)
                        else False
                    )
                )
            )
            &
            (
                (df["type_couverture"] == "ACCIDENT_MALADIE")
                |
                (
                    (df["type_couverture"] == "ACCIDENT_SEULEMENT")
                    & acte_est_accident
                )
            )
        )

        regles = df[mask]

        if regles.empty:
            return {
                "erreur": f"Aucune règle trouvée pour {formule}/{code_acte}",
                "montant_facture": montant,
                "code_acte": code_acte,
                "formule_utilisee": formule,
                "taux_applique": 0,
                "remboursement_brut": 0,
                "remboursement_final": 0,
                "reste_a_charge": montant,
                "plafond_formule": 0,
            }

        reg = regles.iloc[0]
        taux = reg["taux_remboursement"] / 100
        plafond = reg["plafond_annuel"]

        brut = montant * taux
        final = min(brut, plafond)

        return {
            "montant_facture": montant,
            "code_acte": code_acte,
            "taux_applique": taux * 100,
            "remboursement_brut": brut,
            "remboursement_final": final,
            "reste_a_charge": montant - final,
            "plafond_formule": plafond,
            "formule_utilisee": formule,
        }

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        client = self.client_qwen if llm_provider.lower() == "qwen" else self.client_mistral
        response = client.analyze_invoice_image(image_bytes, formule)
        content = response.choices[0].message.content

        start = content.find("{")
        if start == -1:
            raise ValueError("JSON non trouvé dans la réponse LLM.")
        depth, end = 0, start
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        json_str = content[start : end + 1]
        data = json.loads(json_str)
        if "lignes" not in data:
            raise ValueError("Le LLM n'a pas extrait de lignes exploitables.")
        return data, content

    def process_facture_pennypet(
        self,
        file_bytes: bytes,
        formule_client: str,
        llm_provider: str = "qwen",
    ) -> Dict[str, Any]:
        extraction, raw_content = self.extract_lignes_from_image(
            file_bytes, formule_client, llm_provider
        )

        formule = formule_client or extraction.get("formule_utilisee", "INTEGRAL")
        lignes = extraction.get("lignes", [])
        remboursements: List[Dict[str, Any]] = []

        motifs_accident = {"accident", "urgent", "urgence", "fract", "trauma"}

        for ligne in lignes:
            try:
                montant = float(ligne.get("montant_ht", 0.0))
            except (ValueError, TypeError):
                montant = 0.0

            libelle_brut = (ligne.get("code_acte") or ligne.get("description", "")).strip()
            code_normalise = self.normaliseur.normalise(libelle_brut)

            libelle_lower = libelle_brut.lower()
            est_accident = any(m in libelle_lower for m in motifs_accident)

            ligne["libelle_original"] = libelle_brut
            ligne["code_normalise"] = code_normalise

            remb = self.calculer_remboursement_pennypet(
                montant, code_normalise, formule, est_accident
            )
            remboursements.append({**ligne, **remb})

        total_montant = sum(float(l.get("montant_ht", 0.0)) for l in lignes)
        total_rembourse = sum(float(r["remboursement_final"]) for r in remboursements)

        return {
            "extraction_facture": extraction,
            "remboursements": remboursements,
            "total_facture": total_montant,
            "total_remboursement": total_rembourse,
            "reste_total_a_charge": total_montant - total_rembourse,
            "formule_utilisee": formule,
            "infos_client": extraction.get("informations_client", {}),
            "texte_ocr": extraction.get("texte_ocr", ""),
            "llm_raw": raw_content,
            "mapping_stats": self.normaliseur.get_mapping_stats(),
        }

# Instance globale pour un usage direct
pennypet_processor = PennyPetProcessor()
