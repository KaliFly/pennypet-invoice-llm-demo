import json
import re
from typing import Dict, List, Any, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient

class PennyPetProcessor:
    def __init__(self, client_qwen=None, client_mistral=None, config=None):
        self.config = config or PennyPetConfig()
        self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
        self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
        self.regles_pc_df = self.config.regles_pc_df

    def calculer_remboursement_pennypet(self, montant: float, code_acte: str, formule: str) -> Dict[str, Any]:
        df = self.regles_pc_df.copy()
        mask = (
            (df["formule"] == formule) &
            (
                df["code_acte"].fillna("ALL").eq("ALL") |
                df["actes_couverts"].apply(lambda lst: code_acte in lst if isinstance(lst, list) else False)
            )
        )
        regles = df[mask]
        if regles.empty:
            return {"erreur": f"Aucune règle trouvée pour formule '{formule}' et acte '{code_acte}'"}
        reg = regles.iloc[0]
        taux = reg["taux_remboursement"] / 100
        plafond = reg["plafond_annuel"]
        remboursement_brut = montant * taux
        remboursement_final = min(remboursement_brut, plafond)
        reste_a_charge = montant - remboursement_final
        return {
            "montant_facture": montant,
            "code_acte": code_acte,
            "taux_applique": taux * 100,
            "remboursement_brut": remboursement_brut,
            "remboursement_final": remboursement_final,
            "reste_a_charge": reste_a_charge,
            "plafond_formule": plafond,
            "formule_utilisee": formule
        }

    def extract_lignes_from_image(
        self,
        image_bytes: bytes,
        formule: str,
        llm_provider: str = "qwen"
    ) -> (Dict[str, Any], str):
        client = self.client_qwen if llm_provider == "qwen" else self.client_mistral
        response = client.analyze_invoice_image(image_bytes, formule)
        content = response.choices[0].message.content

        # Extraction robuste du JSON, non-greedy
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"Réponse LLM Vision invalide, JSON non trouvé dans : {content!r}")

        json_str = match.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Impossible de parser le JSON extrait : {e}")

        # Vérification des champs essentiels
        if "lignes" not in data or not isinstance(data["lignes"], list):
            raise ValueError("Le LLM n'a pas extrait de lignes exploitables.")

        return data, content

    def process_facture_pennypet(
        self,
        file_bytes: bytes,
        formule_client: str,
        llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        extraction, raw_content = self.extract_lignes_from_image(
            file_bytes, formule_client, llm_provider
        )

        # Choix de la formule détectée ou fournie
        formule = extraction.get("formule_utilisee", formule_client)
        lignes = extraction.get("lignes", [])
        remboursements: List[Dict[str, Any]] = []

        for ligne in lignes:
            try:
                montant = float(ligne.get("montant_ht", 0.0))
            except (ValueError, TypeError):
                montant = 0.0

            code = ligne.get("code_acte") or ligne.get("description", "ALL")
            remb = self.calculer_remboursement_pennypet(montant, code, formule)
            remboursements.append({**ligne, **remb})

        total_montant = sum(float(l.get("montant_ht", 0.0)) for l in lignes)
        total_rembourse = sum(float(r.get("remboursement_final", 0.0)) for r in remboursements)

        return {
            "extraction_facture": extraction,
            "remboursements": remboursements,
            "total_facture": total_montant,
            "total_remboursement": total_rembourse,
            "reste_total_a_charge": total_montant - total_rembourse,
            "formule_utilisee": formule,
            "infos_client": extraction.get("informations_client", {}),
            "texte_ocr": extraction.get("texte_ocr", ""),
            "llm_raw": raw_content
        }

# Instance globale
pennypet_processor = PennyPetProcessor()
