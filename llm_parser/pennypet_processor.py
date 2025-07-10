import json
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path
from openrouter_client import OpenRouterClient
from config.pennypet_config import PennyPetConfig

class PennyPetProcessor:
    """
    Pipeline 100% LLM Vision pour extraction et calcul de remboursement.
    """

    def __init__(
        self,
        client_qwen: OpenRouterClient = None,
        client_mistral: OpenRouterClient = None,
        config: PennyPetConfig = None
    ):
        self.config = config or PennyPetConfig()
        self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
        self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
        self.regles_pc_df = self.config.regles_pc_df

    def calculer_remboursement_pennypet(
        self,
        montant: float,
        code_acte: str,
        formule: str
    ) -> Dict[str, Any]:
        df = self.regles_pc_df.copy()
        mask = (
            (df["formule"] == formule) &
            (
                df["code_acte"].fillna("ALL").eq("ALL") |
                df["actes_couverts"].apply(
                    lambda lst: code_acte in lst if isinstance(lst, list) else False
                )
            )
        )
        regles = df[mask]
        if regles.empty:
            return {"erreur": f"Aucune règle trouvée pour formule '{formule}' et acte '{code_acte}'"}
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
            "formule_utilisee": formule
        }

    def _extract_json_blocks(self, text: str) -> List[str]:
        """
        Extraction robuste des blocs JSON dans un texte, sans regex récursive.
        Utilise un comptage d'accolades pour délimiter le JSON.
        """
        blocks = []
        stack = []
        start = None
        for i, c in enumerate(text):
            if c == '{':
                if not stack:
                    start = i
                stack.append(c)
            elif c == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        blocks.append(text[start:i+1])
        return blocks

    def extract_lignes_from_image(
        self,
        image_bytes: bytes,
        formule: str,
        llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """
        Envoie l’image/PDF à l’API LLM Vision et extrait le JSON structuré.
        Retourne (data, raw_content) où data est le dict JSON et raw_content
        la réponse brute du LLM pour audit.
        """
        client = self.client_qwen if llm_provider.lower() == "qwen" else self.client_mistral
        response = client.analyze_invoice_image(image_bytes, formule)
        content = response.choices[0].message.content

        # Extraction robuste du JSON via comptage d’accolades
        json_blocks = self._extract_json_blocks(content)
        json_str = max(json_blocks, key=len) if json_blocks else None

        if not json_str:
            raise ValueError(f"Réponse LLM Vision invalide, JSON non trouvé dans : {content!r}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Impossible de parser le JSON extrait : {e}")

        if "lignes" not in data or not isinstance(data["lignes"], list):
            raise ValueError("Le LLM n’a pas extrait de lignes exploitables.")

        return data, content

    def process_facture_pennypet(
        self,
        file_bytes: bytes,
        formule_client: str,
        llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """
        Pipeline complet :
        1. extract_lignes_from_image → data, raw_content
        2. calcul de chaque ligne
        3. agrégation des totaux
        """
        extraction, raw_content = self.extract_lignes_from_image(
            file_bytes, formule_client, llm_provider
        )

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

# Instance globale pour usage direct
pennypet_processor = PennyPetProcessor()
