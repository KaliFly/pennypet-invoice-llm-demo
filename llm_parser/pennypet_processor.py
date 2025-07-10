import json
from typing import Dict, List, Any, Tuple
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient

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

    def extract_lignes_from_image(
        self,
        image_bytes: bytes,
        formule: str,
        llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """
        Extraction robuste du JSON structuré depuis la réponse LLM.
        Retourne (data, raw_content).
        """
        client = self.client_qwen if llm_provider.lower() == "qwen" else self.client_mistral
        response = client.analyze_invoice_image(image_bytes, formule)
        content = response.choices[0].message.content

        # Extraction robuste du JSON via équilibrage des accolades
        start = content.find("{")
        if start == -1:
            raise ValueError(f"JSON non trouvé dans la réponse LLM : {content!r}")

        depth = 0
        end = start
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if depth != 0:
            raise ValueError("Appariement des accolades JSON impossible dans la réponse LLM.")

        json_str = content[start:end+1]

        # Vérification de la présence d'un JSON extrait
        if not json_str:
            raise ValueError(f"Impossible d'extraire un bloc JSON de la réponse LLM : {content!r}")

        try:
            data = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Erreur lors du parsing JSON : {e}\nContenu reçu : {json_str!r}")

        if not data or "lignes" not in data or not isinstance(data["lignes"], list):
            raise ValueError("Le LLM n'a pas extrait de lignes exploitables.\nContenu reçu : {json_str!r}")

        return data, content

    def process_facture_pennypet(
        self,
        file_bytes: bytes,
        formule_client: str,
        llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """
        Pipeline complet :
        1. Extraction LLM → data, raw_content
        2. Calcul de chaque ligne
        3. Agrégation des totaux
        """
        try:
            extraction, raw_content = self.extract_lignes_from_image(
                file_bytes, formule_client, llm_provider
            )
        except Exception as e:
            # Vous pouvez ici logger ou retourner un dict d'erreur selon votre UI
            raise ValueError(f"Erreur lors de l'extraction LLM : {e}")

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
