# llm_parser/pennypet_processor.py

import re
from typing import Dict, List, Any, Optional
from config.pennypet_config import PennyPetConfig
from ocr_module.ocr import OCRProcessor
from openrouter_client import OpenRouterClient
from llm_parser.parser import InvoiceParser

class PennyPetProcessor:
    def __init__(
        self,
        ocr: Optional[OCRProcessor] = None,
        client_qwen: Optional[OpenRouterClient] = None,
        client_mistral: Optional[OpenRouterClient] = None,
        config: Optional[PennyPetConfig] = None,
        llm_provider: str = "qwen",
        parser: Optional[InvoiceParser] = None
    ):
        # Injection de dépendances pour les tests ou création par défaut
        self.config         = config or PennyPetConfig()
        self.ocr            = ocr or OCRProcessor(lang="fra")
        self.client_qwen    = client_qwen or OpenRouterClient(model_key="primary")
        self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
        # Choix du parser LLM : injection ou création selon llm_provider
        self.parser         = parser or InvoiceParser(provider=llm_provider)

        # Chargement des ressources de configuration
        self.mapping_amv    = self.config.mapping_amv
        self.formules       = self.config.formules
        self.actes_df       = self.config.actes_df
        self.medicaments_df = self.config.medicaments_df
        self.regles_pc_df   = self.config.regles_pc_df

    def identifier_actes_sur_facture(self, texte_ocr: str) -> List[Dict]:
        """
        Identifie les actes présents dans le texte OCR via les patterns compilés.
        """
        actes_detectes: List[Dict] = []
        for _, row in self.actes_df.iterrows():
            pattern: Optional[re.Pattern] = row.get("pattern")
            if not pattern:
                continue
            match = pattern.search(texte_ocr)
            if match:
                actes_detectes.append({
                    "categorie": row.get("Catégorie", ""),
                    "sous_acte": row.get("Sous-acte", ""),
                    "code_acte": row.get("code_acte", ""),
                    "ligne_detectee": match.group(0)
                })
        return actes_detectes

    def extract_lignes(self, texte_ocr: str) -> Dict[str, Any]:
        """
        Extrait via LLM les lignes de facture et le montant total.
        Doit retourner {'lignes': [...], 'montant_total': float}
        """
        return self.parser.extract(texte_ocr)

    def calculer_remboursement_pennypet(
        self, montant: float, code_acte: str, formule: str
    ) -> Dict[str, Any]:
        """
        Calcule le remboursement pour un acte donné selon les règles de prise en charge.
        Filtre par code_acte ou règle globale ('ALL').
        """
        df = self.regles_pc_df
        mask = (
            (df["formule"] == formule) &
            ((df.get("code_acte", ["ALL"])[0] == "ALL") |
             (code_acte.isupper() and df["actes_couverts"].apply(lambda lst: code_acte in lst)))
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
            "montant_facture":     montant,
            "code_acte":           code_acte,
            "taux_applique":       taux * 100,
            "remboursement_brut":  remboursement_brut,
            "remboursement_final": remboursement_final,
            "reste_a_charge":      reste_a_charge,
            "plafond_formule":     plafond,
            "formule_utilisee":    formule
        }

    def process_facture(
        self, texte_ocr: str, formule: str
    ) -> Dict[str, Any]:
        """
        Orchestrates the full pipeline:
        1. identifier_actes_sur_facture
        2. extract_lignes
        3. calculer_remboursement_pennypet for each line
        """
        actes = self.identifier_actes_sur_facture(texte_ocr)
        extraction = self.extract_lignes(texte_ocr)
        lignes = extraction.get("lignes", [])
        remboursements: List[Dict] = []

        for ligne in lignes:
            montant = ligne.get("montant_ht", 0.0)
            code = ligne.get("code_acte", ligne.get("description", "ALL"))
            remboursement = self.calculer_remboursement_pennypet(
                montant=montant,
                code_acte=code,
                formule=formule
            )
            remboursements.append({**ligne, **remboursement})

        total_montant = sum(l.get("montant_ht", 0.0) for l in lignes)
        total_rembourse = sum(r.get("remboursement_final", 0.0) for r in remboursements)

        return {
            "actes_detectes":        actes,
            "extraction_lignes":     lignes,
            "remboursements":        remboursements,
            "total_facture":         total_montant,
            "total_remboursement":   total_rembourse,
            "reste_total_a_charge":  total_montant - total_rembourse,
            "formule_utilisee":      formule
        }

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """
        Alias pour compatibilité :
        1. OCR -> texte_ocr
        2. process_facture avec formule_client
        """
        try:
            texte = self.ocr.extract_text_from_pdf_bytes(file_bytes)
        except Exception:
            texte = self.ocr.extract_text_from_image_bytes(file_bytes)

        result = self.process_facture(texte, formule_client)
        result["texte_ocr"] = texte
        return result

# Instance globale pour usage direct
pennypet_processor = PennyPetProcessor()
