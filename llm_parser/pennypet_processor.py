# llm_parser/pennypet_processor.py

import re
from typing import Dict, List, Any
import pandas as pd
from config.pennypet_config import PennyPetConfig
from llm_parser.parser import InvoiceParser


class PennyPetProcessor:
    def __init__(self, llm_provider: str = "qwen"):
        config = PennyPetConfig()
        # Lexiques et règles
        self.mapping_amv     = config.mapping_amv
        self.formules        = config.formules
        self.actes_df        = config.actes_df
        self.medicaments_df  = config.medicaments_df
        self.regles_pc_df    = config.regles_pc_df
        # LLM parser (Qwen ou Mistral)
        self.parser          = InvoiceParser(provider=llm_provider)

    def identifier_actes_sur_facture(self, texte_ocr: str) -> List[Dict]:
        """
        Identifie les actes sur une facture via les patterns compilés.
        """
        actes_detectes = []
        for _, row in self.actes_df.iterrows():
            pattern: re.Pattern = row["pattern"]
            match = pattern.search(texte_ocr)
            if match:
                actes_detectes.append({
                    "categorie":     row.get("Catégorie", ""),
                    "sous_acte":     row.get("Sous-acte", ""),
                    "amv":           int(row.get("AMV", 0)),
                    "code_acte":     row.get("code_acte", ""),
                    "ligne_detectee": match.group(0)
                })
        return actes_detectes

    def extract_lignes(self, texte_ocr: str) -> Dict[str, Any]:
        """
        Extrait via LLM les lignes de facture et le montant total.
        """
        return self.parser.extract(texte_ocr)

    def calculer_remboursement_pennypet(
        self, montant: float, amv: int, formule: str
    ) -> Dict:
        """
        Calcule le remboursement selon les règles de prise en charge.
        """
        # Filtrer la règle correspondant à l'assureur, formule et AMV
        df = self.regles_pc_df
        mask = (
            (df["assureur"] == "ASSUREUR_PRINCIPAL") &
            (df["formule"] == formule) &
            (df["taux_remboursement"] == amv)
        )
        regle = df[mask]
        if regle.empty:
            return {"erreur": "Règle non trouvée pour AMV et formule fournis"}

        regle = regle.iloc[0]
        taux = regle["taux_remboursement"] / 100
        plafond = regle["plafond_annuel"]

        remboursement_brut = montant * taux
        remboursement_final = min(remboursement_brut, plafond)
        reste_a_charge = montant - remboursement_final

        return {
            "montant_facture":     montant,
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
        1. identify acts
        2. extract lines + total via LLM
        3. calculate reimbursement
        """
        actes = self.identifier_actes_sur_facture(texte_ocr)
        extraction = self.extract_lignes(texte_ocr)
        montant_total = extraction["montant_total"]

        # Determine AMV: use highest detected or default 1
        amv_list = [a["amv"] for a in actes]
        amv = max(amv_list) if amv_list else 1

        remboursement = self.calculer_remboursement_pennypet(
            montant=montant_total,
            amv=amv,
            formule=formule
        )

        return {
            "actes_detectes":       actes,
            "extraction_facture":   extraction,
            "amv_detectee":         amv,
            "remboursement_pennypet": remboursement
        }


# Instance globale pour usage
pennypet_processor = PennyPetProcessor()
