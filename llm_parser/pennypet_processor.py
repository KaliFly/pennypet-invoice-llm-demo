# llm_parser/pennypet_processor.py
import json
import re
from typing import Dict, List, Optional
from config.pennypet_config import pennypet_config

class PennyPetProcessor:
    def __init__(self):
        self.config = pennypet_config.load_pennypet_mapping()
        self.actes_df = pennypet_config.load_actes_data()
        self.medicaments_df = pennypet_config.load_medicaments_data()
        
    def identifier_actes_sur_facture(self, texte_ocr: str) -> List[Dict]:
        """
        Identifie les actes sur une facture via regex de vos fichiers
        """
        actes_detectes = []
        
        for _, row in self.actes_df.iterrows():
            if 'regex_pattern' in row and pd.notna(row['regex_pattern']):
                pattern = re.compile(row['regex_pattern'])
                if pattern.search(texte_ocr):
                    actes_detectes.append({
                        'categorie': row['Catégorie'],
                        'sous_acte': row['Sous-acte'],
                        'amv': row['AMV'],
                        'ligne_detectee': texte_ocr
                    })
        
        return actes_detectes
    
    def calculer_remboursement_pennypet(self, montant: float, amv: int, formule: str) -> Dict:
        """
        Calcule le remboursement selon les règles PennyPet
        """
        mapping = self.config["mapping_amv_pennypet"].get(str(amv))
        formule_config = self.config["formules_pennypet"].get(formule)
        
        if not mapping or not formule_config:
            return {"erreur": "Configuration non trouvée"}
        
        # Calcul selon la formule
        if formule == "INTEGRAL":
            taux = mapping["taux_integral"] / 100
        elif formule == "INTEGRAL_PLUS":
            taux = mapping["taux_integral_plus"] / 100
        else:
            taux = formule_config["remboursement"] / 100
        
        remboursement = montant * taux
        plafond = formule_config["plafond"]
        
        return {
            "montant_facture": montant,
            "taux_applique": taux * 100,
            "remboursement_brut": remboursement,
            "remboursement_final": min(remboursement, plafond),
            "reste_a_charge": montant - min(remboursement, plafond),
            "plafond_formule": plafond,
            "formule_utilisee": formule
        }

# Instance pour faciliter l'import
pennypet_processor = PennyPetProcessor()
