# config/pennypet_config.py
import os
import json
import pandas as pd
from pathlib import Path

class PennyPetConfig:
    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.lexiques_dir = self.config_dir / "lexiques"
        
    def load_pennypet_mapping(self):
        """Charge la configuration de mapping AMV vers PennyPet"""
        mapping_file = self.lexiques_dir / "mapping_amv_pennypet.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Configuration par défaut si le fichier n'existe pas
            return self._get_default_mapping()
    
    def _get_default_mapping(self):
        """Configuration par défaut basée sur vos fichiers"""
        return {
            "mapping_amv_pennypet": {
                "1": {
                    "description": "Actes courants",
                    "taux_integral": 50,
                    "taux_integral_plus": 100,
                    "plafond_annuel": 1000,
                    "eligible_integral": True,
                    "eligible_integral_plus": True
                },
                "5": {
                    "description": "Actes chirurgicaux",
                    "taux_integral": 50,
                    "taux_integral_plus": 100,
                    "plafond_annuel": 1000,
                    "eligible_integral": True,
                    "eligible_integral_plus": True
                }
            },
            "formules_pennypet": {
                "START": {"type": "accident", "remboursement": 0, "plafond": 500},
                "PREMIUM": {"type": "accident", "remboursement": 100, "plafond": 500},
                "INTEGRAL": {"type": "complet", "remboursement": 50, "plafond": 1000},
                "INTEGRAL_PLUS": {"type": "complet", "remboursement": 100, "plafond": 1000}
            }
        }
    
    def load_actes_data(self):
        """Charge et nettoie vos données d'actes"""
        actes_file = self.lexiques_dir / "actes.csv"
        
        # Charge le CSV avec le bon séparateur
        df = pd.read_csv(actes_file, sep=';', encoding='utf-8')
        
        # Nettoie les données si nécessaire
        df.columns = [col.strip() for col in df.columns]
        
        return df
    
    def load_medicaments_data(self):
        """Charge vos données de médicaments"""
        medicaments_file = self.lexiques_dir / "medicaments.csv"
        return pd.read_csv(medicaments_file, encoding='utf-8')

# Instance globale pour faciliter l'import
pennypet_config = PennyPetConfig()
