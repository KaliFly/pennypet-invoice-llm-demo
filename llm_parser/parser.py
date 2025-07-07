# llm_parser/parser.py

import json
import re
from typing import List, Dict, Any
from openrouter_client import OpenRouterClient

class InvoiceParser:
    """
    Classe responsable de l’envoi de prompts aux LLM Qwen ou Mistral
    et du parsing de la réponse JSON.
    """

    def __init__(self, provider: str = "qwen"):
        """
        :param provider: "qwen" pour Qwen (primary), "mistral" pour Mistral (secondary)
        """
        model_key = "primary" if provider.lower() == "qwen" else "secondary"
        self.client = OpenRouterClient(model_key=model_key)
        self.provider = provider.lower()

    def build_messages(self, texte: str) -> List[Dict[str, str]]:
        """
        Construit la liste des messages System/User pour l’extraction.
        On choisit un prompt concis pour Qwen, plus détaillé pour Mistral.
        """
        if self.provider == "qwen":
            system_prompt = (
                "Vous êtes un assistant expert en factures vétérinaires. "
                "Extrayez en JSON uniquement un tableau 'lignes' et une clé 'montant_total'. "
                "Chaque ligne doit contenir 'animal_uid' (string), 'montant_ht' (float), 'description' (string)."
            )
        else:
            system_prompt = (
                "Vous êtes un assistant vétérinaire spécialisé dans l’analyse de factures. "
                "À partir d’un texte brut OCR, produisez un objet JSON contenant :\n"
                "- 'lignes' : liste d’objets { 'animal_uid': str, 'montant_ht': float, 'description': str }\n"
                "- 'montant_total': somme totale des montants HT\n"
                "Renvoie uniquement un JSON valide, sans commentaire."
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": texte}
        ]

    def extract(self, texte: str) -> Dict[str, Any]:
        """
        Exécute l’appel LLM et parse la réponse en JSON.

        :param texte: texte OCR de la facture
        :return: dict avec 'lignes' et 'montant_total'
        :raises ValueError: si la réponse n’est pas un JSON valide ou manquants
        """
        messages = self.build_messages(texte)
        response = self.client.chat(messages)
        content = response.choices[0].message.content

        # Extraire le bloc JSON s’il est encadré par des backticks ou autre
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if not match:
            raise ValueError(f"Réponse LLM invalide, JSON non trouvé dans: {content!r}")
        json_str = match.group(1)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Réponse LLM invalide, JSON attendu: {e}")

        if "lignes" not in data or "montant_total" not in data:
            raise ValueError("La réponse JSON doit contenir 'lignes' et 'montant_total'")

        return data
