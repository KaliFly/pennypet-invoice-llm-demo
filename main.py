# main.py (mise à jour)
from openrouter_client import OpenRouterClient
from llm_parser.pennypet_processor import pennypet_processor
import json

# Vos clients existants
client_qwen = OpenRouterClient(model_key="primary")
client_mistral = OpenRouterClient(model_key="secondary")

def process_facture_pennypet(texte_ocr: str, formule_client: str):
    """
    Pipeline complet avec intégration PennyPet
    """
    # 1. Identification des actes
    actes_detectes = pennypet_processor.identifier_actes_sur_facture(texte_ocr)
    
    # 2. Extraction LLM (votre code existant)
    messages = [
        {"role": "system", "content": "Extrais les données de facture vétérinaire en JSON..."},
        {"role": "user", "content": texte_ocr}
    ]
    response_qwen = client_qwen.chat(messages)
    extraction_data = json.loads(response_qwen.choices[0].message.content)
    
    # 3. Calcul remboursement PennyPet
    montant_total = extraction_data.get("montant_total", 0)
    
    # Prendre la première AMV détectée (ou 1 par défaut)
    amv_detectee = actes_detectes[0]["amv"] if actes_detectes else 1
    
    remboursement = pennypet_processor.calculer_remboursement_pennypet(
        montant=montant_total,
        amv=amv_detectee, 
        formule=formule_client
    )
    
    # 4. Résultat final
    return {
        "extraction_facture": extraction_data,
        "actes_detectes": actes_detectes,
        "remboursement_pennypet": remboursement
    }

if __name__ == "__main__":
    # Test rapide
    texte_test = "Consultation générale : 65€"
    resultat = process_facture_pennypet(texte_test, "INTEGRAL")
    print(json.dumps(resultat, indent=2, ensure_ascii=False))
