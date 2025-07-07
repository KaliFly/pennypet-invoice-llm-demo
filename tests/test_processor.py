def test_identifier_actes(processor):
    actes = processor.identifier_actes_sur_facture("texte factice")
    # Selon vos données de config, vérifiez que le résultat est une liste
    assert isinstance(actes, list)

def test_extraction_et_remboursement(processor):
    result = processor.process_facture_pennypet(
        file_bytes=b"%PDF", 
        formule_client="INTEGRAL", 
        llm_provider="qwen"
    )
    # Vérification de la structure du résultat
    assert "texte_ocr" in result
    assert result["montant_total"] == 10.0
    assert "remboursement_pennypet" in result
