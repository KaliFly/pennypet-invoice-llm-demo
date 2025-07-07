import pandas as pd
import re

def normalize_actes(input_path, output_path):
    acts = pd.read_csv(input_path, sep=';')
    corrections = {
        r"thoraxthorax": r"(?:thorax)",
        r"abdomenabdomen": r"(?:abdomen)",
        r"osos": r"os"
    }
    for old, new in corrections.items():
        acts["regex_pattern"] = acts["regex_pattern"].str.replace(old, new, regex=True)
    acts["code_acte"] = acts.apply(
        lambda row: re.sub(r"[^A-Za-z0-9]+", "_",
                           f"{row['Catégorie']}_{row['Sous-acte']}").upper(),
        axis=1
    )
    acts.to_csv(output_path, index=False)

def normalize_medicaments(input_path, output_path):
    meds = pd.read_csv(input_path)
    meds.rename(columns={
        "Médicament (générique)": "medicament",
        "Principaux médicaments commerciaux / génériques": "libelle_commercial",
        "Catégorie thérapeutique": "categorie_therapeutique",
        "Variantes / Synonymes OCR": "synonymes_ocr"
    }, inplace=True)
    meds["synonymes_ocr"] = meds["synonymes_ocr"].str.split(r"\s*,\s*")
    meds.to_json(output_path, orient="records", force_ascii=False)
