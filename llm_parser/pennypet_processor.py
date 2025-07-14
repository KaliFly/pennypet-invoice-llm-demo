import json
import re
from typing import Dict, List, Any, Tuple, Optional
from config.pennypet_config import PennyPetConfig
from openrouter_client import OpenRouterClient
import unicodedata

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available. Fuzzy matching will be disabled.")

def pseudojson_to_json(text: str) -> str:
    """
    Correction minimale pour JSON mal formé.
    """
    text = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', text)
    text = text.replace("'", '"')
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

def normaliser_accents(texte: str) -> str:
    """
    Normalise les accents et caractères spéciaux.
    Utilise unicodedata pour une normalisation robuste.
    """
    if not texte:
        return ""
    
    # Normalisation NFD (décomposition) puis suppression des accents
    texte_nfd = unicodedata.normalize('NFD', texte)
    texte_sans_accents = ''.join(c for c in texte_nfd if unicodedata.category(c) != 'Mn')
    
    # Conversion en minuscules et suppression des caractères spéciaux
    texte_clean = re.sub(r'[^\w\s]', ' ', texte_sans_accents.lower())
    
    # Normalisation des espaces
    return ' '.join(texte_clean.split())

class NormaliseurAMVAmeliore:
    """
    Normaliseur amélioré des actes et médicaments avec :
      1. Normalisation des accents et caractères spéciaux
      2. Gestion des variantes orthographiques
      3. Recherche partielle optimisée
      4. Fuzzy matching intelligent
    """
    def __init__(self, config: PennyPetConfig):
        self.config = config
        self.cache: Dict[str, Optional[str]] = {}
        
        # Glossaire actes (mots-clés en minuscules)
        self.termes_actes = set(config.actes_df["field_label"].str.lower())
        self.actes_df = config.actes_df.dropna(subset=["pattern"])
        
        # Glossaire médicaments normalisé
        self.termes_medicaments = config.glossaire_pharmaceutique
        self.medicaments_df = config.medicaments_df
        self.mapping_amv = config.mapping_amv
        
        # Préprocessage du glossaire pharmaceutique
        self.glossaire_normalise = self._preprocess_glossaire()
        
        # Patterns regex pour médicaments
        self.patterns_medicaments = [
            r'\b\d+\s*(mg|ml|g|l|ui|iu|mcg|µg)\b',
            r'\b(comprimé|gélule|cp|gél|sol|inj|ampoule|flacon|tube|boîte)\.?\s*\d*',
            r'\b(antibiotic|anti-inflammatoire|antiparasitaire|antifongique|antiviral)\b',
            r'\b(vaccin|vaccination|rappel|primo-vaccination)\b',
            r'\b(seringue|pipette|spray|pommade|crème|lotion)\b',
            r'\b\d+\s*x\s*\d+\s*(mg|ml|g|l)\b',
            r'\b(principe|actif|laboratoire|generique|specialite)\b'
        ]
        
        # Variantes orthographiques courantes
        self.variantes = {
            'medicament': ['médicament', 'medicaments', 'médicaments'],
            'gelule': ['gélule', 'gélules', 'gelules'],
            'comprimes': ['comprimé', 'comprimés', 'comprimes'],
            'solution': ['solutions', 'sol'],
            'injection': ['injections', 'inj'],
            'milligramme': ['mg', 'milligrammes'],
            'millilitre': ['ml', 'millilitres'],
            'gramme': ['g', 'grammes'],
            'litre': ['l', 'litres']
        }

    def _preprocess_glossaire(self) -> Dict[str, str]:
        """Préprocesse le glossaire pharmaceutique pour optimiser la recherche"""
        glossaire_normalise = {}
        
        for terme in self.termes_medicaments:
            # Normalisation principale
            terme_norm = normaliser_accents(terme)
            if terme_norm:
                glossaire_normalise[terme_norm] = terme
            
            # Ajouter les variantes
            for variante in self._generer_variantes(terme):
                variante_norm = normaliser_accents(variante)
                if variante_norm:
                    glossaire_normalise[variante_norm] = terme
        
        return glossaire_normalise

    def _generer_variantes(self, terme: str) -> List[str]:
        """Génère des variantes orthographiques d'un terme"""
        variantes = [terme]
        
        # Variantes avec/sans 's' final
        if terme.endswith('s'):
            variantes.append(terme[:-1])
        else:
            variantes.append(terme + 's')
        
        # Variantes avec abréviations
        for base, abbrevs in self.variantes.items():
            if base in terme.lower():
                for abbrev in abbrevs:
                    variantes.append(terme.lower().replace(base, abbrev))
        
        return variantes

    def _detecter_patterns_medicaments(self, texte: str) -> bool:
        """Détecte les patterns typiques des médicaments"""
        texte_norm = normaliser_accents(texte)
        
        for pattern in self.patterns_medicaments:
            if re.search(pattern, texte_norm, re.IGNORECASE):
                return True
        
        return False

    def normalise_acte(self, libelle_brut: str) -> Optional[str]:
        """Normalise un acte médical"""
        if not libelle_brut:
            return None
        
        cle = libelle_brut.upper().strip()
        if cle in self.cache:
            return self.cache[cle]
        
        # Normalisation pour recherche
        libelle_norm = normaliser_accents(libelle_brut)
        
        # 1. Pattern CSV exact
        for _, row in self.actes_df.iterrows():
            if row["pattern"].search(cle):
                self.cache[cle] = row["code_acte"]
                return row["code_acte"]
        
        # 2. Fallback sémantique actes
        for terme in self.termes_actes:
            terme_norm = normaliser_accents(terme)
            if re.search(rf"(?<!\w){re.escape(terme_norm)}(?!\w)", libelle_norm):
                code = terme.upper()
                self.cache[cle] = code
                return code
        
        # 3. Fuzzy matching sur les actes
        if RAPIDFUZZ_AVAILABLE:
            codes = self.actes_df["code_acte"].dropna().tolist()
            if codes:
                match, score, _ = process.extractOne(cle, codes, scorer=fuzz.token_sort_ratio)
                if score >= 80:
                    self.cache[cle] = match
                    return match
        
        self.cache[cle] = None
        return None

    def normalise_medicament(self, libelle_brut: str) -> Optional[str]:
        """Normalise un médicament avec recherche améliorée"""
        if not libelle_brut:
            return None
        
        cle = libelle_brut.upper().strip()
        if cle in self.cache:
            return self.cache[cle]
        
        # Normalisation pour recherche
        libelle_norm = normaliser_accents(libelle_brut)
        
        # 1. Détection par patterns regex
        if self._detecter_patterns_medicaments(libelle_brut):
            self.cache[cle] = "MEDICAMENTS"
            return "MEDICAMENTS"
        
        # 2. Recherche exacte dans glossaire normalisé
        if libelle_norm in self.glossaire_normalise:
            self.cache[cle] = "MEDICAMENTS"
            return "MEDICAMENTS"
        
        # 3. Recherche partielle dans glossaire
        for terme_norm in self.glossaire_normalise.keys():
            # Recherche bidirectionnelle
            if (terme_norm in libelle_norm or 
                libelle_norm in terme_norm or
                any(word in libelle_norm for word in terme_norm.split() if len(word) > 3)):
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
        
        # 4. Recherche dans la base de médicaments
        if hasattr(self.medicaments_df, 'medicament'):
            meds_normalises = [normaliser_accents(m) for m in self.medicaments_df["medicament"].dropna()]
            if libelle_norm in meds_normalises:
                self.cache[cle] = "MEDICAMENTS"
                return "MEDICAMENTS"
        
        # 5. Fuzzy matching intelligent
        if RAPIDFUZZ_AVAILABLE:
            # Fuzzy sur le glossaire
            if self.glossaire_normalise:
                match, score, _ = process.extractOne(
                    libelle_norm, 
                    list(self.glossaire_normalise.keys()), 
                    scorer=fuzz.partial_ratio
                )
                if score >= 85:
                    self.cache[cle] = "MEDICAMENTS"
                    return "MEDICAMENTS"
            
            # Fuzzy sur la base de médicaments
            if hasattr(self.medicaments_df, 'medicament'):
                choices = [normaliser_accents(m) for m in self.medicaments_df["medicament"].dropna()]
                if choices:
                    match, score, _ = process.extractOne(
                        libelle_norm, 
                        choices, 
                        scorer=fuzz.token_set_ratio
                    )
                    if score >= 80:
                        self.cache[cle] = "MEDICAMENTS"
                        return "MEDICAMENTS"
        
        self.cache[cle] = None
        return None

    def normalise(self, libelle_brut: str) -> Optional[str]:
        """Normalise un libellé (acte ou médicament)"""
        # Priorité : actes d'abord, puis médicaments
        result = self.normalise_acte(libelle_brut)
        if result:
            return result
        
        result = self.normalise_medicament(libelle_brut)
        if result:
            return result
        
        # Fallback : retourner le libellé original normalisé
        return libelle_brut.strip().upper()

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de mapping"""
        return {
            "cache_size": len(self.cache),
            "actes": len(self.termes_actes),
            "medicaments": len(self.termes_medicaments),
            "glossaire_normalise": len(self.glossaire_normalise),
            "patterns_medicaments": len(self.patterns_medicaments),
            "variantes": len(self.variantes),
            "rapidfuzz": RAPIDFUZZ_AVAILABLE
        }

class PennyPetProcessor:
    """
    Pipeline extraction LLM, normalisation améliorée, calcul remboursement PennyPet.
    """
    def __init__(
        self,
        client_qwen: OpenRouterClient = None,
        client_mistral: OpenRouterClient = None,
        config: PennyPetConfig = None,
    ):
        self.config = config or PennyPetConfig()
        self.client_qwen = client_qwen or OpenRouterClient(model_key="primary")
        self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
        self.regles_pc_df = self.config.regles_pc_df
        self.normaliseur = NormaliseurAMVAmeliore(self.config)
        
        # Statistiques de traitement
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0
        }

    def extract_lignes_from_image(
        self, image_bytes: bytes, formule: str, llm_provider: str = "qwen"
    ) -> Tuple[Dict[str, Any], str]:
        """Extrait les lignes d'une image avec le LLM"""
        client = self.client_qwen if llm_provider.lower() == "qwen" else self.client_mistral
        resp = client.analyze_invoice_image(image_bytes, formule)
        content = resp.choices[0].message.content
        
        # Extraction JSON robuste
        start = content.find("{")
        if start < 0:
            raise ValueError("JSON non trouvé dans la réponse LLM.")
        
        depth = 0
        json_str = None
        for i, ch in enumerate(content[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = content[start : i + 1]
                    break
        
        if not json_str:
            raise ValueError("JSON malformé dans la réponse LLM.")
        
        # Nettoyage et parsing JSON
        json_str = pseudojson_to_json(json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur parsing JSON: {e}")
        
        if "lignes" not in data:
            raise ValueError("Le LLM n'a pas extrait de lignes exploitables.")
        
        return data, content

    def calculer_remboursement(
        self, montant: float, code_acte: str, formule: str, est_accident: bool
    ) -> Dict[str, Any]:
        """Calcule le remboursement selon les règles"""
        df = self.regles_pc_df.copy()
        
        # Construction du masque de filtrage
        mask = (
            (df["formule"] == formule)
            & (
                df["code_acte"].eq(code_acte)
                | (
                    df["code_acte"].fillna("ALL").eq("ALL")
                    & df["actes_couverts"].apply(lambda l: code_acte in l if isinstance(l, list) else False)
                )
            )
            & (
                (df["type_couverture"] == "ACCIDENT_MALADIE")
                | ((df["type_couverture"] == "ACCIDENT_SEULEMENT") & est_accident)
            )
        )
        
        reg = df[mask]
        if reg.empty:
            return {
                "erreur": f"Aucune règle trouvée pour {formule}/{code_acte}",
                "montant_ht": montant,
                "taux": 0.0,
                "remb_final": 0.0,
                "reste": montant
            }
        
        # Calcul du remboursement
        r = reg.iloc[0]
        taux = r["taux_remboursement"] / 100
        plafond = r.get("plafond_annuel", float('inf'))
        
        remb_brut = montant * taux
        remb_final = min(remb_brut, plafond)
        
        return {
            "montant_ht": montant,
            "taux": taux * 100,
            "remb_final": remb_final,
            "reste": montant - remb_final
        }

    def process_facture_pennypet(
        self, file_bytes: bytes, formule_client: str, llm_provider: str = "qwen"
    ) -> Dict[str, Any]:
        """Traite une facture PennyPet complète"""
        
        # Réinitialisation des stats
        self.stats = {
            'lignes_traitees': 0,
            'medicaments_detectes': 0,
            'actes_detectes': 0,
            'erreurs_normalisation': 0
        }
        
        try:
            # Extraction des données
            data, raw_content = self.extract_lignes_from_image(file_bytes, formule_client, llm_provider)
            
            resultats: List[Dict[str, Any]] = []
            accidents = {"accident", "urgent", "urgence", "fract", "trauma", "traumatisme"}
            
            # Traitement de chaque ligne
            for ligne in data["lignes"]:
                try:
                    # Extraction des données de ligne
                    libelle = (ligne.get("code_acte") or ligne.get("description", "")).strip()
                    montant = float(ligne.get("montant_ht", 0) or 0)
                    
                    # Normalisation du code acte
                    code_norm = self.normaliseur.normalise(libelle)
                    
                    # Détection d'accident
                    est_acc = any(mot in libelle.lower() for mot in accidents)
                    
                    # Calcul du remboursement
                    remb = self.calculer_remboursement(montant, code_norm, formule_client, est_acc)
                    
                    # Mise à jour des statistiques
                    self.stats['lignes_traitees'] += 1
                    if code_norm == "MEDICAMENTS":
                        self.stats['medicaments_detectes'] += 1
                    else:
                        self.stats['actes_detectes'] += 1
                    
                    # Ajout des résultats
                    ligne.update({
                        "code_norm": code_norm,
                        "est_accident": est_acc,
                        **remb
                    })
                    resultats.append(ligne)
                    
                except Exception as e:
                    self.stats['erreurs_normalisation'] += 1
                    print(f"Erreur traitement ligne {ligne}: {e}")
                    continue
            
            # Calcul des totaux
            total_remb = sum(r.get("remb_final", 0) for r in resultats)
            total_facture = sum(r.get("montant_ht", 0) for r in resultats)
            
            return {
                "success": True,
                "lignes": resultats,
                "total_remb": total_remb,
                "total_facture": total_facture,
                "reste_a_charge": total_facture - total_remb,
                "stats": self.stats,
                "mapping_stats": self.normaliseur.get_mapping_stats(),
                "raw_llm_response": raw_content
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }

    def get_processor_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes du processeur"""
        return {
            **self.stats,
            **self.normaliseur.get_mapping_stats()
        }

# Instance globale pour usage direct
pennypet_processor = PennyPetProcessor()
