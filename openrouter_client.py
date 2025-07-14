import streamlit as st
import openai
import base64
import time
import random
import json
import logging
from typing import List, Dict, Any, Optional, Union
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import io
from PIL import Image

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    Wrapper amélioré pour l'API OpenRouter.ai avec gestion PDF et validation JSON.
    Supports deux providers (primary → Qwen, secondary → Mistral), incluant les modèles vision.
    """

    def __init__(self, model_key: str):
        # Récupération des clés API et modèles depuis Streamlit secrets
        if model_key == "primary":
            api_key = st.secrets["openrouter"]["API_KEY_QWEN"]
            self.model = st.secrets["openrouter"].get("MODEL_PRIMARY", "qwen/qwen2.5-vl-32b-instruct:free")
        elif model_key == "secondary":
            api_key = st.secrets["openrouter"]["API_KEY_MISTRAL"]
            self.model = st.secrets["openrouter"].get("MODEL_SECONDARY", "mistralai/mistral-small-3.2-24b-instruct:free")
        else:
            raise ValueError(f"Unknown model_key '{model_key}'. Use 'primary' or 'secondary'.")

        if not api_key:
            raise ValueError(f"Missing API key for model_key={model_key!r}")

        self.client = openai.Client(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def _is_pdf(self, file_bytes: bytes) -> bool:
        """Détermine si le fichier est un PDF"""
        return file_bytes.startswith(b'%PDF')

    def _convert_pdf_to_image(self, file_bytes: bytes) -> bytes:
        """Convertit un PDF en image pour l'analyse LLM"""
        try:
            # Méthode 1: PyMuPDF (plus rapide)
            pdf_document = fitz.open("pdf", file_bytes)
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Résolution 2x
            img_bytes = pix.tobytes("png")
            pdf_document.close()
            logger.info("PDF converti en image avec PyMuPDF")
            return img_bytes
        except Exception as e:
            logger.warning(f"Échec PyMuPDF: {e}, utilisation de pdf2image")
            try:
                # Méthode 2: pdf2image (fallback)
                images = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=300)
                img_bytes = io.BytesIO()
                images[0].save(img_bytes, format='PNG')
                return img_bytes.getvalue()
            except Exception as e2:
                logger.error(f"Échec conversion PDF: {e2}")
                raise ValueError(f"Impossible de convertir le PDF: {e2}")

    def _optimize_image(self, image_bytes: bytes) -> bytes:
        """Optimise l'image pour réduire la taille tout en gardant la qualité"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Redimensionner si trop grande
            max_size = (2048, 2048)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Sauvegarder avec compression optimale
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()
        except Exception as e:
            logger.warning(f"Échec optimisation image: {e}")
            return image_bytes

    def _extract_and_validate_json(self, content: str) -> Dict[str, Any]:
        """Extrait et valide le JSON de la réponse"""
        try:
            # Nettoyer le contenu
            content = content.strip()
            
            # Trouver les délimiteurs JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Pas de JSON trouvé dans la réponse")
            
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Validation de la structure
            if not isinstance(data, dict):
                raise ValueError("La réponse doit être un objet JSON")
            
            if "lignes" not in data:
                raise ValueError("Le champ 'lignes' est manquant")
            
            if not isinstance(data["lignes"], list):
                raise ValueError("Le champ 'lignes' doit être une liste")
            
            # Validation des lignes
            for i, ligne in enumerate(data["lignes"]):
                if not isinstance(ligne, dict):
                    raise ValueError(f"Ligne {i} doit être un objet")
                
                required_fields = ["code_acte", "description", "montant_ht"]
                for field in required_fields:
                    if field not in ligne:
                        raise ValueError(f"Champ manquant '{field}' dans ligne {i}")
                
                # Validation du montant
                try:
                    float(ligne["montant_ht"])
                except (ValueError, TypeError):
                    raise ValueError(f"Montant invalide dans ligne {i}: {ligne['montant_ht']}")
            
            # Validation des informations client
            if "informations_client" not in data:
                data["informations_client"] = {}
            
            logger.info(f"JSON validé avec succès: {len(data['lignes'])} lignes")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON: {e}")
            raise ValueError(f"JSON invalide: {e}")
        except Exception as e:
            logger.error(f"Erreur validation: {e}")
            raise

    def get_improved_prompt(self, formule_client: str) -> str:
        """Prompt amélioré pour une meilleure extraction selon PennyPet"""
        return f"""
Vous êtes un expert en extraction de données de factures vétérinaires françaises.

CONTEXTE PENNYPET:
- Vous analysez une facture pour l'assurance santé animale PennyPet
- Formule client: {formule_client}

INSTRUCTIONS STRICTES:
1. Analysez minutieusement l'image de facture vétérinaire fournie
2. Identifiez TOUS les actes médicaux et médicaments présents
3. Extrayez UNIQUEMENT les montants HT (hors taxes)
4. Ignorez les montants TTC et TVA
5. Détectez les caractéristiques d'accidents (urgence, traumatisme, fracture)

CRITÈRES D'IDENTIFICATION:
- MEDICAMENTS: produits pharmaceutiques, vaccins, compléments, antiparasitaires
- ACTES: consultations, examens, interventions chirurgicales, analyses

SCHEMA JSON OBLIGATOIRE:
{{
    "texte_ocr": "Texte complet extrait de la facture",
    "lignes": [
        {{
            "animal_uid": "identifiant animal si présent",
            "code_acte": "description exacte telle qu'elle apparaît sur la facture",
            "description": "description complète détaillée",
            "montant_ht": nombre_décimal_uniquement
        }}
    ],
    "montant_total": nombre_décimal_total_ht,
    "informations_client": {{
        "nom_proprietaire": "nom du propriétaire",
        "nom_animal": "nom de l'animal",
        "identification": "numéro d'identification, tatouage ou puce"
    }}
}}

RÈGLES SPECIFIQUES PENNYPET:
- Pour START: pas d'assurance (information uniquement)
- Pour PREMIUM: accidents uniquement, 100% jusqu'à 500€/an
- Pour INTEGRAL: accidents et maladies, 50% jusqu'à 1000€/an
- Pour INTEGRAL_PLUS: accidents et maladies, 100% jusqu'à 1000€/an

IMPORTANT: Répondez UNIQUEMENT avec du JSON valide, sans texte explicatif avant ou après.
"""

    def chat(
        self,
        messages: List[Dict[str, Union[str, list, dict]]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
        stop: Optional[List[str]] = None,
        retries: int = 3
    ) -> Any:
        """
        Send a chat completion request with retries and exponential backoff.
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if stop:
            params["stop"] = stop

        last_exception = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(**params)
                return response
            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt + random.uniform(0, 1)
                logger.warning(f"Tentative {attempt + 1}/{retries} échouée: {e}")
                time.sleep(wait_time)
        
        raise RuntimeError(f"OpenRouter API failed after {retries} attempts: {last_exception}")

    def analyze_invoice_image(
        self,
        image_bytes: bytes,
        formule_client: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        retries: int = 3
    ) -> Any:
        """
        Analyse une image de facture avec gestion PDF améliorée et validation JSON.
        """
        try:
            # Vérifier et convertir si PDF
            if self._is_pdf(image_bytes):
                logger.info("PDF détecté, conversion en image...")
                image_bytes = self._convert_pdf_to_image(image_bytes)
            
            # Optimiser l'image
            image_bytes = self._optimize_image(image_bytes)
            
            # Encoder en base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Préparer le prompt amélioré
            prompt = self.get_improved_prompt(formule_client)
            
            # Messages structurés
            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analysez cette facture vétérinaire et extrayez toutes les informations selon le format JSON demandé."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Appel API avec retry
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries
            )
            
            logger.info("Analyse terminée avec succès")
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {e}")
            raise

    def extract_and_validate_response(self, response: Any) -> Dict[str, Any]:
        """
        Extrait et valide la réponse JSON du LLM.
        """
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Réponse vide du LLM")
        
        content = response.choices[0].message.content
        return self._extract_and_validate_json(content)
