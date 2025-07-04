# Pennypet Invoice LLM Demo

## Description
This repository contains a minimal proof-of-concept prototype for automating veterinary invoice processing using OCR and a large language model (LLM). The Streamlit-based demo allows uploading PDF or image invoices, extracts text via Tesseract OCR, parses and structures key fields with OpenAI’s GPT-4, and exports results in JSON or CSV.

## Features
- **Multi-format upload**: PDF (native or scanned), JPG, PNG.
- **OCR extraction**: Uses Tesseract via pytesseract.
- **LLM parsing**: GPT-4 prompts to extract invoice fields (invoice number, dates, provider, client, line items, totals).
- **Normalization & validation**: Glossary-based mapping for codes, VAT rates, and field consistency checks.
- **Interactive UI**: Streamlit app showing original document, raw OCR text, and parsed JSON.
- **Export options**: Download parsed data as JSON or CSV.

## Quick Start

1. Clone the repository  
   \`\`\`
   git clone git@github.com:/pennypet-invoice-llm-demo.git
   cd pennypet-invoice-llm-demo
   \`\`\`

2. Create and activate a virtual environment  
   \`\`\`
   python3 -m venv .venv
   source .venv/bin/activate       # macOS/Linux
   .\\.venv\\Scripts\\Activate.ps1    # Windows PowerShell
   \`\`\`

3. Install dependencies  
   \`\`\`
   pip install --upgrade pip
   pip install -r requirements.txt
   \`\`\`

4. Copy and configure environment variables  
   \`\`\`
   cp .env.example .env
   # Edit .env with your OpenAI API key and Tesseract cmd path
   \`\`\`

5. Run the Streamlit demo  
   \`\`\`
   streamlit run ui/app.py --server.port \$STREAMLIT_SERVER_PORT
   \`\`\`

6. Open your browser at \`http://localhost:8501\` and upload a veterinary invoice.

## Project Structure
\`\`\`
pennypet-invoice-llm-demo/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── .env.example
├── main.py
├── ocr_module/
│   └── ocr.py
├── llm_parser/
│   └── parser.py
├── ui/
│   └── app.py
└── tests/
    └── test_pipeline.py
\`\`\`

## Branch Strategy
- **main**: Stable demo release.
- **feature/ocr-module**: Tesseract integration.
- **feature/llm-parser**: LLM prompting and parsing.
- **feature/ui-streamlit**: Streamlit UI.
- **feature/tests**: Test suite and validation scripts.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details." > README.md

echo "# Byte-compiled / optimized
__pycache__/
*.py[cod]
*\$py.class

# Virtual environments
.venv/
env/
venv/
ENV/

# Environment variables
.env
.envrc

# Distribution / packaging
build/
dist/
*.egg-info/
*.eggs/

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage
htmlcov/
.tox/
.nox/
.coverage*
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints/

# Streamlit
.streamlit/

# IDEs and editors
.vscode/
.idea/

# Logs and exports
logs/
exports/

# Miscellaneous
.DS_Store
.pypirc" > .gitignore

echo "MIT License

Copyright (c) $(date +%Y) Pennypet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full MIT License text here]" > LICENSE

git add README.md .gitignore LICENSE
git commit -m "chore: initialize repository with README, .gitignore, and MIT license"
