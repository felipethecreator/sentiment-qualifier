# Sentiment‑Classifier 🇧🇷

Classificador de sentimento em português  
TF‑IDF + Regressão Logística (tweets PT‑BR).

---

## Como rodar

```bash
# clone o repositório
git clone https://github.com/felipethecreator/sentiment-classifier.git
cd sentiment-classifier

# crie ambiente virtual e instale dependências
python -m venv .venv
.\.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# treine o modelo (leva ~15 s)
python src/train.py

# inicie a interface web
streamlit run src/app.py
