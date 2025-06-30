import streamlit as st
from pysentimiento import create_analyzer

@st.cache_resource
def load_model():
    return create_analyzer(task="sentiment", lang="pt")

analyzer = load_model()

st.title("Classificador de Sentimento (BERT)")

texto = st.text_area("Digite um texto para analisar", height=150)

if st.button("Analisar") and texto.strip():
    result = analyzer.predict(texto)
    label  = result.output

    if label == "POS":
        st.success("Positivo 😊")
    elif label == "NEG":
        st.error("Negativo 🙁")
    else:
        st.info("Neutro 😐")

    st.write({k: f"{v:.2%}" for k, v in result.probas.items()})
