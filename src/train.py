import pathlib, joblib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ─── 1. Carregar tweets PT‑BR ────────────────────────────────
def load_tweets_pt() -> pd.DataFrame:
    ds = load_dataset(
        "cardiffnlp/tweet_sentiment_multilingual",
        name="portuguese",
        split="train"
    )
    df = ds.to_pandas()[["text", "label"]]

    df = df[df["label"] != 1]
    df["sentiment"] = df["label"].map({0: "negative", 2: "positive"})
    df = df.rename(columns={"text": "review"})
    return df[["review", "sentiment"]]

# ─── 2. Paths ────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─── 3. Treinar e salvar ────────────────────────────────────
def main():
    df = load_tweets_pt()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["review"], df["sentiment"],
        test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=40_000,
            stop_words=None
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    pipe.fit(X_tr, y_tr)
    print(classification_report(y_te, pipe.predict(X_te)))
    joblib.dump(pipe, MODEL_DIR / "sentiment_model.joblib")
    print("✅  Modelo salvo em models/sentiment_model.joblib")

if __name__ == "__main__":
    main()
