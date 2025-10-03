
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv.zip",
    encoding="latin-1",
    header=None,
    usecols=[0, 5]  
)
df.columns = ["polarity", "text"]

df = df[df.polarity != 2]
df["polarity"] = df["polarity"].map({0: 0, 4: 1})

print("Class distribution:\n", df["polarity"].value_counts())

def clean_text(text):
    return text.lower().strip()

df["clean_text"] = df["text"].astype(str).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["polarity"], test_size=0.2, random_state=42, stratify=df["polarity"]
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

vectorizer = TfidfVectorizer(
    max_features=10000,  
    ngram_range=(1, 2),   
    stop_words="english"  
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape (train):", X_train_tfidf.shape)
print("TF-IDF shape (test):", X_test_tfidf.shape)

models = {
    "BernoulliNB": BernoulliNB(),
    "LinearSVC": LinearSVC(max_iter=5000),
    "LogReg": LogisticRegression(max_iter=200, solver="liblinear")  
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

sample_tweets = ["I love this!", "I hate that!", "It was okay, not great."]
sample_vec = vectorizer.transform(sample_tweets)

print("\nSample Predictions:")
for name, model in models.items():
    print(f"{name}: {model.predict(sample_vec)}")
