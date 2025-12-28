import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLP resources
nltk.download('punkt')
nltk.download('stopwords')

# Load NEW dataset
data = pd.read_csv("huffpost_merged.csv").sample(30000, random_state=42)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data["clean_text"] = data["text"].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(data["clean_text"])
y = data["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Train improved model
from sklearn.naive_bayes import MultinomialNB

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

print("Logistic Regression Accuracy:", lr_acc)
print("Naive Bayes Accuracy:", nb_acc)

# Choose best model
if nb_acc > lr_acc:
    best_model = nb_model
    print("Naive Bayes selected as best model")
else:
    best_model = lr_model
    print("Logistic Regression selected as best model")

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Best model trained & saved!")


