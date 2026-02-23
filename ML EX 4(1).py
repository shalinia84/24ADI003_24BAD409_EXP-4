print("A Shalini-24BAD409")
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\SHALINI A\Downloads\archive (14).zip", encoding="latin-1")[["v1","v2"]]
df.columns = ["label","text"]

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["text"] = df["text"].apply(clean)

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham","Spam"],
            yticklabels=["Ham","Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

misclassified = X_test[y_test != y_pred]
print("\nMisclassified Samples:")
print(misclassified.head(10))

model_smooth = MultinomialNB(alpha=0.1)
model_smooth.fit(X_train_vec, y_train)
smooth_pred = model_smooth.predict(X_test_vec)

print("\nWith Laplace Smoothing (alpha=0.1)")
print("Accuracy:", accuracy_score(y_test, smooth_pred))
print("Precision:", precision_score(y_test, smooth_pred))
print("Recall:", recall_score(y_test, smooth_pred))
print("F1:", f1_score(y_test, smooth_pred))

feature_names = np.array(vectorizer.get_feature_names_out())
spam_probs = model.feature_log_prob_[1]
top_spam = feature_names[np.argsort(spam_probs)[-15:]]

plt.figure(figsize=(8,4))
plt.barh(top_spam, np.sort(spam_probs)[-15:], color="purple")
plt.title("Top Words Indicating Spam")
plt.show()

spam_words = " ".join(df[df.label==1]["text"]).split()
ham_words = " ".join(df[df.label==0]["text"]).split()

spam_freq = pd.Series(spam_words).value_counts()[:15]
ham_freq = pd.Series(ham_words).value_counts()[:15]

combined = pd.DataFrame({
    "Spam": spam_freq,
    "Ham": ham_freq
}).fillna(0)

combined = combined.sort_values(by="Spam", ascending=True)

plt.figure(figsize=(10,6))
plt.barh(combined.index, combined["Spam"], color="red", alpha=0.7, label="Spam")
plt.barh(combined.index, combined["Ham"], color="green", alpha=0.7, label="Ham")

plt.xlabel("Frequency")
plt.title("Spam vs Ham Word Frequency Comparison")
plt.legend()
plt.tight_layout()
plt.show()
