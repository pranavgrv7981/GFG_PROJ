import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_for_ml(text):
    text = clean_text(text)
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
        
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        lemmatizer = WordNetLemmatizer()

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def clean_comments(comments):
    cleaned_comments = []
    for comment in comments:
        cleaned_comment = comment.copy()
        cleaned_text = clean_text(comment.get("text", ""))
        cleaned_comment["cleaned_text"] = cleaned_text
        cleaned_comments.append(cleaned_comment)
    return cleaned_comments