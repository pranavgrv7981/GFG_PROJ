import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_comments(comments):
    cleaned_comments = []

    for comment in comments:
        cleaned_comment = comment.copy()
        cleaned_comment["text"] = clean_text(comment["text"])
        cleaned_comments.append(cleaned_comment)

    return cleaned_comments