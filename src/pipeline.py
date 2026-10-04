from src.data.loader import load_comments
from src.preprocessing.cleaner import clean_comments


def run_pipeline(filepath):
    comments = load_comments(filepath)
    cleaned_comments = clean_comments(comments)

    return cleaned_comments


if __name__ == "__main__":
    result = run_pipeline("data/raw/test_comments.csv")

    for comment in result:
        print(comment)