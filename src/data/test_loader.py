from src.data.loader import load_comments


comments = load_comments("data/raw/test_comments.csv")

for comment in comments:
    print(comment)