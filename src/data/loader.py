import csv


def load_comments(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)