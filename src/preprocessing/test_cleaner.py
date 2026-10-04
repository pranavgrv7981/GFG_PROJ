from src.preprocessing.cleaner import clean_text


text = "OMG!!! @brand Check https://example.com #WorstService"


cleaned = clean_text(text)

print(cleaned)