import os
import csv
import random
import uuid

BRANDS = ["Amazon", "Flipkart", "Swiggy", "Zomato", "PhonePe", "Paytm"]
PLATFORMS = ["twitter", "instagram", "facebook", "reddit"]
CATEGORIES = [
    "delivery_issues",
    "product_quality",
    "customer_service",
    "billing_issues",
    "app_bugs",
    "refund_problems"
]

POSITIVE_TEMPLATES = [
    "Love using {brand}! Always reliable and fast. 💯",
    "{brand} is the absolute best! Delivery was super quick. https://example.com/xyz",
    "Just had an amazing experience with @{brand}. Thanks for the great service! #happy",
    "Highly recommend {brand} for everyone! Great delivery and lovely packaging. #awesome",
    "Great app, smooth UI and seamless checkout. Keep it up {brand}.",
    "Absolutely fantastic service and prompt support from {brand} today.",
    "Thank you {brand} for the quick resolution and courteous team 😊",
    "Best platform ever, @{brand} you guys rock! Five stars! ✨",
    "Using {brand} makes my daily routine so much easier and pleasant.",
    "Five stars for {brand}! Loved the fast response and on-time order.",
    "Super impressed with @{brand}! The driver was polite and order arrived early.",
    "Outstanding customer support from {brand}. Solved my issue within 5 minutes!",
    "Everything arrived in perfect condition. Thank you @{brand}!",
    "Great delivery by @{brand}, loved it! Super fast and reliable. #satisfied",
    "Kudos to the {brand} team for always delivering top quality products.",
    "Seamless payment experience on {brand}. Really happy with the new update.",
    "Delicious food and timely delivery by {brand}. Will order again!",
    "Very good experience with {brand}. Easy to use, fast refunds when requested.",
    "Loved the discount and lightning fast shipping on {brand} today! 🎉",
    "Pleasantly surprised by how quickly {brand} handled my inquiry."
]

NEUTRAL_TEMPLATES = [
    "Does anyone know how to change payment method on {brand}?",
    "Just downloaded the latest {brand} app update.",
    "I think {brand} changed their user interface today.",
    "Waiting for my scheduled delivery from {brand}.",
    "What are good alternative platforms to {brand}?",
    "Is {brand} servers down for anyone else right now?",
    "Trying out {brand} for the first time this week.",
    "Can you pay with crypto or UPI credit on {brand}?",
    "Saw an advertisement for {brand} on YouTube yesterday.",
    "@{brand} please check your direct messages regarding my ticket.",
    "How long does standard delivery take on {brand}?",
    "Just comparing prices between {brand} and other apps.",
    "Has anyone tried the new premium subscription on {brand}?",
    "Wondering if {brand} delivers to rural postal codes.",
    "Notice: {brand} scheduled maintenance tonight at midnight."
]

NEGATIVE_TEMPLATES = {
    "delivery_issues": [
        "My order from {brand} is 3 days late! 😡 #fail #terrible",
        "Never got my delivery from @{brand}. Worst service ever.",
        "Delivery driver from {brand} was extremely rude and unhelpful.",
        "Tracking status says delivered but I received nothing from {brand}.",
        "Late delivery again, {brand} is so inconsistent and delayed.",
        "Package arrived completely damaged due to careless handling by {brand}.",
        "Waited 4 hours for my food delivery from @{brand}. Disgraceful delay."
    ],
    "product_quality": [
        "The item I received from {brand} is broken and counterfeit.",
        "Terrible quality from {brand}. Defective product right out of the box.",
        "Not what I expected from {brand}. Looks fake and cheap.",
        "Food from {brand} was cold, stale, and smelled awful.",
        "The packaging from @{brand} was completely crushed and leaking.",
        "Received a spoiled item from {brand}. Horrible quality check."
    ],
    "customer_service": [
        "{brand} customer support is totally unresponsive and useless.",
        "Been stuck on hold for 45 mins with {brand} phone support.",
        "The automated bot on {brand} is useless, can never speak to a human.",
        "Why is it impossible to get help from a representative at {brand}?",
        "Worst customer service experience ever with @{brand}. Rude agents.",
        "Support representative closed my ticket without resolving anything on {brand}."
    ],
    "billing_issues": [
        "{brand} charged my account twice for the exact same order.",
        "Hidden fees and unauthorized charges on {brand}? Disgusting practice.",
        "My billing invoice from {brand} has duplicate charges and incorrect tax.",
        "Why did my subscription for {brand} auto-renew after I cancelled?",
        "Overcharged by {brand} on checkout and support refuses to adjust the bill.",
        "Payment failed but the money was deducted from my bank by {brand}."
    ],
    "app_bugs": [
        "The {brand} app keeps crashing constantly when I try to pay.",
        "Can't log into my {brand} account, getting internal error 500.",
        "{brand} website is completely broken on checkout page.",
        "Fix your buggy and glitchy app @{brand}! Buttons don't respond.",
        "App freezes every time I open the search tab on {brand}.",
        "Critical bug in {brand} app: cart empties itself on checkout."
    ],
    "refund_problems": [
        "Still waiting for my pending refund from {brand} after 3 weeks.",
        "{brand} arbitrarily rejected my valid return and refund claim.",
        "Where is my money @{brand}? Refund not processed after return #scam",
        "Refund marked processed but not reflecting in my bank account from {brand}.",
        "I demand my money back {brand}! Fraudulent refusal to refund.",
        "Returned the order 10 days ago, still no sign of refund from {brand}."
    ]
}


def generate_comment(is_train: bool = False) -> dict:
    sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.40, 0.25, 0.35])[0]
    brand = random.choice(BRANDS)
    platform = random.choice(PLATFORMS)

    category = ""
    if sentiment == "positive":
        text = random.choice(POSITIVE_TEMPLATES).format(brand=brand)
    elif sentiment == "neutral":
        text = random.choice(NEUTRAL_TEMPLATES).format(brand=brand)
    else:
        category = random.choice(CATEGORIES)
        text = random.choice(NEGATIVE_TEMPLATES[category]).format(brand=brand)

    if random.random() < 0.1:
        text = text.lower()
    if random.random() < 0.05:
        text = text.replace("!", "!!1!")

    if is_train:
        return {"text": text, "sentiment": sentiment}

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "platform": platform,
        "brand": brand,
        "sentiment": sentiment,
        "complaint_category": category
    }


def main():
    random.seed(42)

    base_dir = r"c:\Users\prana\Desktop\gfg"
    synthetic_dir = os.path.join(base_dir, "data", "synthetic")
    processed_dir = os.path.join(base_dir, "data", "processed")

    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    comments_file = os.path.join(synthetic_dir, "comments.csv")
    with open(comments_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "text", "platform", "brand", "sentiment", "complaint_category"]
        )
        writer.writeheader()
        for _ in range(800):
            writer.writerow(generate_comment())

    train_file = os.path.join(synthetic_dir, "sentiment_train.csv")
    with open(train_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "sentiment"])
        writer.writeheader()
        for _ in range(1200):
            writer.writerow(generate_comment(is_train=True))

    print(f"Generated synthetic comments at {comments_file}")
    print(f"Generated sentiment training data at {train_file}")


if __name__ == "__main__":
    main()
