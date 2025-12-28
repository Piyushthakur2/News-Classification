import json
import pandas as pd

# Category mapping
CATEGORY_MAP = {
    "POLITICS": "Politics",
    "U.S. NEWS": "Politics",
    "THE WORLDPOST": "Politics",
    "WORLD NEWS": "Politics",

    "BUSINESS": "Business",
    "MONEY": "Business",

    "SPORTS": "Sports",

    "TECH": "Technology",

    "ENTERTAINMENT": "Entertainment",
    "ARTS": "Entertainment",
    "COMEDY": "Entertainment",

    "SCIENCE": "Science",

    "HEALTH": "Health",
    "WELLNESS": "Health",

    "TRAVEL": "Lifestyle",
    "FOOD & DRINK": "Lifestyle",
    "HOME & LIVING": "Lifestyle",
    "STYLE": "Lifestyle"
}

with open("News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Combine text
df["text"] = df["headline"] + " " + df["short_description"]

# Keep only mapped categories
df = df[df["category"].isin(CATEGORY_MAP.keys())]

# Map categories
df["category"] = df["category"].map(CATEGORY_MAP)

# Keep required columns
df = df[["text", "category"]].dropna()

# Save final dataset
df.to_csv("huffpost_merged.csv", index=False)

print("Merged dataset created successfully!")
print(df["category"].value_counts())
