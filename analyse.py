import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from prettytable import PrettyTable

def preprocess_text(text_column):
    text_column = text_column.str.lower()
    text_column = text_column.str.replace("[^\w\s]", "")
    text_column = text_column.str.replace("\d", "")
    stop_words = stopwords.words("english")
    text_column = text_column.apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
    text_column = text_column.apply(lambda x: word_tokenize(x))
    ps = PorterStemmer()
    text_column = text_column.apply(lambda x: [ps.stem(y) for y in x])
    lemmatizer = WordNetLemmatizer()
    text_column = text_column.apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    return text_column

def create_wordcloud(data):
    all_words = []
    for line in data:
        all_words.extend(line)
    word_freq = nltk.FreqDist(all_words)
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate_from_frequencies(word_freq)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Veri seti yükleme
df = pd.read_csv("Amazon_Reviews.csv")

# Veri seti istatistikleri
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)

# İstatistik tablosu oluşturma
table = PrettyTable()
table.field_names = ["Name", "work_life_balance", "skill_development", "salary_and_benefits", "job_security", "career_growth", "work_satisfaction", "Overall_rating"]

all_names = df["Name"].unique()
for name in all_names:
    table.add_row([name,
                   df[df["Name"] == name]["work_life_balance"].mean(),
                   df[df["Name"] == name]["skill_development"].mean(),
                   df[df["Name"] == name]["salary_and_benefits"].mean(),
                   df[df["Name"] == name]["job_security"].mean(),
                   df[df["Name"] == name]["career_growth"].mean(),
                   df[df["Name"] == name]["work_satisfaction"].mean(),
                   df[df["Name"] == name]["Overall_rating"].mean()])

print(table)

# Metin sütunlarının işlenmesi ve analizi
text_columns = ["Likes", "Dislikes"]
for column in text_columns:
    df[column] = preprocess_text(df[column])
    create_wordcloud(df[column])
