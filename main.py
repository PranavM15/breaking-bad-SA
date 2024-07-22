import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from textblob import TextBlob
from itertools import combinations

file_path = '/Users/pranavm/Desktop/Sentiment Analysis on Breaking Bad Project/breaking_bad.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
filler_words = {'um', 'got', 'uh', 'like', "youre", "im", 'you', 'ah', 'er', 'mm', 'oh', 'okay', 
                'know', 'the', 'that', 'thats', 'there', 'this'}

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special chars
    text = text.lower()  # lowercase
    words = word_tokenize(text)  # tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in filler_words]  # lemmatize
    return words

df['Cleaned_Transcript'] = df['Transcript'].apply(preprocess_text)

# sentiment polarity
def get_sentiment(text):
    return TextBlob(' '.join(text)).sentiment.polarity

# apply sentiment analysis
df['Sentiment'] = df['Cleaned_Transcript'].apply(get_sentiment)

# plot sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Sentiment'], bins=20, kde=True)
plt.title('Sentiment Distribution of Breaking Bad Transcripts')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# filter by character
def filter_by_character(character, transcripts):
    return transcripts[transcripts['Transcript'].str.contains(character, case=False, na=False)]

# key characters
characters = ['Walt', 'Jesse', 'Skyler', 'Hank', 'Saul', 'Gus']

# character sentiments
character_sentiments_list = []

for character in characters:
    character_df = filter_by_character(character, df).copy()
    character_df['Character'] = character
    character_df['Character_Sentiment'] = character_df['Cleaned_Transcript'].apply(get_sentiment)
    character_sentiments_list.append(character_df)

character_sentiments = pd.concat(character_sentiments_list, ignore_index=True)

# plot character sentiment
plt.figure(figsize=(14, 8))
sns.boxplot(x='Character', y='Character_Sentiment', data=character_sentiments)
plt.title('Sentiment Analysis by Character')
plt.xlabel('Character')
plt.ylabel('Sentiment Polarity')
plt.show()

# sentiment by episode
episode_sentiment = df.groupby(['Season', 'Episode'])['Sentiment'].mean().reset_index()

# plot sentiment trend
plt.figure(figsize=(14, 8))
sns.lineplot(data=episode_sentiment, x='Episode', y='Sentiment', hue='Season', marker='o')
plt.title('Sentiment Trend Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Sentiment Polarity')
plt.legend(title='Season', loc='upper right')
plt.grid(True)
plt.show()

# load nrc emotion lexicon
nrc_lexicon_path = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
nrc_df = pd.read_csv(nrc_lexicon_path, names=['word', 'emotion', 'association'], sep='\t')
nrc_df = nrc_df.pivot(index='word', columns='emotion', values='association').reset_index()

# lexicon to dict
nrc_dict = nrc_df.set_index('word').T.to_dict('list')

# emotion intensity
def get_emotion_intensity(text):
    emotion_intensity = {emotion: 0 for emotion in nrc_df.columns if emotion != 'word'}
    for word in text:
        if word in nrc_dict:
            for emotion, score in zip(nrc_df.columns[1:], nrc_dict[word]):
                emotion_intensity[emotion] += score
    return emotion_intensity

# apply emotion scoring
df['Emotion_Intensity'] = df['Cleaned_Transcript'].apply(get_emotion_intensity)

# extract emotions
emotions = nrc_df.columns[1:]
for emotion in emotions:
    df[emotion] = df['Emotion_Intensity'].apply(lambda x: x[emotion])

# emotion by season
seasonal_emotion_intensity = df.groupby('Season')[emotions].mean()

# plot emotion changes
plt.figure(figsize=(14, 8))
for emotion in emotions:
    plt.plot(seasonal_emotion_intensity.index, seasonal_emotion_intensity[emotion], label=emotion, marker='o')
plt.xlabel('Season')
plt.ylabel('Emotion Intensity')
plt.title('Emotion Intensity Analysis of Breaking Bad by Season')
plt.legend()
plt.grid(True)
plt.show()

# generate word cloud
all_words = [word for tokens in df['Cleaned_Transcript'] for word in tokens if word not in filler_words]
word_freq = nltk.FreqDist(all_words)

wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Important Words/Names in Breaking Bad Transcripts')
plt.show()

# identify interactions
def identify_interactions(transcript, characters):
    interactions = []
    for character_pair in combinations(characters, 2):
        if all(character in transcript for character in character_pair):
            interactions.append(character_pair)
    return interactions

# interaction sentiments
interaction_sentiments_list = []

for _, row in df.iterrows():
    interactions = identify_interactions(row['Transcript'], characters)
    for interaction in interactions:
        interaction_sentiments_list.append({
            'Character1': interaction[0],
            'Character2': interaction[1],
            'Sentiment': row['Sentiment']
        })

interaction_sentiments = pd.DataFrame(interaction_sentiments_list)

# avg sentiment for pairs
average_interaction_sentiments = interaction_sentiments.groupby(['Character1', 'Character2'])['Sentiment'].mean().unstack()

# plot interaction sentiments
plt.figure(figsize=(10, 8))
sns.heatmap(average_interaction_sentiments, annot=True, cmap='coolwarm', cbar=True, linewidths=.5)
plt.title('Average Sentiment of Character Interactions')
plt.xlabel('Character 2')
plt.ylabel('Character 1')
plt.show()
