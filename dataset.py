import pandas as pd
from textblob import TextBlob  # Make sure to install the 'textblob' library if not already installed

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('mental_health_dataset.csv')

# Define a function to analyze speech content and map to emotions
def map_speech_to_emotion(speech_content):
    analysis = TextBlob(str(speech_content))
    sentiment_score = analysis.sentiment.polarity

    if sentiment_score > 0.2:
        return 'Joyful'
    elif -0.2 <= sentiment_score <= 0.2:
        return 'Neutral'
    elif sentiment_score < -0.2:
        return 'Melancholic'
    else:
        return 'Unknown'

# Apply the mapping function to update the 'Emotion' column based on 'Speech Pattern'
df['Emotion'] = df['Speech Pattern'].apply(map_speech_to_emotion)

# Save the updated DataFrame back to the CSV file
df.to_csv('mental_health_dataset_updated.csv', index=False)
