import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from keras.models import load_model
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = load_model('task1_model.h5')
sample_tweet = "A lot of good things are happening." \
               " We are respected again throughout the world, " \
               "and that's a great thing.@realDonaldTrump"
tweet_df = pd.DataFrame([[sample_tweet]])

tweet_df[0] = tweet_df[0].apply(lambda x: x.lower())
tweet_df[0] = tweet_df[0].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in tweet_df.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(tweet_df[0].values)
X = tokenizer.texts_to_sequences(tweet_df[0].values)
X = pad_sequences(X, maxlen=28)
print('\n\n\n\n')

pred = model.predict(X)
sentiment = np.argmax(np.squeeze(pred))
print(sentiment)

if sentiment == 0:
    print('\npredicted result: Negative')
elif sentiment == 1:
    print('\npredicted result: Neutral')
elif sentiment == 2:
    print('\npredicted result: Positive')

