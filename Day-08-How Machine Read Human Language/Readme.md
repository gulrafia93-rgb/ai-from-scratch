# Day 8: How Machines Read Human Language

##  Goal
Build a complete text preprocessing pipeline that converts noisy,
real-world text (social media, news, reviews) into clean, structured,
machine-readable tokens using tokenization, stopword removal, regex-based
punctuation stripping, stemming, and lemmatization.

## What I Built

Three raw text samples with realistic messiness:
- **Social media:** emojis, hashtags, inconsistent casing
- **News article:** formal tone, proper casing
- **Customer review:** casual tone, mixed punctuation

```python
import nltk, re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

samples = {"Social Media": socialMedia, "News": news, "Review": reviews}

for name, text in samples.items():
    word_tokens = word_tokenize(text)
    sent_tokens = sent_tokenize(text)

    before_vocab = len(set(word_tokens))
    filtered = [w for w in word_tokens if w.lower() not in stop_words]
    after_vocab = len(set(filtered))

    # Punctuation, emojis, and symbols carry no useful meaning for word-counting
    # or similarity-based models, and can cause the same word to be treated as
    # different tokens (e.g. "good" vs "good!" vs "good,") if not removed.
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_tokens = word_tokenize(cleaned_text)

    stemmed = [stemmer.stem(w) for w in cleaned_tokens]
    lemmatized = [lemmatizer.lemmatize(w) for w in cleaned_tokens]

    print(name, "- Vocab before:", before_vocab, "| after:", after_vocab)
    print(name, "- Stemmed:", stemmed)
    print(name, "- Lemmatized:", lemmatized)
```

## Vocabulary Size: Before vs After Stopword Removal

| Sample | Before | After |
|---|---|---|
| Social Media | 25 | 17 |
| News | 28 | 19 |
| Review | 27 | 14 |

## Stemming vs Lemmatization Examples

| Original | Stemmed | Lemmatized |
|---|---|---|
| really | realli | really |
| government | govern | government |
| delivery | deliveri | delivery |
| technology | technolog | technology |
| was | wa | wa (lemmatizer defaults to noun mode) |

## Explanation: Why Each Step Matters

**Tokenization:** Without breaking text into tokens, no per-word operation
(stopword removal, stemming, lemmatization) is possible — the model would
just see one unstructured string.

**Stopword removal:** Reduced vocabulary size significantly (25→17, 28→19,
27→14). Common filler words carry little distinguishing meaning and waste
model capacity if kept.

**Regex punctuation removal:** Successfully stripped emojis and hashtag
symbols from the social media sample. Without this, "good!!" and "good"
would be treated as different tokens, fragmenting the vocabulary.

**Stemming vs lemmatization:** Stemming produced non-dictionary words
(realli, deliveri, technolog) — fast but rough. Lemmatization preserved
real words but isn't perfect either (was → wa), since it assumes nouns
by default without part-of-speech tagging.

## Bug I Fixed
- Vocabulary comparison code was outside the loop, only calculating for the last sample instead of all three — fixed by correcting indentation

## Status
Day 8 of #60DaysAIChallenge — Complete