# Day 3: Text is Data - Turning Words into Numbers

##  Goal
Learn the core NLP preprocessing steps that turn raw text into a
format a machine learning model can actually use: tokenization,
stop word/punctuation removal, and bag-of-words representation.

##  What I Built

### Full Preprocessing Pipeline

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "AI is amazing! It is changing the world."
token = word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in token if word not in stop_words and word.isalpha()]

word_count = {}
for word in filtered_tokens:
    word_count[word] = word_count.get(word, 0) + 1

print("Text:", text)
print("Token:", token)
print("Filtered Tokens:", filtered_tokens)
print("Word Count:", word_count)
```

### Before/After Example

| Stage | Output |
|---|---|
| Raw text | `AI is amazing! It is changing the world.` |
| Tokens | `['ai', 'is', 'amazing', '!', 'it', 'is', 'changing', 'the', 'world', '.']` |
| Filtered tokens | `['ai', 'amazing', 'changing', 'world']` |
| Bag-of-words | `{'ai': 1, 'amazing': 1, 'changing': 1, 'world': 1}` |

##  Key Takeaways
- `word_tokenize()` separates punctuation into its own tokens, unlike `.split()`
- Lowercasing before tokenizing prevents capitalized stop words from being missed
- `.isalpha()` filters out punctuation/number tokens
- Bag-of-words counts word frequency but loses word order/sentence structure

## Bugs I Fixed Along the Way
- `punkt_tab` resource missing → added `nltk.download('punkt_tab')`
- Used `&&` instead of Python's `and`
- Applied `.isalpha()` to the wrong variable (the stopwords set instead of each word)
- Capitalized stop words like `'It'` weren't removed until text was lowercased first

##  Status
Day 3 of #60DaysAIChallenge — Complete