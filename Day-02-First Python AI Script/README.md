# Day 2: Your First Python AI Script

##  Goal
Get comfortable running Python in Google Colab and build the first two
building blocks of text preprocessing used in real NLP/AI pipelines:
word frequency counting and text cleaning.

## What I Built

### 1. Word Frequency Counter
Takes a sentence, splits it into words, and counts how many times each
word appears using a dictionary.

```python
text = input("enter text: ")
words = text.lower().split()
word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1
print("\nWord Frequency:")
for word, count in word_count.items():
    print(f"{word}: {count}")
```

### Text Cleaner
Takes messy input text and returns a cleaned version: lowercase,
no punctuation, and normalized whitespace.

```python
import string
text = input("enter text: ")
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))
text_cleaned = " ".join(text.split())
print("cleaned text:")
print(text_cleaned)
```

## Key Takeaways
- Real-world text is messy — cleaning it is always the first step before any AI model processes it
- `dict.get(key, default)` is a clean way to count things without extra if/else checks
- `str.maketrans()` + `.translate()` is an efficient way to strip unwanted characters
- `" ".join(text.split())` is the standard trick to normalize whitespace

## Status
Day 2 of #60DaysAIChallenge — Complete