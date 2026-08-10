# Day 7: Your First Classifier

## Goal
Train a real sentiment classifier using scikit-learn — the same core
pattern behind spam detection, content moderation, and recommendation
filtering.

## What I Built

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

reviews_text = [
    # 15 positive reviews, 15 negative reviews
    ...
]
reviews_labels = [1]*15 + [0]*15

reviews = CountVectorizer(stop_words='english')
countVector = reviews.fit_transform(reviews_text)

model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(
    countVector, reviews_labels, test_size=0.3, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Testing on 5 Self-Written Sentences

```python
new_sentences = [
    "The movie had brilliant acting and a wonderful story.",
    "The movie was dull and the story was poorly written.",
    "I really enjoyed the beautiful music and engaging plot.",
    "The ending was disappointing and the characters were weak.",
    "The movie had good acting but the story was quite boring."
]
new_vector = reviews.transform(new_sentences)  # transform, not fit_transform!
predictions = model.predict(new_vector)
```

**Results:** 4 out of 5 correct. The 5th (mixed-sentiment) sentence was
predicted negative — a genuinely ambiguous case even for a human reader.

##  Reflection

With a small dataset (only 20-30 sentences), the model's accuracy was
unstable — it changed noticeably (from 0.5 to 0.167) just by adjusting
test_size, showing that small datasets produce unreliable results. When
I tested a mixed-sentiment sentence ("The movie had good acting but the
story was quite boring"), the model predicted negative, even though the
sentence had one positive part and one negative part. This shows that a
simple bag-of-words model can get confused by mixed or conflicting
feelings — it doesn't understand nuance, it just weighs the words it sees.

##  Bugs I Fixed Along the Way
- Called `CountVectorizer` as a method on my list instead of creating it separately
- Mismatched list lengths between reviews_text (20) and reviews_labels (17) — a silent bug
- Forgot to store `.predict()` output in a variable, causing a NameError
- Used `.fit_transform()` on new test sentences instead of `.transform()`, causing a feature mismatch error

##  Key Takeaways
- Small datasets cause unstable, unreliable accuracy scores
- `.transform()` reuses a fitted vectorizer's vocabulary; `.fit_transform()` rebuilds it from scratch
- Bag-of-words models struggle with sentences containing mixed sentiment
- A single accuracy number can be misleading — always sanity-check with manual test cases

## Status
Day 7 of #60DaysAIChallenge — Complete | Sprint 1 (Days 1-7) Complete 