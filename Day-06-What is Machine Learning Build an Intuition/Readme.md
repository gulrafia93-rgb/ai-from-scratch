# Day 6: What is Machine Learning? Build an Intuition

##  Goal
Build hands-on intuition for how ML models learn from data, using
Google's Teachable Machine to train an image classifier and observe
how data quantity and balance affect accuracy.

##  What I Built
Trained a 2-class image classifier (Thumbs Up vs Thumbs Down) using
webcam images, across 3 rounds of experimentation.

##  Experiment Results

| Round | Data | Result |
|---|---|---|
| Round 1 | 4 samples per class | Weak accuracy — only 64% confidence |
| Round 2 | ~23-24 samples per class (balanced) | Strong accuracy — 100% confidence |
| Round 3 | 41 vs 23 samples (imbalanced) | Model became biased — predicted the wrong gesture with 99% confidence |

## Observations

1. With very few training samples (4 per class), the model's accuracy
   was weak — only 64% confident, barely better than a guess.
2. After adding more samples evenly to both classes (23-24 each),
   accuracy jumped to 100%.
3. When the dataset became unbalanced (41 vs 23 samples), the model
   became biased toward the larger class — it confidently (99%)
   predicted the wrong gesture. This showed that more data alone
   isn't enough; the data also needs to be balanced across classes,
   or the model develops a systematic bias.

##  Key Takeaways
- More training data generally improves accuracy — but only if it's balanced
- Class imbalance causes models to become biased toward the majority class
- A model can be highly confident and still wrong — confidence is not the same as correctness
- Real-world impact: imbalanced data is a serious risk in fraud detection, medical diagnosis, and other high-stakes ML applications

##  Status
Day 6 of #60DaysAIChallenge — Complete