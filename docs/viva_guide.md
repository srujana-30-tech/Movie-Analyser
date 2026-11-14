# 🎓 ReviewSense: Comprehensive Viva & Technical Interview Guide

> **Project Name:** ReviewSense · Movie Review Sentiment Classifier  
> **Domain:** Natural Language Processing (NLP) & Supervised Machine Learning  
> **Target Audience:** B.Tech AI/ML Student, External Examiners, Faculty Evaluation & Placement Interviews  

---

## 1. Executive Summary & Architecture Overview

ReviewSense is an end-to-end sentiment classification system designed to categorize movie critiques into **Positive** (*Standing Ovation*) or **Negative** (*Walkout*).

### Architectural Pipeline
```
[User / IMDb Review]
        │
        ▼
[Shared Preprocessing: clean_text()]
  ├── Lowercase conversion
  ├── HTML tag stripping (<br />, etc.)
  ├── Contraction expansion (didn't -> did not)
  ├── Non-alphanumeric noise removal
  └── Stop-word removal (PROTECTING 20+ NEGATION WORDS)
        │
        ▼
[TF-IDF Vectorizer (ngram=(1,2), max_features=50000)]
  └── Fitted strictly on 80% Train Split (Leakage-Free)
        │
        ▼
[Champion Model: Logistic Regression]
  ├── Probability calibration via Sigmoid / Softmax
  └── Evaluated on held-out 20% Test Split (9,917 reviews)
        │
        ▼
[Cinematic Web Interface: "The Projection Room"]
  └── Streamlit with dynamic metrics telemetry & edge-case guards
```

---

## 2. Top Viva Questions & In-Depth Model Answers

### Q1. Why did you use TF-IDF instead of standard Bag-of-Words (CountVectorizer)?
**Answer:**  
CountVectorizer only computes the raw frequency of a word in a document. In a movie review corpus, words like *"movie"*, *"film"*, *"director"*, or *"character"* appear in almost every single review. A CountVectorizer gives these words very high weights simply because they occur frequently, even though they carry almost **zero sentiment polarity**.

**TF-IDF** (*Term Frequency · Inverse Document Frequency*) solves this by applying an inverse document penalty:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
- **Term Frequency ($\text{TF}$):** How often term $t$ appears in review $d$.
- **Inverse Document Frequency ($\text{IDF}$):** Measures how rare the word is across the entire corpus $D$.

Words like *"masterpiece"*, *"dreadful"*, or *"mesmerizing"* appear frequently in specific sentiment reviews but are rare across the general corpus. TF-IDF amplifies these distinctive sentiment tokens while suppressing generic stopwords.

---

### Q2. Why did you explicitly preserve negation words during preprocessing?
**Answer:**  
Standard stop-word lists (such as `scikit-learn` or `NLTK`) treat words like `not`, `no`, `never`, `neither`, `none`, and `cannot` as meaningless stop words. 

If you remove them indiscriminately:
- Raw text: *"The movie was not good at all."*
- Naive stop-word removal: *"movie good"*

Removing negations **inverts the polarity** of the entire sentence from distinctly negative to positive! ReviewSense constructs a custom stop-word set:
$$\text{StopWords}_{\text{custom}} = \text{StopWords}_{\text{sklearn}} \setminus \text{Negations}$$
We also normalize contractions (e.g., expanding *"didn't"* into *"did not"*) before removing punctuation, ensuring that critical polarity qualifiers are preserved into the vectorizer.

---

### Q3. What is Data Leakage in NLP, and how did you prevent it?
**Answer:**  
Data leakage occurs when information from outside the training partition (i.e., the test set) is inadvertently used to train the machine learning pipeline. In NLP, a common beginner mistake is calling `.fit_transform()` on the entire dataset *before* splitting into train and test sets.

If you fit the vectorizer on the whole dataset:
1. The vocabulary dictionary includes words exclusive to the test set.
2. The $\text{IDF}$ weights incorporate document frequency statistics from the test set.

**How ReviewSense prevents leakage:**
1. Dataset is deduplicated and cleaned first.
2. Stratified 80/20 split partitions the dataset into independent `train` and `test` collections.
3. `TfidfVectorizer.fit_transform()` is called **only on the training split**.
4. The test set is only transformed via `TfidfVectorizer.transform()`.

---

### Q4. Why is Multinomial Naive Bayes so much faster to train than Logistic Regression?
**Answer:**  
- **Multinomial Naive Bayes (0.05 seconds):** It is a generative, closed-form probabilistic model. It does not iterate or optimize an objective function using gradient descent. Instead, it simply counts word frequencies per class in a single pass over the training matrix and applies Laplace smoothing:
$$P(w_i \mid C) = \frac{N_{C, w_i} + \alpha}{N_C + \alpha |V|}$$
- **Logistic Regression (0.51 seconds):** It is a discriminative model trained via an iterative optimization algorithm (e.g., L-BFGS). It repeatedly computes gradients of the binary cross-entropy loss function over multiple epochs until convergence.

While Naive Bayes trains almost instantaneously, Logistic Regression achieved superior discriminative accuracy (90.18% vs. 88.44%) because it does not assume feature independence.

---

### Q5. Why did Logistic Regression achieve a higher F1-score than Naive Bayes?
**Answer:**  
Naive Bayes makes the **conditional independence assumption**—it assumes that the presence of word $A$ is completely independent of word $B$ given the sentiment class. In human natural language, words are strongly co-dependent (e.g., *"not"* and *"good"* form *"not good"*, which has a vastly different meaning than *"not"* and *"bad"*).

Logistic Regression learns weights $w_i$ for each individual feature through regularized cross-entropy optimization without assuming independence:
$$P(Y = 1 \mid X) = \sigma\left(\sum_{i=1}^M w_i x_i + b\right)$$
Because our TF-IDF representation included both unigrams and bigrams (`ngram_range=(1,2)`), Logistic Regression was able to calibrate weights for word pairs directly, yielding higher Precision (89.35% vs 87.94%) and F1-score (90.32% vs 88.57%).

---

### Q6. How do you interpret the Confusion Matrix for binary sentiment classification?
**Answer:**  
On our held-out test set of 9,917 reviews (4,940 Negative, 4,977 Positive):

| True \ Predicted | Predicted Negative | Predicted Positive | Total True |
| :--- | :---: | :---: | :---: |
| **Actual Negative (0)** | **4,398 (TN)** | 542 (FP) | 4,940 |
| **Actual Positive (1)** | 432 (FN) | **4,545 (TP)** | 4,977 |

- **True Negative (TN = 4,398):** Negative reviews correctly classified as Negative.
- **False Positive (FP = 542):** Negative reviews incorrectly classified as Positive (Type I error).
- **False Negative (FN = 432):** Positive reviews incorrectly classified as Negative (Type II error).
- **True Positive (TP = 4,545):** Positive reviews correctly classified as Positive.

The model demonstrates balanced performance across both classes with high diagonal concentration ($4,398 + 4,545 = 8,943$ correct predictions out of 9,917, yielding 90.18% accuracy).

---

### Q7. What is the difference between Precision, Recall, and F1-Score? Why use F1-Score to pick the best model?
**Answer:**  
- **Precision:** Out of all reviews predicted as Positive, how many were truly Positive?
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{4545}{4545 + 542} = 89.35\%$$
- **Recall:** Out of all actual Positive reviews, how many did the model identify?
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{4545}{4545 + 432} = 91.32\%$$
- **F1-Score:** The harmonic mean of Precision and Recall:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 90.32\%$$

We use F1-Score as the primary champion criterion because accuracy can be misleading in skewed distributions, and F1 penalizes extreme imbalances between false positives and false negatives.

---

### Q8. Why did you use a Stratified 80/20 train/test split?
**Answer:**  
In stratified sampling, the train and test subsets are split so that they preserve the **exact percentage of each class** found in the complete dataset.
- Clean dataset balance: 50.19% Positive, 49.81% Negative.
- Training split: 19,907 Positive (50.19%), 19,758 Negative (49.81%).
- Testing split: 4,977 Positive (50.19%), 4,940 Negative (49.81%).

This eliminates sampling bias and ensures test set performance is an honest proxy for real-world distributions.

---

### Q9. What are n-grams, and why did you use `ngram_range=(1,2)`?
**Answer:**  
An n-gram is a contiguous sequence of $n$ items from a given text.
- **Unigrams ($n=1$):** Single words (`["not", "recommended"]`).
- **Bigrams ($n=2$):** Two-word pairs (`["not recommended", "truly magnificent"]`).

Sentiment in human language is often expressed through modifiers. If you only use unigrams, the phrase *"not good"* is split into *"not"* and *"good"*. The model sees *"good"* and might assign positive weight. With bigrams, *"not good"* becomes its own token with an explicit negative weight.

---

### Q10. What does `sublinear_tf=True` do in `TfidfVectorizer`?
**Answer:**  
In standard term frequency, a word appearing 20 times in a single review receives 20 times the weight of a word appearing once. However, in human language, a review using the word *"boring"* 10 times is not 10 times more negative than a review using it twice.

`sublinear_tf=True` replaces raw term frequency with logarithmic scaling:
$$\text{TF}_{\text{sublinear}} = 1 + \log(\text{TF}) \quad \text{for } \text{TF} > 0$$
This dampens the influence of overly repetitive words and prevents outlier reviews with keyword stuffing from dominating the feature space.

---

### Q11. What is the purpose of `min_df=2` and `max_features=50000`?
**Answer:**  
- `min_df=2` (Minimum Document Frequency): Ignores words or n-grams that appear in fewer than 2 reviews across the 39,665 training examples. This eliminates rare typos, random usernames, and noise tokens that cause overfitting.
- `max_features=50000`: Limits the vocabulary to the top 50,000 most informative features ranked by term frequency. This controls memory consumption and keeps the exported vectorizer under 1 MB.

---

### Q12. What is the limitation of selecting the champion model directly on the test set, and what is the future improvement?
**Answer:**  
**Limitation:** When multiple models are evaluated on the test set and the highest performer is selected, the test set has subtly influenced model selection. In strict statistical methodology, this is called *meta-overfitting* or test set snooping.

**Future Improvement (K-Fold Cross-Validation):**  
Use 5-fold stratified cross-validation on the training set to evaluate hyperparameters and select the champion architecture. The test set should remain untouched until the single chosen model is deployed for final validation.

---

### Q13. How does `predict_proba` compute confidence in Logistic Regression?
**Answer:**  
Logistic Regression computes the logit $z = \mathbf{w}^T \mathbf{x} + b$. To convert this unbounded linear margin into a calibrated probability distribution between 0 and 1, it applies the Sigmoid function:
$$P(\text{Positive} \mid \mathbf{x}) = \frac{1}{1 + e^{-z}}, \quad P(\text{Negative} \mid \mathbf{x}) = 1 - P(\text{Positive} \mid \mathbf{x})$$
The confidence reported in the dashboard represents the maximum class probability:
$$\text{Confidence} = \max\left(P(\text{Negative}), P(\text{Positive})\right) \times 100\%$$

---

### Q14. How does the application handle empty reviews, whitespace, or pure symbols?
**Answer:**  
ReviewSense employs strict input validation guards:
1. **Empty / Whitespace Check:** If `user_input.strip() == ""`, the app intercepts execution and renders an amber "BLANK FILM STRIP" warning without invoking the model.
2. **Post-Cleaning Token Check:** If a user submits only punctuation (e.g., `"?!! ... @@@"`), the `clean_text` pipeline filters all non-alphanumeric symbols. If the cleaned string is empty, the app surfaces an "UNINTELLIGIBLE AUDIO REEL" alert.

---

### Q15. How does Streamlit state management work for the Clear and Preset buttons?
**Answer:**  
Streamlit re-executes the Python script from top to bottom on each user interaction. To maintain text across runs and allow programmatic reset, ReviewSense utilizes `st.session_state`:
- Key: `st.session_state.review_text`
- When a Preset button is clicked, `st.session_state.review_text` is updated with the sample review.
- When "CLEAR" is clicked, the state is cleared (`""`) and `st.rerun()` forces an immediate clean re-render.
