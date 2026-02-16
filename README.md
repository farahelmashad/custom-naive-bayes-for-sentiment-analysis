# Tweet Sentiment Classification - Naive Bayes from Scratch

Binary sentiment classifier (positive/negative) built on tweet data, with a full NLP preprocessing pipeline and a Multinomial Naive Bayes implementation written from scratch in NumPy.

---

## What this does

Takes raw tweets, runs them through a preprocessing pipeline, vectorizes the text, and classifies sentiment using a custom Naive Bayes model. The custom model is then compared directly against sklearn's `MultinomialNB` to validate the implementation.

**Final accuracy: 87% on the test set**; custom model and sklearn match exactly.

---


## Preprocessing pipeline


1. **Encoding fixes** — `ftfy` to repair garbled unicode
2. **HTML removal** — BeautifulSoup
3. **URL removal** — regex
4. **Mention removal** — `@username` stripped
5. **Hashtag handling** — `#` symbol removed, word kept
6. **Emoji conversion** — converted to text descriptions using the `emoji` library rather than removed, to preserve sentiment signal
7. **Contraction expansion** — `contractions` library, applied before punctuation removal so `can't` → `cannot` not `cant`
8. **Punctuation removal**
9. **Lowercasing**
10. **Number removal**
11. **Repeated character reduction** — `loooove` → `loove`
12. **Whitespace normalization**
13. **Tokenization** — `nltk.word_tokenize`
14. **Stopword removal**  NLTK English list, with negation words (`not`, `no`, `nor`, `never`, etc.) explicitly kept since they carry sentiment signal
15. **Lemmatization**  NLTK `WordNetLemmatizer` with POS tagging so verbs lemmatize correctly (`running` → `run`, not `running`)

---

## Naive Bayes implementation

Multinomial Naive Bayes implemented from scratch in NumPy.

---


## Results

|Model|Precision|Recall|F1|Accuracy|
|---|---|---|---|---|
|Custom NaiveBayes (α=2.0)|0.87|0.87|0.87|87%|
|sklearn MultinomialNB (α=2.0)|0.87|0.87|0.87|87%|
|sklearn MultinomialNB + TF-IDF|0.87|0.87|0.87|87%|

Custom implementation matches sklearn exactly; same predictions on every test example.
