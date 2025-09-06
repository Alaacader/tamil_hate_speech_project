from sklearn.feature_extraction.text import TfidfVectorizer

# Hybrid word + char TF-IDF often works well for short, noisy social text

def build_tfidf_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),  # char n-grams
        min_df=3,
        max_features=100_000,
    )

def build_word_tfidf_vectorizer():
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),  # unigrams+bigrams
        min_df=3,
        max_features=100_000,
    )
