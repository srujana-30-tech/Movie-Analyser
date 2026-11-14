# Dataset Directory

Place `IMDB Dataset.csv` (50,000 IMDb movie reviews) in this folder if you wish to run `python train.py` and retrain the models from scratch.

- **Dataset Source:** [IMDb Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Production Note:** Pre-trained champion model weights (`best_sentiment_model.pkl`) and the vocabulary vectorizer (`tfidf_vectorizer.pkl`) are already tracked and included in the `models/` directory, allowing `streamlit run app.py` to run immediately without needing the raw 66MB CSV file.
