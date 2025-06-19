# MBTI Personality Prediction Using Machine Learning

This project predicts Myers-Briggs Type Indicator (MBTI) personality types from text data using machine learning models. It is based on a dataset collected from social media posts labeled with MBTI types.

## Objective
Predict one of the 16 MBTI personality types (like INTJ, ENFP, ISTP, etc.) using natural language processing (NLP) and classification algorithms on users' text data.

## Dataset
- The dataset used is: [(MBTI) Myers-Briggs Personality Type Dataset from Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- It contains approximately 8,600 posts labeled with 16 MBTI types.
- Due to GitHub's file size limit, the full dataset is not stored in this repository.
- You can download it directly from the Kaggle link above.

## Files in This Repository
- `mbti_personality_prediction.ipynb`: The main Colab notebook containing:
  - Data loading and preprocessing
  - Text vectorization using TF-IDF or CountVectorizer
  - Label encoding of MBTI types
  - Model training (e.g., Logistic Regression, Naive Bayes)
  - Evaluation metrics (accuracy, confusion matrix)

## Technologies Used
- Python
- Google Colab
- Pandas, NumPy
- Scikit-learn
- NLTK or spaCy (if used)
- Matplotlib or Seaborn

## How to Run
1. Open the notebook in Google Colab.
2. Download and upload the dataset from the Kaggle MBTI Dataset link.
3. Run each cell to train and evaluate the model.

## Results
- The model attempts to classify users into one of the 16 MBTI types based on their writing style and word usage.
- Accuracy scores and confusion matrix visualizations are shown in the notebook output.

## Future Improvements
- Use more advanced models like BERT or RNNs
- Perform personality trait–wise prediction (e.g., I/E, S/N, T/F, J/P)
- Add sentiment analysis or topic modeling for deeper insights

## Contact
For queries or suggestions, feel free to open an issue in this repository or fork the project to experiment with your own models.
