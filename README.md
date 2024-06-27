Developed a sequence classifier prototype to label abbreviations and long forms in scientific literature using the BIO format. Conducted experiments with various preprocessing techniques, algorithms (SVM, HMM, CRF, RNN, FFNN), and text encodings (tf-idf, word2vec, GloVe). Evaluated models using F1-score, precision, recall, and ROC curves. Deployed a web service for the model, implemented performance testing, and basic monitoring.

Repository Contents:

Visualization.ipynb: Visualization notebook.

Experiment 1.ipynb: Comparison of vectorization techniques and 1D Convolutional Neural Network (1D CNN).

Experiment 2.ipynb: Comparison of 1D CNN and BiLSTM models with TF-IDF vectorizer.

Experiment 3.ipynb: Comparison of grid search and random search techniques with 1D CNN model.

Experiment 4.ipynb: Comparison of categorical crossentropy and sparse categorical crossentropy with 1D CNN model.

NLP Individual CW - 6835716 Wilson.pdf: Individual report.

app.py, data.json, requirements.txt, testing.ipynb, model_predictions.log: Deployment files.

Group37_Report.pdf: Final group report.
