import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import adam_v2
from keras.callbacks import ReduceLROnPlateau

porter = PorterStemmer()  # Used to lemmatize tokens
lemmatizer = WordNetLemmatizer()

data = pd.read_csv('C:/Users/Freddy Mendoza/Desktop/final-project/Train.csv')
print(data.shape)

def token_prtr(text):
    return [porter.stem(word) for word in text.split()]

def token_lemma(text):
    return [lemmatizer.lemmatize(word) for word in text.split()]

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None, ngram_range=(1,2))
tfidf_prtr = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=token_prtr, use_idf=True,smooth_idf=True,ngram_range=(1,2))
tfidf_lemma = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=token_lemma, ngram_range=(1,2))
count = CountVectorizer(strip_accents=None,lowercase=False,preprocessor=None,ngram_range=(1,2))
count_prtr = CountVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=token_prtr,ngram_range=(1,2))
count_lemma = CountVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=token_lemma,ngram_range=(1,2))

y = data.label.values
x_tfidf = tfidf.fit_transform(data.text)
x_tfidf_prtr = tfidf_prtr.fit_transform(data.text)
x_tfidf_lemma = tfidf_lemma.fit_transform(data.text)
x_cv = count.fit_transform(data.text)
x_cv_prtr = count_prtr.fit_transform(data.text)
x_cv_lemma = count_lemma.fit_transform(data.text)

def clf_logregCV(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=1, test_size=0.5, shuffle = False)

    clf = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0, n_jobs=-1, verbose=3, max_iter=500).fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
    print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))
    print("Precision Recall Curve:", metrics.PrecisionRecallDisplay.from_predictions(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))

def clf_logreg(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=1, test_size=0.5, shuffle = False)

    clf = LogisticRegression(random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))


def glove_attempt(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=0)

    X_temp = X_test
    y_temp = y_test

    max_feats = 10000
    max_len = 128

    tknzr = text.Tokenizer(num_words=max_feats)
    tknzr.fit_on_texts(X_train)

    tknzd_train = tknzr.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(tknzd_train, maxlen=max_len)

    tknzd_test = tknzr.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(tknzd_test, maxlen=max_len)

    glove_file = "C:/Users/Freddy Mendoza/Desktop/final-project/glove.subset.50d.txt"
    embed_dict = dict(get_glove(*o.rstrip().rsplit(' ')) for o in open(glove_file))

    embs = np.stack(embed_dict.values())
    emb_mean, emb_std = embs.mean(), embs.std()
    emb_size = embs.shape[1]

    word_index = tknzr.word_index
    num_words = min(max_feats, len(word_index))

    emb_matrix = np.random.normal(emb_mean, emb_std, (num_words, emb_size))

    for word, i in word_index.items():
        if i >= num_words: continue
        emb_vector = embed_dict.get(word)
        if emb_vector is not None:
            emb_matrix[i] = emb_vector
        
    batch_size = 256
    epochs = 10
    emb_size = 50

    learn_rate_reduct = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    model = Sequential()

    model.add(Embedding(max_feats, output_dim=emb_size, weights=[emb_matrix], input_length=max_len, trainable=False))
    model.add(Bidirectional(LSTM(units=128)))
    model.add(Dropout(rate=0.8))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=adam_v2.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[learn_rate_reduct])
    print(history)


def get_glove(word, *arr):
    return word, np.asarray(arr, dtype='float32')


clf_logregCV(x_tfidf, y)
