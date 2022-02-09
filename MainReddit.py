import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
RedditData=pd.read_csv("train-balanced-sarcasm.csv",sep=",", dtype="string")


na_comments = RedditData["comment"].isna()
na_comments = RedditData[na_comments].index
RedditData.drop(labels = na_comments,axis = 0, inplace = True)
desired_var = ['comment', 'parent_comment', 'subreddit', 'label']
RedditData = RedditData[desired_var]



train_data, test_data = train_test_split( RedditData, test_size = 0.25,  random_state = 339)


stop_words_en = stopwords.words("english")

desired_var_no_target = ['comment', 'parent_comment', 'subreddit']

train_comp_v = train_data[desired_var_no_target]
test_comp_v = test_data[desired_var_no_target]

train_subreddit = train_data['subreddit']
test_subreddit = test_data['subreddit']

train_parent_comment = train_data['parent_comment']
test_parent_comment = test_data['parent_comment']


train_comp = train_data['comment']
test_comp = test_data['comment']


train_target = train_data['label']
test_target = test_data['label']


tc_matrix= CountVectorizer(strip_accents = 'unicode', stop_words = stop_words_en, min_df = 0.0001,max_df = 0.90)
train_trans = tc_matrix.fit_transform(train_comp)
tfidf_trans_normalized = TfidfTransformer()
train_tfidf = tfidf_trans_normalized.fit_transform(train_trans)
test_trans = tc_matrix.fit_transform(test_comp)


RedditData.label = pd.to_numeric(RedditData.label, errors='coerce')

most_frequent_subreddits = train_subreddit.value_counts()#[0:N_MOST_FREQ_SUBREDDITS] # get the [:n] most freq subreddit
most_frequent_subreddits = list(most_frequent_subreddits.index)
most_frequent_subreddits_percentual = RedditData.groupby("subreddit").agg({"label" : "mean"})
most_frequent_subreddits_percentual_sorted = most_frequent_subreddits_percentual.sort_values(by = 'label',
                                                ascending = False)

print(most_frequent_subreddits_percentual_sorted)
most_frequent_subreddits_percentual_sorted.plot()


log_reg_model = LogisticRegression(random_state = 339)
cross_validate(log_reg_model,
                train_trans,
                train_target,
                cv = 2,
                scoring = "accuracy",
                n_jobs =- 3)

log_reg_model = LogisticRegression(random_state = 42,
                                    penalty = "elasticnet",
                                    solver = "saga")


log_reg_model.fit( train_trans,train_target)

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
logreg_disp = RocCurveDisplay.from_estimator(log_reg_model, train_trans, train_target, ax=ax, alpha=0.8)
logreg_disp.plot(ax=ax, alpha=0.8)

pipeline = Pipeline([
    ('vect', tc_matrix),
    ('tftrans', tfidf_trans_normalized),
    ('model', log_reg_model)
])

param_grid = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vect__max_features': (5000, 15000, 30000),
    'model__l1_ratio': (0.0, 0.15, 0.40, 0.60, 0.85, 1.0)
}


grid_logreg = GridSearchCV(pipeline, param_grid, scoring = "accuracy", cv = 2, n_jobs =- 1)
grid_logreg.fit(train_comp, train_target)


svm_model = SGDClassifier(  penalty = "elasticnet",
                            random_state = 42,
                            n_jobs =- 1)


model = Pipeline([
    ('vect', tc_matrix),
    ('tftrans', tfidf_trans_normalized),
    ('model', svm_model)
])

param_grid = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vect__max_features': (5000, 15000, 30000),
    'model__l1_ratio': (0.0, 0.15, 0.40, 0.60, 0.85, 1.0)
}

grid_svm = GridSearchCV(estimator = model, param_grid = param_grid, scoring = "accuracy", cv = 2, n_jobs =- 1)
grid_svm.fit(train_comp, train_target)

grid_svm.fit( train_trans,train_target)

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
grid_svm_disp = RocCurveDisplay.from_estimator(grid_svm, train_comp, train_target, ax=ax, alpha=0.8)
grid_svm_disp.plot(ax=ax, alpha=0.8)



filename = 'grid_logreg.sav'
pickle.dump(grid_logreg, open(filename, 'wb'))
log_model = pickle.load(open("grid_logreg.sav", 'rb'))

filename = 'grid_svm.sav'
pickle.dump(grid_svm, open(filename, 'wb'))
svm_model = pickle.load(open("grid_svm.sav", 'rb'))


#token_count_matrix = CountVectorizer(strip_accents='unicode', stop_words = stop_words_en, min_df = 0.0001, max_df = 0.70)
#tf_trans = TfidfTransformer()

# instantiate the Naive Bayes model
nb_model = MultinomialNB()

# instantiate Pipeline for model
model = Pipeline([
    ('vect', tc_matrix),
    ('tftrans', tfidf_trans_normalized),
    ('model', nb_model)
])

# instantiate dictionary with parameters
param_grid = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vect__max_features': (5000, 15000, 30000)
}

# instantiate a GridSearchCV object
grid_nb = GridSearchCV(estimator = model, param_grid = param_grid, scoring = "accuracy", cv = 2, n_jobs =- 1)
grid_nb.fit(train_comp , train_target)


ax = plt.gca()
grid_nb_disp = RocCurveDisplay.from_estimator(grid_nb, train_comp, train_target, ax=ax, alpha=0.8)
grid_nb_disp.plot(ax=ax, alpha=0.8)

filename = 'grid_nb.sav'
pickle.dump(grid_nb, open(filename, 'wb'))
nb_model = pickle.load(open("grid_nb.sav", 'rb'))


vocab_size = 15255
oov_tok = "<oov>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_parent_comment)
word_index = tokenizer.word_index
max_length = 150
trunc_type='post'
padding_type='post'


training_sequences = tokenizer.texts_to_sequences(train_comp)
training_padded = pad_sequences(training_sequences,
                                maxlen = max_length,
                                padding = padding_type,
                                truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_comp)
testing_padded = pad_sequences(testing_sequences,
                               maxlen = max_length,
                               padding = padding_type,
                               truncating = trunc_type)

embedding_dim = 24
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,
                              embedding_dim, # default 'uniform'
                              input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

training_padded = np.array(training_padded)
training_labels = np.array(train_target)
training_labels = np.float32(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(test_target)
testing_labels = np.float32(testing_labels)
num_epochs = 10
from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)

history = model.fit( x = training_padded,
                    y = training_labels,
                    epochs = num_epochs,
                    validation_data = (testing_padded, testing_labels),
                    verbose = 2,
                    callbacks = [tbCallBack])


model_2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,
                              embedding_dim, # default 'uniform'
                              input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation = 'relu'),
    tf.keras.layers.Dense(12,activation='relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model_2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_2.summary()

history = model_2.fit( x = training_padded,
                    y = training_labels,
                    epochs = num_epochs,
                    validation_data = (testing_padded, testing_labels),
                    verbose = 2,
                    callbacks = [tbCallBack])


from keras.wrappers.scikit_learn import KerasClassifier
# keras_model = build_model()
# keras_model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)
from sklearn.metrics import roc_curve
y_pred_keras = model.predict(testing_padded).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(testing_labels, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.plot(fpr_keras, tpr_keras)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

from sklearn.metrics import roc_curve
y_pred_keras = model_2.predict(testing_padded).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(testing_labels, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.plot(fpr_keras, tpr_keras)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

model.save('my_model')
NN = tf.keras.models.load_model('/my_model/saved_model.pb')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

lr = LogisticRegression()
gnb = GaussianNB()
rfc = RandomForestClassifier()

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (rfc, "Random forest"),
]

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit( train_trans, train_target)
    display = CalibrationDisplay.from_estimator(
        clf,
        test_trans,
        test_target,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()


from sklearn.model_selection import learning_curve
from sklearn.svm import SVC


