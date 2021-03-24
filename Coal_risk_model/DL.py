import pandas as pd

train = pd.read_csv("D:\\DataSet\\Coal_risk_model\\train.csv").values  # .tolist()
test = pd.read_csv("D:\\DataSet\\Coal_risk_model\\test.csv").values  # .tolist()

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# split into input (X) and output (Y) variables
X = train[:, 0:10]
Y = train[:, 10].tolist()

# create model
model = Sequential()
model.add(Dense(12, input_dim=10, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='sigmoid'))
print(model.summary())

"""
train
"""


def train():
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, Y, nb_epoch=800, batch_size=50)

    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    model.save_weights('D:\\DataSet\\Coal_risk_model\\Coal_model_03')


"""
load
"""


def load():
    model.load_weights('D:\\DataSet\\Coal_risk_model\\Coal_model_02')

    X_test = test[:, 0:10]
    Y_test = test[:, 10]
    pred = model.predict(X_test)
    a = tf.argmax(pred, 1)
    sess = tf.Session()
    res = a.eval(session=sess)

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print("accuracy_score: " + str(accuracy_score(Y_test, res)))
    print("precision_score: " + str(precision_score(Y_test, res)))
    print("recall_score: " + str(recall_score(Y_test, res)))


# train()
load()
