import pandas as pd

train = pd.read_csv("D:\\DataSet\\Ins_qua_model\\train.csv").values  # .tolist()
test = pd.read_csv("D:\\DataSet\\Ins_qua_model\\test.csv").values  # .tolist()

import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(12, input_dim=7, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='sigmoid'))
print(model.summary())

def train_model():
    # split into input (X) and output (Y) variables
    X = train[:, 2:9]
    Y = train[:, 1].tolist()

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, Y, nb_epoch=100, batch_size=50)

    # Save weights to a TensorFlow Checkpoint file
    model.save_weights('./Inspect_model2.h5')
    model.save('./Inspect_model_all.h5')
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def pred_model():

    #model.load_weights('./Inspect_model.h5')
    model = load_model('./Inspect_model_all.h5')
    X_test = test[:, 2:9]
    Y_test = test[:, 1]

    pred = model.predict(X_test)

    a = tf.argmax(pred, 1)
    sess = tf.Session()
    res = a.eval(session=sess)

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print("accuracy_score: " + str(accuracy_score(Y_test, res)))
    print("precision_score: " + str(precision_score(Y_test, res)))
    print("recall_score: " + str(recall_score(Y_test, res)))



train_model()
pred_model()
