from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import losses
from keras import metrics


def modelConf():
    # Initialize model

    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(10000,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    return model
