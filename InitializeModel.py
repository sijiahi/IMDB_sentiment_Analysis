from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras import layers
from keras.layers import LSTM
from keras import losses
from keras import metrics
from keras.utils import to_categorical


def default_modelConf(size=10000):
    # Initialize model

    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(size,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model
'''    
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])
'''

# model with dropout layer
def dp_modelConf():
    # Input - Layer
    model = Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(10000,)))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model
