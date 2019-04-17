base_model = Sequential()
base_model.add(Dense(64, input_dim = dimension, activation = 'relu'))
base_model.add(Dense(32, input_dim = dimension))
base_model.add(Dense(6, activation = 'sigmoid'))

base_model.summary()