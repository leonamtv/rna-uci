from core.data_prep.prepare_data import filter_dataset
from core.train.treinamento_mammographic_masses import TrainMammographicMasses

import tensorflow as tf

file_path = './data/mammographic-masses/mammographic_masses.data'

data = filter_dataset (file_path, format={ 'input_size' : 5 })

train_split = int( 0.8 * len(data))

train_data, test_data = data[:train_split], data[train_split:]

input_layer   = tf.keras.layers.Input(shape=(1, 5), batch_size=10)
dense_layer_1 = tf.keras.layers.Dense(10, activation='sigmoid')(input_layer)
drop_layer_1  = tf.keras.layers.Dropout(0.2)(dense_layer_1)
dense_layer_2 = tf.keras.layers.Dense(25, activation='sigmoid')(drop_layer_1)
drop_layer_2  = tf.keras.layers.Dropout(0.2)(dense_layer_2)
dense_layer_3 = tf.keras.layers.Dense(25, activation='relu')(drop_layer_2)
drop_layer_3  = tf.keras.layers.Dropout(0.2)(dense_layer_3)
dense_layer_4 = tf.keras.layers.Dense(10, activation='sigmoid')(drop_layer_3)
drop_layer_4  = tf.keras.layers.Dropout(0.2)(dense_layer_4)
dense_layer_4 = tf.keras.layers.Dense(1, activation='sigmoid')(drop_layer_4)

model = tf.keras.Model(inputs=[ input_layer ], outputs=[ dense_layer_4 ])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tf.keras.utils.plot_model(model, to_file='file.png', show_shapes=True)

itrain = [ sample[0] for sample in train_data ]
otrain = [ sample[1] for sample in train_data ]


model.fit(itrain, otrain, batch_size=10, epochs=400)

print(model.evaluate( [ sample[0] for sample in test_data ], [ sample[1] for sample in test_data ] ))

