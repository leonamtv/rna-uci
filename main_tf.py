from core.data_prep.prepare_data import filter_dataset
from core.train.treinamento_mammographic_masses import TrainMammographicMasses

import tensorflow as tf

file_path = './data/mammographic-masses/mammographic_masses.data'

data = filter_dataset (file_path, format={ 'input_size' : 5 }, normalize=True)

train_split = int( 0.8 * len(data))

train_data, test_data = data[:train_split], data[train_split:]

def create_model ( ) :

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

    return model

def plot_model ( model ) :
    tf.keras.utils.plot_model(model, to_file='file.png', show_shapes=True)


def train ( model, train_input, train_label ) :
    model.fit(train_input, train_label, batch_size=10, epochs=200)    

def test ( model, test_input, test_label ) :
    model.evaluate( test_input, test_label )

train_input = [ sample[0] for sample in train_data ]
train_label = [ sample[1] for sample in train_data ]

test_input  = [ sample[0] for sample in test_data ]
test_label  = [ sample[1] for sample in test_data ]

model = create_model()

train( model, train_input, train_label )
test ( model, test_input, test_label )

