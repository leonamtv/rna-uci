from core.data_prep.prepare_data import filter_dataset
from core.train.treinamento_mammographic_masses import TrainMammographicMasses
from core.perceptron.perceptron import Perceptron

file_path = './data/mammographic-masses/mammographic_masses.data'
data = filter_dataset (file_path, format={ 'input_size' : 5 }, normalize=True)

print(f'Tamanho do dataset {len(data)}')

train_split = int( 0.8 * len(data))

train_data, test_data = data[:train_split], data[train_split:]

perceptronTreinar = TrainMammographicMasses(1000, 5, 1, data, 0.0001)
perceptronTreinar.fit()
