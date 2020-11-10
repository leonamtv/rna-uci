from core.perceptron.perceptron import Perceptron

import matplotlib.pyplot as plt
import random as rnd

class TrainMammographicMasses :

    def __init__ ( self, epocas, qtd_in, qtd_out, data, taxa_aprendizado=0.3) :
        self.data = data
        self.epocas = epocas
        self.perceptron = Perceptron(qtd_in=qtd_in, qtd_out=qtd_out, ni=taxa_aprendizado)

    def fit ( self ) :  
        for i in range(self.epocas) :
            erroAproxEpoca = 0
            erroClassEpoca = 0
            data = self.data
            rnd.shuffle(data)
            for sample in data :
                erro_aprox, erro_class = self.perceptron.treinar( sample[0], sample[1])
                erroAproxEpoca += erro_aprox
                erroClassEpoca += erro_class
            print(f"Época {i + 1} | Erro aprox: {erroAproxEpoca} | Erro class: {erroClassEpoca} ")
        return self.perceptron  
