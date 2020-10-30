from core.perceptron.perceptron import Perceptron

import matplotlib.pyplot as plt
import random as rnd

class TrainMammographicMasses :

    def __init__ ( self, epocas, qtd_in, qtd_out, data) :
        self.data = data
        self.epocas = epocas
        self.perceptron = Perceptron(qtd_in=qtd_in, qtd_out=qtd_out)

    def fit ( self ) :  
        erros = []
        for i in range(self.epocas) :
            erroEpoca = 0
            data = self.data
            rnd.shuffle(data)
            for sample in data :
                erro = self.perceptron.treinar( sample[0], sample[1])
                erroEpoca += erro
            print(f"Época {i + 1} | Erro: {erroEpoca}")
            erros.append(erroEpoca)
        return self.perceptron  
