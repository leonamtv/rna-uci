from core.perceptron.perceptron import Perceptron

import random as rnd

class TrainMammographicMasses :

    def __init__ ( self, epocas, qtd_in, qtd_out, data, taxa_aprendizado=0.3) :
        """
        Parâmetros:
        -----------
        epocas: int
            Número de épocas de treinamento
        qtd_in: int
            Número de entradas do perceptron
        qtd_out: int
            Número de saídas (classes) do perceptron
        data: Lista de tuplas contendo entrada e saída
            Base de dados para treinamento
        taxa_aprendizado: float, opcional
            Taxa de aprendizado do perceptron. Padrão 0.3
        """
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
