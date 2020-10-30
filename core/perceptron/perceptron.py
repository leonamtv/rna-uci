import numpy as np
import math

class Perceptron :

    def __init__ ( self, qtd_in, qtd_out, ni=0.3 ) :
        self.qtd_in = qtd_in
        self.qtd_out = qtd_out
        self.ni = ni
        self.pesos = np.random.rand(qtd_in + 1, qtd_out)

    def treinar ( self, x, y ) :
        input_x = x.copy()
        input_x.append(1)
        
        def activation ( x ) :
            try:
                den = ( 1 + math.exp(-x))
            except OverflowError:
                den = float('inf')
            return 1. / den

        u = np.dot( np.array(input_x), self.pesos )
        o = np.array([ activation(x) for x in u ])
        
        erro = np.subtract( np.array(y), o )
        
        for i, e in enumerate(erro) :
            deltas = self.ni * e * np.array(input_x)
            self.pesos[:, i] += deltas

        return np.sum(np.abs(erro))
