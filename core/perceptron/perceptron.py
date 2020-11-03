import numpy as np
import math

class Perceptron :

    def __init__ ( self, qtd_in, qtd_out, ni=0.3 ) :
        self.qtd_in = qtd_in
        self.qtd_out = qtd_out
        self.ni = ni
        self.pesos = np.random.rand(qtd_in + 1, qtd_out)

    def treinar ( self, x, y, threshold=0.5 ) :
        
        input_x = x.copy()
        input_x.append(1)
        
        def sigmoid ( x ) :
            try :
                result = 1. / ( 1 + np.exp( x ))
            except e :
                result = 1. / ( 1 + np.exp( x ))
                print(result)
            return result

        u = np.float128(np.dot( np.array(input_x), self.pesos ))
        o = np.float128(sigmoid(u))
        
        erro = np.float128(np.subtract( np.array(y), o ))

        classif = [ 0 if output <= threshold else 1 for output in o ]

        erros_classif = np.subtract( y, np.array(classif) )
        erro_classif = np.sum(erros_classif)
        
        for i, e in enumerate(erro) :
            deltas = self.ni * e * np.array(input_x)
            self.pesos[:, i] += deltas

        return np.sum(np.abs(erro)), 0 if erro_classif == 0 else 1
