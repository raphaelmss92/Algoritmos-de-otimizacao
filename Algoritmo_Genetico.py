import numpy as np


class GA(object):

    def __init__(self, nInd, nCrom, probCruz, probMut, nGer, fCusto, tipoSel='roleta', tipoCruz='ponto', tipoMut='bit-a-bit', elit=True, verbose=True):
        '''
        Algoritmo genético para problemas de minimização de custo.

        nInd - Número de indivíduos
        nCrom - Número de cromossomos (dimensão do problema)
        probCruz - Probabilidade de cruzamento
        probMut - Probabilidade de mutação
        tipoSel - Tipo de seleção: 'roleta' ou 'torneio'. Roleta é o padrão
        tipoCruz - Tipo de cruzamento: 'ponto' ou 'uniforme'. Ponto a ponto é o padrão
        tipoMut - Tipo de mutação: 'bit-a-bit' ou 'aleatBit'. Bit a bit é o padrão
        Elit - Se terá elitismo. Inicialmente True.
        nGer - Número de gerações
        fcusto - Função custo
        verbose - Se a otimização deve mostrar logs de geração e menor custo atual.
        '''

        self.nInd = nInd
        self.nCrom = nCrom
        self.probCruz = probCruz
        self.probMut = probMut
        self.nGer = nGer
        self.fCusto = fCusto
        self.tipoSel = tipoSel
        self.tipoCruz = tipoCruz
        self.tipoMut = tipoMut
        self.elit = elit
        self.verbose = verbose

        self.bestSol = None
        self.bestCusto = np.inf # Inicia como infinito, para que o proximo custo sempre seja menor

        # Criando população de maneira aleatória
        self.pop = np.random.randint(0, 2, (self.nInd, self.nCrom))

        self.custos = np.ones(self.nInd)

        # Iniciando otimização
        self.inicia_otim()

        
    def avalia_pop(self):
        '''
        Faz a avaliação da população e atualiza o vetor de custos
        '''
        for i in range(self.nInd):
            self.custos[i] = self.fCusto(self.pop[i])


    def selecao_roleta(self):
        '''
        Método de seleção do tipo roleta viciada
        '''

       # Transformação do vetor de custos para suportar valores negativos e inverter os valores maiores nos menores e vice-versa.
       # O número 2 é para que o maior custo normalizado fique com valor unitário, possuindo ainda uma chance de ser selecionado.
       # Caso não exista custos com valores negativos, o menor custo ficará com valor 2 e o maior com 1.
       
        novo_custo = 2 - self.custos/np.abs(self.custos.max())

        soma_custo = novo_custo.sum()

        custo_acum = np.zeros(self.nInd)

        pais = self.pop.copy()

        for i in range(self.nInd):
            custo_acum[i] = np.sum(novo_custo[:i+1])

        for j in range(self.nInd):
            val_selected = soma_custo*np.random.random()
            pos_pai = np.where(custo_acum>=val_selected)[0][0]
            pais[j] = self.pop[pos_pai]

        return pais
    

    def selecao_torneio(self):

        pais = self.pop.copy()

        for i in range(self.nInd):

            pos_indv1 = np.random.randint(0, self.nInd)
            pos_indv2 = np.random.randint(0, self.nInd)
        
            if self.custos[pos_indv1]<=self.custos[pos_indv2]:
                pais[i] = self.pop[pos_indv1]
            else:
                pais[i] = self.pop[pos_indv2]
            
        return pais
        

    def cruzamento_ponto(self, pais):

        filhos = pais.copy()

        for i in range(0, self.nInd, 2):

            pai1 = pais[i]
            pai2 = pais[i+1]

            filho1 = pai1.copy()
            filho2 = pai2.copy()

            if np.random.random() <= self.probCruz:

                ponto_cruz = np.random.randint(0, self.nCrom)

                filho1[:ponto_cruz] = pai2[:ponto_cruz]
                filho2[:ponto_cruz] = pai1[:ponto_cruz]
            
            filhos[i] = filho1
            filhos[i+1] = filho2

        return filhos


    def cruzamento_uniforme(self, pais):

        filhos = pais.copy()

        for i in range(0, self.nInd, 2):

            pai1 = pais[i]
            pai2 = pais[i+1]

            filho1 = pai1.copy()
            filho2 = pai2.copy()

            if np.random.random() <= self.probCruz:
                for c in range(self.nCrom):
                    if np.random.randint(0,2)==1:
                        filho1[c] = pai2[c]
                        filho2[c] = pai1[c]
            
            filhos[i] = filho1
            filhos[i+1] = filho2

        return filhos


    def mutacao_bit(self, filhos):
        
        filhos_m = filhos.copy()

        for i, f in enumerate(filhos):
            for j in range(self.nCrom):
                if np.random.random() <= self.probMut:
                    f[j] = 1-f[j] # Inverte de 1 para 0 e de 0 para 1
            
            filhos_m[i] = f
        
        return filhos_m


    def mutacao_aleatbit(self, filhos):

        filhos_m = filhos.copy()

        for i, f in enumerate(filhos):
            if np.random.random() <= self.probMut:
                bit_mut = np.random.randint(0, self.nCrom)
                f[bit_mut] = 1 - f[bit_mut] # Inverte de 1 para 0 e de 0 para 1 no bit aleatório
                filhos_m[i] = f
        
        return filhos_m
        

    def set_best(self):

        best_pos = np.argmin(self.custos)
        custo = self.custos[best_pos]

        if custo < self.bestCusto:
            self.bestCusto = custo
            self.bestSol = self.pop[best_pos]
        

    def inicia_otim(self):

        print("=== Iniciando otimização ===\n")

        for g in range(self.nGer):

            self.avalia_pop()

            if self.elit: self.set_best()

            if self.verbose:
                if self.elit:
                    print("{:.0f}º geração - Melhor custo: {:.3f}".format(g+1, self.bestCusto))
                else:
                    print("{:.0f}º geração - Melhor custo: {:.3f}".format(g+1, self.custos.min()))

            # Seleção dos pais
            if self.tipoSel == 'roleta':
                pais = self.selecao_roleta()
            else:
                pais = self.selecao_torneio()
            
            # Cruzamento para criação dos filhos
            if self.tipoCruz == 'ponto':
                filhos = self.cruzamento_ponto(pais)
            else:
                filhos = self.cruzamento_uniforme(pais)
            
            # Mutação dos filhos
            if self.tipoMut == 'bit-a-bit':
                filhos_m = self.mutacao_bit(filhos)
            else:
                filhos_m = self.mutacao_aleatbit(filhos)
            
            self.pop = filhos_m

        self.set_best()

        if self.verbose:
            if self.elit:
                print("{:.0f}º geração - Melhor custo: {:.3f}".format(g+1, self.bestCusto))
            else:
                print("{:.0f}º geração - Melhor custo: {:.3f}".format(g+1, self.custos.min()))

        print("\n=== Fim da otimização ===")


# ============================================



        
            

