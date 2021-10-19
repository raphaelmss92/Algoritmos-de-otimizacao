import numpy as np

class TLBO(object):

    def __init__(self, n_cand, n_iters, dim, lim_inf, lim_sup, f_custo, verbose):
	
	'''
	n_cand: Número de soluções candidatas
	n_iters: Número de iterações
	lim_inf: Limite inferior das soluções
	lim_sup: Limite superior das soluções
	dim: Dimensão do problema de otimização
	f_custo: Função custo da otimização
	verbose: Se a otimização deve apresentar resultados em tempo real [True/False]
	'''

        self.n_cand = n_cand
        self.n_iters = n_iters
        self.lim_sup = lim_sup
        self.lim_inf = lim_inf
        self.dim = dim
        self.f_custo = f_custo
        self.verbose = verbose

        self.vet_cand = None
        self.vet_custos = np.zeros(self.n_cand)

        self.vet_best_iter = np.zeros(self.n_iters)
        self.best_cand = None
        self.best_custo = None

        self.cria_classe()
        self.inicia_otim()


    def cria_classe(self):

        pos_range = self.lim_sup - self.lim_inf
        self.vet_cand = np.array([self.lim_inf,]*self.n_cand) + np.array([pos_range,]*self.n_cand)*np.random.rand(self.n_cand, self.dim)

        # Avaliando a classe criada
        for i, cand in enumerate(self.vet_cand):
            self.vet_custos[i] = self.f_custo(cand)


    def limitacao(self, cand):
        
        novo_cand = cand.copy()

        novo_cand = np.where(cand>self.lim_sup, 0.95*self.lim_sup, cand)
        novo_cand = np.where(cand<self.lim_inf, 0.95*self.lim_inf, cand)

        return novo_cand


    def inicia_otim(self):

        print("====================\nIniciando otimização...\n")

        for it in range(self.n_iters):
            
            # professor: definido como solução candidata de menor custo
            prof_pos = np.argmin(self.vet_custos)
            prof = self.vet_cand[prof_pos]
            prof_custo = self.vet_custos[prof_pos]

            # Obtendo a média de todas as soluções candidatas
            media = np.mean(self.vet_cand, axis=0)

            # Salvando o melhor custo da iteração
            self.vet_best_iter[it] = prof_custo

            if self.verbose: print("Iteração {}: Menor custo -> {:.3f}".format(it+1, prof_custo))

            # Fase professor ===================
            for i in range(self.n_cand):

                # Aluno da iteração
                aluno = self.vet_cand[i]
                custo = self.vet_custos[i]

                TF = np.random.randint(1,3)

                # Definindo novo aluno
                novo_aluno = aluno + np.random.random()*(prof-TF*media)
                novo_aluno = self.limitacao(novo_aluno)
                novo_custo = self.f_custo(novo_aluno)

                
                if novo_custo < custo:
                    self.vet_cand[i] = novo_aluno
                    self.vet_custos[i] = novo_custo
            
            # Fase aluno ===================
            for j in range(self.n_cand):

                # Aluno da iteração
                aluno = self.vet_cand[j]
                custo = self.vet_custos[j]

                # Escolha do aluno aleatório e garantia que seja diferente do aluno da iteração atual
                k = np.random.randint(0, self.n_cand)
                while j == k:
                    k = np.random.randint(0, self.n_cand)

                aluno_aleat = self.vet_cand[k]
                custo_aleat = self.vet_custos[k]
                
                # Definindo o passo na direção do que possui menor custo
                if custo <= custo_aleat:
                    passo = aluno - aluno_aleat
                else:
                    passo = aluno_aleat - aluno
                
                # Definindo novo aluno
                novo_aluno = aluno + np.random.random()*passo
                novo_aluno = self.limitacao(novo_aluno)
                novo_custo = self.f_custo(novo_aluno)

                if novo_custo < custo:
                    self.vet_cand[j] = novo_aluno
                    self.vet_custos[j] = novo_custo
        
        min_cust_pos = np.argmin(self.vet_custos)
        self.best_cand = self.vet_cand[min_cust_pos]
        self.best_custo = self.vet_custos[min_cust_pos]

        print("Fim da otimização\n====================")
        
    def get_best(self):
        '''
        Retorna a melhor solução e seu respectivo custo
        '''
        return self.best_cand, self.best_custo


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    def optFunction(x):
	    #http://www.sfu.ca/~ssurjano/bukin6.html - Min: [-10,1]
	    out = 100*np.sqrt(np.abs(x[1]-0.01*x[0]**2))+0.01*np.abs(x[0]+10)
	    return out

    otim = TLBO(40, 100, 2, np.array([-15, -3]), np.array([-5, 3]), optFunction, True)

    print(otim.get_best())

    plt.figure()
    plt.title("Custo por iteração")
    plt.plot(otim.vet_best_iter)
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.show()
