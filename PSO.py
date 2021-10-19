import numpy as np

class PSO(object):
    
    def __init__(self, n_cand, dim, lim_inf, lim_sup, n_iter, f_custo, c1=2, c2=2, w_min=0.4, w_max=0.6):
        
        '''
        PSO -> Particle Swarm Optimization (Otimização por nuvem de partículas)
        n_cand: Número de soluções candidatas
        dim: Dimensão do problema de otimização
        lim_inf: Limite inferior dos valores do problema de otimização [colocar como numpy array]
        lim_sup: Limite superior dos valores do problems de otimização [colocar como numpy array]
        n_iter: Número de iterações
        f_custo: Função custo
        c1 e c2: Parâmetros cognitivo e social (o quanto vão em direção do melhor custo pessoal e o melhor custo global)
        w_min e w_max: Parâmetros que definem a inercia da partícula, decaindo do valor máximo ao mínimo.
        '''
        
        self.n_cand = n_cand
        self.dim = dim
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup
        self.n_iter = n_iter
        self.f_custo = f_custo
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        
        # Criando população
        self.pop = np.zeros((n_cand, dim))
        self.pop_custos = np.zeros(n_cand)
        
        # Criando variáveis p_best e g_best
        self.p_best = None
        self.p_best_custo = None
        self.g_best = None
        self.g_best_custo = None
        
        # Matriz de velocidades
        self.v = np.zeros((n_cand, dim))
        
        # Criando população e dando início à otimização
        self.cria_populacao()
        self.inicia_otim()
        
        
    def cria_populacao(self):
        
        self.pop = np.array([self.lim_inf,]*self.n_cand) + np.random.random((self.n_cand, self.dim))*np.array([self.lim_sup-self.lim_inf,]*self.n_cand)
        self.pop_custos = np.array(list(map(self.f_custo, self.pop)))


    def limita_cand(self, cand):
        
        cand_out = np.where(cand > self.lim_sup, 0.95*self.lim_sup, cand)
        cand_out = np.where(cand_out < self.lim_inf, 0.95*self.lim_inf, cand_out)
        
        return cand_out
    
    
    def inicia_otim(self):
        
        # Definindo g_best e p_best da população inicial
        g_best_pos = np.argmin(self.pop_custos)
        self.g_best = self.pop[g_best_pos]
        self.g_best_custo = self.pop_custos[g_best_pos]
            
        self.p_best = self.pop.copy()
        self.p_best_custo = self.pop_custos.copy()
        
        # Iniciando processo de otimização
        for it in range(self.n_iter):
            
            print(f"Iteração {it+1}: Melhor custo -> {self.g_best_custo}")
            
            w = self.w_max - it*(self.w_max-self.w_min)/(self.n_iter-1) # Atualizando a ponderação da inércia

            for i in range(self.n_cand):
                
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                cand = self.pop[i]                  # Candidato atual
                p_best_i = self.p_best[i]           # Personal best atual
                p_custo_i = self.p_best_custo[i]    # Personal custo atual
                
                v_i = self.v[i] # Velocidade atual do candidato
            
                # Cálculo de nova velocidade:
                v_i = w*v_i + self.c1*r1*(p_best_i - cand) + self.c2*r2*(self.g_best - cand)
                
                self.v[i] = v_i # Atualizando nova velocidade
                
                novo_cand = cand + v_i # Atualizando nova posição do candidato
                
                novo_cand = self.limita_cand(novo_cand) # Aplicando limitação
                novo_custo = self.f_custo(novo_cand)    # Obtendo custo do novo candidado
                
                # Atualizando o novo candidato na população
                self.pop[i] = novo_cand
                self.pop_custos = novo_custo
                
                # Comparando custo do novo candidato para personal best e global best
                if novo_custo < p_custo_i:
                    
                    self.p_best[i] = novo_cand
                    self.p_best_custo[i] = novo_custo
                    
                    if novo_custo < self.g_best_custo:
                        
                        self.g_best_custo = novo_custo
                        self.g_best = novo_cand

# ================================================================================


def main():
    opt = PSO(50, 2, np.array([-15, -3]), np.array([-5, 3]), 30, optFunction)
    print(opt.g_best, opt.g_best_custo)
    
    
if __name__ == '__main__':
    
    def optFunction(x):
        #http://www.sfu.ca/~ssurjano/bukin6.html - Min: [-10,1]
        out = 100*np.sqrt(np.abs(x[1]-0.01*x[0]**2))+0.01*np.abs(x[0]+10)
        return out
    
    main()
