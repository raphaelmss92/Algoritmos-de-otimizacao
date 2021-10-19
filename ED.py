import random
import numpy as np

class ED(object):
		
	def __init__(self, tam_pop, dim, n_ger, prob_mut, min_vals, max_vals, f_custo):
		
		'''
		ED: Evolução diferencial
		tam_pop: Número de soluções candidatas
		dim: Dimensão do problema de otimização	
		n_ger: Número de gerações/iterações da otimização
		prob_mut: Probabilidade de mutação
		min_vals: Limite inferior das soluções candidatas a otimização
		max_vals: Limite superior das soluções candidatas a otimização
		f_custo: Função custo
		'''
		
		self.tam_pop = tam_pop		# Tamanho da população
		self.prob_mut = prob_mut        # Probabilidade de mutação
		self.n_ger = n_ger		# Número de gerações
		self.dim = dim			# Dimensão do problema
		self.min_vals = np.array(min_vals)	# Valores máximos de cada solução candidata
		self.max_vals = np.array(max_vals)	# Valores mínimos de cada solução candidata
		self.f_custo = f_custo          # Função custo
		
		self.F = random.random()
		self.best_indiv = None
		self.best_custo = None

		self.vet_cand = np.zeros((self.tam_pop, self.dim))	# Vetor da população - soluções candidatas
		self.vet_cust = np.ones(tam_pop)			# Vetor de custos da população

		self.cria_populacao()   # Cria a população inicial e custos iniciais
		self.inicia_otim()      # Dá inicio à otimização
  
	def cria_populacao(self):
		'''
		Cria a população de soluções candidatas e avalia cada candidato criado
		'''
		for i in range(self.tam_pop):
			cand = self.min_vals + (self.max_vals-self.min_vals)*np.random.random(self.dim)
			self.vet_cust[i] = self.f_custo(cand)
			self.vet_cand[i] = cand
			

	def define_limites(self, cand):
		'''
		Função que limita a 95% do limite máximo ou mínimo caso algum valor da solução candidata ultrapasse um dos limites.
		'''
		return np.clip(cand, 0.95*self.min_vals, 0.95*self.max_vals)
	

	def inicia_otim(self):
				
		print("Iniciando otimização...\n")
		
		for g in range(self.n_ger):

			best_pos = np.argmin(self.vet_cust)
			self.best_indiv = self.vet_cand[best_pos]
			self.best_custo = self.vet_cust[best_pos]
			
			print("Geração {} - Melhor custo: {:.3f}".format(g+1, self.best_custo))

			for i in range(self.tam_pop):
								
				# Selecionando 3 números aleatórios e diferentes entre si e de "i"
				lst = [j for j in range(self.tam_pop) if j!=i]
				r1, r2, r3 = random.sample(lst, 3) 

				# Obtendo os individuos aleatórios que participarão da mutação de características
				Ir1 = self.vet_cand[r1]
				Ir2 = self.vet_cand[r2]
				Ir3 = self.vet_cand[r3]

				pos_aleat_caract = random.randrange(0, self.dim) # Posição aleatória das características
				new_indiv = self.vet_cand[i].copy() # Novo indivíduo a ser mutado

				# Processo de mutação por característica
				for j in range(self.dim):
					mult = random.random()
					if (mult<=self.prob_mut or j==pos_aleat_caract):
						new_indiv[j] = Ir1[j] + self.F*(Ir2[j]-Ir3[j])
				
				new_indiv = self.define_limites(new_indiv) # Checando se as características não ultrapassaram os limites impostos

				new_cust = self.f_custo(new_indiv) # Custo do novo indivíduo mutado

				if new_cust < self.vet_cust[i]:
					self.vet_cand[i] = new_indiv
					self.vet_cust[i] = new_cust

# ==========================================================================================

def main():
    
    from math import sqrt, fabs
    
    def optFunction(x): # http://www.sfu.ca/~ssurjano/bukin6.html - Min: [-10,1]
        out = 100*sqrt(fabs(x[1]-0.01*x[0]**2))+0.01*fabs(x[0]+10)
        return out
    
    otim = ED(30, 2, 40, 0.5, [-15, -5], [-3, 3], optFunction)
    print(otim.best_indiv, otim.best_custo)


if __name__ == "__main__":
    main()
