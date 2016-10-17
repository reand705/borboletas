# coding: utf-8
#Primeiro Modelo - Borboletas
import scipy.integrate
from scipy.stats import norm
import math ## desnecessário
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt ## repetido
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
import matplotlib
from matplotlib import rc ## desnecessário
matplotlib.rcParams['text.usetex'] = True

#Parametros
M = 2
rbar = 1 #taxa de crescimento
alpha = 1 #? ## amplitude da oscilação da taxa de crescimento, alpha \in [0, 1]
## não entendi por que np.pi + 2. Talvez seja melhor definir todas as fases com
## relação aos 365 dias do ano, escrevendo sin(2.0*np.pi*/365 * (t + beta)),
## assim também fica fácil de comparar com phic.
beta1 = np.pi + 2.0 #fase da taxa de crescimento na mata
beta2 = np.pi + 2.0
K1 = 2.0 #carrying capacities na mata
K2 = 2.0 #carrying capacities na mata

## corresponde a 0.1 dia^-1, ou seja, o tempo de vida é ~10 dias só
mu1 = 0.1 #taxas de mortalidade
mu2 = 0.1

rho = 0.3 #taxa de predacao
v = 1 #populacao fixa de predadores
v00 = 0.5 #populacao inicial de predadores que NAO aprenderam
c = 0.3 #taxa de aprendizagem
C3 = 0.1 #taxa de DESaprendizagem
phic = 100 #epoca do ano em que comecam a entrar novos predadores
duracao = 90 #tamanho do intervado em que predadores entram

phibar1 = 1.5*np.pi #phi_barra
phibar2 = np.pi
Nphi = 50 #numero de divisoes do intervalo [0,2pi)
x = [2.0*np.pi*i/Nphi for i in range(0,Nphi+1)] #intervalo dos phis
## ok, mas mais fácil com np.linspace(0, 2*np.pi, Nphi+1)
## Aqui você inclui o 2*pi no grid, precisa cuidar pra mantê-lo igual ao 0, ou
## descartá-lo, não achei isso. De qualquer forma, deve dar um erro pequeno anyway
u0 = 1

u0m1 = 1 #Populacoes Iniciais - Pico da triangular - (QUASE A) integral da populacao na distribuicao inicial
u0m2 = 1
u0b1 = 0 
u0b2 = 0


T = 2000 #Tempo de simulacao (Dias)
## em tese, você não precisa se preocupar com a discretização no tempo, o
## integrador cuida disso pra você, então isso serve mais pra saber com qual
## resolução você quer analisar os resultados
N = 3000 #discretizacao

def u_inicial(x,u0,phibar,tipo):
	'''Calcula a funcao inicial da distribuicao das populacoes (no ponto x) de acordo com a forma funcional desejada, centrada
	em phibar
	'''
	#Triangular - Pico de altura u0
	if tipo == 't':
		if x < phibar:
			return u0*x/phibar
		else:
			return (2.0*np.pi - x)*u0/(2*np.pi - phibar)

	#Gaussiana, centrada phibar, de integral u0 e var = 1
	if tipo == 'g':
		x = x%365
		return u0*norm.pdf(x,phibar)

# ----------------------------------------------------------------------------------------------------
def m(t,phi):
    #taxa de migracao de saida para o bolsao
    return M*(1 + np.sin(2.0*np.pi*t/365 + phi))
# ----------------------------------------------------------------------------------------------------
def c3(t,phic,duracao):
    #entrada de predadores
    if phic < t%365 and t%365 < phic + duracao:
    	return C3
    else:
    	return 0.0
# ----------------------------------------------------------------------------------------------------
def r(t,phi,beta):
    ## não entendi por que essa função depende de phi, você quis dizer alpha?
    #taxa de crescimento na mata
    return rbar*(1 + alpha*np.sin(2.0*np.pi*t/365 + beta))
# ----------------------------------------------------------------------------------------------------
def um_chapeu(um,int1m,int2m,u0):
    #Calcula a densidade da mata
    return um/(u0 + int1m + int2m)

# ----------------------------------------------------------------------------------------------------
def ddt(y,t):
	'''
	Calcula du_M/dt, du_B/dt e dv_0/dt
	y contem as 4 populacoes de borboletas e a de predadores
	'''
	#Monte as populacoes
	u1m = y[0:Nphi+1]
	u1b = y[Nphi+1:2*Nphi+2]
	u2m = y[2*Nphi+2:3*Nphi+3]
	u2b = y[3*Nphi+3:4*Nphi+4]
	v0 = y[len(y) - 1] ## fica mais fácil de ler com y[-1]

	#Calcule as integrais

	int1m = scipy.integrate.trapz(u1m,x)
	int2m = scipy.integrate.trapz(u2m,x)

	#derivadas
	du_1mdt = [0]*(Nphi+1)
	du_1bdt = [0]*(Nphi+1)
	du_2mdt = [0]*(Nphi+1)
	du_2bdt = [0]*(Nphi+1)
        ## ok, mas mais fácil com np.zeros(Nphi+1) (cria um array, não uma lista)
	
	#calcule-as para todo phi
	for i in range(0,Nphi+1):
            ## é cedo ainda pra pensar em desempenho, mas seu código fica muito
            ## mais rápido se trocar o loop por operações vetoriais
            ## (somas/produtos/funções sobre arrays)

		phi = 2.0*np.pi*i/Nphi

		du_1mdt[i] = r(t,phi,beta1)*u1m[i]*(1 - int1m/K1) - rho*v0*um_chapeu(u1m[i],int1m,int2m,u0) - m(t,phi)*u1m[i] + m(t - 365/2.0,phi)*u1b[i]
		du_1bdt[i] = -mu1*u1b[i] + m(t,phi)*u1m[i] - m(t - 365/2.0,phi)*u1b[i]

		du_2mdt[i] = r(t,phi,beta2)*u2m[i]*(1 - int2m/K2) - rho*v0*um_chapeu(u2m[i],int1m,int2m,u0) - m(t,phi)*u2m[i] + m(t - 365/2.0,phi)*u2b[i]
		du_2bdt[i] = -mu2*u2b[i] + m(t,phi)*u2m[i] - m(t - 365/2.0,phi)*u2b[i]


	# e para o predador

	dv0dt = -c*v0*(int1m+ int2m)/(u0 + int2m + int2m) + c3(t,phic,duracao)*(v - v0)

        ## assustei com isso até lembrar que são listas e não arrays... por
        ## completeza, com arrays seria np.r_[du_1mdt, ... ]
	return du_1mdt + du_1bdt + du_2mdt + du_2bdt + [dv0dt]

# ----------------------------------------------------------------------------------------------------
def main():
	t = np.linspace(0, T, N)
	lenT = len(t) ## = N+1 por construção?

        ## np.zeros
	um10 = [0]*(Nphi+1)
	ub10 = [0]*(Nphi+1)
	um20 = [0]*(Nphi+1)
	ub20 = [0]*(Nphi+1)

	#Monte as populacoes iniciais - Todas comecam na mata?
        ## acho que não deve ter biestabilidade, essas coisas, então tanto faz,
        ## se você esperar os transientes passarem
	for i in range(0,Nphi+1):
		um10[i] = u_inicial(2.0*np.pi*i/Nphi,u0m1,phibar1,'g')
		um20[i] = u_inicial(2.0*np.pi*i/Nphi,u0m2,phibar2,'g')

	#Merge as listas
	y0 = um10 + ub10 + um20 + ub20 + [v00]

	#plt.plot(np.linspace(0, 2*np.pi, Nphi+1), um10)
	#plt.show()

	#Passe para o solver
	sol = scipy.integrate.odeint(ddt,y0,t)

	#Axes3D.plot_surface([0:T],[]sol) ?
	#print sol
	sol = np.array(sol)

	#Calculde as medias
        ## essas são as médias da população total, to curioso pra saber como
        ## fica a seleção sobre phi. Acho que dá pra ter uma ideia inicial olhando
        ## pra média (dica: np.average)
	mean_u1m = [0]*lenT
	mean_u1b = [0]*lenT
	mean_u2m = [0]*lenT
	mean_u2b = [0]*lenT

	for i in range(0,lenT):
		mean_u1m[i] = np.mean(sol[i,0:(Nphi+1)])
		mean_u1b[i] = np.mean(sol[i,Nphi+1:2*Nphi+2])
		mean_u2m[i] = np.mean(sol[i,2*Nphi+2:3*Nphi+3])
		mean_u2b[i] = np.mean(sol[i,3*Nphi+3:4*Nphi+4])

	#Imprima
	v0 = sol[:,4*Nphi+4]

	labels = [r"$u_{1,M}$", r"$u_{1,B}$", r"$u_{2,M}$", r"$u_{2,B}$", r"$v_0$"]
	plt.plot(np.linspace(0,T,N),mean_u1m, label = labels[0])
	plt.plot(np.linspace(0,T,N),mean_u1b, label = labels[1])
	plt.plot(np.linspace(0,T,N),mean_u2m, label = labels[2])
	plt.plot(np.linspace(0,T,N),mean_u2b, label = labels[3])
	plt.plot(np.linspace(0,T,N),v0, label = labels[4])

	plt.legend(labels, loc = 'best')
	plt.title("Media x Tempo")
	plt.xlabel("t")
	plt.ylabel(r"Media [0:2$\pi$]")
	plt.show()
        ## essa forma de plot dá a impressão de que você não roda
        ## interativamente, quer dizer, você vai no terminal e dá "python
        ## borboletas1.py" e vê só o que você mandou plotar. É isso? Se sim,
        ## aconselho fortemente a usar ipython, rodá-lo, e de dentro dele usar
        ## "%run borboletas1.py". Com isso as variáveis ficam lá pra você ir
        ## olhando e analisando depois que termina a simulação (bom, a main tem
        ## que retorná-las...). Aí você também pode usar plt.interactive(True) e
        ## não precisa do show(). 

	
## adicionar esta condição pra executar a main é prático:
## if __name__ == '__main__':
## assim, esse trecho só é rodado quando você roda o arquivo, não quando você o
## importa. Deu pra entender?
main()
