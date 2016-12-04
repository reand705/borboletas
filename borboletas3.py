# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# coding: utf-8
#Imprimir Rho - Borboletas
%matplotlib inline
import scipy.integrate
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
from scipy.stats import circmean
from scipy.stats import circstd
import matplotlib
from matplotlib import rc ## desnecessário
matplotlib.rcParams['text.usetex'] = True

#Parametros
M = 10
rbar = 10 #taxa de crescimento
alpha = 1 #? ## amplitude da oscilação da taxa de crescimento, alpha \in [0, 1]
beta1 = 20 #fase da taxa de crescimento na mata (dias)
beta2 = 350 # parametro realista
K1 = 50 #carrying capacities na mata
K2 = 50 #carrying capacities na mata

## mortalidade no bolsao
## corresponde a 0.1 dia^-1, ou seja, o tempo de vida e ~10 dias so
#mu1 = 0.04 #taxas de mortalidade
#mu2 = 0.06
mu1 = 1/180.
mu2 = 1/180.

rho = 1 #taxa de predacao
v = 5 #populacao fixa de predadores
v00 = 5 #populacao inicial de predadores que NAO aprenderam
c = 0.5 #taxa de aprendizagem
C3 = 0.1 #taxa de DESaprendizagem
phic = 240 #epoca do ano em que comecam a entrar novos predadores
duracao = 150 #tamanho do intervado em que predadores entram

phibar1 = 1.5*np.pi #phi_barra
phibar2 = np.pi
Nphi = 50 #numero de divisoes do intervalo [0,2pi)
x = np.linspace(0, 2*np.pi, Nphi+1) #intervalo dos phis
phi = 2.0*np.pi* np.arange(0,Nphi+1)/Nphi

## Aqui voce inclui o 2*pi no grid, precisa cuidar pra mante-lo igual ao 0, ou
## descarta-lo, nao achei isso. De qualquer forma, deve dar um erro pequeno anyway
u0 = 5

u0m1 = 40 #Populacoes Iniciais - Pico da triangular - (QUASE A) integral da populacao na distribuicao inicial
u0m2 = 30
u0b1 = 0 
u0b2 = 0


T = 12*365 #Tempo de simulacao (Dias)
N = 12*365 #resoluçao


# In[19]:

@np.vectorize
def u_inicial(x, u0, phibar, tipo):
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

    if tipo == 'u':
        return u0/(2*np.pi)

    if tipo == 'wg':
        # wrapped Gaussian, sigma = 1?
        return u0 * max(norm.pdf(x,phibar), norm.pdf(x+2*np.pi,phibar), norm.pdf(x-2*np.pi,phibar))

# ----------------------------------------------------------------------------------------------------
def m(t,phi):
    #RECEBE EM [0:2pi]
    #taxa de migracao de saida para o bolsao
    #migracao senoidal 
    #return M*(1 + np.sin(2.0*np.pi*t/365 + phi))
    t = t%365
    sigma = 30
    return M * np.max(np.c_[norm.pdf(t-phi*(365/2/np.pi),0,sigma), 
                   norm.pdf(t+365-phi*(365/2/np.pi),0,sigma), 
                   norm.pdf(t-365-phi*(365/2/np.pi),0,sigma)], axis=1)
# ----------------------------------------------------------------------------------------------------
def c3(t,phic,duracao):
    #entrada de predadores
    #se phic + duracao
    if phic + duracao > 365:
        if (phic + duracao)%365 < t%365 and t%365 < phic:
            return 0.0
        else:
            return C3
    else:
        if phic < t%365 and t%365 < phic + duracao:
            return C3
        else:
            return 0.0

# ----------------------------------------------------------------------------------------------------
def r(t,beta):
    ## não entendi por que essa função depende de phi, você quis dizer alpha?
    #taxa de crescimento na mata
    # senoide:
    # beta é o dia em que a estação favorável começa (r > rbar)
    #return rbar*(1 + alpha*np.sin(2.0*np.pi*(t-beta)/365))
    # Gaussiana:
    t = t%365
    sigma = 10
    return rbar * np.max(np.c_[norm.pdf(t-beta, 0, sigma),
                         norm.pdf(t-365-beta, 0, sigma),
                         norm.pdf(t+365-beta, 0,sigma)], axis=1)
# ----------------------------------------------------------------------------------------------------
def um_chapeu(um,int1m,int2m,u0):
    #Calcula a densidade da mata
    return um/(u0 + int1m + int2m)

# ----------------------------------------------------------------------------------------------------
def mediaphi(u):
    #calcula a media sobre phi da populacao - espera-se um vettor de tamanho Nphi+1
    sum_of_sines = 0
    sum_of_cosines = 0
    for i in range(len(x)):
        sum_of_sines += u[i]*np.sin(x[i])
        sum_of_cosines += u[i]*np.cos(x[i])
    
    return np.arctan2(sum_of_sines,sum_of_cosines) + np.pi
# ----------------------------------------------------------------------------------------------------
def circaverage(angles, weights):
    '''Compute the weighted circular mean of angles'''
    angles = np.array(angles)
    weights = np.array(weights)
    sin_sum = (np.sin(angles) * weights).sum()
    cos_sum = (np.cos(angles) * weights).sum()
    total_weights = weights.sum()
    
    return np.arctan2(sin_sum/total_weights, cos_sum/total_weights) + np.pi
        

# ----------------------------------------------------------------------------------------------------
def ddt(y, t, rho):
    '''
    Calcula du_M/dt, du_B/dt e dv_0/dt
    y contem as 4 populacoes de borboletas e a de predadores
    
    Tambem atualiza as medias
    '''
    #Monte as populacoes
    u1m = y[0:Nphi+1]
    u1b = y[Nphi+1:2*Nphi+2]
    u2m = y[2*Nphi+2:3*Nphi+3]
    u2b = y[3*Nphi+3:4*Nphi+4]
    v0 = y[-1]    

    #Calcule as integrais

    int1m = scipy.integrate.trapz(u1m,x)
    int2m = scipy.integrate.trapz(u2m,x)
   
    #calcule-as para todo phi
    du_1mdt = r(t,beta1) * u1m * (1 - int1m/K1) - rho*v0 * um_chapeu(u1m,int1m,int2m,u0) - m(t,phi)*u1m + m(t - 365/2.0,phi)*u1b
    du_1bdt = -mu1*u1b + m(t,phi)*u1m - m(t - 365/2.0,phi)*u1b
    du_2mdt = r(t,beta2)*u2m*(1 - int2m/K2) - rho*v0*um_chapeu(u2m,int1m,int2m,u0) - m(t,phi)*u2m + m(t - 365/2.0,phi)*u2b
    du_2bdt = -mu2*u2b + m(t,phi)*u2m - m(t - 365/2.0,phi)*u2b

    # e para o predador

    dv0dt = -c*v0*(int1m + int2m)/(u0 + int1m + int2m) + c3(t,phic,duracao)*(v - v0)

    return np.r_[du_1mdt,du_1bdt,du_2mdt,du_2bdt,dv0dt]

# <codecell>

#Simule
rhos = [0.0]
rhos = []
aux = [0.05 + i*0.1 for i in range(0,12)]
for elemento in aux: #preguiçoso
    rhos.append(elemento)

plt.interactive(True)
print(rhos)
t = np.linspace(0, T, N)
lenT = len(t) ## = N+1 por construção?

for rho in rhos:
    print(rho) #para sabermos em qual está
    
    um10 = np.zeros(Nphi+1)
    ub10 = np.zeros(Nphi+1)
    um20 = np.zeros(Nphi+1)
    ub20 = np.zeros(Nphi+1)
    
    #Monte as populacoes iniciais - Todas comecam na mata
    um10 = u_inicial(phi,u0m1,phibar1,'wg')
    um20 = u_inicial(phi,u0m2,phibar2,'wg')
    
    #Merge as listas
    y0 = np.r_[um10,ub10,um20,ub20,v00]
    
    #plt.plot(np.linspace(0, 2*np.pi, Nphi+1), um10)
    #plt.show()
    
    #Passe para o solver
    sol = scipy.integrate.odeint(ddt,y0,t, (rho,), hmax=0.5)
    
    sol = np.array(sol)
    
    np.save('solution_rho(newM)=%.3f' %(rho), sol)

# <codecell>

#Calcule as medias e as populações finais para todos os rhos
means1 = []
errorbars1 = []
means2 = []
errorbars2 = []
finals1m = []
finals2m = []
finals1b = []
finals2b = []

#plt.interactive(True)
rhos = [0.0]
rhos = []
t = np.linspace(0, T, N)
lenT = len(t)
aux = [0.05 + i*0.1 for i in range(0,12)]
for elemento in aux: #preguiçoso
    rhos.append(elemento)

for rho in rhos:
    sol = np.load('solution_rho(newM)=%.3f.npy' %(rho))
    
    #Monte as populacoes finais
    u1m = sol[-1*365:,0:Nphi+1]
    u1b = sol[-1*365:,Nphi+1:2*Nphi+2]
    u2m = sol[-1*365:,2*Nphi+2:3*Nphi+3]
    u2b = sol[-1*365:,3*Nphi+3:4*Nphi+4]
    v0 = sol[-1*365:,-1]    

    #Calcule as integrais
    int1m = scipy.integrate.trapz(u1m,x)
    int2m = scipy.integrate.trapz(u2m,x)
    int1b = scipy.integrate.trapz(u1b,x)
    int2b = scipy.integrate.trapz(u2b,x)
    
    finals1m.append(np.mean(int1m))
    finals1b.append(np.mean(int1b))
    finals2m.append(np.mean(int2m))
    finals2b.append(np.mean(int2b))
    
    #Calcule as medias
    mean_u1m = np.zeros(lenT)
    mean_u1b = np.zeros(lenT)
    mean_u2m = np.zeros(lenT)
    mean_u2b = np.zeros(lenT)
    
    mean_of_phis_um10 = np.zeros(lenT)
    mean_of_phis_ub10 = np.zeros(lenT)
    mean_of_phis_um20 = np.zeros(lenT)
    mean_of_phis_ub20 = np.zeros(lenT)
    
    for i in range(0,lenT):
        #populacao media
        mean_u1m[i] = np.mean(sol[i,0:(Nphi+1)])
        mean_u1b[i] = np.mean(sol[i,Nphi+1:2*Nphi+2])
        mean_u2m[i] = np.mean(sol[i,2*Nphi+2:3*Nphi+3])
        mean_u2b[i] = np.mean(sol[i,3*Nphi+3:4*Nphi+4])
        
        #média dos phis
        mean_of_phis_um10[i] = mediaphi(sol[i,0:(Nphi+1)])
        mean_of_phis_ub10[i] = mediaphi(sol[i,Nphi+1:2*Nphi+2])
        mean_of_phis_um20[i] = mediaphi(sol[i,2*Nphi+2:3*Nphi+3])
        mean_of_phis_ub20[i] = mediaphi(sol[i,3*Nphi+3:4*Nphi+4])
        
    # medias simples
    media_phi_u1m = circmean(mean_of_phis_um10[-5*365:])
    media_phi_u2m = circmean(mean_of_phis_um20[-5*365:])
    
    #desvio padrão
    stdev_phi_u1m = circstd(mean_of_phis_um10[-5*365:])
    stdev_phi_u2m = circstd(mean_of_phis_um20[-5*365:])
    
    # medias ponderadas por pop. total
    media_phi_avg_u1m = circaverage(mean_of_phis_um10[-5*365:], mean_u1m[-5*365:])
    media_phi_avg_u2m = circaverage(mean_of_phis_um20[-5*365:], mean_u2m[-5*365:])
    
    # medidas de sincronia
    print('Diferença entre média simples de phi: %.5f' % ((media_phi_u2m - media_phi_u1m) / 2/ np.pi * 365))
    print('Diferença entre média ponderada de phi: %.5f' % ((media_phi_avg_u2m - media_phi_avg_u1m) / 2/ np.pi * 365))
    
    #adicione às listas
    means1.append(media_phi_avg_u1m/ 2/ np.pi * 365)
    errorbars1.append(stdev_phi_u1m/ 2/ np.pi * 365)
    means2.append(media_phi_avg_u2m/ 2/ np.pi * 365)
    errorbars2.append(stdev_phi_u2m/ 2/ np.pi * 365)

# <codecell>

np.shape(sol)

# <codecell>

#plote os phis medios
plt.errorbar(rhos, means1, yerr = errorbars1, label = r"Esp. 1")
plt.errorbar(rhos, means2, yerr = errorbars2, label = r"Esp. 2")
print("Beta 1 = %d" %beta1)
print("Beta 2 = %d" %beta2)
plt.legend(loc = "best")
plt.xlabel(r"$\rho$")
plt.ylabel(r"$\phi$ medio (dia do ano)")
plt.title(r"-")
plt.show()

# <codecell>

#Plote as populações finais
plt.plot(rhos, finals1m, label = r"1m")
plt.plot(rhos, finals1b, label = r"1b")
plt.plot(rhos, finals2m, label = r"2m")
plt.plot(rhos, finals2b, label = r"2b")
plt.xlabel(r"$\rho$")
plt.ylabel(r"Final pops. (avg. integral - last ano)")
plt.title(r"-")
plt.legend(loc = "best")
plt.show()

# <codecell>

means2

