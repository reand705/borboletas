import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Nphi = 50 #numero de divisoes do intervalo [0,2pi)
x = np.linspace(0, 2*np.pi, Nphi+1) #intervalo dos phis
T = 2000 #Tempo de simulacao (Dias)
N = 4000 #resolucao
data = np.load('solution_rho=0.850.npy')
phic = 100.
duracao = 90.

fig, ax = plt.subplots()
line1, = ax.plot(x, data[0,:Nphi+1], 'b', label='$u_{1m}$')
line2, = ax.plot(x, data[0,Nphi+1:2*(Nphi+1)], '--b', label='$u_{1b}$')
line3, = ax.plot(x, data[0,2*(Nphi+1):3*(Nphi+1)], 'r', label='$u_{2m}$')
line4, = ax.plot(x, data[0,3*(Nphi+1):4*(Nphi+1)], '--r', label='$u_{2b}$')
time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize='x-small')
v0_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize='x-small')
plt.ylim((0, data.max()))
plt.xlim((0, 2*np.pi))
plt.xlabel('$\phi$')
plt.legend(loc='center left', frameon=False)
plt.axvline(phic/365.*2*np.pi, ls=':', c='k')
plt.axvline((phic+duracao)/365.*2*np.pi, ls='-.', c='k')

def animate(i):
    global data, line1, line2, line3, line4, T, N
    line1.set_ydata(data[i,:Nphi+1])
    line2.set_ydata(data[i,Nphi+1:2*(Nphi+1)])
    line3.set_ydata(data[i,2*(Nphi+1):3*(Nphi+1)])
    line4.set_ydata(data[i,3*(Nphi+1):4*(Nphi+1)])
    time_text.set_text('time: %.2f years' % (i*T/(N*365.)))
    v0_text.set_text('v0: %.1f' % data[i,-1])
    return line1, line2, line3, line4, time_text, v0_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, data.shape[0]),
                              interval=20, blit=False)
ani.save('lines.mp4', dpi=300, fps=30, extra_args=['-vcodec', 'libx264'])
#plt.show() # runs forever?!
