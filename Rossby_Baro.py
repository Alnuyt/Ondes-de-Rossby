"""
@auteur : Alexandre Nuyt

Simulation de la dynamique des fluides géophysiques :

Ce script résout numériquement l'évolution d'un champ de vitesse et de vorticité 
sur une grille 2D en utilisant une méthode de différences finies. Il intègre 
les équations de la dynamique des fluides en rotation avec l'effet de la force 
de Coriolis, basé sur la latitude d'entrée. Les résultats incluent la visualisation 
des champs scalaires et vectoriels.

"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

#Constantes du projet----------------------------------------------------------
phi_deg = int(input('Veuillez entrer une latitude de référence (en °): '))
iteration = int(input('Veuillez entrer le nombre d itérations temporelles: '))

g = 9.81 
a = 6371 #rayon de la Terre
Phi = phi_deg*(np.pi)/180 #latitude exprimée en radiant
Omega = 7.27*10**(-5)
f_0 = 2*Omega*np.sin(Phi)
B = 2*np.cos(Phi)/a
U = 20*3,6
M = 60 #nombre de maille sur l'axe x
N = 30 #nombre de maille sur l'axe y
ds = 20 #taille des mailles

W_x = 6000
W_y = 3000

k = 2*np.pi/W_x #nombre d'onde k correspondant aux longueurs d'onde W_x
j = 2*np.pi/W_y #nombre d'onde j correspondant aux longueurs d'onde W_y

#On définit le réseau----------------------------------------------------------

xmin = 0
xmax = 12000
ymin = 0
ymax = 6000
tmax = 96
dt = 1

#Initialisation----------------------------------------------------------------
#Toutes les fonctions avec un "_n" sont les fonctions qui précèdent d'un pas de
#temps celles sans "_n"
#On crée des matrices de la taille du réseau qu'on va remplir de valeurs à 
#chaque itération

u = np.zeros((N, M))
u_n = np.zeros((N, M))
u_0 = np.zeros((N, M))

v = np.zeros((N, M))
v_n = np.zeros((N, M))
v_0 = np.zeros((N, M))

f = np.zeros((N, M))
f_n = np.zeros((N, M))

psi = np.zeros((N,M))
psi_n = np.zeros((N,M))
psi_0 = np.zeros((N,M))

zeta = np.zeros((N,M))
zeta_n = np.zeros((N,M))
zeta_0 = np.zeros((N,M))
zeta_u = zeta_n*u_n
zeta_v = zeta_n*v_n
             
x = np.linspace(xmin,xmax,M)
y = np.linspace(ymin,ymax,N)
X, Y = np.meshgrid(x, y)

"""
-----------------------------Résolution pour C.I-------------------------------
"""
#Champ scalaire Psi initial----------------------------------------------------
psi_0[:,:] = (g/f_0)*100*np.sin(k*X)*np.cos(j*Y)

#Terme source de l'équation de poisson-----------------------------------------
zeta_0[:,:] = (g/f_0)*(-100)*np.sin(k*X)*np.cos(j*Y)*(k**2 + j**2)

#Champ de vitesse initial (u_0, v_0)
u_0[:,:] = (g/f_0)*100*j*np.sin(k*X)*np.sin(j*Y)
v_0[:,:] = (g/f_0)*100*k*np.cos(k*X)*np.cos(j*Y)

######################### Visualistaion des résultats #########################

#Champ de vitesse initial (u_0, v_0)
plt.style.use('_mpl-gallery-nogrid')
fig, ax = plt.subplots()

ax.quiver(X, Y, u_0, v_0, color="C0", angles='xy',
          scale_units='xy', scale=80, width=.001)

ax.set(xlim=(0, 12000), ylim=(0, 6000))
ax.set_title('Champ de vitesse initial')
plt.show()

#Champ scalaire Psi initial
def plot3D(x, y, psi_0):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    
    ax.plot_surface(X, Y, psi_0, rstride=1, cstride=1, cmap=cm.viridis,linewidth=0,
                    antialiased=False)
    ax.view_init(30, 225)
    ax.set_title('Amplitude du champ scalaire initial')
    ax.set_xlabel('X Spacing')
    ax.set_ylabel('Y Spacing')
    ax.set_zlabel('Amplitude du champ scalaire initial')

plot3D(x,y,psi_0)
plt.show()

#Vorticité relative initiale
def plot3D(x, y, zeta_0):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    
    ax.plot_surface(X, Y, zeta_0, rstride=1, cstride=1, cmap=cm.viridis,linewidth=0,
                    antialiased=False)
    ax.view_init(30, 225)
    ax.set_title('Vorticité relative initiale')
    ax.set_xlabel('L_x')
    ax.set_ylabel('L_y')
    ax.set_zlabel('Vorticité relative initiale')

plot3D(x,y,zeta_0)
plt.show()

"""
-----------------------------Résolution par itération--------------------------
"""
for it in range(iteration):  
    
    psi_n = psi.copy()
    u_n = u.copy()
    v_n = v.copy()
    f_n = f.copy()
    zeta_n = zeta.copy()
    
    #Méthode de différence finie
    
    psi[1:-1,1:-1] = (1/4)*((psi_n[1:-1,2:] + psi_n[1:-1,:-2]) + (psi_n[2:,1:-1] 
                    + psi_n[:-2,1:-1]) - zeta_0[1:-1,1:-1]*ds**2)
    u[1:-1,1:-1] = -(psi_n[1:-1,2:]-psi_n[1:-1,:-2])/2*ds
    v[1:-1,1:-1] = (psi_n[2:,1:-1]-psi_n[:-2,1:-1])/2*ds
    f[1:-1,1:-1] = (zeta_u[2:,1:-1]-zeta_u[:-2,1:-1])/2*ds + (zeta_v[1:-1,2:]-
                    zeta_v[1:-1,:-2])/2*ds + B*u_n[1:-1,1:-1]
    zeta[1:-1,1:-1] = -2*f_n[1:-1,1:-1]*dt + zeta_n[1:-1,1:-1]
    
    #Conditions aux bords : Dirichlet (on impose une valeur à la frontière)
    
    psi[0,:] = 0
    psi[N-1,:] = 0
    psi[:,0] = 0
    psi[:,M-1] = 0
    
    zeta[0,:] = 0
    zeta[N-1,:] = 0
    zeta[:,0] = 0
    zeta[:,M-1] = 0
    
    u[0,:] = 0
    u[N-1,:] = 0
    u[:,0] = 0
    u[:,M-1] = 0
    
    v[0,:] = 0
    v[N-1,:] = 0
    v[:,0] = 0
    v[:,M-1] = 0
    
    f[0,:] = 0
    f[N-1,:] = 0
    f[:,0] = 0
    f[:,M-1] = 0
    
######################### Visualistaion des résultats #########################

#Champ Psi (fonction scalaire) 
def plot3D(x, y, psi):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    
    ax.plot_surface(X, Y, psi[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0,
                    antialiased=False)
    ax.view_init(30, 225)
    ax.set_title('Champ Psi intégré')
    ax.set_xlabel('L_x')
    ax.set_ylabel('L_y')
    ax.set_zlabel('Amplitude du champ Psi')

plot3D(x,y,psi)
plt.show()

#Vorticité relative
def plot3D(x, y, zeta):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(X, Y, zeta[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0,
                    antialiased=False)
    ax.view_init(30, 225)
    ax.set_title('Vorticité relative intégrée')
    ax.set_xlabel('L_x')
    ax.set_ylabel('L_y')
    ax.set_zlabel('Amplitude vorticité relative')

plot3D(x,y,zeta)
plt.show()

#Champ vectoriel de vitesse (u,v)
plt.style.use('_mpl-gallery-nogrid')
fig, ax = plt.subplots()

ax.quiver(X, Y, u, v, color="C0", angles='xy',
          scale_units='xy', scale=1000, width=.001)

ax.set(xlim=(0, 12000), ylim=(0, 6000))
ax.set_xlabel('L_x')
ax.set_ylabel('L_y')
ax.set_title('Champ de vitesse intégré')
plt.show()
