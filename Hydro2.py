import numpy as np
import time
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


Length=3
MM=1000
blue = (31/255.0, 119/255.0, 180/255.0)
orange = (255/255.0, 128/255.0, 0/255.0)
purple = (153/255 , 0/255 , 76/255)
green = (0/255 , 102/255 , 51/255)

#np.seterr(all='raise')
    
#inital condition for scalar field
def initialphi(z,p):
    if z > 0.001:
        x=p/((z)**5 * (np.exp(1/z-1)))
    else:
        x=0
    return x


#Rieamnn solver
def HLL(ul,ur,fl,fr,alpha,a,j):#This is the real i
    i=j
    N = len(a)
    if i<N-1:
        vl = alpha[i]/a[i]#velocity at i_th interface
        vr = alpha[i+1]/a[i+1]#velocity at i+1_th interface 1.5 to 0.5
    else:
        vl = alpha[N-1]/a[N-1]#velocity at i_th interface
        vr = alpha[N-1]/a[N-1]#velocity at i+1_th interface 1.5 to 0.5
    alphaplus = max(0,vl,vr,-vl,-vr)
    alphaminus = max(0,vl,vr,-vl,-vr)
    hll = (alphaplus*fl + alphaminus*fr- alphaplus*alphaminus*(ur-ul))/(alphaminus+alphaplus)
    return hll

#First order solver 
def fo(inphi,p,M,N,T):
    u=np.zeros((M,N+4,2),np.longdouble)
    a=np.ones((N))
    alpha=np.ones((N))
    dt=T/M
    dx=Length/N
    for i in range(2,N+2):#imposing boundary condition
        xi=(i-2+0.5)*dx
        u[0,i,:]=np.array([inphi(xi,p),0])#p is critical exponent
    u[0,0,:]=u[0,2,:]
    u[0,1,:]=u[0,2,:]
    u[0,N+2,:]=u[0,N+1,:]
    u[0,N+3,:]=u[0,N+1,:]
#u=((1-z^2)\phi,z^2\pi)
#evolution part
    for i in range(1,M):# i is time
        print(i)
        u[i-1,N+2,:] = u[0,N+2,:]
        u[i-1,N+3,:] = u[0,N+3,:]
        u[i-1,0,0] = -u[i-1,3,0]
        u[i-1,1,0] = -u[i-1,2,0]
        u[i-1,0,1] = u[i-1,3,1]
        u[i-1,1,1] = u[i-1,2,1]
        #print(u[i-1,0,0],u[i-1,1,0],u[i-1,2,0],u[i-1,3,0])
        for j in range(2,N+2):# j is sapce
            #print(i,j)
            rj=dx*(j-2)
            cj=dx*(j-1)
            if j>2 and j<N+1:
                fl = np.array([-u[i-1,j-1,1]*alpha[j-3]/(a[j-3]),-alpha[j-3]*u[i-1,j-1,0]/a[j-3]])
                fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
                fr = np.array([-u[i-1,j+1,1]*alpha[j-1]/a[j-1],-alpha[j-1]*u[i-1,j+1,0]/a[j-1]])
               # flux = 3*(-(rj**2)*HLL(u[i-1,j-1,:],u[i-1,j,:],fl,fc,alpha,a,j-2)+(cj**2)*HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1))/(cj**3-rj**3) 
                flux=0.5*(-fl+fr)/dx
            if j>N:
                fl = np.array([-u[i-1,j-1,1]*alpha[j-3]/(a[j-3]),-alpha[j-3]*u[i-1,j-1,0]/a[j-3]])
                fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
                fr = np.array([-u[i-1,j+1,1],-u[i-1,j+1,0]])
                flux = (-HLL(u[i-1,j-1,:],u[i-1,j,:],fl,fc,alpha,a,j-2)+HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1))/dx
            #print(i,j)
            #flux = -HLL(u[i-1,j-1,:],u[i-1,j,:],fl,fc,alpha,a,j-2)+HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1)
            if j==2:
                fl = np.array([-u[i-1,j-1,1]*alpha[j-3]/a[j-3],-alpha[j-3]*u[i-1,j-1,0]/a[j-3]])
                fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
                fr = np.array([-u[i-1,j+1,1]*alpha[j-1]/a[j-1],-alpha[j-1]*u[i-1,j+1,0]/a[j-1]])
                flux = 0.0 +HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1)/cj
                flux=0.5*fr/dx
            u[i,j,:]=0.5*u[i-1,j+1,:]+0.5*u[i-1,j-1,:]-dt*flux+dt*np.array([2*u[i-1,j,1]/(rj+0.5*dx),0])
       # print(u[i,2:,0],u[i,2:,1])
        a=aevolve(a,alpha,u,i)
        #print(a)
        alpha=alphaevolve(a,alpha,u,i)
       # print(alpha)
    return u,a


#Evolution function for a
def aevolve(a,alpha,u,i):
    N = len(a)
    temp = np.ones(N,np.longdouble)
    for j in range(N-1):
        #print(j)
        phi=u[i,N-j-1,0]
        pi=u[i,N-j-1,1]
        z=(N-j-0.5)/N
        if abs(u[i,N-j-1,0])<np.exp(-140):
            phi=0
        if abs(u[i,N-j-1,1])<np.exp(-140):
            pi=0
        #x=(1-((1-temp[N-j-1]**2)/(2*z)+2*np.pi*z*(pi**2+phi**2))*1/N)
        temp[N-j-2]=temp[N-j-1]+(temp[N-j-1]*((1-temp[N-j-1]**2)/(2*z)+2*np.pi*z*(pi**2+phi**2))*1/N)
    return temp
#Evolution function for alpha
def alphaevolve(a,alpha,u,i):
    N = len(a)
    temp = np.empty(N,np.longdouble)
    temp[N-1]=1
    for j in range(N-1):
        z=(N-j-0.5)/N
        pi=u[i,N-j-1,1]
        phi=u[i,N-j-1,0]
        if abs(u[i,N-j-1,0])<np.exp(-140):
            phi=0
        if abs(u[i,N-j-1,1])<np.exp(-140):
            pi=0
        temp[N-j-2]=temp[N-j-1]-(temp[N-j-1]*((-1+temp[N-j-1]**2)/(2*z)+2*np.pi*z*(pi**2+phi**2))*1/N)
    return temp


def plotter(u,numplots):
    N = u.shape[0]
    O=15#slicing
    NN=len(u[0,2:u.shape[1]-2:O,0])
    b=(np.arange(NN)/NN+0.5/NN)*Length
    c=np.empty(NN)
    c=b*(-1/a[0:N:O]+1)
    minim=np.amin(u[:,:,0])
    maxim=np.amax(u[:,:,0])
    for i in range(numplots):
        print("Plot {0:d}".format(i))
        dN = int(N/numplots)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        rho=u[i*dN,2:u.shape[1]-2:O,0]
        z=Length*(np.arange(NN)/NN+0.5/NN)
        revolve_steps = np.linspace(0, np.pi*2, NN).reshape(1,NN)
        theta = revolve_steps
        #convert rho to a column vector
        rho_column = z.reshape(NN,1)
        x = rho_column.dot(np.cos(theta))
        y = rho_column.dot(np.sin(theta))
        # expand z into a 2d array that matches dimensions of x and y arrays..
        # i used np.meshgrid
        zs, rs = np.meshgrid(z, rho)
        surf = ax.plot_surface(x, y,rs, rstride=1, cstride=1,alpha=0.3, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #ax.set_xlim(-2.5, 2.5)
        #ax.set_ylim(-2.5, 2.5)
        #ax.set_zlim(-1, 5)
        #view orientation
        #ax.elev = 40#30 degrees for a typical isometric view
        #ax.azim= 30
        #ax.dist=7.5
        #turn off the axes to closely mimic picture in original question
        ax.set_axis_off()
        fig.colorbar(surf, shrink=0.5, aspect=5)
        cset = ax.contourf(x, y, rs, zdir='z',offset=minim-1,cmap=cm.winter)    
        cset = ax.contourf(x, y, rs, zdir='x', offset=-4,cmap=cm.cool)
        cset = ax.contourf(x, y, rs, zdir='y', offset=4, cmap=cm.coolwarm)
        ax.set_xlabel('X')
        ax.set_xlim(-3, 3)
        ax.set_ylabel('Y')
        ax.set_ylim(-3, 3)
        ax.set_zlabel('Z')
        ax.set_zlim(minim-3, maxim)
        ax.set_title('3D surface and contours', va='bottom')

        ax.view_init(elev=25, azim=-58)           # elevation and angle
        ax.dist=9 
        fig.savefig("p{0:03d}.png".format(i))
        plt.close(fig)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(c,'r.-',ms=5)
    fig.savefig("c{0:03d}.png".format(i))
    plt.close(fig)



##MMMAAAIIINNNN
u,a=fo(initialphi,0.25,8000,3000,2.8)

numplots = 100
#plotter(u,numplots)

N = u.shape[0]
NN=u.shape[1]-4
b=(np.arange(NN)/NN+0.5/NN)*Length
c=np.empty(NN)
c=b*(-1/a+1)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(c,'r.-',ms=5)
fig.savefig("c")

for i in range(numplots):
    print("Plot {0:d}".format(i))
    dN = int(N/numplots)
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(u[i*dN,:,0],'r.-',ms=5)
    ax = fig.add_subplot(2,1,2)
    ax.plot(u[i*dN,:,1],'r.-',ms=5)
    fig.savefig("p1{0:03d}.png".format(i))
    plt.close(fig)


