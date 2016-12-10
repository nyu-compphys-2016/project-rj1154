import numpy as np
import time
import pylab as pl
import matplotlib.pyplot as plt
import cmath as cm
import matplotlib.animation as animation


Length=3
MM=1000
NN=250
blue = (31/255.0, 119/255.0, 180/255.0)
orange = (255/255.0, 128/255.0, 0/255.0)
purple = (153/255 , 0/255 , 76/255)
green = (0/255 , 102/255 , 51/255)


    
#inital condition for scalar field
def initialphi(z,p):
    if z > 0:
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
    #if j<5:
        #print(j,hll,ur-ul)
    return hll

#First order solver 
def fo(inphi,p,M,N,T):
    u=np.zeros((M,N+4,2))
    a=np.ones((N))
    alpha=np.ones((N))
    dt=T/M
    dx=Length/N
    for i in range(2,N+2):#imposing boundary condition
        xi=(i-2+0.5)*dx
        u[0,i,:]=np.array([inphi(xi,p),0])#p is critical exponent
    u[0,0,:]=-u[0,3,:]
    u[0,1,:]=-u[0,2,:]
    u[0,N+2,:]=u[0,N+1,:]
    u[0,N+3,:]=u[0,N+1,:]
#u=((1-z^2)\phi,z^2\pi)
#evolution part
    for i in range(1,M):# i is time
        u[i-1,N+2,:] = u[0,N+2,:]
        u[i-1,N+3,:] = u[0,N+3,:]
        u[i-1,0,0] = -u[i-1,3,0]
        u[i-1,1,0] = -u[i-1,2,0]
        u[i-1,0,1] = u[i-1,3,1]
        u[i-1,1,1] = u[i-1,2,1]
        for j in range(2,N+2):# j is sapce index
            #print(i,j)
            rj=dx*(j-2)
            cj=dx*(j-1)
            if j>2 and j<N+1:
                fl = np.array([-u[i-1,j-1,1]*alpha[j-3]/(a[j-3]),-alpha[j-3]*u[i-1,j-1,0]/a[j-3]])
                fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
                fr = np.array([-u[i-1,j+1,1]*alpha[j-1]/a[j-1],-alpha[j-1]*u[i-1,j+1,0]/a[j-1]])
                leftflux=HLL(u[i-1,j-1,:],u[i-1,j,:],fl,fc,alpha,a,j-2)
                flux = 3*(-(rj**2)*leftflux+(cj**2)*HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1))/(cj**3-rj**3) 
                #flux=0.5*(-fl+fr)/dx
            if j>N:
                fl = np.array([-u[i-1,j-1,1]*alpha[j-3]/(a[j-3]),-alpha[j-3]*u[i-1,j-1,0]/a[j-3]])
                fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
                fr = np.array([-u[i-1,j+1,1],-u[i-1,j+1,0]])
                flux = (-HLL(u[i-1,j-1,:],u[i-1,j,:],fl,fc,alpha,a,j-2)+HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1))/dx
            #print(i,j)
            #flux = -HLL(u[i-1,j-1,:],u[i-1,j,:],fl,fc,alpha,a,j-2)+HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1)
            if j==2:
                fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
                fr = np.array([-u[i-1,j+1,1]*alpha[j-1]/a[j-1],-alpha[j-1]*u[i-1,j+1,0]/a[j-1]])
                #rightflux=HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1)
                #flux = rightflux/cj
                #print(j,rightflux)
                flux=0.5*fr/dx
            u[i,j,:]=u[i-1,j,:]-dt*flux+dt*np.array([3*u[i-1,j,1]*(cj+rj)/(cj**2+rj**2+cj*rj),0])
       # print(u[i,2:,0],u[i,2:,1])
        #a=aevolve(a,alpha,u,i)
        #print(a)
        #alpha=alphaevolve(a,alpha,u,i)
       # print(alpha)
    return u


#Evolution function for a
def aevolve(a,alpha,u,i):
    N = len(a)
    temp = np.ones((N))
    for j in range(N-1):
        #print(j)
        z=(N-j-0.5)/N
        pi=(1-z)**4*u[i,N-j-1,1]/z**2
        phi=(1-z)**2*u[i,N-j-1,0]
        x=(1-((1-temp[N-j-1]**2)*(1-z)/(2*z)+2*np.pi*z*(pi**2+phi**2)/(1-z))*1/N)/(1-z)**2
       # print(pi,phi,temp[N-j-1],x,z)
        temp[N-j-2]=temp[N-j-1]+(temp[N-j-1]*((1-temp[N-j-1]**2)*(1-z)/(2*z)+2*np.pi*z*(pi**2+phi**2)/(1-z))*1/N)/(1-z)**2
    return temp
#Evolution function for alpha
def alphaevolve(a,alpha,u,i):
    N = len(a)
    temp = np.empty((N))
    temp[N-1]=1
    for j in range(N-1):
        z=(N-j-0.5)/N
        pi=(1-z)**4*u[i,N-j-1,1]/z**2
        phi=(1-z)**2*u[i,N-j-1,0]
        temp[N-j-2]=temp[N-j-1]-(temp[N-j-1]*((-1+temp[N-j-1]**2)*(1-z)/(2*z)+2*np.pi*z*(pi**2+phi**2)/(1-z))*1/N)/(1-z)**2
    return temp



def minmod(a,b,c):
    if a>0 and b>0 and c>0:
        return min(a,b,c)
    else:
        if a<0 and b<0 and c<0:
            return max(a,b,c)
        else:
            return 0

#The advcl and advcr are the functions to find the left and right state in higher order method
def advcl(ua,ub,uc,theta):
    rhol=ub[0]+0.5*minmod(theta*(ub[0]-ua[0]),0.5*(uc[0]-ua[0]),theta*(uc[0]-ub[0]))
    el=ub[1]+0.5*minmod(theta*(ub[1]-ua[1]),0.5*(uc[1]-ua[1]),theta*(uc[1]-ub[1]))
    ul=np.array([rhol,el])
    return ul
def advcr(ua,ub,uc,theta):
    rhor=ub[0]-0.5*minmod(theta*(ub[0]-ua[0]),0.5*(uc[0]-ua[0]),theta*(uc[0]-ub[0]))
    er=ub[1]-0.5*minmod(theta*(ub[1]-ua[1]),0.5*(uc[1]-ua[1]),theta*(uc[1]-ub[1]))
    ur=np.array([rhor,er])
    return ur
#L=dU/dt, the definition is in the problem set
def L(u,theta,alpha,a,N):
    Flux= np.zeros((N+4,2))
    for i in range(2,N+2):
        ur=advcr(u[i,:],u[i+1,:],u[i+2,:],theta)
        ul=advcl(u[i-1,:],u[i,:],u[i+1,:],theta)
        if i<N+1:
            c=(a[i-2]+a[i-1])/(alpha[i-1]+alpha[i-2])
        else:
            c=a[i-2]/alpha[i-2]
        fr=-c*np.array([ur[1],ur[0]])
        fl=-c*np.array([ul[1],ul[0]])
        F1=HLL(ul,ur,fl,fr,alpha,a,i-2)
        vr=advcr(u[i-1,:],u[i,:],u[i+1,:],theta)
        vl=advcl(u[i-2,:],u[i-1,:],u[i,:],theta)
        kr=-c*np.array([vr[1],vr[0]])
        kl=-c*np.array([vl[1],vl[0]])
        F2=HLL(vl,vr,kl,kr,alpha,a,i-2)
        Flux[i,:]=F2-F1
    return Flux
#Higher order solver 
def higherorder(inphi,p,M,N,theta,T):
    a=np.ones((N))
    alpha=np.ones((N))
    u=np.empty((M,N+4,2))
    f=np.empty((N-1,2))
    dt=T/M
    dx=Length/N
    for i in range(2,N+2):
        xi=(i-2+0.5)*dx
        u[0,i,:]=np.array([inphi(xi,p),0])
    u[0,0,:]=u[0,2,:]
    u[0,1,:]=u[0,2,:]
    u[0,N+2,:]=u[0,N+1,:]
    u[0,N+3,:]=u[0,N+1,:]
    u1=np.zeros((N+4,2))
    u2=np.zeros((N+4,2))
    r=dx*np.arange(N+4)-1.5*dx
    r[0]=1.5*dx
    r[1]=0.5*dx#ghost cell coordiantes
    temp=np.ones((N+4,2))
    temp[:,1]=0
    for i in range(1,M):
        for j in range(N):
            temp[j+2,0]=alpha[j]/a[j]
        temp[:,0]=temp[:,0]*dt*2*u[i-1,:,1]/r[:]
        u[i-1,N+2,:]=u[i-1,N+1,:]
        u[i-1,N+3,:]=u[i-1,N+1,:]
        u[i-1,0,0] = -u[i-1,3,0]
        u[i-1,1,0] = -u[i-1,2,0]
        u[i-1,0,1] = u[i-1,3,1]
        u[i-1,1,1] = u[i-1,2,1]
        u1=u[i-1,:,:]+L(u[i-1,:,:],theta,alpha,a,N)*dt/dx
        u2=0.75*u[i-1,:,:]+0.25*u1+0.25*L(u1,theta,alpha,a,N)*dt/dx
        u[i,:,:]=u[i-1,:,:]/3+2*u2/3+2*L(u2,theta,alpha,a,N)*dt/(3*dx)+temp
    return u
"""
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
z=(np.arange(N+4)-1.5)/(N)
ax.plot(z[:],u[i-1,:,0],'r.-',ms=5)
ax = fig.add_subplot(2,1,2)
ax.plot(u[i-1,:,1],'r.-',ms=5)
plt.show() 
"""

##MMMAAAIIINNNN
u=fo(initialphi,0.1,800,300,2)

numplots = 100

for i in range(numplots):
    print("Plot {0:d}".format(i))
    N = u.shape[0]
    dN = int(N/numplots)
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(u[i*dN,:,0],'r.-',ms=5)
    ax = fig.add_subplot(2,1,2)
    ax.plot(u[i*dN,:,1],'r.-',ms=5)
    fig.savefig("ph{0:03d}.png".format(i))
    plt.close(fig)
 

plt.show()

