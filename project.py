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
    if z > 0.005:
        x=p/((z)**5 * (np.exp(1/z-1)))
    else:
        x=0
    return x


#Rieamnn solver
def HLL(ul,ur,fl,fr,alpha,a,j):#This is the real i
    i=j-1
    N = len(a)
    vl = alpha[i]/a[i]#velocity at i_th interface
    vr = alpha[i+1]/a[i+1]#velocity at i+1_th interface 1.5 to 0.5
    alphaplus = max(0,vl,vr,-vl,-vr)
    alphaminus = max(0,vl,vr,-vl,-vr)
    hll = (alphaplus*fl + alphaminus*fr- alphaplus*alphaminus*(ur-ul))/(alphaminus+alphaplus)
    return hll

#First order solver 
def fo(inphi,p,M,N,T):
    u=np.zeros((M,N+4,2),np.longdouble)
    a=np.ones((N+2))
    alpha=np.ones((N+2))
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
        if i == 0:
            u[i-1,N+2,:] = u[0,N+1,:]
        else:
            u[i-1,N+2,:] = u[i-2,N+1,:]-dt*(u[i-2,N+1,:]-u[i-2,N+2,:])/dx
        u[i-1,N+3,:] = 0#u[i-1,N,:]
        u[i-1,0,0] = -u[i-1,3,0] 
        u[i-1,1,0] = -u[i-1,2,0]
        u[i-1,0,1] = u[i-1,3,1]
        u[i-1,1,1] = u[i-1,2,1]
        Flux= np.zeros((N+4,2))
        for j in range(2,N+2):# j is sapce
            #print(i,j)
            rj=dx*(j-2)
            cj=dx*(j-1)
            fr = np.array([-u[i-1,j+1,1]*alpha[j-1]/a[j-1],-alpha[j-1]*u[i-1,j+1,0]/a[j-1]])
            fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
            Flux[j,:]=3*(cj**2)*HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1)
            u[i,j,:]=u[i-1,j,:]+dt*(Flux[j-1,:]-Flux[j,:])/(cj**3-rj**3)-dt*np.array([2*u[i-1,j,1]*alpha[j-2]/((rj+0.5*dx)*a[j-2]),0])
        m=mevolve(u[i,:,:])
        #alpha=alphaevolve(a,alpha,u,i)
      
        #m=rk4(u[i,:,:],dm,0)
        rs=Length*(np.arange(N+2)+0.5)/(N+2)
        a=1/(1-2*m/rs)
        #alp=rk4(u[i,:,:],dalpha,1)
        alp=alphaevolve(u[i,:,:])
        alpha=alp/a
        q=np.amax(alpha)
        alpha=alpha/q
        
        if i%50==1:
            print("time:",i)
            #print(Flux[390:,:],u[i-1,390:,:])   
            fig = plt.figure()
            ax = fig.add_subplot(4,1,1)
            ax.plot(alpha,'r.-',ms=5)
            ax = fig.add_subplot(4,1,2)
            ax.plot(a,'r.-',ms=5)
            ax = fig.add_subplot(4,1,3)
            ax.plot(u[i,:,0],'r.-',ms=5)
            ax = fig.add_subplot(4,1,4)
            ax.plot(u[i,:,1],'r.-',ms=5)
            #plt.show()
            fig.savefig("a{0:03d}.png".format(int(i/50)))
            plt.close(fig)
        return u,a


#First order solver 
def fzo(inphi,p,M,N,T):
    u=np.zeros((M,N+4,2),np.longdouble)
    a=np.ones((N+2))
    alpha=np.ones((N+2))
    dt=T/M
    dx=.99/N
    for i in range(2,N+2):#imposing boundary condition
        xi=(i-2+0.5)*dx
        xi=xi/(1-xi)
        u[0,i,:]=np.array([inphi(xi,p),0])#p is critical exponent
    u[0,0,:]=u[0,2,:]
    u[0,1,:]=u[0,2,:]
    u[0,N+2,:]=u[0,N+1,:]
    u[0,N+3,:]=u[0,N+1,:]
#u=((1-z^2)\phi,z^2\pi)
#evolution part
    #fig = plt.figure()
    #time=np.zeros(N+2)
    for i in range(1,M):# i is time
        u[i-1,N+2,:] = u[0,N+1,:]
        u[i-1,N+3,:] = 0#u[i-1,N,:]
        u[i-1,0,0] = -u[i-1,3,0] 
        u[i-1,1,0] = -u[i-1,2,0]
        u[i-1,0,1] = u[i-1,3,1]
        u[i-1,1,1] = u[i-1,2,1]
        Flux= np.zeros((N+4,2))
        for j in range(2,N+2):# j is sapce
            #print(i,j)
            rj=dx*(j-2)
            z=rj+0.5*dx
            cj=dx*(j-1)
            v=(-3*cj**2+3*cj-1)/(3*(cj-1)**3)-(-3*rj**2+3*rj-1)/(3*(rj-1)**3)
            fr = np.array([-u[i-1,j+1,1]*alpha[j-1]/a[j-1],-alpha[j-1]*u[i-1,j+1,0]/a[j-1]])
            fc = np.array([-u[i-1,j,1]*alpha[j-2]/a[j-2],-alpha[j-2]*u[i-1,j,0]/a[j-2]])
            Flux[j,:]=((cj/(1-cj))**2)*HLL(u[i-1,j,:],u[i-1,j+1,:],fc,fr,alpha,a,j-1)
            u[i,j,:]=u[i-1,j,:]+dt*(Flux[j-1,:]-Flux[j,:])/v-dt*np.array([2*u[i-1,j,1]*alpha[j-2]*(1/z-1)/(a[j-2]),0])

        m=mzevolve(u[i,:,:])
        #m=rk4(u[i,:,:],dm,0)
        rs=(np.arange(N+2)+0.5)/(N+2)
        a=1/(1-2*m*(1/rs-1))
        #alp=rk4(u[i,:,:],dalpha,1)
        alp=alphazevolve(u[i,:,:])
        alpha=alp/a
        q=np.amax(alpha)
        alpha=alpha/q
        ms=m[0:int(0.7*N)]
        mmax=np.amax(ms)
        #time=time+dt*alpha#for making time constant slices
        #plot snap shots of the metric components and save them
        if i==1100:
            print("time:",i,np.amax(a[0:int(0.7*N)]))  
            fig = plt.figure()
            ax = fig.add_subplot(3,1,1)
            ax.plot(rs[0:int(0.7*N)],alpha[0:int(0.7*N)],'r.',ms=5,color=green)
            ax.set_ylabel(r"$\alpha$",fontsize=17)
            #ax.set_ylim(.9975,1)
            ax = fig.add_subplot(3,1,2)
            ax.plot(rs[0:int(0.7*N)],a[0:int(0.7*N)],'x',ms=5,color=blue)
            ax.set_ylabel(r"a(r)",fontsize=17)
            #ax.set_ylim(1,1.0006)
            ax = fig.add_subplot(3,1,3)
            ax.plot(rs[0:int(0.7*N)],m[0:int(0.7*N)],'x',ms=5,color=orange)
            ax.set_ylabel(r"m(r)",fontsize=17)
            #ax.set_ylim(0, 0.00015)
            plt.show()
            #fig.savefig("bh{0:03d}.png".format(int(i/100)),dpi=400)
            #plt.close(fig)
        
    #fig.savefig("time.png",dpi=400)
    #plt.close(fig)
    return a

#Evolution function for a
def mzevolve(u):
    pi=u[2,1]
    phi=u[2,0]
    if abs(phi)<np.exp(-140):
        phi=0
    if abs(pi)<np.exp(-140):
        pi=0
    N=u.shape[0]-2
    xpoint = np.empty(N)
    h=.99/(N-2)
    xpoint[0]=(pi**2+phi**2)*np.pi*h**3/(3*8)
    xpoint[1]=(pi**2+phi**2)*np.pi*9*h**3/8
    for j in range(N-2):
        r=h*(j+2.5)
        pi=u[j+3,1]
        phi=u[j+3,0]
        if abs(phi)<np.exp(-140):
            phi=0
        if abs(pi)<np.exp(-140):
            pi=0
        xpoint[j+2]=xpoint[j]+h*2*np.pi*r*(pi**2+phi**2)*(r/(1-r)-2*xpoint[j])/(1-r)**3
    return xpoint
#Evolution function for alpha
def alphazevolve(u):
    N=u.shape[0]-2
    xpoint = np.empty(N)
    h=.99/(2*(N-2))
    pi=u[2,1]
    phi=u[2,0]
    if abs(phi)<np.exp(-140):
        phi=0
    if abs(pi)<np.exp(-140):
        pi=0
    xpoint[0]=0
    x=0
    y=4*np.pi*h*(pi**2+phi**2)/(1-h)**3
    for j in range(N-1):
        r=2*h*(j+1.5)
        pi=u[j+3,1]
        phi=u[j+3,0]
        if abs(phi)<np.exp(-140):
            phi=0
        if abs(pi)<np.exp(-140):
            pi=0
        xpoint[j+1]=xpoint[j]+h*(4*np.pi*r*(pi**2+phi**2)/(1-r)**3+y)
        y=4*np.pi*r*(pi**2+phi**2)
    #print(np.amax(xpoint))
    xpoint=np.exp(xpoint)
    return xpoint

#RK4 implementation for evolution of a and alpha
def rk4(u,f,init):
    N=u.shape[0]-2
    xpoints = np.empty(N)
    h=2*Length/(N-2)
    x=init
    pi=0.5*u[2,1]+0.5*u[3,1]
    phi=0.5*u[2,0]+0.5*u[3,0]
    if abs(phi)<np.exp(-140):
        phi=0
    if abs(pi)<np.exp(-140):
        pi=0
    y=init+f(0,Length/N,phi,pi)*h/2
    xpoints[0]=x
    xpoints[1]=y
    for i in range(int(N/2+0.5)-1):
        #even part of the array
        pi0=u[2*i+2,1]
        phi0=u[2+2*i,0]
        pi1=u[2*i+3,1]
        phi1=u[2*i+3,0]
        pi2=u[2*i+4,1]
        phi2=u[2*i+4,0]
        if abs(phi0)<np.exp(-100):
            phi0=0
        if abs(pi0)<np.exp(-100):
            pi0=0
        if abs(phi1)<np.exp(-100):
            phi1=0
        if abs(pi1)<np.exp(-100):
            pi1=0
        if abs(phi2)<np.exp(-100):
            phi2=0
        if abs(pi2)<np.exp(-100):
            pi2=0
        r=((2*i)+0.5)*Length/N
        k1 = h*f(x,r,phi0,pi0)
        k2 = h*f(x+0.5*k1,r+0.5*h,phi1,pi1)
        k3 = h*f(x+0.5*k2,r+0.5*h,phi1,pi1)
        k4 = h*f(x+k3,r+h,phi2,pi2)
        x = x+ (k1 + 2*k2 + 2*k3 + k4)/6
        #odd part of the array
        pi0=u[2*i+3,1]
        phi0=u[2*i+3,0]
        pi1=u[2*i+4,1]
        phi1=u[2*i+4,0]
        pi2=u[2*i+5,1]
        phi2=u[2*i+5,0]
        if abs(phi0)<np.exp(-100):
            phi0=0
        if abs(pi0)<np.exp(-100):
            pi0=0
        if abs(phi1)<np.exp(-100):
            phi1=0
        if abs(pi1)<np.exp(-100):
            pi1=0
        if abs(phi2)<np.exp(-100):
            phi2=0
        if abs(pi2)<np.exp(-100):
            pi2=0
        r=((1+2*i)+0.5)*Length/N
        k1 = h*f(y,r,phi0,pi0)
        k2 = h*f(y+0.5*k1,r+0.5*h,phi1,pi1)
        k3 = h*f(y+0.5*k2,r+0.5*h,phi1,pi1)
        k4 = h*f(y+k3,r+h,phi2,pi2)
        y = y+ (k1 + 2*k2 + 2*k3 + k4)/6
        xpoints[2*i+2]=x
        xpoints[2*i+3]=y
    ypoints=np.empty(len(xpoints))
    ypoints[0]=xpoints[0]
    ypoints[-1]=xpoints[-1]
    for i in range(1,len(xpoints)-1):
        ypoints[i]=0.25*(xpoints[i-1]+xpoints[i+1])+0.5*xpoints[i]
    return ypoints

#Evolution function for a
def dm(m,r,phi,pi):
    temp=2*np.pi*r*(pi**2+phi**2)*(r-2*m)
    return temp
#Evolution function for alpha
def dalpha(alpha,r,phi,pi):
    temp=alpha*4*np.pi*r*(pi**2+phi**2)
    return temp

#Evolution function for a
def mevolve(u):
    pi=u[2,1]
    phi=u[2,0]
    if abs(phi)<np.exp(-140):
        phi=0
    if abs(pi)<np.exp(-140):
        pi=0
    N=u.shape[0]-2
    xpoint = np.empty(N)
    h=Length/(N-2)
    xpoint[0]=(pi**2+phi**2)*np.pi*h**3/(3*8)
    theta=(16/h**4)*((pi**2+phi**2)*np.pi*(h**2/4-4*xpoint[0]))
    xpoint[1]=(pi**2+phi**2)*np.pi*9*h**3/8+theta*(3*h/2)**4*h
    for j in range(N-2):
        r=h*(j+2.5)
        pi=u[j+3,1]
        phi=u[j+3,0]
        if abs(phi)<np.exp(-140):
            phi=0
        if abs(pi)<np.exp(-140):
            pi=0
        xpoint[j+2]=xpoint[j]+h*2*np.pi*r*(pi**2+phi**2)*(r-2*xpoint[j])
    return xpoint
#Evolution function for alpha
def alphaevolve(u):
    N=u.shape[0]-2
    xpoint = np.empty(N)
    h=Length/(2*(N-2))
    pi=u[2,1]
    phi=u[2,0]
    if abs(phi)<np.exp(-140):
        phi=0
    if abs(pi)<np.exp(-140):
        pi=0
    xpoint[0]=0
    x=0
    y=4*np.pi*h*(pi**2+phi**2)
    for j in range(N-1):
        r=2*h*(j+1.5)
        pi=u[j+3,1]
        phi=u[j+3,0]
        if abs(phi)<np.exp(-140):
            phi=0
        if abs(pi)<np.exp(-140):
            pi=0
        xpoint[j+1]=xpoint[j]+h*(4*np.pi*r*(pi**2+phi**2)+y)
        y=4*np.pi*r*(pi**2+phi**2)
    #print(np.amax(xpoint))
    xpoint=np.exp(xpoint)
    return xpoint


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
    dx=Length/N
    Flux= np.zeros((N+4,2))
    temp= np.zeros((N+4,2))
    for i in range(2,N+2):
        ri=dx*(i-1)
        ur=advcr(u[i,:],u[i+1,:],u[i+2,:],theta)
        ul=advcl(u[i-1,:],u[i,:],u[i+1,:],theta)
        c=(a[i-2]+a[i-1])/(alpha[i-1]+alpha[i-2])
        fr=-c*np.array([ur[1],ur[0]])
        fl=-c*np.array([ul[1],ul[0]])
        F1=HLL(ul,ur,fl,fr,alpha,a,i-2)*3*ri**2
        Flux[i,:]=F1
    for j in range(2,N+2):# j is sapce
        #print(i,j)
        rj=dx*(j-2)
        cj=dx*(j-1)   
        temp[j]=(Flux[j-1,:]-Flux[j,:])/(cj**3-rj**3)
    return temp
#Higher order solver 
def ho(inphi,p,M,N,theta,T):
    a=np.ones((N+2))
    alpha=np.ones((N+2))
    u=np.empty((M,N+4,2))
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
    temp=np.ones((N+4,2))
    temp[:,1]=0
    for i in range(1,M):
        for j in range(N):
            temp[j+2,0]=alpha[j]/a[j]
        temp[:,0]=-temp[:,0]*dt*2*u[i-1,:,1]/r[:]
        temp[0,:]=0
        temp[1,:]=0
        u[i-1,N+2,:]=u[i-1,N+1,:]
        u[i-1,N+3,:]=u[i-1,N+1,:]
        u[i-1,0,0] = -u[i-1,3,0]
        u[i-1,1,0] = -u[i-1,2,0]
        u[i-1,0,1] = u[i-1,3,1]
        u[i-1,1,1] = u[i-1,2,1]
        u1=u[i-1,:,:]+L(u[i-1,:,:],theta,alpha,a,N)*dt
        u2=0.75*u[i-1,:,:]+0.25*u1+0.25*L(u1,theta,alpha,a,N)*dt
        u[i,:,:]=u[i-1,:,:]/3+2*u2/3+2*L(u2,theta,alpha,a,N)*dt/(3)+temp


        #m=mevolve(u[i,:,:])
        #alpha=alphaevolve(a,alpha,u,i)
      
        #m=rk4(u[i,:,:],dm,0)
        rs=Length*(np.arange(N+2)+0.5)/(N+2)
        #a=1/(1-2*m/rs)
        #alp=rk4(u[i,:,:],dalpha,1)
        #alp=alphaevolve(u[i,:,:])
        #alpha=alp/a
        #q=np.amax(alpha)
        #alpha=alpha/q
        
        if i%200==1:
            print("time:",i)
            #print(Flux[390:,:],u[i-1,390:,:])   
            fig = plt.figure()
            ax = fig.add_subplot(4,1,1)
            ax.plot(alpha,'r.-',ms=5)
            ax = fig.add_subplot(4,1,2)
            ax.plot(a,'r.-',ms=5)
            ax = fig.add_subplot(4,1,3)
            ax.plot(u[i,0:11,0],'r.-',ms=5)
            ax = fig.add_subplot(4,1,4)
            ax.plot(u[i,0:11,1],'r.-',ms=5)
            #plt.show()
            fig.savefig("a{0:03d}.png".format(int(i/50)))
            plt.close(fig)

    return u,a

def plotter(u,numplots):
    N = u.shape[0]
    O=15#slicing
    NN=len(u[0,2:int(0.7*N):O,0])
    minim=np.amin(u[:,:int(0.7*N),0])
    maxim=np.amax(u[:,:int(0.7*N),0])
    for i in range(numplots):
        print("Plot {0:d}".format(i))
        dN = int(N/numplots)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        rho=u[i*dN,2:int(0.7*N):O,0]
        z=Length*(np.arange(NN)/NN+0.5/NN)
        revolve_steps = np.linspace(0, np.pi*2, NN).reshape(1,NN)
        theta = revolve_steps
        #convert rho to a column vector
        rho_column = z.reshape(NN,1)
        x = rho_column.dot(np.cos(theta))
        y = rho_column.dot(np.sin(theta))
        # expand z into a 2d array that matches dimensions of x and y arrays..
        zs, rs = np.meshgrid(z, rho)
        surf = ax.plot_surface(x, y,rs, rstride=1, cstride=1,alpha=0.6, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #turn off the axes to closely mimic picture in original question
        ax.set_axis_off()
        fig.colorbar(surf, shrink=0.5, aspect=5)
        cset = ax.contourf(x, y, rs, zdir='z',offset=minim*1.5,cmap=cm.winter)    
        cset = ax.contourf(x, y, rs, zdir='x', offset=-6,cmap=cm.cool)
        #cset = ax.contourf(x, y, rs, zdir='y', offset=6, cmap=cm.coolwarm)
        ax.set_xlabel('X')
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylabel('Y')
        ax.set_ylim(-4.5, 4.5)
        ax.set_zlabel('Z')
        ax.set_zlim(minim*1.5, maxim)
        #ax.set_title('3D surface and contours', va='bottom')

        ax.view_init(elev=10+55*i/(numplots), azim=-58)           # elevation and angle
        ax.dist=8
        fig.savefig("plot{0:03d}.png".format(i),dpi=400)
        plt.close(fig)
    plt.close(fig)
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

u=fzo(initialphi,0.1,1200,1000,6)

#critical exponent finder
"""
p=.0225
parray=[]
marray=[]
for i in range(34):
    m=fzo(initialphi,p,8000,400,40)
    parray.append(p)
    marray.append(m)
    p=p+.005
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(parray,marray,'s',ms=5,color=orange)
fig.savefig("mvsp.png".format(i),dpi=400)
plt.close(fig)
np.savetxt('ms.txt',marray)
np.savetxt('ps.txt',parray)
numplots = 200
#3D plotter
plotter(u,numplots)
"""
