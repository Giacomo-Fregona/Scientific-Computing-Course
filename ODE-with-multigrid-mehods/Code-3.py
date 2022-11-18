#___________________ Scientific Computing ___________________ Project 3 ___________________ Giacomo Fregona ___________________

import numpy as np 
from time import time

#Multigrid in 1d

#1.

def gs_step_1d(uh, fh):#Gauss Seidel method
    N = len(uh) -1
    h = 4 / N
    old_uh = uh.copy()#copyng the imput uh. It will be used in the computation of uh and of the pseudo residual
    for i in range(1,N) :
        uh[i] = ((h**2)*fh[i]+uh[i-1]+old_uh[i+1])/(2+h**2)
    return max(     [   abs(uh[i]-old_uh[i]) for i in range(N+1)    ]      ) #returning the pseudo residual


#2.

def f_1d(x):# function f
    return (x-1)**3

def g_1d(x):# function g
    return (x**2)/10

def grid_1d(N): #returns the grid corresponding to the given N
    return np.linspace(-2, 2, N+1)

if __name__ == "__main__":
    values_N = [2**l for l in range(3,8)]# stores the values N chosen for our investigations
    iterations2 = [0 for i in range(len(values_N))]# stores the needed iterations
    times2 = [0. for i in range(len(values_N))]#stores the needed CPU times
    for i in range(len(values_N)) :
        N = values_N[i]
        grid = grid_1d(N)

        #defining fh
        fh = np.array([f_1d(x) for x in grid])
        fh[0] = 2/5
        fh[N] = 2/5

        #defining uh
        uh = np.zeros(N+1)
        uh[0] = 2/5
        uh[N] = 2/5

        #measuring times and iterations needed
        counter = 1
        tic = time()
        while gs_step_1d(uh, fh) > 10**(-8) :
            counter+=1
        times2[i] = time()-tic
        iterations2[i] = counter
        
    
    print("\n\n\n\ntimes for .2:\n",times2,"\n\niterations for .2:\n",iterations2)#printing the results

#3.

def to_coarser_grid(fv): #from finer vector to coarser vector (restriction)
    N = len(fv)-1
    if N %2 != 0 : raise Exception("The grid has an odd number of intervals")#we can not apply the restiction in grids with an odd number of intervals
    n = N // 2
    cv = np.zeros(n+1)# preallocating the coarser vector
    cv[0] = fv [0]
    cv[n] = fv [N]
    for i in range(1,n):
        cv[i]= (fv[2*i-1]+ 2*fv[2*i]+fv[2*i+1])/4#filling the entries of cv with weighted values
    return cv



def to_finer_grid(cv):#from coarser vector to finer vector (prolongation)
    n = len(cv)-1
    N = 2 * n
    fv = np.zeros(N+1)
    fv[0] = cv[0]
    fv[N] = cv[n]
    for i in range(1,n):
        fv[2*i] = cv[i]
        fv[2*i-1] = (fv[2*i-2] + fv[2*i]) / 2
    fv[N-1] = (fv[N-2] + fv[N]) / 2
    return fv
        

def two_grid_step(uh,fh):
    #temp = uh.copy()
    N = len(uh) -1
    h = 4 / N

    #Defining Ah
    Ah = (2*np.diag(np.ones(N+1))-np.diag(np.ones(N),1) - np.diag(np.ones(N),-1)  )/(h**2)+  np.diag(np.ones(N+1))
    Ah[0,0] = 1
    Ah[0,1] = 0
    Ah[N,N] = 1
    Ah[N,N-1] = 0
    
    #pre-smoothing
    gs_step_1d(uh, fh)

    #Compute the residual
    rh = fh - Ah@uh

    #Restriction of residual
    crh = to_coarser_grid(rh)#coarser residual

    #Approximating the solution with the GS 
    ceh = np.zeros((N //2)+1)
    for _ in range(5):
        gs_step_1d(ceh, crh)
    
    #Coarse grid correction
    uh+=to_finer_grid(ceh)

    #Post-smoothing and returning the pseudo residual    
    return gs_step_1d(uh, fh)



if __name__ == "__main__":# exercise .3 investigations (same as ex .2 but using the two grid step method)
    values_N = [2**l for l in range(3,8)]
    iterations3 = [0 for _ in range(len(values_N))]
    times3 = [0. for _ in range(len(values_N))]
    for i in range(len(values_N)) :
        N = values_N[i]
        grid = grid_1d(N)
        fh = np.array([f_1d(x) for x in grid])
        fh[0] = 2/5
        fh[N] = 2/5
        uh = np.zeros(N+1)
        uh[0] = 2/5
        uh[N] = 2/5
        counter = 1
        tic = time()
        while two_grid_step(uh,fh) > 10**(-8) :
            counter+=1
        iterations3[i] = counter
        times3[i] = time()-tic

    print("\n\n\n\ntimes for .3:\n",times3,"\n\niterations for .3:\n",iterations3)#printing the results

#4.

def v_cycle_step_1d(uh, fh, alpha1, alpha2):
    N = len(uh) -1
    h = 4 / N

    #Defining Ah
    Ah = (2*np.diag(np.ones(N+1))-np.diag(np.ones(N),1) - np.diag(np.ones(N),-1)  )/(h**2)+  np.diag(np.ones(N+1))
    Ah[0,0] = 1
    Ah[0,1] = 0
    Ah[N,N] = 1
    Ah[N,N-1] = 0
    
    if N == 1: #Case we are in the coarsest grid
        uh = np.linalg.solve(Ah,fh) #Solving exactly the linear system
        return 0
    else:
        #pre-smoothing
        for _ in range(alpha1):
            gs_step_1d(uh, fh)

        #Compute the residual
        rh = fh - Ah@uh

        #Restriction of residual
        crh = to_coarser_grid(rh)#coarser residual

        #Recoursively solve A_2h @ e_2h == r_2h (with initial guess e_2h = 0)
        ceh = np.zeros((N //2)+1)
        v_cycle_step_1d(ceh, crh, alpha1, alpha2)
        
        #Coarse grid correction
        uh+=to_finer_grid(ceh)

        #Post-smoothing and returning the pseudo_residual
        for _ in range(alpha2-1):
            gs_step_1d(uh, fh)
        return gs_step_1d(uh, fh)

#5.

if __name__ == "__main__": # investigations for exercise .5 (times and iterations for v_cycle_step)
    values_N = [2**l for l in range(3,15)]#values of N prescribed in the assignment
    iterations5_1 = [0 for i in range(len(values_N))]#storing iterations and times
    times5_1 = [0. for i in range(len(values_N))]
    alpha1 =1
    alpha2 =1
    for i in range(len(values_N)) :
        N = values_N[i]
        grid = grid_1d(N)

        #defining fh
        fh = np.array([f_1d(x) for x in grid])
        fh[0] = 2/5
        fh[N] = 2/5

        #defining uh and adding its boudaries conditions
        uh = np.zeros(N+1)
        uh[0] = 2/5
        uh[N] = 2/5

        #counting iterations and times
        counter = 1
        tic = time()
        while v_cycle_step_1d(uh, fh, alpha1, alpha2) > 10**(-8) :
            counter+=1
        iterations5_1[i] = counter
        times5_1[i] = time()-tic

    print("\n\n\n\ntimes for .5 alpha=1:\n",times5_1,"\n\niterations for .5 alpha =1:\n",iterations5_1)#printing the results

    iterations5_2 = [0 for i in range(len(values_N))]#the same procedure as above repeated in with different parameters alpha1 and alpha2
    times5_2 = [0. for i in range(len(values_N))]
    alpha1 =2
    alpha2 =2
    for i in range(len(values_N)) :
        N = values_N[i]
        grid = grid_1d(N)
        fh = np.array([f_1d(x) for x in grid])
        fh[0] = 2/5
        fh[N] = 2/5
        uh = np.zeros(N+1)
        uh[0] = 2/5
        uh[N] = 2/5
        counter = 1
        tic = time()
        while v_cycle_step_1d(uh, fh, alpha1, alpha2) > 10**(-8) :
            counter+=1
        iterations5_2[i] = counter
        times5_2[i] = time()-tic

    print("\n\n\n\ntimes for .5 alpha=2:\n",times5_2,"\n\niterations for .5 alpha =2:\n",iterations5_2)#printing the results

#6.

def to_coarser_grid_ex6(fv): #From finer to coarser vector. Version with natural restriction used in the full multigrid method
    N = len(fv)-1
    if N %2 != 0 : raise Exception("The grid has an odd number of points")
    n = N // 2
    cv = np.zeros(n+1) # defining the coarser vector
    for i in range(n+1):
        cv[i]= fv[2*i]
    return cv


def full_mg_1d(uh, fh, alpha1, alpha2, nu):
    N = len(uh) -1
    h = 4 / N

    if N == 1: #Case we are in the coarsest grid (base case of the recursion)

        #Defining Ah
        Ah = (2*np.diag(np.ones(N+1))-np.diag(np.ones(N),1) - np.diag(np.ones(N),-1)  )/(h**2)+  np.diag(np.ones(N+1))
        Ah[0,0] = 1
        Ah[0,1] = 0
        Ah[N,N] = 1
        Ah[N,N-1] = 0

        uh[:] = np.linalg.solve(Ah,fh) #Solving exactly the linear system
        return 0
    else:
        cfh = to_coarser_grid_ex6(fh)
        cuh = np.zeros_like(cfh)
        full_mg_1d(cuh, cfh, alpha1, alpha2, nu) #recursive call
        uh[:] = to_finer_grid(cuh)
        for _ in range(nu-1):
            v_cycle_step_1d(uh, fh, alpha1, alpha2)
        return v_cycle_step_1d(uh, fh, alpha1, alpha2)

#7.

if __name__ == "__main__":# investigations for the full multigrid method
    parameters = [(1,1,1),(1,1,2),(2,2,1),(2,2,2)]#different choiches of the parameters. They will be used as keys in dictionarioes where we store the data
    values_N = [2**l for l in range(3,15)]
    pseudoresiduals = {parameter : [0 for i in range(len(values_N))] for parameter in parameters}
    residuals = {parameter : [0 for i in range(len(values_N))] for parameter in parameters}
    times7 = {parameter : [0. for i in range(len(values_N))] for parameter in parameters}
    for parameter in parameters:
        alpha1 =parameter[0]
        alpha2 =parameter[1]
        nu = parameter[2]
        for i in range(len(values_N)) :
            N = values_N[i]
            h = 4/N
            grid = grid_1d(N)

            #defining fh
            fh = np.array([f_1d(x) for x in grid])
            fh[0] = 2/5
            fh[N] = 2/5

            #defining uh
            uh = np.zeros(N+1)
            uh[0] = 2/5
            uh[N] = 2/5

            #defining Ah (used in the calculations for the residual)
            Ah = (2*np.diag(np.ones(N+1))-np.diag(np.ones(N),1) - np.diag(np.ones(N),-1)  )/(h**2)+  np.diag(np.ones(N+1))
            Ah[0,0] = 1
            Ah[0,1] = 0
            Ah[N,N] = 1
            Ah[N,N-1] = 0

            #counting time and storing residual and psudo residual size
            tic = time()
            pseudoresiduals[parameter][i] = full_mg_1d(uh, fh, alpha1, alpha2, nu)
            times7[parameter][i] = time()-tic
            residuals[parameter][i] = max(     [   abs(fh[i]-(Ah@uh)[i]) for i in range(N+1)    ]      )
            

    print(times7,pseudoresiduals,residuals)  #printing the results



#8.

if __name__ == "__main__":#printing the approximation of u and extimating its minimum
    import matplotlib.pyplot as plt
    #choosing the parameters
    alpha1 =2
    alpha2 =2

    #choosing the grid size
    N = 2**14
    grid = grid_1d(N)

    #definig fh
    fh = np.array([f_1d(x) for x in grid])
    fh[0] = 2/5
    fh[N] = 2/5
    
    #defining uh
    uh = np.zeros(N+1)
    uh[0] = 2/5
    uh[N] = 2/5

    #calculating the approximation
    while v_cycle_step_1d(uh, fh, alpha1, alpha2) > 10**(-8) :
        pass
    
    #printing the solution
    plt.plot(grid, uh, markersize=3, linewidth=1.5)
    plt.axes([-2,2,-4,1])
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()

    #printing the minimum value
    minim=min(uh)
    print("minimum value = ", minim)




#Multigrid in 2d


def f_2d(x1, x2): return x1**2-x2**2 #two dimensional version of the function f 

def g_2d(x1, x2): return (x1**2+x2**2)/10 #two dimensional version of the function g

def gs_step_2d(uh, fh): #Gauss Seidel method for the two dimensional case
    N = len(uh)-1# number of interval per axis
    hto2 = (4/N)**2
    old_uh = uh.copy()#storing a copy of the input

    #in situ computation of the method
    for i in range(1,N):
        for j in range(1,N):   
            uh[i,j] = (hto2*fh[i,j] + uh[i-1,j] + uh[i+1,j] + uh[i,j-1] + uh[i,j+1]) / (4 + hto2)

    return max(     [   abs(uh[i,j]-old_uh[i,j]) for i in range(1,N) for j in range(1,N)   ]      )#returning the pseudo residual


def to_finer_mesh(cv):#from coarser 2d vector to finer 2d vector
    n = len(cv)-1
    N = 2*n
    fv = np.zeros((N+1,N+1))

    #applying the formulas of the lecture notes
    for i in range(n):
        for j in range(n):
            fv[2*i,2*j] = cv[i,j]
            fv[2*i+1,2*j] = (cv[i,j] + cv[i+1,j]) /2
            fv[2*i,2*j+1] = (cv[i,j] + cv[i,j+1]) /2
            fv[2*i+1, 2*j+1] = (cv[i,j] + cv[i+1,j] + cv[i,j+1] + cv[i+1,j+1]) /4
    return fv


def to_coarser_mesh(fv):#from finer 2d vector to coarser 2d vector
    N = len(fv)-1
    n = N//2
    cv = np.zeros((n+1,n+1))

    #applying the formulas of the lecture notes
    for i in range(1,n):
        for j in range(1,n):
            cv[i,j] = (fv[2*i-1,2*j-1] + fv[2*i-1,2*j+1] + fv[2*i+1,2*j-1] + fv[2*i+1,2*j+1] + 2*(fv[2*i,2*j-1] + fv[2*i, 2*j+1] + fv[2*i-1,2*j] + fv[2*i+1,2*j]) + 4*fv[2*i,2*j]) /16
    return cv



def f_minus_A_times_u(uh,fh): #calculating the residual rh = fh-Ah@uh
    N = len(uh) - 1
    hto2 = (4/N)**2
    rh = np.zeros((N+1,N+1))
    for i in range(1,N):
        for j in range(1,N):
            rh[i,j] = fh[i,j]-((4*uh[i,j] - uh[i-1,j] - uh[i+1,j] - uh[i,j-1] - uh[i,j+1]) / hto2 + uh[i,j])
    return rh



def v_cycle_step_2d(uh, fh, alpha1, alpha2):
    N = len(uh) -1

    if N == 1: #Case we are in the coarsest mesh (base case of recursion)

        #Solving exactly the linear system, that in this case is trivial and completely determined by the boundary conditions
        uh = np.array([[4/5,4/5],[4/5,4/5]])
        return 0

    else:
        #pre-smoothing
        for _ in range(alpha1):
            gs_step_2d(uh, fh)

        #Compute the residual
        rh = f_minus_A_times_u(uh,fh)

        #Restriction of residual
        crh = to_coarser_mesh(rh)#coarser residual

        #Recoursively solve A_2h @ e_2h == r_2h (with initial guess e_2h = 0)
        ceh = np.zeros(((N //2)+1,(N //2)+1))
        v_cycle_step_2d(ceh, crh, alpha1, alpha2)
        
        #Coarse grid correction
        uh[:]+=to_finer_mesh(ceh)

        #Post-smoothing and returning the pseudo_residual
        for _ in range(alpha2-1):
            gs_step_2d(uh, fh)

        return gs_step_2d(uh, fh)



def to_coarser_mesh_natural(fv):#from finer 2d vector to coarser 2d vector without wheightening
    N = len(fv)-1
    n = N//2
    cv = np.zeros((n+1,n+1))
    for i in range(1,n):
        for j in range(1,n):
            cv[i,j] = fv[2*i,2*j]
    return cv


def add_boundaries(uh):#adds the values of uh that we know from the boundary conditions
    N = len(uh) -1
    uh[0,:]=0
    uh[N,:]=0
    uh[:,0]=0
    uh[:,N]=0
    x = grid_1d(N)
    y = grid_1d(N)
    x, y = np.meshgrid(x, y)
    g = g_2d(x,y)
    g[1:N,1:N]=0
    return uh+g



def full_mg_2d(uh, fh, alpha1, alpha2, nu):
    N = len(uh) -1
    if N == 1: #Case we are in the coarsest grid

        uh[:] = np.array([[4/5,4/5],[4/5,4/5]]) #Solving exactly the linear system in the base case of recursion
        return 0

    else:
        #defining fh for the coarser mesh problem
        cfh = to_coarser_mesh_natural(fh)

        #defining uh for the coarser mesh problem
        cuh = np.zeros_like(cfh)
        cuh = add_boundaries(cuh)# adding boundaries

        full_mg_2d(cuh, cfh, alpha1, alpha2, nu) #recursive call of the funcion

        #updating the value of uh on the finer grid 
        uh[:] = to_finer_mesh(cuh)
        uh[:] = add_boundaries(uh) # adding boundaries

        #applying nu times V cycle and returning the pseudo residual size in infinity norm
        for _ in range(nu-1):
            v_cycle_step_2d(uh, fh, alpha1, alpha2)
        return v_cycle_step_2d(uh, fh, alpha1, alpha2)



if __name__ == "__main__":# investigations for the GS method
    values_N = [2**l for l in range(2,7)]# stores the values N chosen for our investigations
    iterations2_1 = [0 for i in range(len(values_N))]# stores the needed iterations
    times2_1 = [0. for i in range(len(values_N))]#stores the needed CPU times
    for i in range(len(values_N)) :
        N = values_N[i]
        grid = grid_1d(N)

        #defining fh
        x, y=np.meshgrid(np.linspace(-2,2, N+1), np.linspace(-2,2, N+1))
        fh=f_2d(x,y)
        fh[:,0]=np.array(g_2d(np.linspace(-2,2,N+1),-2))
        fh[:,N]=np.array(g_2d(np.linspace(-2,2,N+1),2))
        fh[0,:]=np.array(g_2d(-2,np.linspace(-2,2,N+1)))
        fh[N,:]=np.array(g_2d(2,np.linspace(-2,2,N+1)))

        #defining uh
        uh=np.zeros((N+1,N+1))
        uh[:]=add_boundaries(uh)

        #measuring times and iterations needed
        counter = 1
        tic = time()
        while gs_step_2d(uh, fh) > 10**(-8) :
            counter+=1
        times2_1[i] = time()-tic
        iterations2_1[i] = counter
        
    
    print("\n\n\n\ntimes for .2.1:\n",times2_1,"\n\niterations for .2.1:\n",iterations2_1)#printing the results




if __name__ == "__main__":# investigations for the V-cycle method
    values_N = [2**l for l in range(2,9)]# stores the values N chosen for our investigations
    iterations2_2 = [0 for i in range(len(values_N))]# stores the needed iterations
    times2_2 = [0. for i in range(len(values_N))]#stores the needed CPU times
    for i in range(len(values_N)) :
        N = values_N[i]
        grid = grid_1d(N)
        alpha1 =1
        alpha2=1

        #defining fh
        x, y=np.meshgrid(np.linspace(-2,2, N+1), np.linspace(-2,2, N+1))
        fh=f_2d(x,y)
        fh[:,0]=np.array(g_2d(np.linspace(-2,2,N+1),-2))
        fh[:,N]=np.array(g_2d(np.linspace(-2,2,N+1),2))
        fh[0,:]=np.array(g_2d(-2,np.linspace(-2,2,N+1)))
        fh[N,:]=np.array(g_2d(2,np.linspace(-2,2,N+1)))

        #defining uh
        uh=np.zeros((N+1,N+1))
        uh[:]=add_boundaries(uh)

        #measuring times and iterations needed
        counter = 1
        tic = time()
        while v_cycle_step_2d(uh, fh,alpha1,alpha2) > 10**(-8) :
            counter+=1
        times2_2[i] = time()-tic
        iterations2_2[i] = counter
        
    
    print("\n\n\n\ntimes for .2.2:\n",times2_2,"\n\niterations for .2.2:\n",iterations2_2)#printing the results


if __name__ == "__main__":# investigations for the V-cycle method
    values_N = [2**l for l in range(2,9)]# stores the values N chosen for our investigations
    iterations2_2 = [0 for i in range(len(values_N))]# stores the needed iterations
    times2_2 = [0. for i in range(len(values_N))]#stores the needed CPU times
    for i in range(len(values_N)) :
        N = values_N[i]
        grid = grid_1d(N)
        alpha1 =2
        alpha2=2

        #defining fh
        x, y=np.meshgrid(np.linspace(-2,2, N+1), np.linspace(-2,2, N+1))
        fh=f_2d(x,y)
        fh[:,0]=np.array(g_2d(np.linspace(-2,2,N+1),-2))
        fh[:,N]=np.array(g_2d(np.linspace(-2,2,N+1),2))
        fh[0,:]=np.array(g_2d(-2,np.linspace(-2,2,N+1)))
        fh[N,:]=np.array(g_2d(2,np.linspace(-2,2,N+1)))

        #defining uh
        uh=np.zeros((N+1,N+1))
        uh[:]=add_boundaries(uh)

        #measuring times and iterations needed
        counter = 1
        tic = time()
        while v_cycle_step_2d(uh, fh,alpha1,alpha2) > 10**(-8) :
            counter+=1
        times2_2[i] = time()-tic
        iterations2_2[i] = counter
        
    
    print("\n\n\n\ntimes for .2.2:\n",times2_2,"\n\niterations for .2.2:\n",iterations2_2)#printing the results


if __name__ == "__main__":# investigations for the full multigrid method
    parameters = [(1,1,1),(1,1,2),(2,2,1),(2,2,2)]#different choiches of the parameters. They will be used as keys in dictionarioes where we store the data
    values_N = [2**l for l in range(2,9)]
    pseudoresiduals = {parameter : [0 for i in range(len(values_N))] for parameter in parameters}
    residuals = {parameter : [0 for i in range(len(values_N))] for parameter in parameters}
    times2_3 = {parameter : [0. for i in range(len(values_N))] for parameter in parameters}
    for parameter in parameters:
        alpha1 =parameter[0]
        alpha2 =parameter[1]
        nu = parameter[2]
        for i in range(len(values_N)) :
            N = values_N[i]
            h = 4/N
            grid = grid_1d(N)

            #defining fh
            x, y=np.meshgrid(np.linspace(-2,2, N+1), np.linspace(-2,2, N+1))
            fh=f_2d(x,y)
            fh[:,0]=np.array(g_2d(np.linspace(-2,2,N+1),-2))
            fh[:,N]=np.array(g_2d(np.linspace(-2,2,N+1),2))
            fh[0,:]=np.array(g_2d(-2,np.linspace(-2,2,N+1)))
            fh[N,:]=np.array(g_2d(2,np.linspace(-2,2,N+1)))

            #defining uh
            uh=np.zeros((N+1,N+1))
            uh[:]=add_boundaries(uh)

            #counting time and storing residual and psudo residual size
            tic = time()
            pseudoresiduals[parameter][i] = full_mg_2d(uh, fh, alpha1, alpha2, nu)
            times2_3[parameter][i] = time()-tic
            res = f_minus_A_times_u(uh,fh)
            residuals[parameter][i] = max(     [   abs(res[i,j]) for i in range(N+1)  for j in range(N+1)   ]      )
            

    print(times2_3,pseudoresiduals,residuals)  #printing the results


if __name__ == "__main__":#printing an approximation of the solution and finding the minimum
    N = 2**8
    h = 4/N
    grid = grid_1d(N)
    alpha1=2
    alpha2=2

    #defining fh
    x, y=np.meshgrid(np.linspace(-2,2, N+1), np.linspace(-2,2, N+1))
    fh=f_2d(x,y)
    fh[:,0]=np.array(g_2d(np.linspace(-2,2,N+1),-2))
    fh[:,N]=np.array(g_2d(np.linspace(-2,2,N+1),2))
    fh[0,:]=np.array(g_2d(-2,np.linspace(-2,2,N+1)))
    fh[N,:]=np.array(g_2d(2,np.linspace(-2,2,N+1)))

    #defining uh
    uh=np.zeros((N+1,N+1))
    uh[:]=add_boundaries(uh)

    #plotting the approximation
    x=np.linspace(-2,2,N+1)
    y=np.linspace(-2,2,N+1)
    x,y=np.meshgrid(x,y)

    while v_cycle_step_2d(uh, fh,alpha1,alpha2) > 10**(-8) :
        pass


    fig=plt.figure()

    minim = min(uh[i,j] for i in range(1,N+1) for j in range(1,N+1))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,uh,cmap=plt.cm.cool)
    ax.view_init(25, 50)

    plt.show()

    print(minim)
