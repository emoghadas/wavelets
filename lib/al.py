
#!/usr/bin/python
"""
Python module for the calculation of the Atomic Limit
original functions created by Dominik Robert Fus
adapted by Matthias Reitner
"""
__author__='Dominik Robert Fus, Matthias Reitner'
import numpy as np
import matplotlib as mpl
import scipy as sp


class atom(object):
    """
    A class to calulate quantites of the atomic limit
    """
    def __init__(self, U, beta, mu, Niwf=100):
        self.U = U
        self.beta = beta
        self.mu = mu
        self.Niwf = Niwf

    def Z(self):
        return (1+2*np.exp(self.mu*self.beta)+np.exp((2*self.mu-self.U)*self.beta))

    def n(self):
        """returns density"""
        U = self.U
        mu = self.mu
        beta = self.beta 
        # overflow handling for extreme parameters
        N = np.where(
            #if 
            mu>9*U/2,
                #then
                (np.exp(-mu*beta)+np.exp(-U*beta))/(np.exp(-2*mu*beta)+2*np.exp(-mu*beta)+np.exp(-U*beta))
                #elif
                ,np.where(mu<-9*U/2,
                    #then
                    (np.exp(mu*beta)+np.exp((2*mu-U)*beta))/(1+2*np.exp(mu*beta)+np.exp((2*mu-U)*beta)) 
                    #else
                    ,(1+np.exp((mu-U)*beta))/(np.exp(-mu*beta)+2+np.exp((mu-U)*beta)) # times exp(-mu*beta)
        ))
        return N

    def iw(self, omega=0):
        """returns fermionic matsuabra frequencies i(w + omega)"""
        return 1j*np.pi/self.beta*(2*np.arange(-self.Niwf + omega,self.Niwf + omega)+1) 

    def g(self, omega=0.):
        """returns Green's function"""
        U = self.U
        mu = self.mu
        N = self.n()
        #z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta)) 
        #N=(1+np.exp((mu-U)*beta))/z  #time np.exp(-mu*beta)
        iw = self.iw(omega)
        return N/(iw+mu-U) + (1-N)/(iw+mu)


    def dg_diw(self, omega=0):
        """returns Green's function"""
        U = self.U
        mu = self.mu
        N = self.n()
        #z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta)) 
        #N=(1+np.exp((mu-U)*beta))/z  #time np.exp(-mu*beta)
        iw = self.iw(omega)
        return -N/(iw+mu-U)**2 - (1-N)/(iw+mu)**2

    def sigma(self):
        """returns self energy function"""
        G = self.g()
        iw = self.iw()
        g0_1 = iw + self.mu
        return g0_1 - 1/G

    def g2uu(self, omega=0.):
        """
        returns iw, iw' matrix of connected two-particle Green's function
        spin up up up up in ph-convention <c+ c c+ c>
        """
        U = self.U
        mu = self.mu
        beta = self.beta
        N = self.n()

        iw = self.iw()
        iw_o = self.iw(omega)
        nu_o = iw_o[:,None]
        nup = iw[None,:]

        x1 = nu_o+mu
        x_1 = nu_o+mu-U
        x2 = nup+mu
        x_2 = nup+mu-U


        delta = np.eye(2*self.Niwf)
        if omega == 0.:
            return beta * U**2 * N*(1-N) * (1-delta)/(x1*x_1*x2*x_2)
        else:
            return - beta * U**2 * N*(1-N) *delta/(x1*x_1*x2*x_2)


    def g2ud(self, omega=0.):
        """
        returns iw, iw' matrix of connected two-particle Green's function
        spin up up down down in ph-convention <c+ c c+ c> for the bosonic
        frequency omega
        """
        U = self.U
        mu = self.mu
        beta = self.beta
        N = self.n()
        z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta))

        iw = self.iw()
        iw_o = self.iw(omega)
        nu = iw[:,None]
        nu_o = iw_o[:,None]
        nup = iw[None,:]
        nup_o = iw_o[None,:]

        x1 = nu_o+mu
        x_1 = nu_o+mu-U
        x2 = nup+mu
        x_2 = nup+mu-U
        x3 = nup_o+mu
        x_3 = nup_o+mu-U
        x4 = nu+mu
        x_4 = nu+mu-U

        # delta(nu = nu')
        delta = np.eye(2*self.Niwf)
        # delta(nu + omega = -nu')
        delta_12 = np.eye(2*self.Niwf,k=omega)[:,::-1]

        if mu == U/2:
            hf_term = beta*delta_12/(2*(1+np.exp(beta*mu))) \
                    *(1/x_1+1/x_2)*(1/x_3+1/x_4)
        else:
            hf_term = (2*N-1)/(nu_o+nup+2*mu-U)\
                    *(1/x_1+1/x_2)*(1/x_3+1/x_4)
            
        h0_term = - beta*delta/z*(1/x1-1/x_3)*(1/x4-1/x_2)

        if omega == 0:
            w0_term = beta*U**2 *(np.exp(-beta*U)-1)/z**2 \
                    * 1/(x1*x_1*x2*x_2)
        else:
            w0_term = 0. 

        diag = hf_term +h0_term + w0_term

        offdiag = (N-1)/(x1*x_3*x4)  + (1-N)/(x1*x_2*x_3) \
                + (1-N)/(x_1*x2*x_3) + (N-1)/(x2*x_3*x4)  \
                + (1-N)/(x1*x2*x4)   + (1-N)/(x1*x2*x3)   \
                + (1-N)/(x_1*x3*x_4) + (N-1)/(x_1*x2*x3)  \
                + (1-N)/(x_2*x3*x_4) + (N-1)/(x1*x_2*x3)  \
                -    N/(x_1*x_2*x_4) -  N/(x_1*x_2*x_3)
        
        return diag+offdiag
    
    def f_uu(self,omega=0):
        return -self.g2uu(omega=omega)/(self.g()[:,None]*self.g(omega=omega)[:,None] * self.g()[None,:]*self.g(omega=omega)[None,:])
    
    def f_ud(self,omega=0):
        return -self.g2ud(omega=omega)/(self.g()[:,None]*self.g(omega=omega)[:,None] * self.g()[None,:]*self.g(omega=omega)[None,:])

    def chi_0(self,omega=0):
        return - self.beta *np.diag(self.g()*self.g(omega=omega))
    
    def chi_c(self,omega=0):
        return self.chi_0(omega=omega) + self.g2uu(omega=omega) + self.g2ud(omega=omega)

    def chi_uu(self,omega=0):
        return self.chi_0(omega=omega) + self.g2uu(omega=omega)

    def chi_ud(self,omega=0):
        return self.g2ud(omega=omega)

    def chi_s(self,omega=0):
        return self.chi_0(omega=omega) + self.g2uu(omega=omega) - self.g2ud(omega=omega)

    def gamma_c(self,omega=0):
        return np.linalg.inv(self.chi_c(omega=omega)) \
             - np.linalg.inv(self.chi_0(omega=omega))

    def skolimowskiDelta(self):
        def fermi(z):
            return 1/(np.exp(z)+1)
        N = self.n()
        return - fermi(-self.beta*self.mu)*N \
               + fermi(-self.beta*(self.mu-self.U))*(N-1) \
               + fermi(-self.beta*(self.mu+(N-1)*self.U))

# for older scripts
def n(U, beta, mu):
    N = np.where(
        #if 
        mu>9*U/2,
            #then
            (np.exp(-mu*beta)+np.exp(-U*beta))/(np.exp(-2*mu*beta)+2*np.exp(-mu*beta)+np.exp(-U*beta))
            #elif
            ,np.where(mu<-9*U/2,
                #then
                (np.exp(mu*beta)+np.exp((2*mu-U)*beta))/(1+2*np.exp(mu*beta)+np.exp((2*mu-U)*beta)) 
                #else
                ,(1+np.exp((mu-U)*beta))/(np.exp(-mu*beta)+2+np.exp((mu-U)*beta)) # times exp(-mu*beta)
    ))
    return N

def chi_c(U, beta, mu, Niwf):
    size = 2*Niwf
    z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta)) # times exp(-mu*beta)
    N=n(U, beta, mu)
    #N=(1+np.exp((mu-U)*beta))/z
    iw =1j*np.pi/beta*(2*np.arange(-Niwf,Niwf)+1) 
    nu = iw[:,None]
    nup =  iw[None,:]
    nu_m  = nu+mu
    nu_mU = nu+mu-U
    np_m  = nup+mu
    np_mU = nup+mu-U
    delta = np.eye(size)
    if mu == U/2:
        quot = delta[:,::-1]*beta/2*1/(1+np.exp(beta*U/2))
    else:
        quot = (2*N-1)/(nu+nup+2*mu-U)
    return (-beta*((1-N)/nu_m + N/nu_mU)**2*delta \
        + beta * U**2 * N*(1-N) * (1-delta)/(nu_m*nu_mU*np_m*np_mU) \
        + quot *(1/nu_mU+1/np_mU)**2 \
        - beta * 1/z * delta * (1/nu_m - 1/np_mU)**2 \
        + beta * U**2 * (np.exp(-U*beta)-1)/z**2 * 1/(nu_m*nu_mU*np_m*np_mU) \
        + (N-1)/(nu_m**2 * np_mU) + (1-N)/(nu_m*np_mU**2) \
        + 2*((1-N)/(nu_mU * np_m * np_mU) + (N-1)/(nu_m * np_m * np_mU)) \
        + (1-N)/(nu_m**2*np_m) + (1-N)/(nu_m*np_m**2) \
        + (1-N)/(nu_mU**2*np_m) + (N-1)/(nu_mU*np_m**2) \
        - N/(nu_mU**2*np_mU) - N/(nu_mU*np_mU**2))


def gradient(x, y):
    '''returns central differences and simple
    differences at begin and end of output vector'''
    assert len(x) == len(y), 'arguments must be of same length'
    yoverx=np.divide(np.diff(y), np.diff(x))
    diff = np.empty(len(x),dtype='complex')
    diff[0]=yoverx[0]
    for i in range(len(x)-2):
        f = np.diff(x)[i]/(np.diff(x)[i]+np.diff(x)[i+1])
        diff[i+1]= (1-f)*yoverx[i]+ f*yoverx[i+1]
    diff[-1]=yoverx[-1]
    return diff
    
# plotting helper:
# -------------------------------------
# colormap inspired by Patrick Chalupa
# -------------------------------------
cdict_white = {'blue':  [[0.0, 0.6, 0.6],
                   [0.499, 1.0, 1.0],
                   #[0.5, 0.0, 0.0],
                   [0.5, 1.0, 1.0],
                   [0.501, 0.0, 0.0],
                   [1.0, 0., 0.]],
         'green': [[0.0, 0.0, 0.0],
                   [0.02631578947368421, 7.673360394717657e-06, 7.673360394717657e-06],
                   [0.05263157894736842, 0.00012277376631548252, 0.00012277376631548252],
                   [0.07894736842105263, 0.0006215421919721302, 0.0006215421919721302],
                   [0.10526315789473684, 0.0019643802610477203, 0.0019643802610477203],
                   [0.13157894736842105, 0.004795850246698536, 0.004795850246698536],
                   [0.15789473684210525, 0.009944675071554084, 0.009944675071554084],
                   [0.18421052631578946, 0.018423738307717093, 0.018423738307717093],
                   [0.21052631578947367, 0.031430084176763524, 0.031430084176763524],
                   [0.23684210526315788, 0.050344917549742546, 0.050344917549742546],
                   [0.2631578947368421, 0.07673360394717657, 0.07673360394717657],
                   [0.2894736842105263, 0.11234566953906126, 0.11234566953906126],
                   [0.3157894736842105, 0.15911480114486534, 0.15911480114486534],
                   [0.3421052631578947, 0.21915884623353094, 0.21915884623353094],
                   [0.3684210526315789, 0.2947798129234735, 0.2947798129234735],
                   [0.39473684210526316, 0.3884638699825815, 0.3884638699825815],
                   [0.42105263157894735, 0.5028813468282164, 0.5028813468282164],
                   [0.4473684210526315, 0.6408867335272133, 0.6408867335272133],
                   [0.47368421052631576, 0.8055186807958807, 0.8055186807958807],
                   [0.499, 1.0, 1.0],
                   #[0.5, 0.0, 0.0],
                   [0.5, 1.0, 1.0],
                   [0.501, 1.0, 1.0],
                   [0.5263157894736843, 0.8055186807958807, 0.8055186807958807],
                   [0.5526315789473685, 0.6408867335272133, 0.6408867335272133],
                   [0.5789473684210527, 0.5028813468282164, 0.5028813468282164],
                   [0.6052631578947368, 0.3884638699825815, 0.3884638699825815],
                   [0.631578947368421, 0.2947798129234735, 0.2947798129234735],
                   [0.6578947368421053, 0.21915884623353094, 0.21915884623353094],
                   [0.6842105263157895, 0.15911480114486534, 0.15911480114486534],
                   [0.7105263157894737, 0.11234566953906126, 0.11234566953906126],
                   [0.736842105263158, 0.07673360394717657, 0.07673360394717657],
                   [0.7631578947368421, 0.050344917549742546, 0.050344917549742546],
                   [0.7894736842105263, 0.031430084176763524, 0.031430084176763524],
                   [0.8157894736842105, 0.018423738307717093, 0.018423738307717093],
                   [0.8421052631578947, 0.009944675071554084, 0.009944675071554084],
                   [0.868421052631579, 0.004795850246698536, 0.004795850246698536],
                   [0.8947368421052632, 0.0019643802610477203, 0.0019643802610477203],
                   [0.9210526315789473, 0.0006215421919721302, 0.0006215421919721302],
                   [0.9473684210526316, 0.00012277376631548252, 0.00012277376631548252],
                   [0.9736842105263158, 7.673360394717657e-06, 7.673360394717657e-06],
                   [1.0, 0.0, 0.0]],
         'red':   [[0.0, 0., 0.],
                   [0.499, 0.0, 0.0],
                   [0.5, 1.0, 1.0],
                   #[0.5, 0.0, 0.0],
                   [0.501, 1.0, 1.0],
                   [1.0, 0.6, 0.6]]}

cmap_w = mpl.colors.LinearSegmentedColormap('chalupa_white',segmentdata = cdict_white,N=10000)

# ---------------------------------------
# normalize colormap from stackoverflow
# ---------------------------------------
class norm(mpl.colors.Normalize):
    def __init__(self, matrix, midpoint=0, clip=False):
        vmin = np.amin(matrix.real)
        vmax = np.amax(matrix.real)
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.vmax == 0:
            normalized_min = 0
        else:
            normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        if self.vmin == 0:
            normalized_max = 1
        else:
            normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))