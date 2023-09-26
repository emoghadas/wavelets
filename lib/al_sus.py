
#!/usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp

### filling
def N(U, beta, mu):
    if mu > 9*U/2:
        n=(np.exp(-mu*beta)+np.exp(-U*beta))/(np.exp(-2*mu*beta)+2*np.exp(-mu*beta)+np.exp(-U*beta))
    elif mu < -9*U/2:
        n=(np.exp(mu*beta)+np.exp((2*mu-U)*beta))/(1+2*np.exp(mu*beta)+np.exp((2*mu-U)*beta)) 
    else:
        n=(1+np.exp((mu-U)*beta))/(np.exp(-mu*beta)+2+np.exp((mu-U)*beta))
    return n

def nu_compute(beta, Niwf):
    i = np.arange(-Niwf, Niwf)
    return np.pi / beta * (2 * i + 1)

##double occupancy
def d(U, beta, mu):
    return np.exp(beta*(2*mu-U))/(1+2*np.exp(beta*mu)+np.exp(beta*(2*mu-U)))

##Greensfunction
def g(U, beta, mu, Niwf):
    n = N(U, beta, mu)
    i = np.arange(-Niwf, Niwf)
    iw = 1j*np.pi/beta*(2*i+1)
    return n/(iw+mu-U) + (1-n)/(iw+mu)

#chi up up (omega!=0 implemented but not tested)
def chi_uu(U, beta, mu, Niwf,omega=0):
    iw = nu_compute(beta, Niwf)
    if np.any((omega/(np.pi/beta)) % 2 != 0):
        print('Warning omega is no bosonic Matsubara frequency!')
    
    size = len(iw)
    n=N(U, beta, mu)

    nu = iw[:,None]
    nup =  iw[None,:]

    nu_m  = 1j*nu+mu
    nu_mU = 1j*nu+mu-U

    np_m  = 1j*nup+mu
    np_mU = 1j*nup+mu-U
    delta = np.eye(size)

            
    
    if omega==0:
        return (-beta*((1-n)/nu_m + n/nu_mU)**2*delta \
                + beta * U**2 * n*(1-n) * (1-delta)/(nu_m*nu_mU*np_m*np_mU))
    else:
        nu_o_m  = 1j*(nu+omega)+mu
        nu_o_mU = 1j*(nu+omega)+mu-U
        
        return (-beta*((1-n)/nu_m + n/nu_mU)*((1-n)/nu_o_m + n/nu_o_mU)*delta \
                - beta * U**2 * n*(1-n) * delta/(nu_o_m*nu_o_mU*np_m*np_mU))
    
#chi up down (omega!=0 implemented but not tested)
def chi_ud(U, beta, mu, Niwf, omega=0):
    iw = nu_compute(beta, Niwf)
    if (omega/(np.pi/beta))%2!=0:
        print('Warning omega is no bosonic Matsubara frequency!')
    
    size = len(iw)
    z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta))
    n=N(U, beta, mu)

    nu = iw[:,None]
    nup =  iw[None,:]

    nu_m  = 1j*nu+mu
    nu_mU = 1j*nu+mu-U

    np_m  = 1j*nup+mu
    np_mU = 1j*nup+mu-U
    delta = np.eye(size)

    J=np.zeros((size,size))
    np.fill_diagonal(np.fliplr(J),1)
    
    if mu == U/2:
        quot = beta/2*((np.exp(mu*beta)+2)/(1+np.exp(mu*beta))-1)*J
    else:
        quot = (2*n-1)/(1j*(nu+omega)+1j*nup+2*mu-U)
    
    if omega==0:
        return (quot *(1/nu_mU+1/np_mU)**2 \
                - beta * 1/z * delta * (1/nu_m - 1/nu_mU)**2 \
                + beta * U**2 * (np.exp(-U*beta)-1)/z**2 * 1/(nu_m*nu_mU*np_m*np_mU) \
                + (n-1)/(nu_m**2 * np_mU) + (1-n)/(nu_m*np_mU**2) \
                + 2*((1-n)/(nu_mU * np_m * np_mU) + (n-1)/(nu_m * np_m * np_mU)) \
                + (1-n)/(nu_m**2*np_m) + (1-n)/(nu_m*np_m**2) \
                + (1-n)/(nu_mU**2*np_m) + (n-1)/(nu_mU*np_m**2) \
                - n/(nu_mU**2*np_mU) - n/(nu_mU*np_mU**2))
    else:
        nu_o_m  = 1j*(nu+omega)+mu
        nu_o_mU = 1j*(nu+omega)+mu-U
        np_o_m  = 1j*(nup+omega)+mu
        np_o_mU = 1j*(nup+omega)+mu-U
        
        return  (quot *(1/nu_o_mU+1/np_mU)*(1/nu_mU+1/np_o_mU) \
                - beta * 1/z * delta * (1/nu_o_m - 1/nu_o_mU)*(1/nu_m - 1/nu_mU) \
                + (n-1)/(nu_o_m*nu_m * np_o_mU) + (1-n)/(nu_o_m*np_mU*np_o_mU) \
                + (1-n)/(nu_o_mU * np_m * np_o_mU) + (n-1)/(nu_m * np_m * np_o_mU) \
                + (1-n)/(nu_m*nu_o_m*np_m) + (1-n)/(nu_m*np_m*np_o_m) \
                + (1-n)/(nu_mU*nu_o_mU*np_o_m) + (n-1)/(nu_o_mU*np_m*np_o_m) \
                + (1-n)/(nu_mU * np_mU * np_o_m) + (n-1)/(nu_o_m * np_o_m * np_mU) \
                - n/(nu_mU*nu_o_mU*np_mU) - n/(nu_o_mU*np_mU*np_o_mU))

    
#chi charge (omega!=0 implemented but not tested)
def chi_c(U, beta, mu, Niwf,omega=0):
    iw = nu_compute(beta, Niwf)
    if np.any((omega/(np.pi/beta)) % 2 != 0):
        raise ValueError('Warning omega is no bosonic Matsubara frequency!')
    
    size = len(iw)
    z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta))
    n=N(U, beta, mu)

    nu = iw[:,None]
    nup =  iw[None,:]

    nu_m  = 1j*nu+mu
    nu_mU = 1j*nu+mu-U

    np_m  = 1j*nup+mu
    np_mU = 1j*nup+mu-U
    delta = np.eye(size)

    J=np.zeros((size,size))
    np.fill_diagonal(np.fliplr(J),1)
    


    if mu == U/2:
        quot = beta/2*((np.exp(mu*beta)+2)/(1+np.exp(mu*beta))-1)*J
    else:
        quot = (2*n-1)/(1j*(nu+omega)+1j*nup+2*mu-U)
        
    
    if omega==0:
        beta_U2 = beta * U**2
        one_minus_n_over_nu_m_squared = (1 - n) / nu_m**2
        one_minus_n_over_np_m_squared = (1 - n) / np_m**2
        one_minus_n_over_nu_mU_squared = (1 - n) / nu_mU**2
        n_over_nu_mU_squared = n / nu_mU**2
        reciprocal_sum = 1 / nu_mU + 1 / np_mU
        
        return (-beta*((1-n)/nu_m + n/nu_mU)**2*delta \
                + beta * U**2 * n*(1-n) * (1-delta)/(nu_m*nu_mU*np_m*np_mU) \
                + quot *(1/nu_mU+1/np_mU)**2 \
                - beta * 1/z * delta * (1/nu_m - 1/nu_mU)**2 \
                + beta * U**2 * (np.exp(-U*beta)-1)/z**2 * 1/(nu_m*nu_mU*np_m*np_mU) \
                + (n-1)/(nu_m**2 * np_mU) + (1-n)/(nu_m*np_mU**2) \
                + 2*((1-n)/(nu_mU * np_m * np_mU) + (n-1)/(nu_m * np_m * np_mU)) \
                + (1-n)/(nu_m**2*np_m) + (1-n)/(nu_m*np_m**2) \
                + (1-n)/(nu_mU**2*np_m) + (n-1)/(nu_mU*np_m**2) \
                - n/(nu_mU**2*np_mU) - n/(nu_mU*np_mU**2))
    else:
        nu_o_m  = 1j*(nu+omega)+mu
        nu_o_mU = 1j*(nu+omega)+mu-U

        np_o_m  = 1j*(nup+omega)+mu
        np_o_mU = 1j*(nup+omega)+mu-U
        
        return (-beta*((1-n)/nu_m + n/nu_mU)*((1-n)/nu_o_m + n/nu_o_mU)*delta \
                - beta * U**2 * n*(1-n) * delta/(nu_o_m*nu_o_mU*np_m*np_mU) \
                + quot *(1/nu_o_mU+1/np_mU)*(1/nu_mU+1/np_o_mU) \
                - beta * 1/z * delta * (1/nu_o_m - 1/nu_o_mU)*(1/nu_m - 1/nu_mU) \
                + (n-1)/(nu_o_m*nu_m * np_o_mU) + (1-n)/(nu_o_m*np_mU*np_o_mU) \
                + (1-n)/(nu_o_mU * np_m * np_o_mU) + (n-1)/(nu_m * np_m * np_o_mU) \
                + (1-n)/(nu_m*nu_o_m*np_m) + (1-n)/(nu_m*np_m*np_o_m) \
                + (1-n)/(nu_mU*nu_o_mU*np_o_m) + (n-1)/(nu_o_mU*np_m*np_o_m) \
                + (1-n)/(nu_mU * np_mU * np_o_m) + (n-1)/(nu_o_m * np_o_m * np_mU) \
                - n/(nu_mU*nu_o_mU*np_mU) - n/(nu_o_mU*np_mU*np_o_mU))


#chi spin (omega!=0 implemented but not tested)
def chi_s(U, beta, mu, Niwf,omega=0):
    iw = nu_compute(beta, Niwf)
    
    if (omega/(np.pi/beta))%2!=0:
        print('Warning omega is no bosonic Matsubara frequency!')
    
    size = len(iw)
    z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta))
    n=N(U, beta, mu)

    nu = iw[:,None]
    nup =  iw[None,:]

    nu_m  = 1j*nu+mu
    nu_mU = 1j*nu+mu-U

    np_m  = 1j*nup+mu
    np_mU = 1j*nup+mu-U
    delta = np.eye(size)

    J=np.zeros((size,size))
    np.fill_diagonal(np.fliplr(J),1)
    
    if mu == U/2:
        quot = beta/2*((np.exp(mu*beta)+2)/(1+np.exp(mu*beta))-1)*J
    else:
        quot = (2*n-1)/(1j*(nu+omega)+1j*nup+2*mu-U)
        
    
    if omega==0:
        return (-beta*((1-n)/nu_m + n/nu_mU)**2*delta \
                + beta * U**2 * n*(1-n) * (1-delta)/(nu_m*nu_mU*np_m*np_mU) \
                - quot *(1/nu_mU+1/np_mU)**2 \
                + beta * 1/z * delta * (1/nu_m - 1/nu_mU)**2 \
                - beta * U**2 * (np.exp(-U*beta)-1)/z**2 * 1/(nu_m*nu_mU*np_m*np_mU) \
                - (n-1)/(nu_m**2 * np_mU) - (1-n)/(nu_m*np_mU**2) \
                - 2*((1-n)/(nu_mU * np_m * np_mU) + (n-1)/(nu_m * np_m * np_mU)) \
                - (1-n)/(nu_m**2*np_m) - (1-n)/(nu_m*np_m**2) \
                - (1-n)/(nu_mU**2*np_m) - (n-1)/(nu_mU*np_m**2) \
                + n/(nu_mU**2*np_mU) + n/(nu_mU*np_mU**2))
    else:
        nu_o_m  = 1j*(nu+omega)+mu
        nu_o_mU = 1j*(nu+omega)+mu-U

        np_o_m  = 1j*(nup+omega)+mu
        np_o_mU = 1j*(nup+omega)+mu-U
        
        return (-beta*((1-n)/nu_m + n/nu_mU)*((1-n)/nu_o_m + n/nu_o_mU)*delta \
                - beta * U**2 * n*(1-n) * delta/(nu_o_m*nu_o_mU*np_m*np_mU) \
                - quot *(1/nu_o_mU+1/np_mU)*(1/nu_mU+1/np_o_mU) \
                + beta * 1/z * delta * (1/nu_o_m - 1/nu_o_mU)*(1/nu_m - 1/nu_mU) \
                - (n-1)/(nu_o_m*nu_m * np_o_mU) - (1-n)/(nu_o_m*np_mU*np_o_mU) \
                - (1-n)/(nu_o_mU * np_m * np_o_mU) - (n-1)/(nu_m * np_m * np_o_mU) \
                - (1-n)/(nu_m*nu_o_m*np_m) - (1-n)/(nu_m*np_m*np_o_m) \
                - (1-n)/(nu_mU*nu_o_mU*np_o_m) - (n-1)/(nu_o_mU*np_m*np_o_m) \
                - (1-n)/(nu_mU * np_mU * np_o_m) - (n-1)/(nu_o_m * np_o_m * np_mU) \
                + n/(nu_mU*nu_o_mU*np_mU) + n/(nu_o_mU*np_mU*np_o_mU))

    
#chi paring (omega!=0 not implemented)
def chi_pp(U, beta, mu, Niwf,omega=0):
    iw = nu_compute(beta, Niwf)
    
    if (omega/(np.pi/beta))%2!=0:
        print('Warning omega is no bosonic Matsubara frequency!')
    
    size = len(iw)
    z=(np.exp(-mu*beta)+2+np.exp((mu-U)*beta))
    n=N(U, beta, mu)

    nu = iw[:,None]
    nup =  iw[None,:]

    nu_m  = 1j*nu+mu
    nu_mU = 1j*nu+mu-U

    np_m  = 1j*nup+mu
    np_mU = 1j*nup+mu-U
    delta = np.eye(size)
    
    nu_o_m  = 1j*(omega-nu)+mu
    nu_o_mU = 1j*(omega-nu)+mu-U

    np_o_m  = 1j*(omega-nup)+mu
    np_o_mU = 1j*(omega-nup)+mu-U

    J=np.zeros((size,size))
    np.fill_diagonal(np.fliplr(J),1)

        

    if mu == U/2 and omega!=0:
        quot = 0.
    elif mu==U/2 and omega==0:
        quot = beta/2*((np.exp(mu*beta)+2)/(1+np.exp(mu*beta))-1)
    else:
        quot = (2*n-1)/(1j*omega+2*mu-U)
        
  
    return (beta*((1-n)/nu_m + n/nu_mU)*((1-n)/nu_o_m + n/nu_o_mU)*delta \
            + quot *(1/np_o_mU+1/np_mU)*(1/nu_o_mU+1/nu_mU) \
            - beta * 1/z * J * (1/np_m - 1/nu_o_mU)*(1/nu_m - 1/np_o_mU) \
            + beta * U**2 * (np.exp(-U*beta)-1)/z**2 * delta/(np_o_m*np_o_mU*np_m*np_mU) \
            + (n-1)/(np_m*nu_o_mU * nu_m) + (1-n)/(np_m*np_o_mU*nu_o_mU) \
            + (1-n)/(np_mU * np_o_m * nu_o_mU) + (n-1)/(np_o_m * nu_m * nu_o_mU) \
            + (1-n)/(np_o_m*np_m*nu_m) + (1-n)/(np_o_m*np_m*nu_o_m) \
            + (1-n)/(np_mU*nu_o_m*nu_mU) + (n-1)/(np_mU*np_o_m*nu_o_m) \
            + (1-n)/(np_o_mU * nu_o_m * nu_mU) + (n-1)/(np_m * np_o_mU * nu_o_m) \
            - n/(np_o_mU*nu_mU*np_mU) - n/(np_o_mU*np_mU*nu_o_mU))

    



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
# normalized colormap from stackoverflow
# ---------------------------------------
class norm(mpl.colors.Normalize):
    def __init__(self, matrix, midpoint=0, clip=False):
        vmin = np.amin(matrix)
        vmax = np.amax(matrix)
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__7(self, vmin, vmax, clip)

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
