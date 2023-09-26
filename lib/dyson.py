#!/usr/bin/env python
# %%
import numpy as np
import al
from fast_map import fast_map

def Sigma(G,G2_ud,beta,U):
   """
   calculates the self energy for given
   1 particle Green's function G,
   connected 2 particle Green's function G2_ud,
   inverse temperature beta,
   and Hubbard interaction U
   """
   Niv, Nivp, Niw = G2_ud.shape
   niv, = G.shape
   assert Niv == Nivp
   mid = niv//2
   iv_slice = np.arange(mid-Niv//2,mid+Niv//2)
   iw_slice = np.arange(-Niw//2+1,Niw//2+1)
   iv_range = np.arange(Niv)
   iw_range = np.arange(Niw)
   iv = 1j*(2*np.arange(-niv//2,niv//2)+1)*np.pi/beta

   n = np.sum(G-1/iv)/beta + 0.5

   Σ = np.zeros((Niv),dtype=complex)
   Σ += U*n
   
   temp = np.einsum('ijw, i -> i', G2_ud, 1/G)
   Σ += U/beta**2 * temp
   """
   for ω,w in zip(iw_range,iw_slice):
      for i,v in zip(iv_range,iv_slice):
         for j,vp in zip(iv_range,iv_slice):
            Σ[i] += U/beta**2 * G2_ud[i,j,ω]/G[v]
   """
   return Σ

# %%
if __name__ == "__main__":
   #test
   import matplotlib.pyplot as plt
   #parameters 
   U, beta, µ = 1, 100, 1/2 +0.2
   atom = al.atom(U,beta,µ,Niwf=300)

   # for the Green's function more Matsubara 
   # frequencies are needed
   at_more_fq = al.atom(U,beta,µ,Niwf=300)
   g = at_more_fq.g()

   # bosonic Matsubara frequencies
   Niwb = atom.Niwf
   iO = np.arange(-Niwb,Niwb+1)
   v = np.imag(atom.iw())

   # g2_ud : iv, iv', iO
   #g2_ud = np.moveaxis(np.array([atom.g2ud(omega=io) for io in iO]),0,-1)
   g2_ud = np.moveaxis(np.array(list(fast_map(atom.g2ud, iO, threads_limit=100))),0,-1)

   Σ = Sigma(g,g2_ud,beta,U)

   # exact selfenergy
   Self = atom.sigma()

   # compare
   plt.figure()
   plt.subplot(121)
   plt.suptitle("Re$\Sigma$")
   plt.plot(v,Σ.real,'-o',mfc='None')
   plt.plot(v,Self.real,'-x',)
   plt.xlabel("i$\\nu$")
   plt.xlim(-1,1)
   plt.subplot(122)
   plt.suptitle("Im$\Sigma$")
   plt.plot(v,Σ.imag,'-o',mfc='None',label="Dyson equation")
   plt.plot(v,Self.imag,'-x',label="exact")
   plt.xlabel("i$\\nu$")
   plt.xlim(-1,1)
   plt.legend()
   plt.show()


# %%