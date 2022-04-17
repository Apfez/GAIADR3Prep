import numpy as np
import matplotlib.pyplot as plt

M = 100000

i_s = np.random.rayleigh(8*np.pi/180,M)

lmbda = np.random.uniform(-180,180,M)*np.pi/180

i_o = np.random.rayleigh(8*np.pi/180,M)
i_o = 8*np.pi/180
psi = np.arccos(np.cos(i_o)*np.cos(i_s) + np.sin(i_o)*np.sin(i_s)*np.cos(lmbda))

fig, ax = plt.subplots()
ax.hist(i_s*180/np.pi,bins = 40,alpha = 0.3,density = True)
#ax.hist(i_o*180/np.pi,bins = 40,alpha = 0.3,density = True)
ax.hist(psi*180/np.pi,bins = 40,alpha = 0.3,density = True)
ax.set_xlabel('degrees')

print("i_s = %1.1f deg"%np.mean(i_s*180/np.pi))

print("i_o = %1.1f deg"%np.mean(i_o*180/np.pi))

print("psi = %1.1f deg"%np.mean(psi*180/np.pi))

plt.show()

print('am I on git?')