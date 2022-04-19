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

###############################################################################
#%%
# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.set_size_inches(15, 15)


i_o_err = 0.001*np.pi/180
psi_err = 5*np.pi/180


incs = []
for n in range(100):

    i_s = np.arccos(np.random.uniform(0, 1))
    lmbda = np.random.uniform(-np.pi, np.pi)

    n_s = [np.sin(i_s * np.pi / 180) * np.sin(lmbda * np.pi / 180),
           np.sin(i_s * np.pi / 180) * np.cos(lmbda * np.pi / 180),
           np.cos(i_s * np.pi / 180)]

    ax.scatter(n_s[0], n_s[1], n_s[2])

    i_o = 10000
    while i_o > 20 * np.pi / 180:
        i_o = np.random.rayleigh(i_o_err)

    psi = np.random.rayleigh(psi_err)
    Omega = np.random.uniform(-np.pi * 2, np.pi * 2)

    lmbda = np.arctan(
        np.sin(psi) * np.sin(Omega) / (np.cos(psi) * np.sin(i_o) + np.sin(psi) * np.cos(Omega) * np.cos(i_o)))

    i_s = np.abs(np.arcsin(np.sin(psi) * np.sin(Omega) / np.sin(lmbda)))




plt.show()
