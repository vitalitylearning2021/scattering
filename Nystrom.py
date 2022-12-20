# --- This code solves the Electric Field Integral Equation (EFIE) for Perfectly Electric Conducting (PEC) cylinder with
#     circular or elliptical cross section. It can be easily generalized to other cross-sectional shapes. 

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

def trigInterp(n, j):
    out = 0.
    for m in range(1, n):
        out = out + (1. / m) * np.cos(m * j * np.pi / n)

    out = -(2 * np.pi / n) * out - (-1)**j * np.pi / (n**2)
    return out

##############################
# ELECTROMAGNETIC PARAMETERS #
##############################

epsilon0            = 8.85e-12                      # --- F / m
mu0                 = 4. * np.pi * 1e-7             # --- H / m
epsilonr            = 1                             # --- Relative permittivity of the host medium
mur                 = 1                             # --- Relative permeability of the host medium
epsilon             = epsilon0 * epsilonr           # --- Permittivity of the host medium
mu                  = mu0 * mur                     # --- Permeability of the host medium
lambd               = 1                             # --- Wavelength
k                   = (2. * np.pi) / lambd          # --- Wavenumber
zita                = np.sqrt(mu / epsilon)         # --- Intrinsic impedance of the host medium 
f                   = 3.e8 / lambd                  # --- Working frequency
w                   = 2. * np.pi * f                # --- Working wavenumber

C                   = 0.5772156649

########################
# SCATTERER'S GEOMETRY #
########################
a                   = 2 * lambd                     # --- First semi-axis of the elliptical cross-section
b                   = 2 * lambd                     # --- Second semi-axis of the elliptical cross-section

##########################
# SURFACE DISCRETIZATION #
##########################
N                   = 10                            # --- Initial number of quadrature nodes
err                 = np.inf                        # --- Initial discretization accuracy
prec                = lambd / 30                    # --- Target discretization accuracy

# --- Determining the number of nodes to meet a target accuracy
while (err > prec):
    quadratureNodes = np.linspace(0., 2. * np.pi, N)
    err             = np.sqrt((a * np.cos(quadratureNodes[1]) - b * np.cos(quadratureNodes[0]))**2 + 
                              (a * np.sin(quadratureNodes[1]) - b * np.sin(quadratureNodes[0]))**2)
    if (err > prec):
        N = N + 10

tn                  = np.pi * np.arange(0, 2 * N) / N
Xn                  = a * np.cos(tn)
Yn                  = b * np.sin(tn)

dXn                 = -a * np.sin(tn)
dYn                 =  b * np.cos(tn)

#####################
# RELEVANT MATRICES #
#####################

M1                  = np.zeros ((2 * N, 2 * N))
M2                  = np.zeros ((2 * N, 2 * N), dtype = np.complex128)
R                   = np.zeros ((2 * N, 2 * N))
A                   = np.zeros ((2 * N, 2 * N), dtype = np.complex128)

for i in range(2 * N):          # --- Observation point
    for j in range(2 * N):      # --- Source point
        M1[i, j]     = -(1. / (2. * np.pi)) * sp.special.jv(0, k * np.sqrt((Xn[i] - Xn[j])**2 + (Yn[i] - Yn[j])**2)) * \
                        np.sqrt((dXn[j]**2 + dYn[j]**2))
        R[i, j]      = trigInterp(N, abs(i - j))
        if (i == j):
            M2[i, j] = ((-1j / 2.) - (C / np.pi) - (1. / np.pi) * np.log((k / 2.) * \
                        np.sqrt((dXn[i]**2 + dYn[i]**2)))) * np.sqrt((dXn[i]**2 + dYn[i]**2))    
        else:
            M2[i, j] = (-1j / 2.) * sp.special.hankel2(0, k * np.sqrt((Xn[i] - Xn[j])**2 + (Yn[i] - Yn[j])**2)) * \
                       np.sqrt((dXn[j]**2 + dYn[j]**2)) - M1[i, j] * np.log(4. * (np.sin((tn[i] - tn[j]) / 2.))**2)
      
        A[i, j] = -R[i, j] * (-1j * zita * M1[i, j]) - (np.pi / N) * (-1j * zita * M2[i, j])

###################
# IMPINGING FIELD #
###################
E0                  = 1.                            # --- Complex amplitude of the impinging field
Ei                  = E0 * np.exp(-1j * k * Xn)

#######################
# CURRENT COMPUTATION #
#######################
J                   = np.linalg.solve(A, Ei) * 2. * zita / (w * mu)

#################################
# REFERENCE CURRENT COMPUTATION #
#################################
Jref                = np.zeros_like(J)
for n in range(-int(np.ceil(3. * k * a)), int(np.ceil(3. * k * a))):
    Jref = Jref + 1j**(-n) * np.exp(1j * n * tn) / sp.special.hankel2(n, k * a)

Jref                = 2. * E0 * Jref / (a * np.pi * w * mu)

rmsRefCurrent       = 100. * np.sqrt(np.sum(np.abs(J - Jref)**2) / np.sum(np.abs(Jref)**2))
print('Root Mean Square error with respect to reference current {}'.format(rmsRefCurrent))

#########################
# FAR-FIELD COMPUTATION #
#########################
Deltal              = np.sqrt((Xn[1] - Xn[0])**2 + (Yn[1] - Yn[0])**2);     # --- Length of the discretization segments
rFar                = 2 * 2 * max(a, b)**2 / lambd                          # --- Implements the condition r >= 2 * D^2 / lambd

Es                  = np.zeros(tn.shape, dtype = np.complex128)
for i in range(tn.size):
    Es[i] = - (w * mu0 / 4.) * (2. * np.pi * a / tn.size) * np.sum(sp.special.hankel2(0, k * np.sqrt((rFar * np.cos(tn[i]) - Xn)**2 + (rFar * np.sin(tn[i]) - Yn)**2)) * J)

###################################
# REFERENCE FAR-FIELD COMPUTATION #
###################################
EsRif               = np.zeros(tn.shape, dtype = np.complex128)

for n in range(-int(np.ceil(3. * k * a)), int(np.ceil(3. * k * a))):
    EsRif           = EsRif + 1j**(-n) * (sp.special.jv(n, k * a) / sp.special.hankel2(n, k * a)) * sp.special.hankel2(n, k * rFar) * np.exp(1j * n * tn)

EsRif               = -E0 * EsRif

rmsRefField         = 100. * np.sqrt(np.sum(np.abs(Es - EsRif)**2) / np.sum(np.abs(EsRif)**2))
print('Root Mean Square error with respect to reference fieldt {}'.format(rmsRefField))

##########
# GRAPHS #
##########

plt.plot(tn, np.abs(J), 'r', linewidth = 2)
plt.plot(tn, np.abs(Jref), 'b--', linewidth = 4)
plt.title('Current amplitude')
plt.xlabel('Angle')
plt.ylabel('|J| [A/m]')
plt.show()

plt.plot(tn, np.angle(J), 'r', linewidth = 2)
plt.plot(tn, np.angle(Jref), 'b--', linewidth = 4)
plt.title('Current phase')
plt.xlabel('Angle')
plt.ylabel('Arg(J) [rad]')
plt.show()

plt.plot(tn, np.abs(Es), 'r', linewidth = 2)
plt.plot(tn, np.abs(EsRif), 'b--', linewidth = 4)
plt.title('Far-field amplitude')
plt.xlabel('Angle')
plt.ylabel('|Es| [A/m]')
plt.show()

plt.plot(tn, np.angle(Es), 'r', linewidth = 2)
plt.plot(tn, np.angle(EsRif), 'b--', linewidth = 4)
plt.title('Far-field phase')
plt.xlabel('Angle')
plt.ylabel('Arg(Es) [rad]')
plt.show()


