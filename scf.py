"""
 SCF calculation for HeH+ and H2 using  STO-SG3.

 Author: Eric M Gossett
 Dependencies: Numpy

 Based the SCF code found on pages 419-428 in Modern Quantum Chemistry
 By Szabo A. and Ostlund N.
"""
import numpy as np
import math

# Globals
DEBUG = False
VERBOSE = True
# For HeH+ we can switch between SG1, SG2 and SG3 where N is the SG number
N = 3


def overlap(alpha, beta, RAB2):
    """
        Calculates and eturns the matrix element S_uv of the overlap matrix.
    """
    return (
        ((math.pi / (alpha + beta)) ** 1.5) *
        math.exp((-alpha * beta * RAB2)/(alpha + beta))
    )


def kinetic(alpha, beta, RAB2):
    """
        Calculates and returns the kinetic part of the matrix element for H_uv.
    """
    return (
        ((alpha * beta)/(alpha + beta)) *
        (3 - ((2 * alpha * beta * RAB2)/(alpha + beta))) *
        ((math.pi/(alpha + beta)) ** 1.5) *
        math.exp((-alpha * beta * RAB2)/(alpha + beta))
    )


def F_0(t):
    """
        Calculates and returns the value for F_0.
    """
    if t == 0:
        return 1.0
    else:
        return 0.5 * (math.sqrt(math.pi / t)) * math.erf(math.sqrt(t))


def nuclearAttract(alpha, beta, RAB2, RCP2, Zc):
    """
        Calculates and returns the nuclear attraction part of the matrix
        element for H_uv.
    """
    return (
        (-2 * (math.pi)/(alpha + beta)) *
        F_0((alpha + beta) * RCP2) * Zc *
        math.exp((-alpha * beta * RAB2)/(alpha + beta))
    )


def coreHamiltonian(alpha, beta, Zc, Ra, Rb, Rc):
    """
        Calculates and returns the matrix element for the core Hamiltonian,
        H_uv.
    """
    return (
        kinetic(alpha, beta, Ra, Rb) +
        nuclear_attract(alpha, beta, Zc, Ra, Rb, Rc)
    )


def twoE(alpha, beta, gamma, delta, RAB2, RCD2, RPQ2):
    """
        Calculates and returns the matrix element of the two electron
        integral (ab|cd).
    """
    return (
        ((2*(math.pi**2.5)) / (
            (alpha + beta) * (gamma + delta) *
            math.sqrt(alpha + beta + gamma + delta)
        )) *
        F_0(
            ((alpha + beta)*(gamma + delta)*RPQ2) /
            (alpha + beta + gamma + delta)
        ) * math.exp(
            ((-alpha*beta*RAB2)/(alpha + beta)) -
            ((gamma*delta*RCD2)/(gamma + delta))
        )
    )


def scaleExpon(expon, coeff, zeta1, zeta2):
    """
        This function calculates alpha, beta, gamma and delta from the
        exponents and coefficients.
    """
    A1 = np.zeros(3)
    A2 = np.zeros(3)
    D1 = np.zeros(3)
    D2 = np.zeros(3)
    for i in range(0, N):
        A1[i] = expon[N-1][i]*(zeta1**2)
        A2[i] = expon[N-1][i]*(zeta2**2)
        D1[i] = coeff[N-1][i]*((2.0*A1[i]/math.pi)**0.75)
        D2[i] = coeff[N-1][i]*((2.0*A2[i]/math.pi)**0.75)
    return A1, A2, D1, D2


def getCoreHamiltonian(A1, A2, D1, D2, R, Za, Zb):
    """
        Calculates and returns the core Hamiltonian matrix H.
    """
    R2 = R * R
    s12 = t11 = t12 = t22 = 0
    v11a = v11b = v12a = v12b = v22a = v22b = 0
    for i in range(0, N):
        for j in range(0, N):
            RAP = A2[j]*R/(A1[i] + A2[j])
            RAP2 = RAP**2
            RBP2 = (R-RAP)**2
            s12 = s12 + overlap(A1[i], A2[j], R2) * D1[i] * D2[j]

            t11 = t11 + kinetic(A1[i], A1[j], 0) * D1[i] * D1[j]
            v11a = (
                v11a + nuclearAttract(A1[i], A1[j], 0, 0, Za) * D1[i] * D1[j]
            )
            v11b = (
                v11b + nuclearAttract(A1[i], A1[j], 0, R2, Zb) * D1[i] * D1[j]
            )
            t12 = t12 + kinetic(A1[i], A2[j], R2) * D1[i] * D2[j]
            v12a = (
                v12a + nuclearAttract(A1[i], A2[j], R2, RAP2, Za) *
                D1[i] * D2[j]
            )
            v12b = (
                v12b + nuclearAttract(A1[i], A2[j], R2, RBP2, Zb) *
                D1[i] * D2[j]
            )
            t22 = t22 + kinetic(A2[i], A2[j], 0) * D2[i] * D2[j]
            v22a = (
                v22a + nuclearAttract(A2[i], A2[j], 0, R2, Za) * D2[i] * D2[j]
            )
            v22b = (
                v22b + nuclearAttract(A2[i], A2[j], 0, 0, Zb) * D2[i] * D2[j]
            )

    if DEBUG:
        print "A1: ", A1
        print "A2: ", A2
        print "D1: ", D1
        print "D2: ", D2
        print "s12: ", s12
        print "t11: ", t11
        print "t12: ", t12
        print "t22: ", t22
        print "v11a: ", v11a
        print "v12a: ", v12a
        print "v22a: ", v22a
        print "v11b: ", v11b
        print "v12b: ", v12b
        print "v22b: ", v22b

    return np.array([
        [t11 + v11a + v11b, t12 + v12a + v12b],
        [t12 + v12a + v12b, t22 + v22a + v22b]
    ])


def getOverlap(A1, A2, D1, D2, R2):
    """
        Constructs and returns the overlap matrix S.
    """
    s12 = 0
    for i in range(0, N):
        for j in range(0, N):
            s12 = s12 + overlap(A1[i], A2[j], R2) * D1[i] * D2[j]
    return np.array([[1.0, s12], [s12, 1.0]])


def getX(M):
    """
        Preforms canonical orthogonalization on S to obtain X.
        Returns the matrix X.
    """
    X = np.zeros((2, 2))
    X[0][0] = 1 / math.sqrt(2 * (1 + M[0][1]))
    X[1][0] = X[0][0]
    X[0][1] = 1 / math.sqrt(2 * (1 - M[0][1]))
    X[1][1] = -X[0][1]
    return X


def getTwoElectron(A1, A2, D1, D2, R):
    """
        Calculates all the two electron integrals and returns a
        data structure holding all the calculated values.
    """
    R2 = R * R
    v = np.zeros((2, 2, 2, 2))
    v1111 = v2111 = v2121 = v2211 = v2221 = v2222 = 0
    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                for l in range(0, N):
                    RAP = (A2[i] * R) / (A2[i] + A1[j])
                    RBP = R - RAP
                    RAQ = (A2[k] * R) / (A2[k] + A1[l])
                    RBQ = R - RAQ
                    RPQ = RAP - RAQ
                    RAP2 = RAP * RAP
                    RBP2 = RBP * RBP
                    RAQ2 = RAQ * RAQ
                    RBQ2 = RBQ * RBQ
                    RPQ2 = RPQ * RPQ
                    v1111 = (
                        v1111 + twoE(A1[i], A1[j], A1[k], A1[l], 0, 0, 0) *
                        D1[i] * D1[j] * D1[k] * D1[l]
                    )
                    v2111 = (
                        v2111 + twoE(A2[i], A1[j], A1[k], A1[l], R2, 0, RAP2) *
                        D2[i] * D1[j] * D1[k] * D1[l]
                    )
                    v2121 = (
                        v2121 +
                        twoE(A2[i], A1[j], A2[k], A1[l], R2, R2, RPQ2) *
                        D2[i] * D1[j] * D2[k] * D1[l]
                    )
                    v2211 = (
                        v2211 + twoE(A2[i], A2[j], A1[k], A1[l], 0, 0, R2) *
                        D2[i] * D2[j] * D1[k] * D1[l]
                    )
                    v2221 = (
                        v2221 + twoE(A2[i], A2[j], A2[k], A1[l], 0, R2, RBQ2) *
                        D2[i] * D2[j] * D2[k] * D1[l]
                    )
                    v2222 = (
                        v2222 + twoE(A2[i], A2[j], A2[k], A2[l], 0, 0, 0) *
                        D2[i] * D2[j] * D2[k] * D2[l]
                    )

    v[0][0][0][0] = v1111
    v[1][0][0][0] = v2111
    v[0][1][0][0] = v2111
    v[0][0][1][0] = v2111
    v[0][0][0][1] = v2111
    v[1][0][1][0] = v2121
    v[0][1][1][0] = v2121
    v[1][0][0][1] = v2121
    v[0][1][0][1] = v2121
    v[1][1][0][0] = v2211
    v[0][0][1][1] = v2211
    v[1][1][1][0] = v2221
    v[1][1][0][1] = v2221
    v[1][0][1][1] = v2221
    v[0][1][1][1] = v2221
    v[1][1][1][1] = v2222

    if DEBUG:
        print '( 1 1 1 1) ', v[0][0][0][0]
        print '( 1 1 1 2) ', v[1][0][0][0]
        print '( 1 1 2 1) ', v[0][0][1][1]
        print '( 1 1 2 2) ', v[0][0][1][1]
        print '( 1 2 1 1) ', v[0][1][0][0]
        print '( 1 2 1 2) ', v[0][1][0][1]
        print '( 1 2 2 1) ', v[0][1][1][0]
        print '( 1 2 2 2) ', v[0][1][1][1]
        print '( 2 1 1 1) ', v[1][0][0][0]
        print '( 2 1 1 2) ', v[1][0][0][1]
        print '( 2 1 2 1) ', v[1][0][1][0]
        print '( 2 1 2 2) ', v[1][0][1][1]
        print '( 2 2 1 1) ', v[1][1][0][0]
        print '( 2 2 1 2) ', v[1][1][0][1]
        print '( 2 2 2 1) ', v[1][1][1][0]
        print '( 2 2 2 2) ', v[1][1][1][1]

    return v


def getG(A1, A2, D1, D2, P, R):
    """
        Constructs and returns the matrix G.
    """
    V = getTwoElectron(A1, A2, D1, D2, R)
    G = np.zeros((2, 2))
    for i in range(0, N-1):
        for j in range(0, N-1):
            G[i][j] = 0
            for k in range(0, N-1):
                for l in range(0, N-1):
                    G[i][j] = (
                        G[i][j] + P[k][l] *
                        (V[i][j][k][l] - 0.5 * V[i][l][k][j])
                    )
    return G


def getNewP(P, C):
    """
        Calculates and returns the new density matrix.
    """
    P_new = np.zeros((2, 2))
    for i in range(0, N-1):
        for j in range(0, N-1):
            P_new[i][j] = P_new[i][j] + 2 * C[i][0] * C[j][0]
    return P_new


def getEne(P, H, F):
    """
        Determines the electronic energy from P, H and F and returns
        the electronic energy.
    """
    E = 0
    for i in range(0, N-1):
        for j in range(0, N-1):
            E = E + 0.5 * P[i][j] * (H[i][j] + F[i][j])
    return E


def getDelta(P_old, P_new):
    """
        returns delta for the convergence test.
    """
    delta = 0.0
    for i in range(0, N-1):
        for j in range(0, N-1):
            delta = delta + ((P_new[i][j] - P_old[i][j]) ** 2)
    delta = math.sqrt(delta/4)
    return delta


def runSCF(Ra, Rb, R, Za, Zb, coeff, expon, zeta1, zeta2):
    """
        Main loop of the SCF calculation.
    """
    R2 = R*R
    A1, A2, D1, D2 = scaleExpon(expon, coeff, zeta1, zeta2)
    iteration = 0
    S = getOverlap(A1, A2, D1, D2, R2)
    H = getCoreHamiltonian(A1, A2, D1, D2, R, Za, Zb)
    X = getX(S)
    XT = X.T
    P = np.zeros((2, 2))
    delta = 1.0
    convergance = 1.0e-4
    max_itr = 200

    if VERBOSE:
            print '%%%%%%   OVERLAP MATRIX  %%%%%%%'
            print ''
            print S
            print ''
            print '%%%%%%     X MATRIX      %%%%%%%%'
            print ''
            print X
            print ''
            print '%%%%%%% CORE HAMILTONIAN %%%%%%%%'
            print ''
            print H
            print ''
            print '%%%%%%%  DENSITY MATRIX  %%%%%%%%'
            print ''
            print P
            print ''

    ene = 0
    while delta > convergance:
        G = getG(A1, A2, D1, D2, P, R)
        F = H + G
        ene = getEne(P, H, F)
        FX = np.dot(F, X)
        F_prime = np.dot(XT, np.dot(F, X))
        e, C_prime = np.linalg.eigh(F_prime)
        E = np.diag(e)
        C = np.dot(X, C_prime)
        P_new = getNewP(P, C)
        iteration = iteration + 1
        delta = getDelta(P, P_new)
        P = P_new

        if VERBOSE:
            print ''
            print '---------------------------------'
            print '        ' + 'Iteration #' + str(iteration)
            print '---------------------------------'
            print ''
            print '%%%%%%%     G MATRIX     %%%%%%%%'
            print ''
            print G
            print ''
            print '%%%%%%%     F MATRIX    %%%%%%%%'
            print ''
            print F
            print ''
            print '%%%%%%%     F PRIME      %%%%%%%%'
            print ''
            print F_prime
            print ''
            print '%%%%%%%     C PRIME      %%%%%%%%'
            print ''
            print C_prime
            print ''
            print '%%%%%%%     E MATRIX     %%%%%%%%'
            print ''
            print E
            print ''
            print '%%%%%%%     C MATRIX     %%%%%%%%'
            print ''
            print C
            print ''
            print '%%%%%%%   NEW P MATRIX   %%%%%%%%'
            print ''
            print P_new
            print ''
            print 'DELTA: ', delta
            print 'ELECTRONIC ENERGY: ', ene

        if iteration == max_itr:
            break

    ene_tot = ene + (Za*Zb/R)
    PS = np.dot(P, S)

    if VERBOSE:
        print ''
        print ''
        print 'CONVERGED AFTER ' + str(iteration) + ' ITERATIONS '
        print ''
        print 'TOTAL ENERGY: ', ene_tot
        print ''
        print '%%%%%%%    PS MATRIX     %%%%%%%%'
        print ''
        print PS


def main():
    """
        This is where the above functions are used to preform
        the SCF calculation. Global variables are initalized here
        for both the HeH+ and H2 case.
    """

    # STARTING WITH HeH+ CASE
    Za = 2.0
    Zb = 1.0
    R = 1.4632
    R2 = R*R
    zeta1 = 2.0925
    zeta2 = 1.24
    Ra = np.array([0.0, 0.0, 0.0])
    Rb = np.array([R, 0.0, 0.0])

    # THESE ARE THE COEFF AND EXPON FOR STO-SG1, STO-SG2, STO-SG3
    coeff = np.array([
        [1.0, 0, 0],
        [0.678914, 0.430129, 0],
        [0.444635, 0.535328, 0.154329]
    ])
    expon = np.array([
        [0.270950, 0, 0],
        [0.151623, 0.851814, 0],
        [0.109818, 0.405771, 2.22766]
    ])

    print '|-------------------------------------------------_'
    print '|                      HeH+                        |'
    print '|_________________________________________________-'
    print ''
    runSCF(Ra,  Rb, R, Za, Zb, coeff, expon, zeta1, zeta2)

    # NOW FOR H2 CASE
    Za = Zb = 1.0
    R = 1.4
    R2 = R*R
    zeta1 = zeta2 = 1.24
    Ra = np.array([0.0, 0.0, 0.0])
    Rb = np.array([R, 0.0, 0.0])

    print ''
    print ''
    print '|-------------------------------------------------_'
    print '|                       H2                         |'
    print '|_________________________________________________-'
    print ''
    runSCF(Ra,  Rb, R, Za, Zb, coeff, expon, zeta1, zeta2)

if __name__ == '__main__':
    main()
