"""
This library contains functions for creating 2D surface plots using scattered
function values.  The functions are designed to work either globally or locally,
although the default behavior is to use a global least squares polynomial.

To approximate locally, the user breaks the original rectangular domain into
many smaller rectangular subdomains.  A local approximation in a given subdomain
attempts to match all data in the subdomain plus adjacent subdomains.

Greg Barnett
August 2022
"""
################################################################################

import numpy as np

################################################################################

def getContourLevels(vals, useMeanOf = (), minDiff = 2, nColors = 64) :
    """
    Get the z-values to be used to make the contour levels in the 2D surf plots.
    """
    if useMeanOf == () :
        useMeanOf = vals
    m = np.mean(useMeanOf)
    D = np.max([np.max(vals) - m, m - np.min(vals), minDiff])
    clevels = np.linspace(m - D, m + D, nColors + 1)
    return clevels

################################################################################

def printPolyCoeffs(lam) :
    """
    Print polynomial coefficients, hopefully in a format that can be easily
    copied and pasted somewhere else.
    """
    ell = len(lam)
    useFormat = "2.6e"
    count = 0
    def printLine(s, count) :
        if lam[count] < 0 :
            s = s + 3 * " "
        else :
            s = s + 4 * " "
        s = s + "{0:" + useFormat + "}"
        print(s . format(lam[count]))
        count = count + 1
        return count
    if ell >= 1 :
        count = printLine("x0y0", count)
    if ell >= 3 :
        count = printLine("x1y0", count)
        count = printLine("x0y1", count)
    if ell >= 6 :
        count = printLine("x2y0", count)
        count = printLine("x1y1", count)
        count = printLine("x0y2", count)
    if ell >= 10 :
        count = printLine("x3y0", count)
        count = printLine("x2y1", count)
        count = printLine("x1y2", count)
        count = printLine("x0y3", count)
    if ell >= 15 :
        count = printLine("x4y0", count)
        count = printLine("x3y1", count)
        count = printLine("x2y2", count)
        count = printLine("x1y3", count)
        count = printLine("x0y4", count)
    if ell >= 21 :
        count = printLine("x5y0", count)
        count = printLine("x4y1", count)
        count = printLine("x3y2", count)
        count = printLine("x2y3", count)
        count = printLine("x1y4", count)
        count = printLine("x0y5", count)
    if ell >= 28 :
        count = printLine("x6y0", count)
        count = printLine("x5y1", count)
        count = printLine("x4y2", count)
        count = printLine("x3y3", count)
        count = printLine("x2y4", count)
        count = printLine("x1y5", count)
        count = printLine("x0y6", count)
    if ell >= 36 :
        count = printLine("x7y0", count)
        count = printLine("x6y1", count)
        count = printLine("x5y2", count)
        count = printLine("x4y3", count)
        count = printLine("x3y4", count)
        count = printLine("x2y5", count)
        count = printLine("x1y6", count)
        count = printLine("x0y7", count)
    if (ell > 36) or (ell < 1) :
        raise ValueError("Polynomial degree less than or equal to 7, please.")

################################################################################

def phs(x, y, rbfParam) :
    """
    Polyharmonic spline radial basis function with odd exponent (rbfParam).
    """
    return (x**2 + y**2) ** (rbfParam/2)

################################################################################

def rbf(x, y, xc, yc, rbfParam) :
    """
    RBF matrix with basis functions arranged in columns.
    """
    A = np.zeros((len(x), len(xc)), float)
    
    for i in range(len(x)) :
        for j in range(len(xc)) :
            A[i,j] = phs(x[i] - xc[j], y[i] - yc[j], rbfParam)
            
    return A
    
################################################################################
    
def poly(x, y, pd) :
    """
    Polynomial matrix with monomials arranged in columns.
    """
    # Maximum polynomial degree allowed is 7.
    maxD = 7
    if pd > maxD :
        exit("Please choose a reasonable polynomial degree (0 <= pd <= " + maxD + ").")
    
    # Make the polynomial matrix one degree at a time.
    p = np.zeros((len(x), int((pd+1)*(pd+2)/2)), float)
    count = 0
    numP = 0
    for i in range(pd + 1) :
        for j in range(numP + 1) :
            if (j == 0) and (numP == 0) :
                p[:,count] = 1
            elif (j == 0) :
                p[:,count] = x**(numP-j)
            elif (numP-j == 0) :
                p[:,count] = y**j
            else :
                p[:,count] = x**(numP-j) * y**j
            count += 1
        numP += 1
    # print(p)
    # exit()
    
    # p = np.zeros((len(x), int((pd+1)*(pd+2)/2)), float)
    # if pd >= 0 :
        # p[:,0] = 1.
    # if pd >= 1 :
        # p[:,1] = x
        # p[:,2] = y
    # if pd >= 2 :
        # p[:,3] = x**2
        # p[:,4] = x * y
        # p[:,5] = y**2
    # if pd >= 3 :
        # p[:,6] = x**3
        # p[:,7] = x**2 * y
        # p[:,8] = x * y**2
        # p[:,9] = y**3
    # if pd >= 4 :
        # p[:,10] = x**4
        # p[:,11] = x**3 * y
        # p[:,12] = x**2 * y**2
        # p[:,13] = x * y**3
        # p[:,14] = y**4
    # if pd >= 5 :
        # p[:,15] = x**5
        # p[:,16] = x**4 * y
        # p[:,17] = x**3 * y**2
        # p[:,18] = x**2 * y**3
        # p[:,19] = x * y**4
        # p[:,20] = y**5
    # if pd >= 6 :
        # p[:,21] = x**6
        # p[:,22] = x**5 * y
        # p[:,23] = x**4 * y**2
        # p[:,24] = x**3 * y**3
        # p[:,25] = x**2 * y**4
        # p[:,26] = x * y**5
        # p[:,27] = y**6
    # if pd >= 7 :
        # p[:,28] = x**7
        # p[:,29] = x**6 * y
        # p[:,30] = x**5 * y**2
        # p[:,31] = x**4 * y**3
        # p[:,32] = x**3 * y**4
        # p[:,33] = x**2 * y**5
        # p[:,34] = x * y**6
        # p[:,35] = y**7
    # if (pd < 0) or (pd > 7) :
        # exit("Please choose a reasonable polynomial degree (0 <= pd <= 7).")
    # print(p)
    # exit()
        
    return p

################################################################################

def inSquare(x, y, xmci, ymci, ell, w) :
    """
    Indices of [x,y] inside the rectangle centered at [xmci,ymci].
    """
    ind = np.array([], int)
    
    for j in range(len(x)) :
        if (np.abs(x[j] - xmci) <= w) and (np.abs(y[j] - ymci) <= ell) :
            ind = np.hstack((ind, j))
            
    return ind

################################################################################

def assignDefaults(x, y, xmc, ymc, ell, w, ELL, W) :
    """
    Assuming only a single subdomain, this function assigns default values to
    the variables that are not needed.  In other words, if the user does not
    give enough information to perform a local approximation, then the default
    is to perform a single global approximation (see functions following this).
    """
    if len(xmc) == 0 :
        xmc = np.array([(np.min(x) + np.max(x)) / 2])
    if len(ymc) == 0 :
        ymc = np.array([(np.min(y) + np.max(y)) / 2])
    if not ell :
        ell = (np.max(y) - np.min(y)) / 2
    if not w :
        w = (np.max(x) - np.min(x)) / 2
    if not ELL :
        ELL = 3 * ell
    if not W :
        W = 3 * w
    
    return xmc, ymc, ell, w, ELL, W

################################################################################

def polyLS(pd, x, y, f, X, Y \
, coeff = [], xmc = [], ymc = [], ell = [], w = [], ELL = [], W = []) :
    """
    After breaking the large rectangular domain into small rectangular
    subdomains of length 2*ell and width 2*w, go through the subdomains and find
    a polynomial least squares approximation using data in the subdomain and all
    adjacent subdomains.

    pd is the desired polynomial degree.
    (x,y,f) defines the known function values.
    (X,Y) are the points where you WANT to know the function.
    coeff contains coefficients to evaluate the approximant (empty first time).
    (xmc,ymc) are the centers of the rectangular subdomains.
    ell is half the length of a rectangular subdomain.
    w is half the width of a rectangular subdomain.
    ELL = 3 * ell.
    W = 3 * w.
    """
    xmc, ymc, ell, w, ELL, W = assignDefaults(x, y, xmc, ymc, ell, w, ELL, W)
    
    numP = int((pd + 1) * (pd + 2) / 2)
    
    if (len(xmc) == 1) and (len(ymc) == 1) :
        

        if coeff == [] :
            p = poly(x, y, pd)
            coeff = np.linalg.lstsq(p, f, rcond=None)[0]

        B = poly(X, Y, pd)
        approx = B.dot(coeff).flatten()
        coeff_copy = coeff
        
    else :
        
        approx = np.zeros(len(X), float)
        
        if coeff == [] :
            for i in range(len(xmc)) :
                IND = inSquare(x, y, xmc[i], ymc[i], ELL, W)
                if len(IND) < int(1.5 * numP) :
                    raise ValueError("Not enough data for this polynomial " \
                    + "degree.\nEither lower the polynomial degree or " \
                    + "decrease the number of subdivisions.")
                p = poly(x[IND], y[IND], pd)
                lam = np.linalg.lstsq(p, f[IND], rcond=None)[0]
                coeff.append(lam)

        coeff_copy = coeff.copy()

        for i in range(len(xmc) - 1, -1, -1) :
            IND = inSquare(X, Y, xmc[i], ymc[i], ell, w)
            B = poly(X[IND], Y[IND], pd)
            lam = coeff.pop()
            approx[IND] = B.dot(lam).flatten()
        
    return approx, coeff_copy

################################################################################

def RBFLS(rbfParam, pd, x, y, f, X, Y \
, xmc = [], ymc = [], ell = [], w = [], ELL = [], W = []) :
    """
    After breaking the large rectangular domain into small rectangular
    subdomains of length 2*ell and width 2*w, go through the subdomains and find
    a radial basis function (plus polynomials) least squares approximation using
    data in the subdomain and all adjacent subdomains.

    rbfParam is the exponent in the polyharmonic spline radial basis function.
    pd is the desired polynomial degree.
    (x,y,f) defines the known function values.
    (X,Y) are the points where you WANT to know the function.
    (xmc,ymc) are the centers of the rectangular subdomains.
    ell is half the length of a rectangular subdomain.
    w is half the width of a rectangular subdomain.
    ELL = 3 * ell.
    W = 3 * w.
    """
    xmc, ymc, ell, w, ELL, W = assignDefaults(x, y, xmc, ymc, ell, w, ELL, W)
    
    if (len(xmc) < 2) and (len(ymc) < 2) :
    
        raise ValueError("Need at least 4 subregions (total) for RBF-LS.")
        
    else :
    
        approx = np.zeros(len(X), float)
        approx_sparse = np.zeros(len(x), float)
        
        for i in range(len(xmc)) :
            IND = inSquare(x, y, xmc[i], ymc[i], ELL, W)
            p = poly(x[IND], y[IND], pd)
            ind = inSquare(x[IND], y[IND], xmc[i], ymc[i], ell, w)
            xc = x[IND][ind]
            yc = y[IND][ind]
            A = rbf(x[IND], y[IND], xc, yc, rbfParam)
            A = np.hstack((A, p))
            lam = np.linalg.lstsq(A, f[IND], rcond=None)[0]
            approx_sparse[IND] = A.dot(lam).flatten()
            IND = inSquare(X, Y, xmc[i], ymc[i], ell, w)
            A = rbf(X[IND], Y[IND], xc, yc, rbfParam)
            B = poly(X[IND], Y[IND], pd)
            B = np.hstack((A, B))
            approx[IND] = B.dot(lam).flatten()
        
    return approx, approx_sparse, lam

################################################################################

def RBFinterp(rbfParam, pd, x, y, f, X, Y \
, coeff = [], xmc = [], ymc = [], ell = [], w = [], ELL = [], W = []) :
    """
    After breaking the large rectangular domain into small rectangular
    subdomains of length 2*ell and width 2*w, go through the subdomains and find
    a radial basis function (plus polynomials) INTERPOLANT using data in the
    subdomain and all adjacent subdomains.

    rbfParam is the exponent in the polyharmonic spline radial basis function.
    pd is the desired polynomial degree.
    (x,y,f) defines the known function values.
    (X,Y) are the points where you WANT to know the function.
    coeff contains coefficients to evaluate the interpolant (empty first time).
    (xmc,ymc) are the centers of the rectangular subdomains.
    ell is half the length of a rectangular subdomain.
    w is half the width of a rectangular subdomain.
    ELL = 3 * ell.
    W = 3 * w.
    """
    xmc, ymc, ell, w, ELL, W = assignDefaults(x, y, xmc, ymc, ell, w, ELL, W)
    
    numP = int((pd + 1) * (pd + 2) / 2)
    zp1 = np.zeros((numP, 1))
    zp2 = np.zeros((numP, numP))
    
    if (len(xmc) == 1) and (len(ymc) == 1) :
    
        if coeff == [] :
            p = poly(x, y, pd)
            A = rbf(x, y, x, y, rbfParam)
            A = np.hstack((A, p))
            tmp = np.hstack(( p.T, zp2))
            A = np.vstack((A, tmp))
            tmp = np.reshape(f, (len(x), 1))
            coeff = np.linalg.solve(A, np.vstack((tmp, zp1)))

        A = rbf(X, Y, x, y, rbfParam)
        B = poly(X, Y, pd)
        B = np.hstack((A, B))
        approx = B.dot(coeff).flatten()
        coeff_copy = coeff
    
    else :
    
        approx = np.zeros(len(X), float)

        if coeff == [] :
            for i in range(len(xmc)) :
                # Find the index IND of all nodes in the subdomain, and all
                # adjacent subdomains.  Create the combined RBF-polynomial
                # interpolation matrix and solve for the coefficients.
                IND = inSquare(x, y, xmc[i], ymc[i], ELL, W)
                if len(IND) < int(1.5 * numP) :
                    raise ValueError("Not enough data for this polynomial " \
                    + "degree.\nEither lower the polynomial degree or " \
                    + "decrease the number of subdivisions.")
                p = poly(x[IND], y[IND], pd)
                A = rbf(x[IND], y[IND], x[IND], y[IND], rbfParam)
                A = np.hstack((A, p))
                tmp = np.hstack((p.T, zp2))
                A = np.vstack((A, tmp))
                tmp = np.reshape(f[IND], (len(IND), 1))
                lam = np.linalg.solve(A, np.vstack((tmp, zp1)))
                coeff.append(lam)
                
        coeff_copy = coeff.copy()

        for i in range(len(xmc) - 1, -1, -1) :
            # Find the index IND of all evaluation points in the subdomain, and
            # evaluate the interpolant there to produce approx[IND].
            IND = inSquare(X, Y, xmc[i], ymc[i], ell, w)
            ind = inSquare(x, y, xmc[i], ymc[i], ELL, W)
            A = rbf(X[IND], Y[IND], x[ind], y[ind], rbfParam)
            B = poly(X[IND], Y[IND], pd)
            B = np.hstack((A, B))
            lam = coeff.pop()
            approx[IND] = B.dot(lam).flatten()
        
    return approx, coeff_copy

################################################################################

