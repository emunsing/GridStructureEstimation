import numpy as np
import time

######### ADMM HELPER FUNCTIONS #############

# First Step Update Functions, B1, S1
def updateB1(S2, B2, B3, Y12, Y13, L, rho):
    
    # Get all the typecasting out of the way
    S2, B2, B3, Y12, Y13, L = (np.matrix(x) for x in (S2, B2, B3, Y12, Y13, L))
    n = L.shape[0]

    out = (S2 * L.T + rho*B2 + rho*B3 - Y12 -Y13) * np.linalg.inv(L*L.T + 2*rho*np.eye(n))
    return out


def S_alpha(x,alpha):
    x = np.matrix(x)
    # Soft threshold operator
    alpha = float(alpha)
    max_op = np.maximum(1-alpha/np.abs(x),0)
    return np.multiply(x, max_op) #This guarantees elementwise multiplication

def updateS1(S2,Y,k2,rho):
    x     = S2 - 1./rho * Y
    alpha = k2 / rho
    return S_alpha(x,alpha)


## Second Step Update Functions: B2, B3, S2
def updateB2(B1_plus, Y12, k1, rho):
    # Inputs can either be matrices or np.ndarrays ...rho is a scalar
    n = Y12.shape[0]
    x = B1_plus + 1./rho * Y12
    x_out = np.minimum(x + k1/rho,0)           # This is the thresholding operator
    np.putmask(x_out,np.eye(n,dtype='bool'),x) # Place the original values of x in the diagonals of x_out
    return x_out
    
def T_alpha(s, U, alpha):
    #Eigenvalue scaling operator used in the computation of B3
    U = np.matrix(U) # Handle inputs

    dvec = 1./2*( s + np.sqrt(s**2 + 4*alpha) )
    d = np.matrix( np.diag(dvec) )
    return U * d * U.T

def updateB3(B1_plus, Y13, k4, rho):
    X = B1_plus + 1./rho * Y13
    X_temp = 1./2 *(X + X.T)
    (s,u) = np.linalg.eig(X_temp)
    return T_alpha( s , u , k4/rho )
    

def P_alpha(x, alpha):
    # Soft thresholding of singular values of matrix, used in computation of S2
    u,s,v = np.linalg.svd(x)
    # Handle inputs
    u = np.matrix(u)
    v = np.matrix(v)
    n = u.shape[0]
    d = np.matrix( np.zeros( [n, v.shape[0]]) ) # Zero matrix. We will later assign the diagonal of this

    # Compose the diagonal matrix
    dvec = np.maximum(s-alpha,0) # Soft threshold
    d[:n,:n] = np.diag(dvec)       # Create diagonals from soft-thresholded singular values
    return u * d * v
    
def updateS2(B1_plus, L, S1_plus, Y, k3, rho):
    B1_plus, L, S1_plus, Y = (np.matrix(x) for x in (B1_plus, L, S1_plus, Y))  #Typecast inputs
    
    x = B1_plus * L + rho * S1_plus + Y
    return 1./(rho+1.) * P_alpha(x,k3)



##############   ADMM IMPLEMENTATION   ###############

def estimateB(mcc,rho,k1,k2,k3,k4,relTol=1e-3):
	# mcc: numpy.ndarray with time in columns, and nodes in rows

	n = mcc.shape[0]
	t = mcc.shape[1]
	L = mcc


	## Initialize B1,B2,B3, S1,S2, Y12,Y13,Y

	B1,B2,B3, Y12,Y13 = [np.matrix(np.zeros([n,n]))]*5
	S1,S2, Y = [np.matrix(np.zeros([n,t]))]*3
	O = np.ones([n,n]) - np.eye(n)  # Off-diagonal matrix, dim n*n   (O := 1 1' - I)


	constraintFlag = True
	i = 1

	startTime = time.time()
	## ADMM Iterations
	while constraintFlag:

	    ## First Step: Update B1, S1
	    B1_plus = updateB1(S2, B2, B3, Y12, Y13, L, rho)
	    S1_plus = updateS1(S2,Y,k2,rho)
	    
	    ## Second step: Update B2,B3,S2
	    B2_plus = updateB2(B1_plus, Y12, k1, rho)
	    B3_plus = updateB3(B1_plus, Y13, k4, rho)
	    S2_plus = updateS2(B1_plus, L , S1_plus, Y, k3, rho)
	    
	    ## Lagrange Multiplier Update
	    Y12_plus = Y12 + rho*(B1_plus - B2_plus)
	    Y13_plus = Y13 + rho*(B1_plus - B3_plus)
	    Y_plus   = Y   + rho*(S1_plus - S2_plus)

	    ## Increment everything
	    B1 = B1_plus
	    B2 = B2_plus
	    B3 = B3_plus
	    S1 = S1_plus
	    S2 = S2_plus
	    Y12 = Y12_plus
	    Y13 = Y13_plus
	    Y = Y_plus
	    

	    b12constraint = np.abs(B1-B2).sum() / max(1e-9, min(np.abs(B1).sum(),np.abs(B2).sum()) )
	    b13constraint = np.abs(B1-B3).sum() / max(1e-9, min(np.abs(B1).sum(),np.abs(B3).sum()) )
	    sConstraint = np.abs(S1-S2).sum() /   max(1e-9, min(np.abs(S1).sum(),np.abs(S2).sum()) )
	    if max(b12constraint, b13constraint, sConstraint) < relTol:
	        constraintFlag = False
	    i += 1

	print("Finished %s iterations in %s seconds"%(i,time.time()-startTime))

	B = np.average([B1,B2,B3],axis=0)

	return B