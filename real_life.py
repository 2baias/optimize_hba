from math import log,log2,ceil,exp
from scipy.special import lambertw as W
from numpy import real as re
from numpy import imag as im
import numpy as np
from matplotlib import pyplot as plt

def Hval(hashrate,price):
    return price/(hashrate*3600)

def Uval(price,T):
    return 20*price*T/(30.4375*24*3600*(1024**3))

rs = [2.64*10**6,22.72*10**6,1.83*10**6,78.01*10**6]#,65139]
p_cs = [0.225,0.1578,0.27,0.918]#,15.78]
p_sval = 0.023
Tval=13.32

def alpha_ec2_s3(T,p_s,p_c,r):
    return (20/(30.4375*24*(1024**3)))*(T*p_s*r/p_c)

def ysel_cost(H,U,N):
    return H*N*ceil(0.5*log2(N))+U*N*ceil(log2(N))

def opt_N_kim(T,p_s,p_c,r,eth,gas):
    Rset = 460*(eth*10**-9)*gas
    H = Hval(r,p_c)
    alpha = alpha_ec2_s3(T,p_s,p_c,r)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    beta = H*(1+alpha*(np.exp(s0+1)-1))/(s0+1)
    return ceil(Rset/beta)

def opt_N_ysel(T,p_s,p_c,r,eth,gas):
    Rset = 460*(eth*10**-9)*gas
    H = Hval(r,p_c)
    U = Uval(p_s,T)
    alpha = alpha_ec2_s3(T,p_s,p_c,r)
    beta = (log(2)**-1)*(H/2+U)
    return ceil(Rset/beta)

def alpha_lower_bound_check(alpha,N):
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = (1+s0)*(1+np.exp(1+s0))
    rhs = log(N)
    return lhs - rhs

def alpha_upper_bound_check(alpha,N):
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = 1/(s0+1)+(alpha/(s0+1))*(np.exp(s0+1)-1)-(log(2)**-1)*(0.5+alpha)
    rhs = 1/log(N)
    return lhs - rhs

for ix in range(0,len(rs)):
    alpha = alpha_ec2_s3(Tval,p_sval,p_cs[ix],rs[ix])
    optNkim = opt_N_kim(Tval,p_sval,p_cs[ix],rs[ix],3225.81,65)
    optNysel = opt_N_ysel(Tval,p_sval,p_cs[ix],rs[ix],3225.81,65)
    H = Hval(rs[ix],p_cs[ix])
    U = Uval(p_sval,Tval)
    yselcost = ysel_cost(H,U,optNysel)
    print("ysel_cost="+str(yselcost))
    print("alpha="
            +str(alpha)
            +str("\nopt_N_kim/10^9=")
            +str((10**-9)*optNkim)
            +str("\nopt_N_ysel/10^9=")
            +str((10**-9)*optNysel))
    print("low_bnd:"
            +str(alpha_lower_bound_check(alpha,optNkim))
            +str("\t")
            +str("upper_bnd:")
            +str(alpha_upper_bound_check(alpha,optNkim)))
    print("\n\n")
