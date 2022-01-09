from math import log,log2,ceil,exp
from scipy.special import lambertw as W
from numpy import real as re
from numpy import imag as im
import numpy as np
from matplotlib import pyplot as plt
#[hashrate] = hashes per second
#[price] = dollars per hour
#[blocktime] = seconds
def alpha(hashrate,memory,price,blocktime):
    S = 20*price/(3600*memory*(1024**3))
    U = S*blocktime
    H = price/(hashrate*3600)
    return U/H

rs = [2.64*10**6,22.72*10**6,1.83*10**6,78.01*10**6]
p_cs = [0.225,0.1578,0.27,0.918]
p_sval = 0.023
Tval=13.32
def alpha_ec2_s3(T,p_s,p_c,r):
    return (20/(30.4375*24*(1024**3)))*(T*p_s*r/p_c)
def opt_N(T,p_s,p_c,r,eth,gas):
    Rset = 460*(eth*10**-9)*gas
    H = Hval(r,p_c)
    alpha = alpha_ec2_s3(T,p_s,p_c,r)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    beta = H*(1+alpha*(np.exp(s0+1)-1))/(s0+1)

for ix in range(0,len(rs)):
    print(alpha_ec2_s3(13.32,p_sval,p_cs[ix],rs[ix]))

def Hval(hashrate,price):
    return price/(hashrate*3600)

def Uval(price,memory,blocktime):
    S = 20*price/(3600*memory*(1024**3))
    return S*blocktime

def cost_g(N):
    return (0.70268380505036398*Hval(23.4*10**6,0.526)+2.2134739918636780*Uval(0.526,16,13.32))*N*log(N)-Hval(23.4*10**6,0.526)*N

def cost_g_ysel(N):
    return N*(Hval(23.4*10**6,0.526)*ceil(0.5*log2(N))+Uval(0.526,16,13.32)*ceil(log2(N)))

def cost_p_ysel(N):
    return N*(Hval(2.05*10**6,0.90)*ceil(0.5*log2(N))+Uval(0.90,61,13.32)*ceil(log2(N)))

def cost_naive_scaled(N,alpha):
    if alpha <= 1:
        return N+alpha*N*(N+1)/2
    else:
        return N(N+1)/2+N*alpha

def cost_kim_scaled(N,alpha):
    s0 = re(W(((1/alpha)-1)/exp(1)))
    m0_0 = np.floor(log(N)/(s0+1)-1)
    m0_1 = np.ceil(log(N)/(s0+1)-1)
    return min(m0_0*N+alpha*N*(m0_0+1)*(exp(s0+1)-1),m0_1*N+alpha*N*(m0_1+1)*(exp(s0+1)-1))

def cost_ysel_scaled(N,alpha):
    return N*ceil(0.5*log2(N))+N*alpha*ceil(log2(N))

alpha_g = Uval(0.526,16,13.32)/Hval(23.4*10**6,0.526)

def plot_alpha_kim_worse_ysel(N,max_alpha,res):
    alpha = np.linspace(10**-9,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = (1/(alpha+2))*(1/(s0+1)+(alpha/(s0+1))*(np.exp(s0+1)-1)-(log(2)**-1)*(0.5+alpha))
    rhs = 1/log(N)
    plt.plot(alpha,lhs)
    plt.plot(alpha,rhs*np.ones(res))
    plt.plot(alpha,0*np.ones(res))
    plt.show()

def lhs_max_N(max_alpha,res):
    alpha = np.linspace(10**-9,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = (1/(alpha+2))*(1/(s0+1)+(alpha/(s0+1))*(np.exp(s0+1)-1)-(log(2)**-1)*(0.5+alpha))
    return exp(1/np.max(lhs))

mill_hash_for_year = 12*30.4375*24*3600/(13.32*10**6)
def which_alpha_year(max_alpha,res):
    #want (1+s_0)(1+e^{s_0+1}) \leq log(N)
    alpha = np.linspace(0.1,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = (1+s0)*(1+np.exp(1+s0))
    rhs = log(mill_hash_for_year*10**6)
    #want lhs below rhs
    plt.plot(alpha,lhs)
    plt.plot(alpha,rhs*np.ones(res))
    plt.show()

def is_it_even_possible(max_alpha,res):
    alpha = np.linspace(0,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = np.exp(s0+1)+alpha*(np.exp(2*(s0+1))-1)-(log(2)**-1)*(0.5+alpha)*(s0+1)*(1+np.exp(s0+1))
    plt.plot(alpha,lhs)
    plt.plot(alpha,np.zeros(res))
    plt.show()

def plot_alpha_kim_better_ysel(N,max_alpha,res):
    alpha = np.linspace(10**-9,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = 1/(s0+1)+(alpha/(s0+1))*(np.exp(s0+1)-1)-(log(2)**-1)*(0.5+alpha)
    rhs = 1/log(N)
    plt.plot(alpha,lhs)
    plt.plot(alpha,rhs*np.ones(res))
    plt.show()

def alpha_lower_bound(max_alpha,N,res):
    alpha = np.linspace(0.001,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = (1+s0)*(1+np.exp(1+s0))
    rhs = log(N)
    idx = np.argwhere(np.diff(np.sign(lhs-rhs))).flatten()
    return alpha[idx]

def alpha_upper_bound(max_alpha,N,res):
    alpha = np.linspace(10**-9,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = 1/(s0+1)+(alpha/(s0+1))*(np.exp(s0+1)-1)-(log(2)**-1)*(0.5+alpha)
    rhs = 1/log(N)
    idx = np.argwhere(np.diff(np.sign(lhs-rhs))).flatten()
    return alpha[idx]

def alpha_limit_bound(max_alpha,res):
    alpha = np.linspace(10**-9,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = 1/(s0+1)+(alpha/(s0+1))*(np.exp(s0+1)-1)-(log(2)**-1)*(0.5+alpha)
    rhs = 0
    idx = np.argwhere(np.diff(np.sign(lhs-rhs))).flatten()
    return alpha[idx]

def alpha_lower_bound_R(max_alpha,res):
    #want (1+s_0)(1+e^{s_0+1}) \leq log(N)
    alpha = np.linspace(0.001,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    #H = Hval(23.4*10**6,0.526)
    H=Hval(2.05*10**6,0.90)
    beta_div_H = (1+alpha*(np.exp(s0+1)-1))/(s0+1)
    R_div_H = (460*(3225.81*10**-9)*65)/H
    N = R_div_H/beta_div_H
    lhs = (1+s0)*(1+np.exp(1+s0))
    rhs = np.log(N)
    idx = np.argwhere(np.diff(np.sign(lhs-rhs))).flatten()
    print("alpha="+str(alpha[idx])+" N="+str(N[idx]))
    plt.plot(alpha,lhs)
    plt.plot(alpha,rhs)
    plt.show()

def alpha_upper_bound_R(max_alpha,res):
    alpha = np.linspace(10**-9,max_alpha,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    lhs = 1/(s0+1)+(alpha/(s0+1))*(np.exp(s0+1)-1)-(log(2)**-1)*(0.5+alpha)
    #H = Hval(23.4*10**6,0.526)
    H=Hval(2.05*10**6,0.90)
    beta_div_H = (1+alpha*(np.exp(s0+1)-1))/(s0+1)
    R_div_H = (460*(3225.81*10**-9)*65)/H
    N = R_div_H/beta_div_H
    rhs = 1/np.log(N)
    idx = np.argwhere(np.diff(np.sign(lhs-rhs))).flatten()
    print("alpha="+str(alpha[idx])+" N="+str(N[idx]))
    plt.plot(alpha,lhs)
    plt.plot(alpha,rhs)
    plt.show()

def R_div_beta(start,stop,res):
    alpha = np.linspace(start,stop,num=res)
    s0 = re(W(((1/alpha)-1)/exp(1)))
    H=Hval(2.05*10**6,0.90)
    beta_div_H = (1+alpha*(np.exp(s0+1)-1))/(s0+1)
    R_div_H = (460*(3225.81*10**-9)*65)/H
    N = R_div_H/beta_div_H
    plt.plot(alpha,N)
    plt.show()

hash_for_month = 30.4375*24*3600/13.32
Ns = [ceil(hash_for_month*n) for n in [1,2,3,4,5,6,7,8,9,10,11,12,24,36]]
for N in Ns:
    print(str(alpha_lower_bound(10,N,10000000))+str(" ")+str(alpha_upper_bound(10,N,10000000)))
