import numpy.linalg as nla
import scipy.linalg as sla
import numpy as np
from numpy import identity as id
from pylab import *
import cmath as cm
import random
import matplotlib.pyplot as plt
import cmath
from itertools import compress, product
import math

def Herm(A):
    A = np.transpose(np.conj(A))
    return A

def NKron(*args):
  """Calculate a Kronecker product over a variable number of inputs"""
  result = np.array([[1.0]])
  for op in args:  
    result = np.kron(result, op)
  return result

def MagnetZ(Psi, N):
    MagnetSum=0
    Mag=NKron(PZ,id(2**(N-1)))
    Magnet=np.dot(Herm(Psi),np.dot(Mag,Psi))
    Magnet=MagnetSum+0.5*float(np.real(Magnet))
    return Magnet
    
def MagnetX(Psi, N):
    MagnetSum=0
    Mag=NKron(PX,id(2**(N-1)))
    Magnet=np.dot(Herm(Psi),np.dot(Mag,Psi))
    MagnetSum=MagnetSum+0.5*float(np.real(Magnet))
    return Magnet

def LongCorr(Psi, N):
    Corr=NKron(PX,id(2**(N-2)),PX)
    XN=NKron(id(2**(N-1)),PX)
    Correlation=float(np.real(np.dot(Herm(Psi),np.dot(Corr, Psi)))) - float(np.real(MagnetX(Psi,N))) * float(np.real(np.dot(Herm(Psi), np.dot(XN,Psi))))
    return float(np.real(Correlation))
def Ham(N):
    H=np.zeros((2**N,2**N),dtype=np.complex128)
    H=H+J*(NKron(PX,PX,id(2**(N-2))) + NKron(id(2**(N-2)),PX,PX))
    H=H+J*(NKron(PY,PY,id(2**(N-2))) + NKron(id(2**(N-2)),PY,PY))
    H=H+U*(NKron(PZ,PZ,id(2**(N-2))) + NKron(id(2**(N-2)),PZ,PZ))
    i=1
    while i<N-1:
        H=H+J*(NKron(id(2**i),PX,PX,id(2**(N-2-i))))
        H=H+J*(NKron(id(2**i),PY,PY,id(2**(N-2-i))))
        H=H+U*(NKron(id(2**i),PZ,PZ,id(2**(N-2-i))))
        i+=1    
    i=0
    while i<N:
        H=H+random.uniform(-h,h)*(NKron(id(2**i),PZ,id(2**(N-i-1))))
        H=H+Xi*(NKron(id(2**i),PX,id(2**(N-i-1))))
        i+=1
        
    EigVals, EigVecs = nla.eigh(H)
    Evolution = np.zeros((2**N,2**N),dtype=np.complex128)
    P = 0    
    while P < len(EigVals):
        Evolution = Evolution + cmath.exp(-t*complex(0,1)*EigVals[P])*(np.outer(np.transpose(EigVecs[:,P]),np.conjugate(EigVecs[:,P])))
        P += 1
    return EigVals, Evolution

def StateVector(N):
    i = 0  
    Psi=1
    while i<N:
        r=random.uniform(-np.pi,np.pi)
        Psi=NKron(Psi,(np.cos(r)*up+np.sin(r)*down))
        i += 1
    Psi=np.transpose(Psi)    
    
    return Psi


def Entangle(Psi,N):
    WaveFunction=Psi.reshape(int(2**(N/2)),int(2**(N/2)))
    Z,S,V = nla.svd(WaveFunction, full_matrices=True)
    i = 0
    Entropy = 0
    while i<len(S):
        if S[i]==0:
            Entropy=Entropy+0
        else:
            Entropy=Entropy-(pow(S[i],2)*math.log(pow(S[i],2), 2))
        i+=1
    return float(Entropy)

#Spin states in the z-basis
up = np.array([1, 0])
down = np.array([0, 1])
Spin=[up,down]
#Spin matrices
PX = np.array([[0, 1],[ 1, 0]])
PZ = np.array([[1, 0],[0, -1]])
PY = np.array([[0, -1.0*cm.sqrt(-1.0)], [cm.sqrt(-1.0), 0]])

Values=[]
Repeats=20
J=1
N=int(10)
h=10
U=0.5
Xi=0.1
Time=10
t=0.01
Basis=[]
SpinTot=[]
Ratio=[]
    
StatEntropy=[]

R=0
while R<Repeats:
    T=0
    EigVals, Evolution=Ham(N)
    Psi=StateVector(N)
    EntropyList=[]
    while T<Time:
        EntropyList.append(Entangle(Psi,N))
        Psi=np.dot(Evolution,Psi)
        
        T+=t
    StatEntropy.append(EntropyList)
    print(R)
    R+=1

Entropy=[]
i=0
while i<int(Time/t):
    j=0
    SumEnt=0
    while j<R:
         SumEnt=SumEnt+(1/R)*float(StatEntropy[j][i])
         j+=1
    Entropy.append(SumEnt)
    i+=1
             
plt.plot(Entropy)
plt.xlabel('Time')
plt.ylabel('Entanglement entropy between equally bipartitioned subsystems')
plt.show()
    