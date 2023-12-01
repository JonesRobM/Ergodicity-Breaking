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
    return EigVals, EigVecs

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
Repeats=100
J=1
U=0.5
Xi=0.5
SpinTot=[]
Ratio=[]
Mean=[] 
EntropyList=[] 
MeanEntropyList=[]

EntropyMatrix1=[]
EntropyMatrix2=[]
EntropyMatrix3=[]
EntropyMatrix4=[]
EntropyMatrix5=[]

EntropyVector1=[]
EntropyVector2=[]
EntropyVector3=[]
EntropyVector4=[]
EntropyVector5=[]

ErrorList1=[]
ErrorList2=[]
ErrorList3=[]
ErrorList4=[]
ErrorList5=[]


N=int(4)
while N < 12:
    R=0

    while R<Repeats:
        h=0
        MeanEntropyList=[]
        while h<10:
        
            EigVals, EigVecs = Ham(N)
            EigVecs=np.transpose(EigVecs)

            j=2**int(N/4)
            EntropyList=[]
            while j<2**int(3*N/4):

                WaveFunction=EigVecs[j].reshape(int(2**(N/2)),int(2**(N/2)))   
               
                Z,S,V = nla.svd(WaveFunction, full_matrices=True)
                i = 0
                Entropy = 0
                while i<len(S):
                    if S[i]==0:
                        Entropy=Entropy+0
                    else:
                        Entropy=Entropy-(pow(S[i],2)*math.log(pow(S[i],2),2))
                    i+=1
                j+=1
                EntropyList.append(Entropy)
            MeanEntropyList.append(np.average(EntropyList))
            
    

            h+=0.25   
    
        R+=1
        print(R)
        if N==4:
            EntropyMatrix1.append(np.transpose(MeanEntropyList))
        elif N==6:
            EntropyMatrix2.append(np.transpose(MeanEntropyList))
        
        elif N==8:
            EntropyMatrix3.append(np.transpose(MeanEntropyList))
             
        elif N==10:
            EntropyMatrix4.append(np.transpose(MeanEntropyList))
        """
        elif N==12:
            EntropyMatrix5.append(np.transpose(MeanEntropyList))
        """
                       
    N+=2
    print(N)       



EntropyMatrix1=np.transpose(EntropyMatrix1)

EntropyMatrix2=np.transpose(EntropyMatrix2)

EntropyMatrix3=np.transpose(EntropyMatrix3)

EntropyMatrix4=np.transpose(EntropyMatrix4)

"""
RatioMatrix5=np.transpose(EntropyMatrix5)
"""

i=0
while i<40:
    EntropyVector1.append(np.average(EntropyMatrix1[i]))
    
    EntropyVector2.append(np.average(EntropyMatrix2[i]))
    
    EntropyVector3.append(np.average(EntropyMatrix3[i]))
    
    EntropyVector4.append(np.average(EntropyMatrix4[i]))
    
    """
    EntropyVector5.append(np.average(EntropyMatrix5[i]))
    """
    
    ErrorList1.append((np.std(EntropyMatrix1[i]))/np.sqrt(len(EntropyVector1)))
    
    ErrorList2.append((np.std(EntropyMatrix2[i]))/np.sqrt(len(EntropyVector1)))
    
    ErrorList3.append((np.std(EntropyMatrix3[i]))/np.sqrt(len(EntropyVector1)))
        
    ErrorList4.append((np.std(EntropyMatrix4[i]))/np.sqrt(len(EntropyVector1)))
    
    """
    ErrorList5.append((np.std(EntropyMatrix5[i]))/np.sqrt(len(EntropyVector1)))
    """
    
    i+=1
    
    
x=np.linspace(0,10,num=40)

plt.plot(x,EntropyVector1, label="N=4")
plt.errorbar(x,EntropyVector1, yerr=ErrorList1, fmt='o')

plt.plot(x,EntropyVector2, label="N=6")
plt.errorbar(x,EntropyVector2, yerr=ErrorList2, fmt='v')

plt.plot(x,EntropyVector3, label="N=8")
plt.errorbar(x,EntropyVector3, yerr=ErrorList3, fmt='s')


plt.plot(x,EntropyVector4, label="N=10")
plt.errorbar(x,EntropyVector4, yerr=ErrorList4, fmt='x')

"""
plt.plot(x,EntropyVector5, label="N=12")
plt.errorbar(x,EntropyVector5, yerr=ErrorList5, fmt='D')
"""

plt.xlabel(' h/J  ')
plt.ylabel(r' $ \langle S_{VN} \rangle $')
plt.legend(loc='upper right', shadow = True)
plt.show()