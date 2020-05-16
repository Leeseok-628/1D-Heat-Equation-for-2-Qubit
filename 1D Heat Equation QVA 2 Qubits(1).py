#!/usr/bin/env python
# coding: utf-8

# In[21]:


import cirq
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# In[22]:


class U(cirq.Gate):
    def __init__(self,l):
        #l := lambda variational parameter
        #we specify the variational parameter when we declare a U gate object
        self.l = l
    
    def _decompose_(self, qubits):
        #U gate object takes in two qubits as input
        q0, q1 = qubits
        
        #U^(1)
        #Y on q0
        yield cirq.ry(self.l[0]).on(q0)
        #X on q0
        yield cirq.rx(self.l[1]).on(q0)
        
        
        #U^(2)
        #Y on q1
        yield cirq.ry(self.l[2]).on(q1)
    
        #XY on q0,q1
        yield cirq.H(q0)
        yield cirq.rx(np.pi/2).on(q1)
        yield cirq.CNOT(q0,q1)
        yield cirq.rz(self.l[3]).on(q1)
        yield cirq.CNOT(q0,q1)
        yield cirq.rx(-np.pi/2).on(q1)
        yield cirq.H(q0)
        
        #X on q1
        yield cirq.rx(self.l[4]).on(q1)
        
        #XX on q0,q1
        yield cirq.H(q0)
        yield cirq.H(q1)
        yield cirq.CNOT(q0,q1)
        yield cirq.rz(self.l[5]).on(q1)
        yield cirq.CNOT(q0,q1)
        yield cirq.H(q1)
        yield cirq.H(q0)
        
    def _num_qubits_(self):
        #We are required to implement this method, 
        #it specifies the number of qubits our gate acts on
        return 2    
    
    def _unitary_(self):
        #Some other method we need in order to get inverses of U to work
        #as we will need them in our circuits
        return cirq.unitary(cirq.Circuit(self._decompose_(cirq.LineQubit.range(2))))


# In[23]:


#Returns Circuit for computing Re<ψ|ψ'>
def QNPU1A(l_old,l_new,n):
    ctrl,q0,q1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)]
    
    U1=cirq.ControlledGate(U(l_old))
    U2=cirq.ControlledGate(cirq.inverse(U(l_new)))
    
    circuit = cirq.Circuit(cirq.H(ctrl), U1.on(ctrl,q0,q1),
                           U2.on(ctrl,q0,q1),
                           cirq.H(ctrl),
                           cirq.measure(ctrl,key='c'),
                           strategy=cirq.InsertStrategy.EARLIEST)
    
    
    simulator = cirq.Simulator()
    result=simulator.run(circuit, repetitions=n)
    m=result.histogram(key='c')
    
    for i,j in m.items():
        if i==0:
            term=2*(j/n)-1
        elif i==1:
            term=1-2*(j/n)
    
    return term


# In[24]:


QNPU1A([0,0,0,0,0,0],[1,1,2,1,1,1],100000)


# In[25]:


#Returns Circuit for computing Im<ψ|ψ'>
def QNPU1B(l_old,l_new,n):
    ctrl,q0,q1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)]
    
    U1=cirq.ControlledGate(U(l_old))
    U2=cirq.ControlledGate(cirq.inverse(U(l_new)))
    
    circuit = cirq.Circuit(cirq.H(ctrl), cirq.S(ctrl), U1.on(ctrl,q0,q1),
                           U2.on(ctrl,q0,q1),
                           cirq.H(ctrl),
                           cirq.measure(ctrl,key='c'),
                           strategy=cirq.InsertStrategy.EARLIEST)
    
    
    simulator = cirq.Simulator()
    result=simulator.run(circuit, repetitions=n)
    m=result.histogram(key='c')
    
    for i,j in m.items():
        if i==0:
            term=2*(j/n)-1
        elif i==1:
            term=1-2*(j/n)
    
    return term


# In[26]:


QNPU1B([0,0,0,0,0,0],[1,1,2,1,1,1],100000)


# In[27]:


#Returns Circuit for computing Re<ψ|A|ψ'>
def QNPU2A(l_old,l_new,n):
    ctrl,q0,q1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)]
    
    U1=cirq.ControlledGate(U(l_old))
    U2=cirq.ControlledGate(cirq.inverse(U(l_new)))
    
    circuit = cirq.Circuit(cirq.H(ctrl), U1.on(ctrl,q0,q1),
                           cirq.CNOT(ctrl,q1),
                           cirq.TOFFOLI(ctrl,q1,q0),
                           U2.on(ctrl,q0,q1),
                           cirq.H(ctrl),
                           cirq.measure(ctrl,key='c'),
                           strategy=cirq.InsertStrategy.EARLIEST)
    
    
    simulator = cirq.Simulator()
    result=simulator.run(circuit, repetitions=n)
    m=result.histogram(key='c')
    
    for i,j in m.items():
        if i==0:
            term=2*(j/n)-1
        elif i==1:
            term=1-2*(j/n)
    
    return term


# In[28]:


QNPU2A([0,0,0,0,0,0],[1,1,2,1,1,1],100000)


# In[29]:


#Returns Circuit for computing Im<ψ|A|ψ'>
def QNPU2B(l_old,l_new,n):
    ctrl,q0,q1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)]
    
    U1=cirq.ControlledGate(U(l_old))
    U2=cirq.ControlledGate(cirq.inverse(U(l_new)))
    
    circuit = cirq.Circuit(cirq.H(ctrl), cirq.S(ctrl),U1.on(ctrl,q0,q1),
                           cirq.CNOT(ctrl,q1),
                           cirq.TOFFOLI(ctrl,q1,q0),
                           U2.on(ctrl,q0,q1),
                           cirq.H(ctrl),
                           cirq.measure(ctrl,key='c'),
                           strategy=cirq.InsertStrategy.EARLIEST)
    
    
    simulator = cirq.Simulator()
    result=simulator.run(circuit, repetitions=n)
    m=result.histogram(key='c')
    
    for i,j in m.items():
        if i==0:
            term=2*(j/n)-1
        elif i==1:
            term=1-2*(j/n)
    
    return term


# In[30]:


QNPU2B([0,0,0,0,0,0],[1,1,2,1,1,1],100000)


# In[32]:


q0,q1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
simulator = cirq.Simulator()
circuit0 = cirq.Circuit(U([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]).on(q0,q1))
simulation0 = simulator.simulate(circuit0)
b0=np.around(simulation0.final_state, 5)


# In[33]:


b0


# In[34]:


q0,q1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
simulator = cirq.Simulator()
circuit0 = cirq.Circuit(U([0,0,0,0,0,0]).on(q0,q1))
simulation0 = simulator.simulate(circuit0)
b1=np.around(simulation0.final_state, 5)


# In[35]:


b1


# In[36]:


Re_f=list(np.absolute(b1)**2)
Re_f.append(Re_f[0])

x=list((1/4)*np.arange(5))
p = CubicSpline(x,Re_f)
xnew = np.arange(0, 1, 0.001)
ynew = p(xnew)
plt.plot(xnew, ynew)
plt.show()


# In[37]:


def costfn(x,l0=[1,0,np.pi/2,0,0,0,0,0],tau=0.025,v=1,n=1000):
    
    term1=QNPU1A(l0[2:],x[2:],n)
    term2=QNPU1B(l0[2:],x[2:],n)
    term3=QNPU2A(l0[2:],x[2:],n)
    term4=QNPU2B(l0[2:],x[2:],n)
    term5=QNPU2A(x[2:],l0[2:],n)
    term6=-1*(QNPU2B(x[2:],l0[2:],n))
    
    expectation=(1-32*tau*v)*(term1+term2*1j)+16*tau*v*((term3+term5)+(term4+term6)*1j)
    
    return x[0]**2+x[1]**2-2*np.real((x[0]-x[1]*1j)*(l0[0]+l0[1]*1j)*expectation)


# In[38]:


x0=[np.random.random(),np.random.random(),2*np.pi*np.random.random(),2*np.pi*np.random.random(),
    2*np.pi*np.random.random(),2*np.pi*np.random.random(),
   2*np.pi*np.random.random(),2*np.pi*np.random.random()]
res=optimize.minimize(costfn, x0, method='Powell')
lnew=res.x


# In[39]:


q0,q1 = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
simulator = cirq.Simulator()
circuit1 = cirq.Circuit(U(lnew[2:]).on(q0,q1))
simulation1 = simulator.simulate(circuit1)
b2=np.around(simulation1.final_state, 5)


# In[40]:


lnew


# In[42]:


Re_f1=list(np.absolute((lnew[0]+lnew[1]*1j)*b2)**2)
Re_f1.append(Re_f1[0])

x=list((1/4)*np.arange(5))
p = CubicSpline(x,Re_f1)
xnew = np.arange(0, 1, 0.001)
ynew = p(xnew)
plt.plot(xnew, ynew)
plt.show()


# In[ ]:




