import numpy as np
import numdifftools as nd

def f(xu):
    x,u = xu
    return 2 * x[:,[0]] + u + x[:,[1]]


grad = nd.Gradient(f)

xs = np.array([[1,2],[3,4]],dtype=np.float32)
us = np.array([[1],[-1]],dtype=np.float32)

print(f([xs,us]))

erg = []
for i in range(len(us)):
    x = xs[i]
    u = us[i]
    erg.append(grad([x,u]))

print(erg)