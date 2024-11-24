import numpy as np

np.seterr(over='raise')

def Ackley(xy):
    return - 20 * np.exp(- 0.2 * np.sqrt(0.5 * (xy[0] ** 2 + xy[1] ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * xy[0]) + np.cos(2 * np.pi * xy[1]))) + np.e + 20

def Alpine(xy):
    return np.sum(np.abs(xy[0] * np.sin(xy[0]) + 0.1 * xy[0]) + np.abs(xy[1] * np.sin(xy[1]) + 0.1 * xy[1]))

def Beale(xy):
    try:
        return (1.5 - xy[0] + xy[0] * xy[1])**2 + (2.25 - xy[0] + xy[0] * xy[1]**2)**2 + (2.625 - xy[0] + xy[0] * xy[1]**3)**2
    except FloatingPointError:
        return np.inf
    
def Booth(xy):
    return (xy[0] + 2 * xy[1]- 7)**2 + (2 * xy[0] + xy[1] - 5)**2

def CrossInTray(xy):
    return -0.0001 * (np.abs(np.sin(xy[0]) * np.sin(xy[1]) * np.exp(np.abs(100 - np.sqrt(xy[0]**2 + xy[1]**2) / np.pi))) + 1)**0.1

def DixonPrice(xy):
    return (xy[0] - 1)**2 + (2 * xy[1]**2 - xy[0])**2

def Dropwave(xy):
    return - (1 + np.cos(12 * np.sqrt(xy[0]**2 + xy[1]**2))) / (0.5 * (xy[0]**2 + xy[1]**2) + 2)

def Easom(xy):
    return -np.cos(xy[0]) * np.cos(xy[1]) * np.exp(-((xy[0] - np.pi)**2 + (xy[1] - np.pi)**2))

def Expanded(xy):
    return -np.exp(-2 * np.log(2) * ((xy[0] - 0.1)**2 / 0.8 + (xy[1] - 0.2)**2 / 0.3)) * np.power(np.abs(np.sin(5 * np.pi * xy[0]) * np.sin(5 * np.pi * xy[1])), 3)

def SchaffersF6(xy):
    return 0.5 + (np.sin(np.sqrt(xy[0]**2 + xy[1]**2))**2 - 0.5) / (1 + 0.001 * (xy[0]**2 + xy[1]**2))**2

def GoldsteinPrice(xy):
    try:
        return (1 + (xy[0] + xy[1] + 1)**2 * (19 - 14 * xy[0] + 3 * xy[0]**2 - 14 * xy[1] + 6 * xy[0] * xy[1] + 3 * xy[1]**2)) * (30 + (2 * xy[0] - 3 * xy[1])**2 * (18 - 32 * xy[0] + 12 * xy[0]**2 + 48 * xy[1] - 36 * xy[0] * xy[1] + 27 * xy[1]**2))
    except FloatingPointError:
        return np.inf
    
def Griewank(xy):
    return 1 + (xy[0]**2 + xy[1]**2) / 4000 - np.cos(xy[0]) * np.cos(xy[1] / np.sqrt(2))

def Himmelblau(xy):
    return (xy[0]**2 + xy[1] - 11)**2 + (xy[0] + xy[1]**2 - 7)**2

def HolderTable(xy):
    return -np.abs(np.sin(xy[0]) * np.cos(xy[1]) * np.exp(np.abs(1 - np.sqrt(xy[0]**2 + xy[1]**2) / np.pi)))

def Levi(xy):
    return np.sin(3 * np.pi * xy[0])**2 + (xy[0] - 1)**2 * (1 + np.sin(3 * np.pi * xy[1])**2) + (xy[1] - 1)**2 * (1 + np.sin(2 * np.pi * xy[1])**2)

def Matyas(xy):
    return 0.26 * (xy[0]**2 + xy[1]**2) - 0.48 * xy[0] * xy[1]

def Michalewicz(xy):
    return -np.sin(xy[0]) * np.sin(xy[0]**2 / np.pi)**20 - np.sin(xy[1]) * np.sin(2 * xy[1]**2 / np.pi)**20

def Rastrigin(xy):
    return 20 + (xy[0]**2 - 10 * np.cos(2 * np.pi * xy[0])) + (xy[1]**2 - 10 * np.cos(2 * np.pi * xy[1]))

def Rosenbrock(xy):
    return 100 * (xy[1] - xy[0]**2)**2 + (1 - xy[0])**2

def Salomon(xy):
    return 1 - np.cos(2 * np.pi * np.sqrt(xy[0]**2 + xy[1]**2)) + 0.1 * np.sqrt(xy[0]**2 + xy[1]**2)

def SchaffersF2(xy):
    return 0.5 + (np.sin(np.sqrt(xy[0]**2 + xy[1]**2))**2 - 0.5) / (1 + 0.001 * (xy[0]**2 + xy[1]**2))**2

def Schwefeles(xy):
    return 418.9829 * 2 - (xy[0] * np.sin(np.sqrt(np.abs(xy[0]))) + xy[1] * np.sin(np.sqrt(np.abs(xy[1]))))

def Sphere(xy):
    return xy[0]**2 + xy[1]**2

def StyblinskiTang(xy):
    return 0.5 * (xy[0]**4 - 16 * xy[0]**2 + 5 * xy[0] + xy[1]**4 - 16 * xy[1]**2 + 5 * xy[1])

def ThreeHumpCamel(xy):
    return 2 * xy[0]**2 - 1.05 * xy[0]**4 + xy[0]**6 / 6 + xy[0] * xy[1] + xy[1]**2
    