import numpy as np
import plotly.graph_objects as go

from optim_algorithms import *
import cost_functions

import argparse


def gradient(weights, cost_function, epsilon):
    grad = np.zeros_like(weights)
    for i in range(len(weights)):
        weights_eps = np.copy(weights)
        weights_eps[i] += epsilon
        cost1 = cost_function(weights_eps)
        weights_eps[i] -= 2 * epsilon
        cost2 = cost_function(weights_eps)
        grad[i] = (cost1 - cost2) / (2 * epsilon)
    return grad


def optimize(weights, optim, cost_function, max_steps, tol, epsilon):
    weights_history = [weights.copy()]
    steps = 0

    while True:
        grads = gradient(weights, cost_function, epsilon)
        weights = optim.update(weights, grads)
        weights_history.append(weights.copy())
        steps += 1

        if np.linalg.norm(grads) < tol or steps > max_steps:
            break

    weights_history = np.array(weights_history)

    return weights_history, steps, grads

def plot(weights_history, cost_function, optim):

    min_x, min_y = np.min(weights_history, axis=0) - 0.5
    max_x, max_y = np.max(weights_history, axis=0) + 0.5

    x = np.linspace(min(min_x,-2), max(max_x,2), 100)
    y = np.linspace(min(min_y,-2), max(max_y,2), 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[cost_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Jet')])
    fig.update_layout(title=f"{cost_function.__name__} + {optim.__class__.__name__}", autosize=True,
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Cost'))
    
    fig.add_trace(go.Scatter3d(x=weights_history[:, 0], y=weights_history[:, 1],
                            z=[cost_function(w) for w in weights_history],
                            mode='markers+lines', marker=dict(size=5, color='magenta')))
    fig.show()


if __name__=='__main__':

    optimizers_list = ['SGD', 'SGD_momentum', 'Adam', 'RMSprop']
    cost_functions_list = dir(cost_functions)[:-9]

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=float, nargs='+', default=[1,0.1], help='Initial weights')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimization algorithm', choices=optimizers_list)
    parser.add_argument('--cost', type=str, default='Matyas', help='Cost function', choices=cost_functions_list)
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum number of steps')
    parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Epsilon for gradient computation')

    args = parser.parse_args()

    optim = eval(args.optim)()
    cost_function = getattr(cost_functions, args.cost)

    weights = np.array(args.weights)

    weights_history, steps, grads = optimize(weights, optim, cost_function, args.max_steps, args.tol, args.epsilon)
    plot(weights_history, cost_function, optim)

    print(f'Optimization algorithm: {args.optim}')
    print(f'Cost function: {args.cost}')
    print(f'Number of steps: {steps}')
    print(f'Final weights: {weights_history[-1]}')
    print(f'Final cost: {cost_function(weights_history[-1])}')
    print(f'Final gradient: {grads}')
    

