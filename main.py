import numpy as np
import argparse

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cost_functions
from optim_algorithms import *

ZOOM = 1e-2

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

        if np.linalg.norm(grads) < tol or steps >= max_steps:
            break

    weights_history = np.array(weights_history)

    return weights_history, steps, grads

def plot(weights_history, cost_function, optim):
    min_x, min_y = np.min(weights_history, axis=0) - ZOOM
    max_x, max_y = np.max(weights_history, axis=0) + ZOOM

    x = np.linspace(min(min_x,-ZOOM), max(max_x,ZOOM), 100)
    y = np.linspace(min(min_y,-ZOOM), max(max_y,ZOOM), 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[cost_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Jet')])
    fig.update_layout(title=f"{cost_function.__name__} + {optim.__class__.__name__}", autosize=True,
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Cost'))
    
    fig.add_trace(go.Scatter3d(x=weights_history[:, 0], y=weights_history[:, 1],
                            z=[cost_function(w) for w in weights_history],
                            mode='markers+lines', marker=dict(size=5, color='magenta')))
    fig.show()


def benchmark(optimizers, cost_functions, cost_functions_list, num_initializations, max_steps, tol, epsilon):
    results = {}
    weights = np.random.uniform(-1.5, 1.5, (num_initializations, 2))
    for optimizer_name in optimizers:
        results[optimizer_name] = {}
        optimizer_class = eval(optimizer_name)
        for cost_function_name in cost_functions_list:
            cost_function = getattr(cost_functions, cost_function_name)
            avg_cost = 0
            for run in range(num_initializations):
                optim = optimizer_class()
                weights_history, steps, grads = optimize(np.array(weights[run]), optim, cost_function, max_steps, tol, epsilon)
                avg_cost += cost_function(weights_history[-1])
            avg_cost /= num_initializations
            results[optimizer_name][cost_function_name] = avg_cost
            results[optimizer_name][f"{cost_function_name}_weights"] = weights_history

    cost_functions_to_plot = ['HolderTable', 'Matyas', 'Rastrigin', 'Rosenbrock', 'SchaffersF6']
    fig = make_subplots(rows=len(cost_functions_to_plot), cols=1, shared_xaxes=True, vertical_spacing=0.01,
                        subplot_titles=[f"{cf}" for cf in cost_functions_to_plot],
                        specs=[[{'type': 'scene'}] for _ in range(len(cost_functions_to_plot))])
    
    fig.update_xaxes(rangeslider=dict(visible=False))

    trace_colors ={ 'SGD': 'magenta', 'SGD_momentum': 'green', 'Adam': 'red', 'RMSprop': 'orange'}

    for i, cost_function_name in enumerate(cost_functions_to_plot):
        cost_function = getattr(cost_functions, cost_function_name)
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf

        for optimizer_name in optimizers:
            if f"{cost_function_name}_weights" in results[optimizer_name]:
                weights_history = results[optimizer_name][f"{cost_function_name}_weights"]
                min_x = min(min_x, np.min(weights_history[:, 0]) - ZOOM)
                min_y = min(min_y, np.min(weights_history[:, 1]) - ZOOM)
                max_x = max(max_x, np.max(weights_history[:, 0]) + ZOOM)
                max_y = max(max_y, np.max(weights_history[:, 1]) + ZOOM)

        x = np.linspace(min(min_x, -ZOOM), max(max_x, ZOOM), 100)
        y = np.linspace(min(min_y, -ZOOM), max(max_y, ZOOM), 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[cost_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Jet', showscale=False), row=i+1, col=1)

        for optimizer_name in optimizers:
            if f"{cost_function_name}_weights" in results[optimizer_name]:
                weights_history = results[optimizer_name][f"{cost_function_name}_weights"]
                fig.add_trace(go.Scatter3d(x=weights_history[:, 0], y=weights_history[:, 1],
                                           z=[cost_function(w) for w in weights_history],
                                           mode='markers+lines', name=optimizer_name, marker=dict(size=5, color=trace_colors[optimizer_name]), showlegend=(i==0)), 
                                           row=i+1, col=1)

    fig.update_layout(title="Cost Surfaces and Optimization Paths", autosize=True,
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Cost'), height=800*len(cost_functions_to_plot))
    
    fig.update_xaxes(rangeslider=dict(visible=False))

    fig.show()
    return results

if __name__=='__main__':
    optimizers_list = ['SGD', 'SGD_momentum', 'Adam', 'RMSprop']
    cost_functions_list = [func for func in dir(cost_functions) if callable(getattr(cost_functions, func))]

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=float, nargs='+', default=[3,2], help='Initial weights')
    parser.add_argument('--optim', type=str, default='SGD', help='Optimization algorithm', choices=optimizers_list)
    parser.add_argument('--cost', type=str, default='GoldsteinPrice', help='Cost function', choices=cost_functions_list)
    parser.add_argument('--max_steps', type=int, default=1, help='Maximum number of steps')
    parser.add_argument('--tol', type=float, default=1e-2, help='Tolerance')
    parser.add_argument('--epsilon', type=float, default=1e-2, help='Epsilon for gradient computation')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--num_initializations', type=int, default=1, help='Number of random initializations for benchmark')

    args = parser.parse_args()

    if args.benchmark:
        results = benchmark(optimizers_list, cost_functions, cost_functions_list, args.num_initializations, args.max_steps, args.tol, args.epsilon)
        for cost_function_name in cost_functions_list:
            print(f'Cost function: {cost_function_name}')
            for optimizer_name in optimizers_list:
                if cost_function_name in results[optimizer_name]:
                    avg_cost = results[optimizer_name][cost_function_name]
                    print(f'\tOptimization algorithm: {optimizer_name}, Average cost: {avg_cost}')
    else:
        optim = eval(args.optim)()
        cost_function = getattr(cost_functions, args.cost)

        weights = np.array(args.weights, dtype=np.float64)

        weights_history, steps, grads = optimize(weights, optim, cost_function, args.max_steps, args.tol, args.epsilon)
        plot(weights_history, cost_function, optim)

        print(f'Optimization algorithm: {args.optim}')
        print(f'Cost function: {args.cost}')
        print(f'Number of steps: {steps}')
        print(f'Final weights: {weights_history[-1]}')
        print(f'Final cost: {cost_function(weights_history[-1])}')
        print(f'Final gradient: {grads}')
