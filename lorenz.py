import argparse
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import solve_ivp
from scipy.stats import linregress


def lorenz(t, y):
    """Evaluate the Lorenz vector field with σ=10, ρ=28, and β=8/3."""
    f = np.zeros(y.shape)
    f[0] = 10 * (y[1] - y[0])
    f[1] = y[0] * (28 - y[2]) - y[1]
    f[2] = y[0] * y[1] - 8 * y[2] / 3
    return f


def solveEnsemble(y0_ens, t0, tf):
    """Integrate each initial state on a shared time grid."""
    t_eval = np.linspace(t0, tf, 2000)
    sol_list = []

    for n in range(y0_ens.shape[0]):
        sol = solve_ivp(
            lorenz,
            (t0,
             tf),
            y0_ens[n],
            t_eval=t_eval,
            rtol=1.e-8,
            atol=1.e-10)
        if not sol.success:
            raise RuntimeError(sol.message)
        sol_list.append(sol.y)
    return t_eval, sol_list


def ensembleStats(sol_list):
    """Return componentwise means and sample standard deviations."""
    nEnsemble = len(sol_list)
    if nEnsemble < 2:
        raise ValueError("At least two ensemble members are required")
    y_mean = np.zeros_like(sol_list[0])
    y_std = np.zeros_like(sol_list[0])

    for item in range(len(y_mean)):
        for i in range(nEnsemble):
            y_mean[item] += sol_list[i][item]

    y_mean /= nEnsemble

    for item in range(len(y_std)):
        for i in range(nEnsemble):
            y_std[item] += (sol_list[i][item] - y_mean[item])**2
    y_std = np.sqrt(y_std / (nEnsemble - 1))

    return y_mean, y_std


def ensembleDistance(sol_list):
    """Return Euclidean separation histories for each unordered pair."""
    nEnsemble = len(sol_list)
    dist = []

    for i in range(nEnsemble):
        for j in range(i):
            diff = sol_list[i] - sol_list[j]
            dist.append(np.linalg.norm(diff, axis=0))
    return dist


def expfit(t, y):
    """Fit exponential separation growth over the supplied time window."""
    lr = linregress(t, np.log(y))
    print('Approximate maximal Lyapunov exponent: %f' % lr.slope)
    fit = np.exp(lr.intercept + lr.slope * t)
    return fit


def plotEnsemble(t_eval, sol_list):

    y_mean, y_std = ensembleStats(sol_list)

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    for y in sol_list:
        for nd in range(3):
            axs[nd, 0].plot(t_eval, y[nd, :])

    for nd in range(3):
        axs[nd, 1].plot(t_eval, y_mean[nd, :], '-b')
        ax_twin = axs[nd, 1].twinx()
        ax_twin.semilogy(t_eval, y_std[nd, :], '--r')

    axs[0, 0].set_title('Full Ensemble', fontsize=16)
    axs[0, 0].set_ylabel('x', fontsize=14)
    axs[0, 0].grid('both')
    axs[1, 0].set_ylabel('y', fontsize=14)
    axs[1, 0].grid('both')
    axs[2, 0].set_ylabel('z', fontsize=14)
    axs[2, 0].grid('both')
    axs[2, 0].set_xlabel('t', fontsize=14)

    axs[0, 1].set_title('Ensemble Mean and Std. Deviation', fontsize=16)
    axs[0, 1].set_ylabel('x', fontsize=14)
    axs[0, 1].grid('both')
    axs[1, 1].set_ylabel('y', fontsize=14)
    axs[1, 1].grid('both')
    axs[2, 1].set_ylabel('z', fontsize=14)
    axs[2, 1].grid('both')
    axs[2, 1].set_xlabel('t', fontsize=14)

    # Proxy artists combine the legends from the twin axes.
    bline = mlines.Line2D([], [], color='blue', label='Mean Solution')
    rline = mlines.Line2D(
        [],
        [],
        color='red',
        linestyle='dashed',
        label='Std. Deviation')
    axs[2, 1].legend(handles=[bline, rline], loc='lower right')

    fig.tight_layout()
    plt.savefig('solutions.png')


def plotDistances(t_eval, dist):

    y_mean, y_std = ensembleStats(dist)

    # Limit the fit to the initial 75% of the separation history.
    end = int(0.75 * len(t_eval))
    t_sub, y_sub = t_eval[:end], y_mean[:end]
    fit = expfit(t_sub, y_sub)

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    for y in dist:
        axs[0].semilogy(t_eval, y)

    axs[1].semilogy(t_eval, y_mean[:], '-b', t_sub, fit, '--k')

    axs[0].set_title('Ensemble Distances', fontsize=16)
    axs[0].set_xlabel('t', fontsize=14)
    axs[0].set_ylabel('Distance', fontsize=14)
    axs[0].grid('both')

    axs[1].set_title('Mean Distance and Fit', fontsize=16)
    axs[1].set_xlabel('t', fontsize=14)
    axs[1].set_ylabel('Mean', fontsize=14)
    axs[1].grid('both')
    axs[1].legend(['Mean Distance', 'Exponential Fit'],
                  fontsize=14, loc='lower right')

    fig.tight_layout()
    plt.savefig('distances.png')


def plotTrajectory3D(sol_list: list, traj_idx_start: int, traj_idx_end: int):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for idx, sol in enumerate(sol_list[traj_idx_start:traj_idx_end + 1]):
        ax.plot(*sol, linewidth=0.3, label=f'sol {idx}')

    ax.legend(loc='upper right', title='Trajectories', fontsize=10)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_zlabel('z', fontsize=14)
    ax.set_title("Lorenz Trajectories")

    plt.savefig('trajectories.png')


def main():

    parser = argparse.ArgumentParser(description="Lorenz ensemble simulation")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--members', type=int, default=10)
    parser.add_argument('--duration', type=float, default=35.0)
    args = parser.parse_args()
    if args.members < 3 or not np.isfinite(
            args.duration) or args.duration <= 0:
        parser.error('Use at least 3 members and a positive finite duration')
    rng = default_rng(args.seed)
    nEnsemble = args.members

    y0_mean = np.array([2, 3, -14])
    y0_var = (1.e-9)**2
    y0_cov = np.diag(y0_var * np.ones(3))
    y0_ens = rng.multivariate_normal(y0_mean, y0_cov, nEnsemble)

    t_eval, sol_list = solveEnsemble(y0_ens, 0.0, args.duration)

    plotEnsemble(t_eval, sol_list)

    dist = ensembleDistance(sol_list)
    plotDistances(t_eval, dist)

    plotTrajectory3D(sol_list, 0, 1)


if __name__ == "__main__":
    main()
