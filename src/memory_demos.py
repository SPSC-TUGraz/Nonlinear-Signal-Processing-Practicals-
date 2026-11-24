import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import maximum_filter1d

# Each system: state s, time t, input u(t) (added as a force / to the first equation)
def damped_pendulum(t, s, u, a=0.5):  return [s[1], -a*s[1] - np.sin(s[0]) + u(t)]
def frictionless_pendulum(t, s, u):   return [s[1], -np.sin(s[0]) + u(t)]
def double_well(t, s, u, a=0.5):      return [s[1], -a*s[1] + s[0] - s[0]**3 + u(t)]
def self_oscillator(t, s, u, m=1.0):  return [s[1], m*(1 - s[0]**2)*s[1] - s[0] + u(t)]   # van der Pol
def lorenz(t, s, u, sg=10., r=28., b=8/3):
    x, y, z = s
    return [sg*(y - x) + u(t), x*(r - z) - y, x*y - b*z]

SYSTEMS = dict(damped_pendulum=damped_pendulum, frictionless_pendulum=frictionless_pendulum,
               double_well=double_well, self_oscillator=self_oscillator, lorenz=lorenz)

def simulate(f, s0, T, u=lambda t: 0., dt=0.01):
    """Trajectory of system f from s0 driven by u(t); returns t and the states (dim x N)."""
    t = np.arange(0, T, dt)
    return t, solve_ivp(f, (0, T), s0, t_eval=t, args=(u,), rtol=1e-9, atol=1e-9, max_step=0.05).y

def smooth_input(T, amp=0.05, dt_knot=2., seed=0):
    """Random but smooth input signal (cubic-free: linear interpolation of random knots, then smoothed)."""
    rng = np.random.default_rng(seed); tk = np.arange(0, T + dt_knot, dt_knot); vk = amp*rng.standard_normal(len(tk))
    return lambda t: np.interp(t, tk, vk)

def pulse(t0, width=0.5, amp=1.):
    """Short rectangular input pulse starting at t0."""
    return lambda t: amp*(t0 <= t < t0 + width)

def distance(a, b, angle=False):
    """Euclidean distance between two trajectories; angle=True wraps the first coordinate (pendulum angle) to (-pi, pi]."""
    d = a - b
    if angle: d[0] = np.angle(np.exp(1j*d[0]))
    return np.linalg.norm(d, axis=0)

def envelope(d, dt=0.01, window=20.):
    """Running maximum of the distance over the past `window` time units (padded with the first value): 'how different are the copies still?'"""
    n = int(window/dt)
    return maximum_filter1d(d, n, origin=(n - 1)//2, mode='nearest')

def portrait_and_test(f, starts, s0, s1, T_free, T_test, u, xlab='$x$', ylab='$\\dot x$', idx=(0, 1), angle=False):
    """Left: free trajectories (no input) from several starts. Right: two copies, same input, different starts."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[1, 1.6, 1.2]))
    for s in starts:
        _, x = simulate(f, s, T_free); ax[0].plot(x[idx[0]], x[idx[1]], lw=0.8); ax[0].plot(*np.array(s)[list(idx)], 'ko', ms=3)
    ax[0].set_xlabel(xlab); ax[0].set_ylabel(ylab); ax[0].set_title('no input: several starting states')
    t, a = simulate(f, s0, T_test, u); _, b = simulate(f, s1, T_test, u)
    ax[1].plot(t, a[0], label='copy A'); ax[1].plot(t, b[0], '--', label='copy B'); ax[1].set_xlabel('t'); ax[1].set_ylabel(xlab)
    ax[1].set_title('same input, different starting states'); ax[1].legend(loc='upper right')
    d = distance(a, b, angle); ax[2].semilogy(t, d, lw=0.6, color='0.6', label='distance'); ax[2].semilogy(t, envelope(d), 'k', label='running max (past 20)')
    ax[2].set_ylim(1e-10, 1e2); ax[2].set_xlabel('t'); ax[2].set_title('how different are A and B?'); ax[2].legend(loc='lower left')
    fig.tight_layout()
    return t, d
