"""Drawing helpers for the motivation notebook (fading vs. non-fading memory)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from memory_demos import simulate, smooth_input, distance, envelope, SYSTEMS

A_COL, B_COL = 'C0', 'C1'

def _arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=18, lw=1.8, color=kw.get('color', 'k')))

def block_diagram():
    """Input -> system with internal state -> output, initial state from above."""
    fig, ax = plt.subplots(figsize=(9, 2.8)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.add_patch(FancyBboxPatch((3.5, 0.6), 3, 1.5, boxstyle='round,pad=0.1', fc='#e8f0fa', ec='k'))
    ax.text(5, 1.6, 'dynamical system', ha='center', fontsize=12, weight='bold'); ax.text(5, 1.05, r'state $\mathbf{z}(t)$', ha='center', fontsize=12)
    t = np.linspace(0, 2.2, 200); ax.plot(0.6 + t, 1.35 + 0.3*np.sin(5*t)*np.exp(-0.2*t), 'C2'); ax.text(1.7, 0.6, r'input $u(t)$', ha='center', fontsize=11)
    _arrow(ax, (2.9, 1.35), (3.4, 1.35)); _arrow(ax, (6.6, 1.35), (7.1, 1.35))
    ax.plot(7.2 + t, 1.35 + 0.25*np.sin(3*t + 1), 'C3'); ax.text(8.3, 0.6, r'output $y(t)$', ha='center', fontsize=11)
    _arrow(ax, (5, 2.85), (5, 2.25), color='0.4'); ax.text(5.15, 2.6, r'initial state $\mathbf{z}(0)$', fontsize=11, color='0.3')
    return fig

def memory_kernel(T=4., seed=3):
    """How much does an input at time t - tau still count now? Fading vs. non-fading weighting of the past."""
    rng = np.random.default_rng(seed); tau = np.arange(0, 25); u = rng.normal(0, 1, len(tau))
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.4), sharey=True)
    for a, w, title in zip(ax, (np.exp(-tau/T), np.ones_like(tau, float)), (rf'fading: $h(\tau)=e^{{-\tau/T}}$, $T={T:g}$', r'non-fading: $h(\tau)=1$ (integrator)')):
        a.bar(-tau, u, color=plt.cm.Greens(0.25 + 0.75*w), edgecolor='k', lw=0.4)
        a.plot(-tau, 2.6*w, 'k--', lw=1.5, label=r'weight $h(\tau)$ (scaled)'); a.axvline(0.5, color='C3', lw=2); a.text(0.8, 2.7, 'now', color='C3')
        a.set_xlabel(r'past time $-\tau$'); a.set_title(title, fontsize=10); a.legend(loc='lower left', fontsize=8)
    ax[0].set_ylabel(r'past input $u(t-\tau)$'); fig.tight_layout(); return fig

def two_integrators(T=4., seed=1):
    """Leaky integrator (fading) vs. ideal integrator (non-fading): same input, different starts; and with noise."""
    dt = 0.01; t = np.arange(0, 40, dt); u = smooth_input(40, amp=0.5, seed=seed)(t)
    fig, ax = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True)
    for j, (leak, name) in enumerate(((1/T, 'leaky integrator  $\\dot x=-x/T+u$'), (0., 'ideal integrator  $\\dot x=u$'))):
        for x0, c, lab in ((2., A_COL, 'copy A'), (-1.5, B_COL, 'copy B')):
            x = np.empty_like(t); x[0] = x0
            for k in range(1, len(t)): x[k] = x[k-1] + dt*(-leak*x[k-1] + u[k-1])
            ax[0, j].plot(t, x, c, label=lab)
        ax[0, j].set_title(name, fontsize=10); ax[0, j].legend(fontsize=8)
        rng = np.random.default_rng(0)
        for r in range(25):
            n = rng.normal(0, 1, len(t))/np.sqrt(dt)*0.3; x = np.zeros_like(t)
            for k in range(1, len(t)): x[k] = x[k-1] + dt*(-leak*x[k-1] + n[k-1])
            ax[1, j].plot(t, x, color='0.4', lw=0.5)
        ax[1, j].set_xlabel('$t$')
    ax[0, 0].set_ylabel('same input, different start'); ax[1, 0].set_ylabel('25 runs, only noise as input'); fig.tight_layout(); return fig

LANDSCAPES = [(lambda x: x**2/2, lambda x: x, 'bowl: one resting place'),
              (lambda x: 0*x, lambda x: 0*x, 'flat: every place is a resting place'),
              (lambda x: x**4/4 - x**2/2, lambda x: x**3 - x, 'two valleys: two resting places')]

def landscapes(starts=(-1.6, 0.45), T=12., u_amp=0.15, seed=2):
    """Balls (copies A, B) rolling with strong friction in different landscapes, driven by the same small input: dx/dt = -V'(x) + u."""
    dt = 0.01; t = np.arange(0, T, dt); u = smooth_input(T, amp=u_amp, seed=seed)(t); xs = np.linspace(-2, 2, 300)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, (V, dV, title) in zip(ax, LANDSCAPES):
        a.plot(xs, V(xs), 'k', lw=2); a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
        for x0, c in zip(starts, (A_COL, B_COL)):
            x = np.empty_like(t); x[0] = x0
            for k in range(1, len(t)): x[k] = x[k-1] + dt*(-dV(x[k-1]) + u[k-1])
            idx = np.linspace(0, len(t) - 1, 9).astype(int)
            for i, k in enumerate(idx): a.add_patch(Circle((x[k], V(x[k]) + 0.12), 0.09, color=c, alpha=0.15 + 0.85*(i == len(idx) - 1 or i == 0)))
            a.annotate('', (x[-1], V(x[-1]) + 0.35), (x[0], V(x[0]) + 0.35), arrowprops=dict(arrowstyle='->', color=c, lw=1.2, connectionstyle='arc3,rad=-0.25'))
        a.set_ylim(-0.45, 2.3); a.set_xlim(-2, 2); a.set_aspect('equal')
    fig.tight_layout(); return fig

def _pend(ax, x0, y0, phi, c, L=0.38):
    xe, ye = x0 + L*np.sin(phi), y0 - L*np.cos(phi); ax.plot([x0, xe], [y0, ye], color=c, lw=2); ax.add_patch(Circle((xe, ye), 0.06, color=c))

def pendulum_filmstrip(system, sA=(1.0, 0.), sB=(-1.5, 0.5), times=(0, 3, 6, 10, 20, 40, 70), seed=1):
    """Snapshots of two pendulum copies (same input, different start)."""
    u = smooth_input(100, seed=seed); f = SYSTEMS[system]; T = max(times) + 1
    t, a = simulate(f, list(sA), T, u); _, b = simulate(f, list(sB), T, u)
    fig, ax = plt.subplots(figsize=(13, 2.3)); ax.axis('off')
    for i, ti in enumerate(times):
        k = int(ti/0.01); x0 = i*1.0; ax.plot([x0 - 0.15, x0 + 0.15], [0, 0], 'k', lw=3)
        _pend(ax, x0, 0, b[0, k], B_COL); _pend(ax, x0, 0, a[0, k], A_COL); ax.text(x0, -0.6, f'$t={ti}$', ha='center')
    ax.set_xlim(-0.6, len(times) - 0.4); ax.set_ylim(-0.7, 0.45); ax.set_aspect('equal'); ax.set_title(system.replace('_', ' ') + ': copy A (blue) and copy B (orange)', fontsize=10)
    return fig

def clocks(sA=(2., 0.), sB=(-1.5, 0.5), times=(0, 3, 7, 12, 25, 41, 64), seed=1):
    """Self-sustained oscillator shown as a clock: hand angle = position on its oscillation."""
    u = smooth_input(100, seed=seed); T = max(times) + 1
    t, a = simulate(SYSTEMS['self_oscillator'], list(sA), T, u); _, b = simulate(SYSTEMS['self_oscillator'], list(sB), T, u)
    fig, ax = plt.subplots(figsize=(13, 2.4)); ax.axis('off')
    for i, ti in enumerate(times):
        k = int(ti/0.01); c0 = (i*1.0, 0); ax.add_patch(Circle(c0, 0.4, fill=False, lw=1.5))
        for s, col in ((a, A_COL), (b, B_COL)):
            ang = np.arctan2(-s[1, k], s[0, k]); r = min(1, np.hypot(*s[:, k])/2.2)*0.37
            ax.plot([c0[0], c0[0] + r*np.sin(ang)], [0, r*np.cos(ang)], color=col, lw=2.5)
        ax.text(c0[0], -0.58, f'$t={ti}$', ha='center')
    ax.set_xlim(-0.6, len(times) - 0.4); ax.set_ylim(-0.7, 0.5); ax.set_aspect('equal')
    ax.set_title('self-sustained oscillator as a clock: hand angle = position in the cycle, hand length = size of the swing', fontsize=10); return fig

def lorenz_pair(T=30., eps=1e-6, seed=1):
    """Two Lorenz copies starting eps apart: 3-D trajectories (colour = time) and distance."""
    u = smooth_input(100, seed=seed); f = SYSTEMS['lorenz']
    t, a = simulate(f, [1, 1, 1], T, u); _, b = simulate(f, [1 + eps, 1, 1], T, u); d = distance(a, b)
    fig = plt.figure(figsize=(13, 3.8)); ax = fig.add_subplot(1, 2, 1)
    ax.plot(t, a[0], color=A_COL, lw=1, label='copy A'); ax.plot(t, b[0], '--', color=B_COL, lw=1, label='copy B'); ax.set_xlabel('$t$'); ax.set_ylabel('$x$')
    ax.legend(fontsize=8, loc='lower left'); ax.set_title(f'identical for a long time, then completely different (start {eps:g} apart)', fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2); ax2.semilogy(t, d, 'k', lw=0.8); ax2.set_xlabel('$t$'); ax2.set_ylabel('distance A–B'); ax2.set_title('the tiny difference grows (log scale) until A and B are unrelated', fontsize=10)
    fig.tight_layout(); return fig

def pulse_memory(seed=1):
    """Does the system remember a short push? x(t) with and without a pulse, for four systems."""
    from memory_demos import pulse
    tests = [('damped_pendulum', [1, 0], 1.0, 'forgets the push'), ('double_well', [1, 0], -1.5, 'push flips the valley: stored'),
             ('self_oscillator', [2, 0], 1.0, 'push shifts the timing: stored'), ('lorenz', [1, 1, 1], 0.01, 'tiny push (0.01): future completely changed')]
    fig, ax = plt.subplots(len(tests), 1, figsize=(12, 8), sharex=True)
    for a_, (name, s0, amp, msg) in zip(ax, tests):
        t, x = simulate(SYSTEMS[name], s0, 80); _, y = simulate(SYSTEMS[name], s0, 80, pulse(30, 1.0, amp))
        a_.plot(t, x[0], color='0.3', lw=1.2, label='without push'); a_.plot(t, y[0], 'C3--', lw=1.2, label='with push'); a_.axvspan(30, 31, color='C3', alpha=0.25)
        a_.set_ylabel(name.replace('_', '\n'), fontsize=9); a_.text(0.99, 0.85, msg, transform=a_.transAxes, ha='right', fontsize=9, bbox=dict(fc='w', ec='0.7'))
    ax[0].legend(loc='lower left', fontsize=8); ax[-1].set_xlabel('$t$'); fig.tight_layout(); return fig

def summary(seed=1):
    u = smooth_input(300, seed=seed)
    cases = dict(damped_pendulum=([1, 0], [-1.5, .5], True), frictionless_pendulum=([1, 0], [-1.5, .5], True),
                 double_well=([1, 0], [-1.5, .5], False), self_oscillator=([2, 0], [-1.5, .5], False), lorenz=([1, 1, 1], [1 + 1e-6, 1, 1], False))
    fig, ax = plt.subplots(figsize=(9, 4))
    for n, (a0, b0, ang) in cases.items():
        t, a = simulate(SYSTEMS[n], a0, 100, u); _, b = simulate(SYSTEMS[n], b0, 100, u); ax.semilogy(t, envelope(distance(a, b, ang)), label=n.replace('_', ' '))
    ax.set_ylim(1e-9, 1e2); ax.set_xlabel('$t$'); ax.set_ylabel('difference A–B (running max)'); ax.legend(fontsize=8, ncol=2); ax.set_title('same input, different start', fontsize=10)
    return fig
