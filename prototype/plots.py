import json, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
C = dict(raw='#2a78d6', inv='#eb6834', fix='#1baf7a', fs='#eda100')
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'font.family': 'DejaVu Sans'})
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=.8); ax.set_axisbelow(True)

# Fig 1: pooled RMSE vs sensor-noise level
R = {ns: json.load(open(f'../data/results{suf}.json'))['metrics'] for ns, suf in [(0, '_ns0'), (0.3, '_ns0.3'), (1, '')]}
xs = [0, 0.3, 1]; lab = ['0\n(lag only)', '0.3', '1\n(simulator noise,\nsd≈12 mg/dL)']
fig, ax = plt.subplots(figsize=(7.4, 4.3)); clean(ax)
for key, name, col, ls in [('raw', 'Raw CGM', C['raw'], '-'), ('inv', 'Fixed-τ inversion', C['inv'], '-'), ('ekf_fix', 'EKF, fixed τ = 10 min', C['fix'], '-'),
                           ('ekf_est_fs', 'EKF, online τ + fingersticks', C['fs'], '-'), ('ekf_oracle', 'EKF, true τ (oracle)', MUTED, '--')]:
    y = [R[x][key]['rmse'] for x in xs]
    ax.plot(range(3), y, color=col, lw=2, ls=ls, marker='o', ms=7, mec=SURF, mew=2, label=name)
ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_xlim(-.15, 2.15); ax.set_ylim(0, 17.5); ax.legend(frameon=False, loc='upper left', fontsize=9)
ax.set_xlabel('Sensor-noise scale (× simulator noise)'); ax.set_ylabel('RMSE vs true plasma glucose (mg/dL)')
ax.set_title('Lag compensation only pays off when sensor noise is small', loc='left', color=INK, fontsize=11.5, fontweight='bold')
fig.tight_layout(); fig.savefig('../figs/fig1_rmse_vs_noise.png', dpi=170); plt.close(fig)

# Fig 2: example trace, noise-free CGM, around a hypoglycemic event
d = pickle.load(open('../data/preds_ns0.pkl', 'rb')); test, st = d['test'], d['store']
best = None
for i, p in enumerate(test):
    low = np.where(p['G'][120:] < 70)[0]
    if len(low): best = (i, 120 + low[0]); break
i, k0 = best; p = test[i]; sl = slice(k0 - 50, k0 + 30); t = (np.arange(len(p['G']))[sl] - k0) * 3 / 60.0
fig, ax = plt.subplots(figsize=(7.4, 4.3)); clean(ax)
ax.axhline(70, color=GRID, lw=1.2); ax.text(t[0], 72, '70 mg/dL', color=MUTED, fontsize=8.5)
ax.plot(t, p['G'][sl], color=INK, lw=2.4, label='True plasma glucose')
ax.plot(t, p['y'][sl], color=C['raw'], lw=2, label='CGM (lagged)')
ax.plot(t, st['ekf_fix'][i][sl], color=C['fix'], lw=2, label='EKF, fixed τ = 10 min')
ax.legend(frameon=False, loc='upper right', fontsize=9)
ax.set_xlabel('Hours relative to true glucose crossing 70 mg/dL'); ax.set_ylabel('Glucose (mg/dL)')
ax.set_title(f'CGM trails a falling glucose; the estimator tracks it\n({p["name"]}, true τ = {p["tau"]:.1f} min, noise-free sensor)', loc='left', color=INK, fontsize=10.5, fontweight='bold')
fig.tight_layout(); fig.savefig('../figs/fig2_example_trace.png', dpi=170); plt.close(fig)

# Fig 3: tau recovery
r = json.load(open('../data/results_ns0.json'))['tau']['ekf_est_fs']
fig, ax = plt.subplots(figsize=(5.2, 4.6)); clean(ax); ax.grid(axis='x', color=GRID, lw=.8)
ax.plot([5, 22], [5, 22], color=MUTED, ls='--', lw=1.2); ax.text(15.3, 14.0, 'perfect', color=MUTED, fontsize=8.5, rotation=36)
ax.scatter(r['t_true'], r['t_hat'], s=60, color=C['fs'], edgecolor=SURF, linewidth=2, zorder=3)
ax.axhline(10, color=C['raw'], lw=1.5); ax.text(5.2, 10.4, 'prior guess (10 min)', color=INK, fontsize=8.5)
ax.set_xlim(5, 22); ax.set_ylim(0, 40); ax.set_xlabel('True τ (min)'); ax.set_ylabel('Estimated τ (min)')
ax.set_title('Online τ estimate does not track true τ', loc='left', color=INK, fontsize=11, fontweight='bold')
fig.tight_layout(); fig.savefig('../figs/fig3_tau_recovery.png', dpi=170); plt.close(fig)
print('ok', best)
