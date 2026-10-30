"""python3 -I plot_baselines.py <baseline_results.json> <out.png>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); scs = ['S0_matched', 'S1_sensor_chain', 'S2_second_order', 'S3_tau_varying', 'S4_calib_artifact', 'S5_full']
ests = [('B0_raw', 'B0 raw', '#5C6B73'), ('B2_smooth', 'B2 causal smoothing', '#A3C9A8'), ('B3_inverse', 'B3 causal inversion (tau 10)', '#2D6A6F'),
        ('B3_oracle_tau', 'B3 with true tau (oracle)', '#7FB7BE'), ('B1_shift', 'B1 time shift (non-causal)', '#E4572E')]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for a, ns in zip(ax, (2, 8)):
    w = 0.16
    for j, (k, lab, col) in enumerate(ests):
        v = np.array([R['%s|%g' % (s, ns)][k]['rmse_rapid'] for s in scs]); x = np.arange(len(scs)) + (j - 2) * w
        a.bar(x, v[:, 0], w, color=col, label=lab, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], error_kw=dict(lw=0.8, capsize=1.5))
    a.set_xticks(range(len(scs))); a.set_xticklabels([s.split('_')[0] for s in scs]); a.set_title('sensor noise sd %d mg/dL' % ns, fontsize=11); a.grid(axis='y', alpha=.25)
ax[0].set_ylabel('RMSE on rapid-change samples (mg/dL)\nmedian over 27 subjects, 95% bootstrap CI'); fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=8, frameon=False)
fig.suptitle('Piece 3 baselines: only the non-causal shift halves rapid-change error; causal methods help modestly', fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(sys.argv[2], dpi=130)
