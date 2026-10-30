"""python3 -I plot_kalman.py <kalman_results.json> <out.png>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); scs = ['S0_matched', 'S1_sensor_chain', 'S2_second_order', 'S3_tau_varying', 'S4_calib_artifact', 'S5_full']
ests = [('B0_raw', 'B0 raw', '#5C6B73', ''), ('B3_inverse', 'B3 inversion (real-time)', '#A3C9A8', ''), ('M1f_fixed', 'M1 Kalman filter, tau 10 (real-time)', '#2D6A6F', ''),
        ('M1f_taupop', 'M1 Kalman filter, tuned tau (real-time)', '#7FB7BE', ''),
        ('B1_shift', 'B1 time shift (uses future)', '#F2A541', '//'), ('M1s_fixed', 'M1 RTS smoother, tau 10 (uses future)', '#E4572E', '//'),
        ('M1s_taupop', 'M1 RTS smoother, tuned tau (uses future)', '#B03A1A', '//')]
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True); w = 0.115
for a, ns in zip(ax, (2, 8)):
    for j, (k, lab, col, h) in enumerate(ests):
        v = np.array([R['%s|%g' % (s, ns)][k]['rmse_rapid'] for s in scs]); x = np.arange(len(scs)) + (j - 3) * w
        a.bar(x, v[:, 0], w, color=col, hatch=h, edgecolor='white' if h else col, label=lab, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], error_kw=dict(lw=0.7, capsize=1))
    a.set_xticks(range(len(scs))); a.set_xticklabels([s.split('_')[0] for s in scs]); a.set_title('sensor noise sd %d mg/dL' % ns, fontsize=11); a.grid(axis='y', alpha=.25)
ax[0].set_ylabel('RMSE on rapid-change samples (mg/dL)\nmedian over 27 subjects, 95% bootstrap CI')
fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=8, frameon=False)
fig.suptitle('Piece 4: real-time Kalman filtering helps at low noise but not at noise sd 8; retrospective smoothing helps in both', fontsize=11)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(sys.argv[2], dpi=130)
