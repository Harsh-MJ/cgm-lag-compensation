"""python3 -I plot_ident.py <ident_results.json> <fstune_results.json> <out.png>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
I = json.load(open(sys.argv[1])); F = json.load(open(sys.argv[2]))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw=dict(width_ratios=[1, 1.6]))
a = ax[0]; t = np.array(I['B']['tau_eff'])
for tag, key, lab, col, mk in [('B', 'ML_fs', 'likelihood (CGM + fingersticks), noise 2', '#2D6A6F', 'o'), ('C', 'ML_fs', 'likelihood (CGM + fingersticks), noise 8', '#7FB7BE', 's'),
                               ('B', 'FSreg', 'fingerstick regression, noise 2', '#E4572E', '^')]:
    a.scatter(I[tag]['tau_eff'], I[tag]['estimates'][key], s=28, color=col, marker=mk, label=lab, alpha=.9)
a.plot([4, 28], [4, 28], color='#5C6B73', lw=1, ls='--', label='perfect estimate'); a.axhline(10, color='#A3C9A8', lw=1, label='population prior (10 min)')
a.set_xlim(4, 24); a.set_ylim(3, 29); a.set_xlabel('true lag tau (min), scenario S0'); a.set_ylabel('estimated lag (min), per patient'); a.grid(alpha=.25); a.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
a.set_title('Per-patient likelihood estimates are confidently wrong\n(95% intervals cover truth in 15-19% of patients)', fontsize=10)
b = ax[1]; tags = [('B', 'S0\nnoise 2'), ('C', 'S0\nnoise 8'), ('D', 'S1\nnoise 2'), ('E', 'S1\nnoise 8'), ('G', 'S5\nnoise 2'), ('F', 'S5\nnoise 8')]
rows = [('fixed', 'tau fixed at 10 min', '#5C6B73'), ('lik', 'per-patient likelihood tau', '#A3C9A8'), ('fs', 'population tau from fingersticks', '#E4572E'),
        ('truth', 'population tau from true glucose (sim only)', '#F2A541'), ('oracle', 'true effective tau (oracle)', '#2D6A6F')]
get = dict(fixed=lambda g: F[g]['M1s_fixed'], lik=lambda g: I['downstream'][g]['M1s_EB'], fs=lambda g: F[g]['M1s_tau_fs'], truth=lambda g: F[g]['M1s_tau_truth'], oracle=lambda g: F[g]['M1s_oracle_eff'])
w = 0.16
for j, (k, lab, col) in enumerate(rows):
    v = np.array([get[k](g)['rmse_rapid'] for g, _ in tags]); x = np.arange(len(tags)) + (j - 2) * w
    b.bar(x, v[:, 0], w, color=col, label=lab, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], error_kw=dict(lw=0.7, capsize=1))
b.set_xticks(range(len(tags))); b.set_xticklabels([l for _, l in tags], fontsize=9); b.set_ylabel('RMSE on rapid-change samples, days 4-5 (mg/dL)\nRTS smoother, median over 27 subjects, 95% CI')
b.grid(axis='y', alpha=.25); b.legend(fontsize=7.5, loc='upper left'); b.set_title('A population lag chosen from fingersticks recovers most of the oracle gain', fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=130)
