import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","DejaVu Serif"],
    "font.size":7,"axes.labelsize":7,"axes.titlesize":7.5,"xtick.labelsize":6.5,"ytick.labelsize":6.5,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"mathtext.fontset":"stix"})
import pathlib; ROOT=pathlib.Path(__file__).resolve().parents[1]; R=str(ROOT/"results")+"/"
b=pd.read_csv(R+"results_bootstrap_ci.csv")
core=pd.read_csv(R+"results_core_all_models.csv")
tier={"SmolLM2-135M":1,"SmolLM2-360M":1,"TinyLlama-1.1B":2,"Qwen2.5-0.5B":2,"SmolLM2-1.7B":3,"Qwen2.5-1.5B":3,"Qwen2.5-3B":4,"Qwen2.5-7B":4}
par={"SmolLM2-135M":.135,"SmolLM2-360M":.36,"Qwen2.5-0.5B":.5,"TinyLlama-1.1B":1.1,"Qwen2.5-1.5B":1.5,"SmolLM2-1.7B":1.7,"Qwen2.5-3B":3,"Qwen2.5-7B":7}
col={1:"#9a9a9a",2:"#8e6bb0",3:"#3f9a6b",4:"#2c5aa0",0:"#222222"}
b["tier"]=b.model.map(lambda m: tier.get(m,0)); b["p"]=b.model.map(lambda m: par.get(m,1e9))
b=b.sort_values("p",ascending=False)
short=lambda m: "EBR (oracle)" if "EBR" in m else m
fig,ax=plt.subplots(1,4,figsize=(7.16,2.05),gridspec_kw={"width_ratios":[1.25,1.0,0.95,0.95],"wspace":0.78})
a=ax[0]; y=np.arange(len(b))
a.barh(y,b["mean"],xerr=[b["mean"]-b.ci_lo,b.ci_hi-b["mean"]],color=[col[t] for t in b.tier],height=0.62,
       error_kw=dict(lw=0.7,capsize=1.5,capthick=0.6))
a.set_yticks(y); a.set_yticklabels([short(m) for m in b.model]); a.invert_yaxis()
a.set_xlim(0,20); a.set_xlabel("Mean total score (/20), 95% CI"); a.set_title("(a) Simulated tiers",loc="left",fontweight="bold")
import matplotlib.patches as mp
a.legend(handles=[mp.Patch(color=col[k],label=l) for k,l in [(1,"Tier 1"),(2,"Tier 2"),(3,"Tier 3"),(4,"Tier 4"),(0,"Oracle")]],
         fontsize=5.5,frameon=False,loc="lower right",handlelength=1,borderpad=0.2,labelspacing=0.25)
e=core[core.model.str.contains("EBR")]
d=e.groupby("domain").total_score.agg(["mean","std","count"]).sort_values("mean")
a=ax[1]; y=np.arange(len(d))
a.barh(y,d["mean"],xerr=d["std"],color="#555555",height=0.62,error_kw=dict(lw=0.7,capsize=1.5,capthick=0.6))
a.set_yticks(y); a.set_yticklabels([f"Math. ({int(n)})" if i=="Mathematics" else f"{i} ({int(n)})" for i,n in zip(d.index,d["count"])])
a.set_xlim(0,22); a.set_xlabel("EBR mean $\\pm$ SD (/20)"); a.set_title("(b) Oracle by domain",loc="left",fontweight="bold")
for (k,xl,title,lab) in [(2,"depth_depth_score","(c) Depth signal","Depth-module score (0\u201310)"),(3,"sai","(d) Alignment","SAI")]:
    a=ax[k]; x=e[xl]; yv=e.total_score
    a.scatter(x,yv,s=9,color="#2c5aa0",alpha=0.8,linewidths=0)
    rho,p=stats.spearmanr(x,yv)
    a.set_xlabel(lab); a.set_title(title,loc="left",fontweight="bold")
    a.text(0.04,0.95,f"$\\rho$ = {rho:+.2f}\n$p$ {'< 0.001' if p<1e-3 else '= '+format(p,'.3f')}",transform=a.transAxes,va="top",fontsize=6.5)
    a.set_ylim(8,21)
ax[2].set_ylabel("Total score (/20)")
fig.savefig(str(ROOT/"paper"/"fig_results.pdf"),bbox_inches="tight",pad_inches=0.02)
print("ok")
