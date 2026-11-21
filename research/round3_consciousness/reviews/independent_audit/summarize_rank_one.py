"""Summarize all rank-one outcomes, using base-history bootstrap units."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    source = HERE / "rank_one_uplift"
    rows = json.loads((source / "summary.json").read_text())
    rng = np.random.default_rng(83105)
    bootstrap = rng.integers(0, 8, size=(20000, 8))
    summaries = []
    table = ["| Checkpoint | Held-out source energy in mode | Common uplift / mean current | Current sign after rank-one opposite | Later history margin, matched full opposite | Later history margin, rank-one opposite | Rebuilt history margin, rank-one own |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cs = {r["condition"]:r for r in row["conditions"]}
        with np.load(source/(row["model"]+".npz")) as arr:
            y=arr["margin"];base=arr["base_context"]
            conditions=list(cs)
            effects={}
            for condition in ("intact_uplift_full_opposite", "intact_uplift_rank1_opposite", "all_mean_uplift_full_own", "all_mean_uplift_rank1_own"):
                v=np.mean(y[:,conditions.index(condition),:,1]*np.array([1,-1]),axis=1)
                b=np.array([v[base==i].mean() for i in range(8)])
                effects[condition]=dict(mean=float(b.mean()),ci95=np.quantile(b[bootstrap].mean(axis=-1),[.025,.975]).tolist(),base_context_values=b.tolist())
            fi=conditions.index("all_mean_uplift_full_own");ri=conditions.index("all_mean_uplift_rank1_own")
            full=y[:,fi,:,1];rank=y[:,ri,:,1]
            sign_agreement=float(np.mean((full>0)==(rank>0)))
            rms=float(np.sqrt(np.mean((full-rank)**2)))
        entry=dict(model=row["model"],reduction=row["reduction"],effects=effects,
                   reconstruction_history_sign_agreement_full_vs_rank1=sign_agreement,
                   reconstruction_history_output_rms_full_vs_rank1=rms,
                   intact_rank1_opposite=cs["intact_uplift_rank1_opposite"],
                   all_mean_rank1_own=cs["all_mean_uplift_rank1_own"])
        summaries.append(entry)
        a=cs["intact_uplift_rank1_opposite"];b=cs["all_mean_uplift_rank1_own"];f=cs["intact_uplift_full_opposite"]
        short=row["model"].replace("_RateRNNstructNUM", " ")
        table.append(f"| {short} | {row['reduction']['heldout_rank1_energy_fraction']:.1%} | {row['reduction']['common_uplift_norm_over_source_mean_norm']:.2%} | {a['current_sign_accuracy']:.1%} | {f['history_signed_history_margin']:.3f} | {a['history_signed_history_margin']:.3f} | {b['history_signed_history_margin']:.3f} |")
    (HERE/"rank_one_summary.json").write_text(json.dumps(summaries,indent=2))
    (HERE/"rank_one_table.md").write_text("\n".join(table)+"\n")
    print("\n".join(table))
    print(json.dumps({r["model"]:{k:v["ci95"] for k,v in r["effects"].items()} for r in summaries},indent=2))


if __name__ == "__main__":
    main()
