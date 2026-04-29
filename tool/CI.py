import codecs
import datetime
import math
import numpy as np

def readQRELS(fname):
    result = {}
    reader = codecs.open(filename=fname, mode='r', encoding='utf-8')
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        ss = line.split('\t')
        if ss[0] not in result:
            result[ss[0]] = []
        result[ss[0]].append(ss[1])
    reader.close()
    return result

def computePerformanceForOnePatent(sids_ret, sids_qrel):
    nCount = 0
    ap_like = 0.0
    sumRank = 0.0

    qrel_set = set(sids_qrel)

    for i in range(len(sids_ret)):
        sid_ret = sids_ret[i]
        if sid_ret in qrel_set:
            nCount += 1
            ap_like += nCount / (i + 1)
            sumRank += i + 1

    if len(sids_qrel) == 0:
        recall = 0.0
    else:
        recall = float(nCount) / len(sids_qrel)

    precision = float(nCount) / len(sids_ret) if len(sids_ret) > 0 else 0.0

    if nCount == 0:
        ap_like = 0.0
    else:
        ap_like = ap_like / nCount

    n = len(sids_qrel)
    nMax = len(sids_ret)
    pres = 0.0
    if n * nMax != 0:
        nCollection = nMax + n
        remain = n - nCount
        sumRank += remain * (nCollection - (remain - 1) / float(2))
        pres = 1 - (sumRank - (n * (n + 1) / float(2))) / (n * (nCollection - n))
        if (pres < 0.0) or (pres > 1.0):
            print('Error:PRES-->' + str(pres) + ' 不符合规范！')

    return recall, precision, ap_like, pres

def _t_critical_approx(df, alpha=0.05):
    try:
        from scipy.stats import t
        return float(t.ppf(1 - alpha/2, df=df))
    except Exception:
        return 1.959963984540054

def mean_ci_t(values, alpha=0.05):
    x = np.asarray(values, dtype=np.float64)
    Q = int(x.size)
    mean = float(x.mean()) if Q > 0 else 0.0
    sd = float(x.std(ddof=1)) if Q > 1 else 0.0
    se = sd / math.sqrt(Q) if Q > 0 else 0.0
    tcrit = _t_critical_approx(df=max(Q-1, 1), alpha=alpha)
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return mean, (lo, hi), se, sd, Q, tcrit

def computePerformance_with_ci(results, QRELS, alpha=0.05):
    per_query = []

    for tid, sids_ret in results.items():
        if tid not in QRELS:
            continue

        sids_qrel = QRELS[tid]
        recall, precision, ap_like, pres = computePerformanceForOnePatent(sids_ret, sids_qrel)

        per_query.append({
            "query_id": tid,
            "recall": recall,
            "precision": precision,   # 兼容你原来的 accuracy 含义
            "map_like": ap_like,      # 兼容你的实现
            "pres": pres,
            "num_rel": len(sids_qrel),
            "topN": len(sids_ret),
        })

    # 转成数组计算 CI
    recalls = [r["recall"] for r in per_query]
    precisions = [r["precision"] for r in per_query]
    maplikes = [r["map_like"] for r in per_query]
    presses = [r["pres"] for r in per_query]

    out = {
        "recall": mean_ci_t(recalls, alpha=alpha),
        "precision": mean_ci_t(precisions, alpha=alpha),
        "map_like": mean_ci_t(maplikes, alpha=alpha),
        "pres": mean_ci_t(presses, alpha=alpha),
        "per_query": per_query,
        "Q": len(per_query)
    }
    return out

def evalute_with_ci(ret_results, QRELS, tid='All Patents', alpha=0.05, save_path='save/result_with_ci.txt'):
    out = computePerformance_with_ci(ret_results, QRELS, alpha=alpha)
    Q = out["Q"]

    def fmt(name, pack):
        mean, (lo, hi), se, sd, Q_, tcrit = pack
        return f"{name}: {mean:.6f}  (95% CI: [{lo:.6f}, {hi:.6f}], SE={se:.6f}, SD={sd:.6f}, Q={Q_})"

    datetime_object = datetime.datetime.now()
    lines = []
    lines.append(f"Current Time: {datetime_object}")
    lines.append(f"adding patent: {tid}")
    lines.append(f"#queries used: {Q}")
    lines.append(fmt("Average Recall", out["recall"]))

    lines.append(fmt("Average 'Accuracy'(Precision@N)", out["precision"]))
    lines.append(fmt("Average MAP_like", out["map_like"]))
    lines.append(fmt("Average PRES", out["pres"]))
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if save_path:
        writer = codecs.open(filename=save_path, mode='a+', encoding='utf-8')
        writer.write(report + "\n")
        writer.close()

    return out

def evalute_with_ci_hits_only(ret_results, QRELS, tid='All Patents', alpha=0.05, save_path='save/result_with_ci.txt'):
    out = computePerformance_with_ci(ret_results, QRELS, alpha=alpha)
    per_query = out["per_query"]
    Q_all = out["Q"]

    per_query_hit = [r for r in per_query if r["recall"] > 0]
    Q_hit = len(per_query_hit)
    hit_rate = Q_hit / Q_all if Q_all > 0 else 0.0

    def pack_from_rows(rows, key):
        vals = [r[key] for r in rows]
        return mean_ci_t(vals, alpha=alpha)

    all_recall = out["recall"]
    all_prec  = out["precision"]
    all_map   = out["map_like"]
    all_pres  = out["pres"]

    hit_recall = pack_from_rows(per_query_hit, "recall") if Q_hit > 0 else None
    hit_prec   = pack_from_rows(per_query_hit, "precision") if Q_hit > 0 else None
    hit_map    = pack_from_rows(per_query_hit, "map_like") if Q_hit > 0 else None
    hit_pres   = pack_from_rows(per_query_hit, "pres") if Q_hit > 0 else None

    def fmt(name, pack):
        mean, (lo, hi), se, sd, Q, tcrit = pack
        return f"{name}: {mean:.6f}  (95% CI: [{lo:.6f}, {hi:.6f}], SE={se:.6f}, SD={sd:.6f}, Q={Q})"

    datetime_object = datetime.datetime.now()
    lines = []
    lines.append(f"Current Time: {datetime_object}")
    lines.append(f"adding patent: {tid}")
    lines.append("")

    lines.append("[All queries]")
    lines.append(f"#queries used: {Q_all}")
    lines.append(fmt("Average Recall", all_recall))
    lines.append(fmt("Average 'Accuracy'(Precision@N)", all_prec))
    lines.append(fmt("Average MAP_like", all_map))
    lines.append(fmt("Average PRES", all_pres))
    lines.append("")

    lines.append("[Only queries with hits>0 (equiv. recall>0)]")
    lines.append(f"#queries used: {Q_hit} / {Q_all}  (Hit@N={hit_rate:.4f})")
    if Q_hit == 0:
        lines.append("No queries with hits>0 under this topN.")
    else:
        lines.append(fmt("Average Recall", hit_recall))
        lines.append(fmt("Average 'Accuracy'(Precision@N)", hit_prec))
        lines.append(fmt("Average MAP_like", hit_map))
        lines.append(fmt("Average PRES", hit_pres))
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if save_path:
        writer = codecs.open(filename=save_path, mode='a+', encoding='utf-8')
        writer.write(report + "\n")
        writer.close()

    return {
        "all": out,
        "hit_only": {
            "Q_hit": Q_hit,
            "hit_rate": hit_rate,
            "recall": hit_recall,
            "precision": hit_prec,
            "map_like": hit_map,
            "pres": hit_pres,
        }
    }
