import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer

#同小类内连边，不够再到其他数据里找
@torch.no_grad()
def encode_texts(text_list, tokenizer, model, device, max_length=128, batch_size=32):
    model.to(device)
    model.eval()

    out = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        outputs = model(**enc)
        cls = outputs.last_hidden_state[:, 0, :]
        out.append(cls.detach().cpu())

    if len(out) == 0:
        return torch.zeros(0, model.config.hidden_size)
    return torch.cat(out, dim=0)


def parse_ipc_cell(cell: str, keep_len=4):
    cell = (cell or "").strip()
    if not cell:
        return []

    if ";" in cell:
        parts = [x.strip() for x in cell.split(";") if x.strip()]
    elif "；" in cell:
        parts = [x.strip() for x in cell.split("；") if x.strip()]
    else:
        parts = [x.strip() for x in cell.split() if x.strip()]

    out = []
    for x in parts:
        if len(x) >= keep_len:
            out.append(x[:keep_len])

    return list(dict.fromkeys(out))

def load_ipc_desc(ipc_desc_path: str):
    df = pd.read_csv(ipc_desc_path, sep="\t", header=None)
    code = df[0].astype(str).str.strip()
    desc = df[2].astype(str).fillna("")
    return dict(zip(code, desc))

def build_mappings_and_texts(
    patent_tsv_path,
    cache_path="data/mappings.pkl",
):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading mappings from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["mappings"], cache["patent_text_dict"]

    print("[INFO] Cache not found, scanning patent TSV...")

    patent2idx = {}
    ipc2idx = {}
    patent_text_dict = {}

    patent_cnt = 0
    ipc_cnt = 0

    patent_ipc_edges = []
    patent2codes = defaultdict(list)
    code2patents = defaultdict(list)

    def get_patent_idx(pid):
        nonlocal patent_cnt
        if pid not in patent2idx:
            patent2idx[pid] = patent_cnt
            patent_cnt += 1
        return patent2idx[pid]

    def get_ipc_idx(code4):
        nonlocal ipc_cnt
        if code4 not in ipc2idx:
            ipc2idx[code4] = ipc_cnt
            ipc_cnt += 1
        return ipc2idx[code4]

    with open(patent_tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header:
            header[0] = header[0].lstrip("\ufeff")

        idx_id = header.index("id")
        idx_title = header.index("title")
        idx_abs = header.index("abstract")
        idx_ipc = header.index("xiaolei")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            pid = str(parts[idx_id]).strip()

            p = get_patent_idx(pid)

            title = parts[idx_title].strip()
            abs_ = parts[idx_abs].strip()
            text = (title + " " + abs_).strip()
            if pid not in patent_text_dict:
                patent_text_dict[pid] = text

            codes4 = parse_ipc_cell(parts[idx_ipc], keep_len=4)
            for c4 in codes4:
                c = get_ipc_idx(c4)
                patent_ipc_edges.append((p, c))
                patent2codes[p].append(c)
                code2patents[c].append(p)

    # df：每个小类连接的不同专利数
    code_df = {c: len(set(plist)) for c, plist in code2patents.items()}

    mappings = {
        "patent2idx": patent2idx,
        "ipc2idx": ipc2idx,
        "patent_ipc_edges": patent_ipc_edges,
        "patent2codes": dict(patent2codes),
        "code_df": code_df,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"mappings": mappings, "patent_text_dict": patent_text_dict}, f)

    print(f"[INFO] Saved mappings to: {cache_path}")
    print(f"[INFO] patents={len(patent2idx)} ipc={len(ipc2idx)} p-ipc edges={len(patent_ipc_edges)}")
    return mappings, patent_text_dict

def build_node_features(
    bert_path,
    device,
    mappings,
    patent_text_dict,
    ipc_desc_path,
    cache_file="data/node_features.pt",
    bert_bs_patent=32,
    bert_bs_ipc=32,
    max_len_patent=128,
    max_len_ipc=128,
):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"[INFO] Loading node features cache: {cache_file}")
        obj = torch.load(cache_file, map_location="cpu")
        return obj["patent_text_x"].float(), obj["ipc_x"].float(), obj["patent_x"].float()

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path)

    patent2idx = mappings["patent2idx"]
    ipc2idx = mappings["ipc2idx"]

    # ---- patent texts by idx ----
    num_patent = len(patent2idx)
    id2pat = [None] * num_patent
    for pid, idx in patent2idx.items():
        id2pat[idx] = pid
    patent_texts = [patent_text_dict.get(pid, "") for pid in id2pat]

    # ---- ipc desc by idx ----
    ipc_desc_dict = load_ipc_desc(ipc_desc_path)

    num_ipc = len(ipc2idx)
    id2ipc = [None] * num_ipc
    for code4, idx in ipc2idx.items():
        id2ipc[idx] = code4
    ipc_texts = [ipc_desc_dict.get(code4, code4) for code4 in id2ipc]

    print("[INFO] Encoding patent_text_x ...")
    patent_text_x = encode_texts(
        patent_texts, tokenizer, model, device,
        max_length=max_len_patent, batch_size=bert_bs_patent
    )

    print("[INFO] Encoding ipc_x (subclass descriptions) ...")
    ipc_x = encode_texts(
        ipc_texts, tokenizer, model, device,
        max_length=max_len_ipc, batch_size=bert_bs_ipc
    )

    # ---- 聚合 mean(ipc_x) 到专利 ----
    px_edges = torch.tensor(mappings["patent_ipc_edges"], dtype=torch.long)
    p_idx = px_edges[:, 0]
    c_idx = px_edges[:, 1]

    # 用 CPU 聚合（省显存），也可放 GPU
    agg = torch.zeros(num_patent, ipc_x.size(-1), dtype=torch.float32)
    cnt = torch.zeros(num_patent, dtype=torch.float32)

    agg.index_add_(0, p_idx, ipc_x[c_idx].float())
    cnt.index_add_(0, p_idx, torch.ones_like(p_idx, dtype=torch.float32))
    mean_ipc = agg / cnt.clamp(min=1.0).unsqueeze(-1)

    patent_x = torch.cat([patent_text_x.float(), mean_ipc.float()], dim=-1)  # [Np,768]

    # 保存：float16 省空间
    torch.save(
        {
            "patent_text_x": patent_text_x,
            "ipc_x": ipc_x,
            "patent_x": patent_x,
            "meta": {
                "bert_path": bert_path,
                "max_len_patent": max_len_patent,
                "max_len_ipc": max_len_ipc,
                "patent_x_rule": "patent_text_x + mean(ipc_x)",
            }
        },
        cache_file
    )
    print(f"[INFO] Saved node features to: {cache_file}")
    return patent_text_x.float(), ipc_x.float(), patent_x.float()


def build_pp_edges_subclass_pref_then_global_hnsw(
    vec_for_sim_cpu_f32: torch.Tensor,   # [N, D] CPU float32
    mappings,
    out_path="data/pp_edges.pt",
    topk_per_patent=20,
    min_cos=0.6,
    ann_candidates=3000,                 # 候选越大，越容易“同小类优先”挑满 20（但更慢/更占内存）
    hnsw_m=32,
    ef_search=128,
    batch=4096,
):
    """
    逻辑：
    1) 全量专利向量建 HNSW (IP)，L2 normalize 后 inner product = cosine
    2) 对每个专利检索 ann_candidates 个候选
    3) 过滤：score>=min_cos、去掉自己、去重
    4) 选边：优先选“共享任一小类”的候选，不足 topk 再用其他候选补齐
    5) 每个专利最多 topk_per_patent 条；不足就保留已有
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"[INFO] Loading cached p-p edges: {out_path}")
        obj = torch.load(out_path, map_location="cpu")
        return obj["pp_edge_index"]

    import faiss

    # -------- 0) 准备：专利 -> 小类列表（去重后的小列表，避免 set 占太大内存）--------
    patent2codes = mappings["patent2codes"]  # dict: p_idx -> list[code_idx]
    N = vec_for_sim_cpu_f32.size(0)

    codes_list = [[] for _ in range(N)]
    for p, codes in patent2codes.items():
        if 0 <= p < N and codes:
            # 去重并保持为小列表（通常每篇专利的小类很少）
            codes_list[p] = list(dict.fromkeys(codes))

    def share_any_code(codes_a, codes_b):
        # 两个都很短时，用双层小循环比建 set 更省内存
        if not codes_a or not codes_b:
            return False
        for ca in codes_a:
            for cb in codes_b:
                if ca == cb:
                    return True
        return False

    # -------- 1) 向量 normalize --------
    X = vec_for_sim_cpu_f32.detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(X)
    N, D = X.shape
    print(f"[INFO] Build edges (subclass-pref + global HNSW): N={N}, D={D}, topk={topk_per_patent}, min_cos={min_cos}")

    # -------- 2) 建全局 HNSW (INNER_PRODUCT) --------
    print("[INFO] Building FAISS HNSW index (INNER_PRODUCT)...")
    index = faiss.IndexHNSWFlat(D, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.add(X)

    # -------- 3) 批量 search + 逐点选边（同小类优先）--------
    src_all, dst_all, sim_all = [], [], []
    k_search = ann_candidates + 1  # +1 包含 self，方便剔除

    print("[INFO] ANN search + subclass-preferred selection ...")
    for st in range(0, N, batch):
        ed = min(N, st + batch)
        q = X[st:ed]
        Dv, Iv = index.search(q, k_search)  # Dv: cosine(IP), Iv: neighbor ids

        for i in range(ed - st):
            p = st + i
            kept = 0
            used = set()

            same_bucket = []   # (score, qid) 共享小类
            other_bucket = []  # (score, qid) 其他

            codes_p = codes_list[p]

            for score, qid in zip(Dv[i], Iv[i]):
                if qid < 0:
                    continue
                qid = int(qid)
                if qid == p:
                    continue
                if score < min_cos:
                    continue
                if qid in used:
                    continue
                used.add(qid)

                # 同小类优先：共享任一小类即可
                if share_any_code(codes_p, codes_list[qid]):
                    same_bucket.append((float(score), qid))
                else:
                    other_bucket.append((float(score), qid))

            # 注意：Faiss 返回本身已按 score desc 排好；bucket 内保持原顺序即可
            for _, qid in same_bucket:
                src_all.append(p)
                dst_all.append(qid)
                sim_all.append(0.0)  # 可不存；若你想存分数可存 score
                kept += 1
                if kept >= topk_per_patent:
                    break

            if kept < topk_per_patent:
                for _, qid in other_bucket:
                    src_all.append(p)
                    dst_all.append(qid)
                    sim_all.append(0.0)
                    kept += 1
                    if kept >= topk_per_patent:
                        break

    if len(src_all) == 0:
        pp_edge_index = torch.zeros((2, 0), dtype=torch.long)
        torch.save({"pp_edge_index": pp_edge_index, "meta": {"empty": True}}, out_path)
        print(f"[WARN] No p-p edges generated. saved empty to {out_path}")
        return pp_edge_index

    # -------- 4) 全局去重：保证 (src,dst) 唯一 --------
    src_np = np.asarray(src_all, dtype=np.int64)
    dst_np = np.asarray(dst_all, dtype=np.int64)

    order = np.lexsort((dst_np, src_np))  # src asc, dst asc
    src_np = src_np[order]
    dst_np = dst_np[order]

    keep = np.ones(len(src_np), dtype=bool)
    keep[1:] = (src_np[1:] != src_np[:-1]) | (dst_np[1:] != dst_np[:-1])

    src_np = src_np[keep]
    dst_np = dst_np[keep]

    pp_edge_index = torch.tensor([src_np.tolist(), dst_np.tolist()], dtype=torch.long)

    torch.save(
        {
            "pp_edge_index": pp_edge_index,
            "meta": {
                "strategy": "subclass_pref_then_global_hnsw",
                "min_cos": float(min_cos),
                "topk_per_patent": int(topk_per_patent),
                "ann_candidates": int(ann_candidates),
                "hnsw_m": int(hnsw_m),
                "ef_search": int(ef_search),
                "batch": int(batch),
                "directed": True,
                "N": int(N),
                "D": int(D),
                "E": int(pp_edge_index.size(1)),
                "dedup": True,
            }
        },
        out_path
    )
    print(f"[INFO] Saved p-p edges to: {out_path}, E={pp_edge_index.size(1)} (deduped)")
    return pp_edge_index

def build_all(
    patent_tsv_path,
    ipc_desc_path,
    out_dir="data",
    bert_path="bert-base-uncased",
    device="cuda",
):
    os.makedirs(out_dir, exist_ok=True)

    mappings_path = os.path.join(out_dir, "mappings.pkl")
    feats_path = os.path.join(out_dir, "node_features.pt")
    pp_path = os.path.join(out_dir, "pp_edges.pt")
    struct_path = os.path.join(out_dir, "graph_structure.pkl")

    # 1) mappings + 文本 + 专利-ipc 边
    mappings, patent_text_dict = build_mappings_and_texts(
        patent_tsv_path,
        cache_path=mappings_path
    )

    # 2) 节点特征
    patent_text_x, ipc_x, patent_x = build_node_features(
        bert_path=bert_path,
        device=device,
        mappings=mappings,
        patent_text_dict=patent_text_dict,
        ipc_desc_path=ipc_desc_path,
        cache_file=feats_path
    )

    # 3) p-ipc edge_index
    px = torch.tensor(mappings["patent_ipc_edges"], dtype=torch.long).t().contiguous()

    # 4) p-p edges: 全局ANN + 共享小类过滤
    # 默认用纯专利文本向量做 ANN（更符合“相似度只用专利文本”）
    vec_for_sim = patent_text_x.detach().cpu().float()

    pp = build_pp_edges_subclass_pref_then_global_hnsw(
        vec_for_sim_cpu_f32=vec_for_sim,
        mappings=mappings,
        out_path=pp_path,
        topk_per_patent=10,
        min_cos=0.7,
        ann_candidates=2000,
        hnsw_m=48,
        ef_search=256,
        batch=4096
    )

    edge_index_dict = {
        ("patent", "has_ipc", "ipc"): px,
        ("ipc", "rev_has_ipc", "patent"): px.flip(0),
        ("patent", "sim", "patent"): pp,
        ("patent", "rev_sim", "patent"): pp.flip(0),
    }

    # 5) 保存结构（方便你直接喂给 PyG）
    with open(struct_path, "wb") as f:
        pickle.dump(
            {
                "edge_index_dict": edge_index_dict,
                "num_patent": patent_x.size(0),
                "num_ipc": ipc_x.size(0),
            },
            f
        )

    print("[INFO] DONE.")
    print(f"  mappings : {mappings_path}")
    print(f"  features : {feats_path}  (contains patent_text_x, ipc_x, patent_x)")
    print(f"  pp edges : {pp_path}")
    print(f"  structure: {struct_path}")


if __name__ == "__main__":
    PATENT_TSV = "/home/pc3/zy/data/src/TOPK_SubIPC/patent.tsv"
    IPC_DESC_TSV = "/home/pc3/zy/data/src/TOPK_SubIPC/ipc.tsv"

    build_all(
        patent_tsv_path=PATENT_TSV,
        ipc_desc_path=IPC_DESC_TSV,
        out_dir="/home/pc3/zy/soft/data",
        bert_path="/home/pc3/zy/premodel/bert_base_uncase",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
