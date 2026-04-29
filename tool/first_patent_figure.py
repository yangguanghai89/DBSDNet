import os
import pandas as pd

data_dir = "/home/pc3/zy/first/save/topk_subipc/42/weight/za"
k_list = [1, 3, 5, 10, 20, 50]

def get_first_relevant_rank(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["idx", "candidate_id", "score", "label"]
        )
    except Exception as e:
        print(f"读取失败: {file_path}, 错误: {e}")
        return None

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    relevant_rows = df[df["label"] == 1.0]

    if relevant_rows.empty:
        return None

    return int(relevant_rows.index[0] + 1)

rows = []
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    if not os.path.isfile(file_path):
        continue

    first_rank = get_first_relevant_rank(file_path)
    rows.append({
        "query_id": filename,
        "first_relevant_rank": first_rank
    })

first_rank_df = pd.DataFrame(rows)

print("=" * 100)
print("每个查询的第一个相关专利 rank")
print("=" * 100)
print(first_rank_df.sort_values("query_id").to_string(index=False))

valid_df = first_rank_df[first_rank_df["first_relevant_rank"].notna()].copy()
total_queries = len(valid_df)

curve_rows = []
for k in k_list:
    hit_count = (valid_df["first_relevant_rank"] <= k).sum()
    hit_ratio = hit_count / total_queries if total_queries > 0 else 0.0
    curve_rows.append({
        "K": k,
        "hit_count": int(hit_count),
        "total_queries": int(total_queries),
        "hit_ratio": float(hit_ratio)
    })

curve_df = pd.DataFrame(curve_rows)

print("\n" + "=" * 100)
print("折线图数据")
print("=" * 100)
print(curve_df.to_string(index=False))

first_rank_df.to_csv("za_first_relevant_rank.csv", index=False, encoding="utf-8-sig")
curve_df.to_csv("za_first_relevant_curve.csv", index=False, encoding="utf-8-sig")
