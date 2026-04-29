import os
import pandas as pd
from collections import defaultdict

# ========= 你的数据目录 =========
data_dir = "/home/pc3/zy/first/save/topk_subipc/42/weight/za"

# ========= rank 分桶函数 =========
def get_bucket(rank):
    # rank 从 1 开始
    if 1 <= rank <= 10:
        return "Top10"
    elif 11 <= rank <= 50:
        return "11-50"
    elif 51 <= rank <= 100:
        return "51-100"
    elif 101 <= rank <= 500:
        return "101-500"
    elif 501 <= rank <= 1000:
        return "501-1000"
    else:
        return None

# 保存每个桶里的具体数据
bucket_data = defaultdict(list)

# 保存总体相关专利数
total_relevant = 0

# ========= 遍历 za 文件夹 =========
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)

    if not os.path.isfile(file_path):
        continue

    # 文件名默认就是 query patent ID
    query_id = filename

    try:
        # 假设文件是 tab 分隔，无表头
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["idx", "candidate_id", "score", "label"],
            dtype={"idx": int, "candidate_id": str, "score": float, "label": float}
        )
    except Exception as e:
        print(f"读取失败: {file_path}, 错误: {e}")
        continue

    # rank = 行号 + 1
    df["rank"] = df.index + 1

    # 只保留 relevant 文档（label == 1）
    relevant_df = df[df["label"] == 1].copy()

    total_relevant += len(relevant_df)

    for _, row in relevant_df.iterrows():
        bucket = get_bucket(int(row["rank"]))
        if bucket is None:
            continue

        bucket_data[bucket].append({
            "query_id": query_id,
            "rank": int(row["rank"]),
            "candidate_id": row["candidate_id"],
            "score": float(row["score"]),
            "label": float(row["label"])
        })

# ========= 桶顺序 =========
bucket_order = ["Top10", "11-50", "51-100", "101-500", "501-1000"]

# ========= 输出统计结果 =========
print("=" * 80)
print(f"总相关专利数: {total_relevant}")
print("=" * 80)

summary_rows = []

for bucket in bucket_order:
    cnt = len(bucket_data[bucket])
    ratio = cnt / total_relevant if total_relevant > 0 else 0.0
    summary_rows.append([bucket, cnt, ratio])

summary_df = pd.DataFrame(summary_rows, columns=["bucket", "count", "ratio"])

print("\n每个桶的统计结果：")
print(summary_df.to_string(index=False))

# ========= 输出每个桶内的具体数据 =========
for bucket in bucket_order:
    print("\n" + "=" * 80)
    print(f"桶: {bucket}")
    print(f"数量: {len(bucket_data[bucket])}")
    if total_relevant > 0:
        print(f"占比: {len(bucket_data[bucket]) / total_relevant:.6f}")
    print("-" * 80)

    if len(bucket_data[bucket]) == 0:
        print("无数据")
        continue

    bucket_df = pd.DataFrame(bucket_data[bucket])
    bucket_df = bucket_df.sort_values(by=["query_id", "rank"])

    print(bucket_df.to_string(index=False))
