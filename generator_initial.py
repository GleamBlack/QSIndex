import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def generate_xy(x_min=0, x_max=10_000_000, y_min=0, y_max=400, target_count=2_000_000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(target_count, dtype=float)
    y = np.zeros(target_count, dtype=float)
    k = int(target_count * 0.2)
    x[:k] = rng.normal(loc=(10_000_000 + 8_000_000) / 2, scale=350_000, size=k)
    x[k:] = rng.normal(loc=(0 + 8_000_000) / 2, scale=2_000_000, size=target_count - k)
    y = rng.integers(y_min, y_max + 1, target_count)
    mask = (x - y >= x_min) & (x + y <= x_max)
    x = x[mask]
    y = y[mask]
    x = np.round(x * 2) / 2
    y = np.round(y)
    return x, y


def shuffle_integer_parts(x, seed=42):
    rng = np.random.default_rng(seed)
    maxi = int(np.ceil(np.max(x))) + 1
    numbers = np.arange(0, maxi, dtype=float)
    rng.shuffle(numbers)
    x_shuf = []
    for v in x:
        integer_part = int(v)
        decimal_part = v - integer_part
        x_shuf.append(numbers[integer_part] + decimal_part)
    return np.array(x_shuf, dtype=float)


def split_train_eval(x_shuf, y, eval_size=1_000_000, seed=42):
    data = np.vstack((x_shuf, y)).T
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    data_eval = data[:eval_size]
    data_train = data[eval_size:]
    return data_train, data_eval


def run_dbscan(data_train, eps=40, min_samples=150):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_train)
    labels = db.labels_
    core_mask = np.zeros_like(labels, dtype=bool)
    if hasattr(db, "core_sample_indices_"):
        core_mask[db.core_sample_indices_] = True
    return labels, core_mask


def cluster_boundaries_via_diff(data_train, labels):
    bounds = {}
    uniq = set(labels)
    uniq.discard(-1)
    for k in uniq:
        pts = data_train[labels == k]
        if len(pts) == 0:
            continue
        diffs = pts[:, 0] - pts[:, 1]
        bounds[k] = (float(np.min(diffs)), float(np.max(diffs)))
    return bounds


def merge_intervals(intervals):
    if not intervals:
        return []
    arr = sorted(intervals, key=lambda it: it[0])
    merged = [list(arr[0])]
    for s, e in arr[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def relabel_by_merged(labels, x_shuf, y, merged_bounds):
    new_labels = np.array(labels, copy=True)
    if len(merged_bounds) == 0:
        return new_labels
    diffs = x_shuf - y
    for idx in range(len(diffs)):
        if labels[idx] == -1:
            continue
        d = diffs[idx]
        for k, (l, r) in enumerate(merged_bounds):
            if l <= d <= r:
                new_labels[idx] = k
                break
    return new_labels


def plot_clusters(data_train, labels, core_mask, title):
    uniq = sorted(set(labels) - {-1})
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(1, len(uniq)))]
    plt.figure(figsize=(10, 6))
    for k, col in zip(uniq, colors):
        sel_core = (labels == k) & core_mask
        sel_border = (labels == k) & (~core_mask)
        if np.any(sel_core):
            xy = data_train[sel_core]
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=8)
        if np.any(sel_border):
            xy = data_train[sel_border]
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=4)
    plt.title(title)
    plt.xlabel("x_values_shuffled")
    plt.ylabel("y_values")
    plt.tight_layout()
    plt.show()


def fixed_grid_intervals(lo=1, hi=1_000_000, width=50):
    return np.array([[i, i + (width - 1)] for i in range(lo, hi + 1, width)], dtype=int)


def cover_grid_with_clusters(grid, merged_bounds):
    merged_ranges = []
    covered = np.zeros(len(grid), dtype=bool)
    for l, r in merged_bounds:
        sel = []
        for idx, (a, b) in enumerate(grid):
            if not (b < l or a > r):
                sel.append(idx)
                covered[idx] = True
        if sel:
            merged_ranges.append([grid[sel[0]][0], grid[sel[-1]][1]])
    merged_ranges += [g.tolist() for g, c in zip(grid, covered) if not c]
    merged_ranges = sorted(merged_ranges, key=lambda z: z[0])
    return merged_ranges


def eval_ranges(data_eval):
    return np.array([[x - y, x + y] for x, y in data_eval], dtype=float)


def count_accesses(query_ranges, segments):
    segs = np.array(segments, dtype=float)
    counts = np.zeros(len(segs), dtype=np.int64)
    for qs, qe in query_ranges:
        for i, (ls, le) in enumerate(segs):
            if not (qe < ls or qs > le):
                counts[i] += 1
    return counts


def summarize_segment_sizes(segments, width=50):
    sizes = [(r - l + 1) / width for (l, r) in segments]
    cnt = pd.Series(sizes).value_counts().sort_index()
    return pd.DataFrame({"节点大小": cnt.index.astype(int), "节点个数": cnt.values})


def main():
    x, y = generate_xy(target_count=2_000_000, seed=1)
    x_shuf = shuffle_integer_parts(x, seed=1)
    df = pd.DataFrame({"col1": np.round(x_shuf), "col2": np.round(y)})
    print(f"rows={len(df)}")

    data_train, data_eval = split_train_eval(x_shuf, y, eval_size=1_000_000, seed=1)
    print("train shape:", data_train.shape, "eval shape:", data_eval.shape)

    labels, core_mask = run_dbscan(data_train, eps=40, min_samples=150)
    bounds_dict = cluster_boundaries_via_diff(data_train, labels)
    cluster_intv = list(bounds_dict.values())
    merged_bounds = merge_intervals(cluster_intv)
    new_labels = relabel_by_merged(labels, data_train[:, 0], data_train[:, 1], merged_bounds)


    grid = fixed_grid_intervals(lo=1, hi=1_000_000, width=50)
    merged_segments = cover_grid_with_clusters(grid, merged_bounds)

    q_ranges = eval_ranges(data_eval)
    counts_merged = count_accesses(q_ranges, merged_segments)
    counts_orig = count_accesses(q_ranges, grid)

    print("total accesses (original):", int(counts_orig.sum()))
    print("total accesses (merged)  :", int(counts_merged.sum()))
    if counts_orig.sum() > 0:
        saving = (counts_orig.sum() - counts_merged.sum()) / counts_orig.sum() * 100.0
        print(f"saving (%): {saving:.2f}")

    size_table = summarize_segment_sizes(merged_segments, width=50)
    print("\nMerged segment size distribution:")
    print(size_table.to_string(index=False))

    merged_df = pd.DataFrame({"区间": merged_segments, "区间大小": [(r - l + 1) / 50 for (l, r) in merged_segments]})
    print("\nMerged segments (head):")
    print(merged_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
