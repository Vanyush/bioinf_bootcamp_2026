import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Set
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass


@dataclass
class HaplotypeCluster:
    index: int
    label: str
    sequence: str
    member_indices: List[int]
    member_headers: List[str]
    observed_ages: List[Optional[float]]
    
    @property
    def count(self) -> int:
        return len(self.member_indices)
    
    @property
    def observed_age_bp(self) -> Optional[float]:
        ages = [a for a in self.observed_ages if a is not None]
        return float(np.median(ages)) if ages else None

def parse_fasta_data(file: str) -> pd.DataFrame:
    names = []
    seq = []
    year = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                names.append(line[1:])
                parts = line[1:].split('_')
                if parts and parts[-1].isdigit():
                    year.append(int(parts[-1]))
                else:
                    year.append(None)
            else:
                seq.append(line)
    data = {'Name': names, 'Year': year, 'Sequence': seq}
    df = pd.DataFrame(data)
    return df

def compute_distances(sequences: List[str]) -> np.ndarray:
    n = len(sequences)
    dist = np.zeros((n, n), dtype=int)
    seqs = [list(s.upper()) for s in sequences]

    for i in range(n):
        for j in range(i+1, n):
            diff = 0
            comparable = 0
            a = seqs[i]
            b = seqs[j]
            min_len = min(len(a), len(b))
            for k in range(min_len):
                x = a[k]
                y = b[k]
                if x == '-' or y == '-' or x == 'N' or y == 'N':
                    continue
                comparable += 1
                if x != y:
                    diff += 1
            if comparable == 0:
                diff = abs(len(a) - len(b))
            dist[i, j] = diff
            dist[j, i] = diff
    return dist


def build_haplotype_clusters(records: pd.DataFrame) -> List[HaplotypeCluster]:
    grouped = {}
    for idx, row in records.iterrows():
        seq = row['Sequence']
        if seq not in grouped:
            grouped[seq] = []
        grouped[seq].append((idx, row['Name'], row['Year']))
    
    clusters = []
    for cluster_idx, (seq, items) in enumerate(sorted(grouped.items())):
        member_indices = sorted([item[0] for item in items])
        member_headers = [item[1] for item in items]
        observed_ages = [item[2] for item in items if pd.notna(item[2])]
        
        clusters.append(HaplotypeCluster(
            index=cluster_idx,
            label=f"HG{cluster_idx + 1}",
            sequence=seq,
            member_indices=member_indices,
            member_headers=member_headers,
            observed_ages=observed_ages
        ))
    
    return clusters


def minimum_spanning_tree(distances: np.ndarray) -> List[Tuple[int, int, int]]:
    n = distances.shape[0]
    if n <= 1:
        return []
    in_tree = [False] * n
    best_dist = [float('inf')] * n
    parent = [-1] * n
    best_dist[0] = 0

    for _ in range(n):
        u = -1
        min_val = float('inf')
        for i in range(n):
            if not in_tree[i] and best_dist[i] < min_val:
                min_val = best_dist[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True

        for v in range(n):
            if not in_tree[v]:
                d = distances[u, v]
                if d < best_dist[v]:
                    best_dist[v] = d
                    parent[v] = u

    edges = []
    for v in range(1, n):
        if parent[v] != -1:
            edges.append((parent[v], v, distances[parent[v], v]))
    return edges


def build_haplotype_network(distances: np.ndarray) -> List[Tuple[int, int, int]]:
    n = distances.shape[0]
    if n <= 1:
        return []
    
    edges = []
    seen = set()
    
    def add_edge(a: int, b: int):
        x, y = min(a, b), max(a, b)
        key = f"{x}-{y}"
        if key not in seen:
            seen.add(key)
            edges.append((x, y, distances[x, y]))
    
    non_zero = []
    for i in range(n):
        for j in range(i + 1, n):
            d = distances[i, j]
            if d > 0:
                non_zero.append(d)
    
    non_zero.sort()
    q1 = non_zero[len(non_zero) // 4] if non_zero else 1
    threshold = max(1, q1)
    for i in range(n): 
        row_min = float('inf')
        for j in range(n):
            if j != i:
                d = distances[i, j]
                if d > 0 and d < row_min:
                    row_min = d
        
        for j in range(i + 1, n):
            d = distances[i, j]
            if d > 0 and (d <= threshold or d == row_min):
                add_edge(i, j)
  
    in_tree = [False] * n
    best_dist = [float('inf')] * n
    parent = [-1] * n
    best_dist[0] = 0
    
    for _ in range(n):
        u = -1
        min_val = float('inf')
        for i in range(n):
            if not in_tree[i] and best_dist[i] < min_val:
                min_val = best_dist[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        
        for v in range(n):
            if not in_tree[v]:
                d = distances[u, v]
                if d < best_dist[v]:
                    best_dist[v] = d
                    parent[v] = u
    
    for v in range(1, n):
        if parent[v] != -1:
            add_edge(parent[v], v)
   
    edges.sort(key=lambda e: (e[2], e[0], e[1]))
    return edges


def weighted_age_prior(index: int,
                       records: pd.DataFrame,
                       distances: np.ndarray,
                       observed_indices: Set[int]) -> float:
    sorted_obs = sorted(observed_indices, key=lambda j: distances[index, j])
    nearest = sorted_obs[:5]
    pairs = []
    for j in nearest:
        age = records.loc[j, 'Year']
        if age is None:
            continue
        d = max(1, distances[index, j])
        w = 1.0 / (d * d)
        pairs.append((age, w))
    if not pairs:
        ages = records.loc[list(observed_indices), 'Year'].dropna().tolist()
        return float(np.median(ages)) if ages else 0.0
    sum_w = sum(w for _, w in pairs)
    sum_val = sum(age * w for age, w in pairs)
    return sum_val / max(sum_w, 1e-9)


def infer_ages(records: pd.DataFrame,
               distances: np.ndarray,
               mst_edges: List[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:

    n = len(records)
    adjacency = [[] for _ in range(n)]
    for a, b, d in mst_edges:
        w = 1.0 / (d + 1)
        adjacency[a].append((b, w))
        adjacency[b].append((a, w))

    observed_indices = set(records[records['Year'].notna()].index)
    ages = np.full(n, np.nan, dtype=float)

    for idx in observed_indices:
        ages[idx] = float(records.loc[idx, 'Year'])

    if not observed_indices:
        print("Предупреждение: ни одна последовательность не имеет возраста. "
              "Возраст будет приближён к генетической дистанции от первого образца.")
        pseudo_root = 0
        scale = 1000.0
        for i in range(n):
            ages[i] = float(distances[pseudo_root, i]) * scale
    else:
        for i in range(n):
            if i not in observed_indices:
                vals = []
                for j in observed_indices:
                    d = max(1, distances[i, j])
                    w = 1.0 / (d * d)
                    vals.append((records.loc[j, 'Year'], w))
                if vals:
                    sum_w = sum(w for _, w in vals)
                    sum_val = sum(age * w for age, w in vals)
                    ages[i] = sum_val / max(sum_w, 1e-9)
                else:
                    ages[i] = 0.0

        iterations = 250
        for _ in range(iterations):
            next_ages = ages.copy()
            for i in range(n):
                if i in observed_indices:
                    continue
                neigh = adjacency[i]
                if neigh:
                    sum_w = sum(w for _, w in neigh)
                    sum_val = sum(w * ages[nei] for nei, w in neigh)
                    graph_val = sum_val / max(sum_w, 1e-9)
                else:
                    graph_val = ages[i] 
                prior = weighted_age_prior(i, records, distances, observed_indices)
                next_ages[i] = 0.70 * graph_val + 0.30 * prior
            ages = next_ages

    confidences = np.ones(n, dtype=float)
    for i in range(n):
        if i in observed_indices:
            confidences[i] = 1.0
        else:
            if observed_indices:
                nearest_dist = min(distances[i, j] for j in observed_indices)
            else:
                nearest_dist = 0.0
            c1 = 1.0 / (1.0 + nearest_dist)
            neigh = adjacency[i]
            if neigh:
                local_spread = np.mean([abs(ages[nei] - ages[i]) for nei, _ in neigh])
            else:
                local_spread = 0.0
            c2 = 1.0 / (1.0 + local_spread / 1000.0)
            confidence = 0.55 * c1 + 0.45 * c2
            confidence = max(0.05, min(0.98, confidence))
            confidences[i] = confidence

    return ages, confidences


def infer_ages_clusters(clusters: List[HaplotypeCluster],
                        distances: np.ndarray,
                        network_edges: List[Tuple[int, int, int]],
                        warnings: List[str]) -> Tuple[np.ndarray, np.ndarray]:
 
    n = len(clusters)
    if n == 0:
        return np.array([]), np.array([])
    
    # Строим взвешенную смежность
    adjacency = [[] for _ in range(n)]
    for a, b, d in network_edges:
        w = 1.0 / (d + 1)
        adjacency[a].append((b, w))
        adjacency[b].append((a, w))
    
    observed_indices = set()
    for idx, cluster in enumerate(clusters):
        if cluster.observed_age_bp is not None:
            observed_indices.add(idx)
    
    ages = np.full(n, np.nan, dtype=float)
    for idx in observed_indices:
        ages[idx] = float(clusters[idx].observed_age_bp)
    
    if not observed_indices:
        warnings.append("В FASTA не найдено ни одной датированной последовательности. Возраст будет приблизительно привязан к генетическому сходству")
        root = 0
        scale = 1000.0
        for i in range(n):
            ages[i] = float(distances[root, i]) * scale
    else:
        for i in range(n):
            if i not in observed_indices:
                vals = []
                for j in observed_indices:
                    d = max(1, distances[i, j])
                    age = clusters[j].observed_age_bp
                    if age is not None:
                        w = 1.0 / (d * d)
                        vals.append((age, w))
                if vals:
                    sum_w = sum(w for _, w in vals)
                    sum_val = sum(age * w for age, w in vals)
                    ages[i] = sum_val / max(sum_w, 1e-9)
                else:
                    ages_all = [c.observed_age_bp for c in clusters if c.observed_age_bp is not None]
                    ages[i] = float(np.median(ages_all)) if ages_all else 0.0
        
        iterations = 250
        for _ in range(iterations):
            next_ages = ages.copy()
            for i in range(n):
                if i in observed_indices:
                    continue
                neighbors = adjacency[i]
                if neighbors:
                    sum_w = sum(w for _, w in neighbors)
                    sum_val = sum(w * ages[nei] for nei, w in neighbors)
                    graph_val = sum_val / max(sum_w, 1e-9)
                else:
                    continue
                
                prior = weighted_age_prior_clusters(i, clusters, distances, observed_indices)
                next_ages[i] = 0.70 * graph_val + 0.30 * prior
            ages = next_ages
    
    confidences = np.ones(n, dtype=float)
    for i in range(n):
        if i in observed_indices:
            confidences[i] = 1.0
        else:
            if observed_indices:
                nearest_dist = min(distances[i, j] for j in observed_indices)
            else:
                nearest_dist = 0.0
            c1 = 1.0 / (1.0 + nearest_dist)
            neighbors = adjacency[i]
            if neighbors:
                local_spread = np.mean([abs(ages[nei] - ages[i]) for nei, _ in neighbors])
            else:
                local_spread = 0.0
            c2 = 1.0 / (1.0 + local_spread / 1000.0)
            confidence = 0.55 * c1 + 0.45 * c2
            confidence = max(0.05, min(0.98, confidence))
            confidences[i] = confidence
    
    return ages, confidences


def weighted_age_prior_clusters(index: int,
                                clusters: List[HaplotypeCluster],
                                distances: np.ndarray,
                                observed_indices: Set[int]) -> float:
    sorted_obs = sorted(observed_indices, key=lambda j: distances[index, j])
    nearest = sorted_obs[:5]
    pairs = []
    for j in nearest:
        age = clusters[j].observed_age_bp
        if age is None:
            continue
        d = max(1, distances[index, j])
        w = 1.0 / (d * d)
        pairs.append((age, w))
    
    if not pairs:
        ages = [c.observed_age_bp for c in clusters if c.observed_age_bp is not None]
        return float(np.median(ages)) if ages else 0.0
    
    sum_w = sum(w for _, w in pairs)
    sum_val = sum(age * w for age, w in pairs)
    return sum_val / max(sum_w, 1e-9)


def assign_haplogroups(sequences: List[str]) -> List[str]:
    unique_to_haplo = {}
    next_label = 1 
    haplogroups = []
    for seq in sequences:
        if seq not in unique_to_haplo:
            unique_to_haplo[seq] = f'HG{next_label}'
            next_label += 1
        haplogroups.append(unique_to_haplo[seq])
    return haplogroups


def visualize_haplotype_network(clusters: List[HaplotypeCluster],
                                network_edges: List[Tuple[int, int, int]],
                                distances: np.ndarray,
                                ages_clusters: np.ndarray,
                                confidences_clusters: np.ndarray,
                                output_png: str = 'haplotype_network.png'):
    n = len(clusters)
    if n == 0:
        print("Нет данных для визуализации")
        return
    
    pos = {}
    y_min, y_max = 1, 3
    
    sorted_indices = sorted(range(n), key=lambda i: ages_clusters[i], reverse=True)
    
    for rank, idx in enumerate(sorted_indices):
        age = ages_clusters[idx]
        hash_val = abs(hash(clusters[idx].label)) % 97
        y = y_max - (rank / max(1, n-1)) * (y_max - y_min)
        pos[idx] = (age, y)
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, d in network_edges:
        G.add_edge(a, b, weight=d)
    
    node_colors = []
    for idx in range(n):
        if clusters[idx].observed_age_bp is not None:
            node_colors.append('#1f77b4')  # синий
        else:
            node_colors.append('#ff7f0e')  # оранжевый
    
    node_sizes = [max(300, min(1500, clusters[i].count * 80)) for i in range(n)]
    
    plt.figure(figsize=(16, 10))
    
    for a, b, d in network_edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(a, b)], width=1.5, 
                              edge_color='gray', alpha=0.6, arrows=False)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          edgecolors='white', linewidths=2, alpha=0.9)
    
    labels = {}
    for i in range(n):
        age_text = f"{ages_clusters[i]:.0f}"
        labels[i] = f"{clusters[i].label}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    min_age, max_age = min(ages_clusters), max(ages_clusters)
    if min_age == max_age:
        min_age, max_age = min_age - 1, max_age + 1
    
    plt.xlabel('Возраст, лет назад', 
               fontsize=12, fontweight='bold')
    plt.gca().set_ylabel('')
    plt.gca().set_yticklabels([])
    plt.title('Временная сеть гаплогрупп', 
              fontsize=16, fontweight='bold')
    
    ax = plt.gca()
    ax.set_xlim(max_age + (max_age - min_age) * 0.05, min_age - (max_age - min_age) * 0.05)
    
    x_ticks = np.linspace(max_age, min_age, 6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(x)}" for x in x_ticks])
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='white', label='Датированные гаплогруппы'),
        Patch(facecolor='#ff7f0e', edgecolor='white', label='Прогнозные гаплогруппы'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'График сохранён в {output_png}')


def run_analysis(fasta_path: str, output_csv: Optional[str] = None):
    df = parse_fasta_data(fasta_path)
    if df.empty:
        print('Файл не содержит последовательностей.')
        return
    
    sequences = df['Sequence'].tolist()
    n_original = len(df)
    
    print(f'Загружено {n_original} последовательностей.')
    dated_count_original = df['Year'].notna().sum()
    print(f'Датированных: {dated_count_original}, недатированных: {n_original - dated_count_original}')
   
    lengths = set(df['Sequence'].str.len())
    if len(lengths) != 1:
        print("Предупреждение: последовательности имеют разную длину. Для алгоритма желательно выравнивание.")
  
    print("Формирование гаплогрупп (объединение идентичных последовательностей)...")
    clusters = build_haplotype_clusters(df)
    print(f"Выделено {len(clusters)} уникальных гаплогрупп")
    
    if len(clusters) < n_original:
        print(f"Объединено {n_original - len(clusters)} дублирующихся последовательностей")
   
    cluster_records = []
    for cluster in clusters:
        cluster_records.append({
            'Name': f"{cluster.label} (n={cluster.count})",
            'Year': cluster.observed_age_bp,
            'Sequence': cluster.sequence,
            'Count': cluster.count,
            'MemberHeaders': ', '.join(cluster.member_headers[:3]) + ('...' if len(cluster.member_headers) > 3 else '')
        })
    cluster_df = pd.DataFrame(cluster_records)
    
    print("Вычисление попарных расстояний между гаплогруппами...")
    cluster_sequences = cluster_df['Sequence'].tolist()
    distances = compute_distances(cluster_sequences)
    result_distances = compute_distances(sequences)

    
    print("Построение сети гаплогрупп...")
    network_edges = build_haplotype_network(distances)
    mst_edges = minimum_spanning_tree(distances)
    print(f"Построено {len(network_edges)} рёбер сети")
  
    print("Инференция возрастов...")
    warnings = []
    ages, confidences = infer_ages_clusters(clusters, distances, network_edges, warnings)
    res_ages, res_confidences = infer_ages(df, result_distances, mst_edges)
    result_df = pd.DataFrame()
    result_df['index'] = range(n_original)
    result_df['header'] = df['Name']
    result_df['observed_age_bp'] = df['Year'].apply(lambda x: '' if pd.isna(x) else str(int(x)))
    inferred_formatted = []
    for i in range(n_original):
        inferred_formatted.append(f"{res_ages[i]:.2f}")
    result_df['inferred_age_bp'] = inferred_formatted
    result_df['inferred_age_bp'] = pd.to_numeric(result_df['inferred_age_bp'], errors='coerce')
    result_df['confidence'] = pd.Series(res_confidences).apply(lambda x: f"{x:.3f}")
    result_df['sequence_length'] = df['Sequence'].str.len()
    haplo_labels = assign_haplogroups(df['Sequence'].tolist())
    result_df['haplogroup'] = haplo_labels
    result_df = result_df.sort_values('inferred_age_bp', ascending=True).reset_index(drop=True)

    if output_csv is None:
        output_csv = fasta_path.replace('.fasta', '_haplogroups.csv').replace('.fa', '_haplogroups.csv')
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f'Результаты сохранены в {output_csv}')

    try:
        visualize_haplotype_network(clusters, network_edges, distances, ages, confidences,
                              output_png=output_csv.replace('.csv', '_network.png'))
    except Exception as e:
        print(f"Визуализация не выполнена: {e}")

    dated_clusters = sum(1 for c in clusters if c.observed_age_bp is not None)
    print(f"Итог: {len(clusters)} гаплогрупп, из них датированных: {dated_clusters}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Использование: python script.py <path_to_fasta> [output_csv]')
        sys.exit(1)
    fasta_file = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else None
    run_analysis(fasta_file, out_csv)