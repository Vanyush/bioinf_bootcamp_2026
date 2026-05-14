import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass
import plotly.express as px


class HaplotypeCluster:
    index: int
    label: str
    sequence: str
    member_indices: List[int]
    member_headers: List[str]
    observed_ages: List[Optional[float]]
    inferred_age: float = 0.0
    inferred_confidence: float = 0.0
    
    @property
    def count(self) -> int:
        return len(self.member_indices)
    
    @property
    def observed_age_bp(self) -> Optional[float]:
        ages = [a for a in self.observed_ages if a is not None]
        return ages[0] if ages else None


def parse_fasta_data(file) -> pd.DataFrame:
    names = []
    seq = []
    year = []
    
    if hasattr(file, 'read'):
        content = file.read().decode('utf-8')
        lines = content.split('\n')
    else:
        with open(file) as f:
            lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            names.append(line[1:])
            parts = line[1:].split('_')
            if parts and parts[-1].isdigit():
                year.append(int(parts[-1]))
            else:
                year.append(None)
        elif line:
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

def infer_ages_clusters(clusters: List[HaplotypeCluster],
                        distances: np.ndarray,
                        network_edges: List[Tuple[int, int, int]],
                        warnings: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(clusters)
    if n == 0:
        return np.array([]), np.array([])
    
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
                    graph_val = ages[i]
                
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

# ----------------------------------------------------------------------
# GUI на Streamlit
# ----------------------------------------------------------------------
def create_network_plot(clusters: List[HaplotypeCluster], 
                        network_edges: List[Tuple[int, int, int]], 
                        ages_clusters: np.ndarray,
                        selected_node: Optional[int] = None):
    """Создание интерактивного графа с помощью Plotly (по убыванию возраста)"""
    
    G = nx.Graph()
    for i, cluster in enumerate(clusters):
        G.add_node(i, 
                   label=cluster.label,
                   count=cluster.count,
                   age=ages_clusters[i],
                   observed=cluster.observed_age_bp is not None,
                   confidence=cluster.inferred_confidence if hasattr(cluster, 'inferred_confidence') else 0.0)
    
    for a, b, d in network_edges:
        G.add_edge(a, b, weight=d)
    
    pos = {}
    height = 10
    y_min, y_max = 1, height - 1
    
    sorted_indices = sorted(range(len(clusters)), key=lambda i: ages_clusters[i], reverse=True)
    
    for rank, idx in enumerate(sorted_indices):
        age = ages_clusters[idx]
        base_y = y_max - (rank / max(1, len(clusters)-1)) * (y_max - y_min)
        hash_val = abs(hash(clusters[idx].label)) % 97
        jitter = (hash_val / 97.0 - 0.5) * 1.2
        y = max(y_min, min(base_y + jitter, y_max))
        pos[idx] = (age, y)
  
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        cluster = clusters[node]
        is_observed = cluster.observed_age_bp is not None
        node_colors.append('#1f77b4' if is_observed else '#ff7f0e')
        node_sizes.append(max(20, min(50, cluster.count * 2)))
        
        age_text = f"{ages_clusters[node]:.0f}"
        node_text.append(f"{cluster.label}<br>n={cluster.count}<br>{age_text} лет")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[clusters[i].label for i in range(len(clusters))],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text="Временная сеть гаплогрупп",
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(
                            title="Возраст, лет назад (BP)",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='LightGrey',
                            # Реверсируем ось X, чтобы древние были слева
                            autorange='reversed'
                        ),
                        yaxis=dict(
                            showticklabels=False,
                            showgrid=False,
                            title=""
                        ),
                        height=700,
                        clickmode='event+select'
                    ))
    
    if selected_node is not None:
        fig.add_trace(go.Scatter(
            x=[node_x[selected_node]],
            y=[node_y[selected_node]],
            mode='markers',
            marker=dict(size=node_sizes[selected_node] + 8, 
                       color='red', 
                       line=dict(width=3, color='white')),
            hoverinfo='none',
            showlegend=False
        ))
    
    return fig

def main():
    st.set_page_config(layout="wide", page_title="MMNV - Временная сеть гаплогрупп")
    
    st.title("MMNV")
    st.caption("Тестовое приложение для хакатона 'биоинф буткемп' | Команда: Бригада 9")
    
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    
    # Sidebar
    with st.sidebar:
        st.header("Управление")
        
        uploaded_file = st.file_uploader(
            "Открыть FASTA", 
            type=['fasta', 'fa', 'txt'],
            help="Выберите FASTA файл с выравненными последовательностями"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            load_button = st.button("📂 Открыть FASTA", use_container_width=True)
        with col2:
            analyze_button = st.button("🔄 Переанализировать", use_container_width=True, 
                                      disabled=not st.session_state.analysis_done)
        
        if uploaded_file is not None and load_button:
            with st.spinner("Загрузка и анализ файла..."):
                df = parse_fasta_data(uploaded_file)
                
                if df.empty:
                    st.error("Файл не содержит последовательностей")
                else:
                    st.session_state.df = df
                    st.session_state.filename = uploaded_file.name
                    
                    sequences = df['Sequence'].tolist()
                    
                    clusters = build_haplotype_clusters(df)
                    cluster_sequences = [c.sequence for c in clusters]
                    
                    distances = compute_distances(cluster_sequences)
                    network_edges = build_haplotype_network(distances)
                    
                    warnings = []
                    ages_clusters, confidences_clusters = infer_ages_clusters(
                        clusters, distances, network_edges, warnings
                    )
                    
                    for idx, cluster in enumerate(clusters):
                        cluster.inferred_age = ages_clusters[idx]
                        cluster.inferred_confidence = confidences_clusters[idx]
                    
                    result_rows = []
                    for cluster_idx, cluster in enumerate(clusters):
                        for member_idx in cluster.member_indices:
                            original_row = df.iloc[member_idx]
                            result_rows.append({
                                'index': member_idx,
                                'header': original_row['Name'],
                                'observed_age_bp': '' if pd.isna(original_row['Year']) else str(int(original_row['Year'])),
                                'inferred_age_bp': f"{ages_clusters[cluster_idx]:.2f}",
                                'confidence': f"{confidences_clusters[cluster_idx]:.3f}",
                                'sequence_length': len(cluster.sequence),
                                'haplogroup': cluster.label
                            })
                    
                    result_df = pd.DataFrame(result_rows)
                    result_df['inferred_age_bp_numeric'] = pd.to_numeric(result_df['inferred_age_bp'])
                    result_df = result_df.sort_values('inferred_age_bp_numeric', ascending=True).reset_index(drop=True)
                    result_df.drop('inferred_age_bp_numeric', axis=1, inplace=True)
                    
                    st.session_state.clusters = clusters
                    st.session_state.network_edges = network_edges
                    st.session_state.ages_clusters = ages_clusters
                    st.session_state.confidences_clusters = confidences_clusters
                    st.session_state.result_df = result_df
                    st.session_state.warnings = warnings
                    st.session_state.analysis_done = True
                    st.session_state.selected_node = None
                    
                    st.success(f"Загружено {len(df)} последовательностей, выделено {len(clusters)} гаплогрупп")
                    st.info(f"Файл: {uploaded_file.name}")
        
        if st.session_state.analysis_done:
            st.divider()
            st.subheader("Статус")
            st.text(f"Файл: {st.session_state.filename}")
            st.text(f"Последовательностей: {len(st.session_state.df)}")
            st.text(f"Гаплогрупп: {len(st.session_state.clusters)}")
            dated = sum(1 for c in st.session_state.clusters if c.observed_age_bp is not None)
            st.text(f"Датированных: {dated}")
            st.text(f"Прогнозных: {len(st.session_state.clusters) - dated}")
            
            if st.session_state.warnings:
                st.divider()
                st.warning("Предупреждения")
                for w in st.session_state.warnings:
                    st.caption(f"• {w}")
 
    if st.session_state.analysis_done:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader("Временная сеть гаплогрупп")
            st.caption("🔵 Голубые узлы — датированные образцы | 🟠 Оранжевые — прогнозные")
        with col3:
            csv_data = st.session_state.result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Экспорт CSV",
                data=csv_data,
                file_name=f"{st.session_state.filename}_results.csv",
                mime="text/csv",
                use_container_width=True
            )
       
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            fig = create_network_plot(
                st.session_state.clusters,
                st.session_state.network_edges,
                st.session_state.ages_clusters,
                st.session_state.selected_node
            )
            selected_points = st.plotly_chart(fig, use_container_width=True, key="network_plot")
        
        with right_col:
            st.subheader("Гаплогруппы")
            
            display_nodes = []
            for i, cluster in enumerate(st.session_state.clusters):
                display_nodes.append({
                    'index': i,
                    'haplogroup': cluster.label,
                    'members': cluster.count,
                    'observed_age': f"{int(cluster.observed_age_bp)}" if cluster.observed_age_bp else "",
                    'inferred_age': f"{st.session_state.ages_clusters[i]:.0f}",
                    'confidence': f"{st.session_state.confidences_clusters[i]:.3f}",
                    'type': "observed" if cluster.observed_age_bp else "predicted"
                })
            
            display_df = pd.DataFrame(display_nodes)
            display_df = display_df.sort_values('inferred_age', ascending=False)

            haplo_options = {row['haplogroup']: row['index'] for _, row in display_df.iterrows()}
            selected_haplo = st.selectbox(
                "Выберите гаплогруппу",
                options=list(haplo_options.keys()),
                format_func=lambda x: f"{x} ({display_df[display_df['haplogroup']==x]['members'].values[0]} образцов)"
            )
            
            if selected_haplo:
                idx = haplo_options[selected_haplo]
                cluster = st.session_state.clusters[idx]
                st.session_state.selected_node = idx
               
                with st.expander("Детальная информация", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Гаплогруппа", cluster.label)
                        st.metric("Количество образцов", cluster.count)
                    with col_b:
                        if cluster.observed_age_bp:
                            st.metric("Датированный возраст", f"{int(cluster.observed_age_bp)} лет")
                        else:
                            st.metric("Предсказанный возраст", f"{st.session_state.ages_clusters[idx]:.0f} лет")
                        st.metric("Confidence", f"{st.session_state.confidences_clusters[idx]:.3f}")
            
                st.subheader("Образцы")
                for header in cluster.member_headers[:10]:
                    st.text(f"• {header}")
                if len(cluster.member_headers) > 10:
                    st.caption(f"... и ещё {len(cluster.member_headers) - 10} образцов")

        st.divider()
        st.subheader("Таблица последовательностей")
        st.dataframe(
            st.session_state.result_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "index": "№",
                "header": "Заголовок",
                "observed_age_bp": "Наблюдаемый возраст",
                "inferred_age_bp": "Предсказанный возраст",
                "confidence": "Confidence",
                "sequence_length": "Длина",
                "haplogroup": "Гаплогруппа"
            }
        )
        
        if selected_haplo:
            st.rerun()
    
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info(
                """
                ### Импортируйте FASTA-файл, чтобы построить временную сеть
                
                Приложение рассчитано на выравненные митохондриальные последовательности ДНК 
                и умеет прогнозировать возраст недатированных образцов по генетическому сходству.
                
                **Требования к файлу:**
                - Формат FASTA
                - Выравненные последовательности
                - В заголовке возраст (число после последнего символа '_')
                """
            )

if __name__ == "__main__":
    main()