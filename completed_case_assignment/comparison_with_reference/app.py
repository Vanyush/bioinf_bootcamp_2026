#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение предсказаний гаплогрупп с референсом.
Автоматическое сопоставление названий + визуализация (PNG).
Запуск: python app.py
"""

import os
import csv
from collections import Counter, defaultdict
from itertools import combinations
import math
import random
import re

# ============================================================
# НАСТРОЙКИ
# ============================================================
AUTO_MAPPING = True          # автоматически подбирать соответствия гаплогрупп
MAPPING_THRESHOLD = 0.8      # минимальная доля образцов для надёжного соответствия (0.8 = 80%)
NORMALIZE_HG = True          # HG9 -> H9

# ============================================================
# ПОПЫТКА ИМПОРТА БИБЛИОТЕК ДЛЯ ГРАФИКОВ
# ============================================================
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
    # Установим стиль seaborn
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
except ImportError:
    HAS_VIZ = False
    print("⚠️ Библиотеки matplotlib/seaborn не установлены. Графики созданы не будут.")
    print("   Чтобы установить: pip install matplotlib seaborn")

# ============================================================
# 1. ПРЕОБРАЗОВАНИЕ НАЗВАНИЙ
# ============================================================
def normalize_haplo(h):
    if NORMALIZE_HG:
        h = re.sub(r'^HG(\d+)$', r'H\1', h)
    return h

# ============================================================
# 2. ЧТЕНИЕ TSV
# ============================================================
def read_haplo_tsv(filepath, mapping=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)
        has_header = any(k in first_line.lower() for k in ['member', 'sample', 'accession', 'id', 'hap'])
        reader = csv.reader(f, delimiter='\t')
        if has_header:
            header = next(reader)
            idx_sample = None
            idx_haplo = None
            for i, col in enumerate(header):
                col_low = col.lower()
                if any(k in col_low for k in ['member', 'sample', 'accession', 'id', 'header']):
                    idx_sample = i
                if any(k in col_low for k in ['hap', 'haplo', 'group']):
                    idx_haplo = i
            if idx_sample is None:
                idx_sample = 0
            if idx_haplo is None:
                idx_haplo = 1
        else:
            idx_sample = 0
            idx_haplo = 1
            reader = csv.reader(f, delimiter='\t')
        
        data = {}
        for row in reader:
            if len(row) <= max(idx_sample, idx_haplo):
                continue
            sample = row[idx_sample].strip()
            haplo = row[idx_haplo].strip()
            if sample and haplo:
                haplo = normalize_haplo(haplo)
                if mapping:
                    haplo = mapping.get(haplo, haplo)
                data[sample] = haplo
        return data

# ============================================================
# 3. СТАТИСТИЧЕСКИЕ ФУНКЦИИ
# ============================================================
def cohen_kappa(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    observed = sum(1 for a,b in zip(y_true, y_pred) if a==b) / n
    freq_true = Counter(y_true)
    freq_pred = Counter(y_pred)
    expected = sum((freq_true[c]/n)*(freq_pred[c]/n) for c in classes)
    if expected == 1:
        return 1.0
    return (observed - expected)/(1 - expected)

def bootstrap_kappa(y_true, y_pred, n_bootstrap=1000):
    n = len(y_true)
    kappas = []
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n-1) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        kappas.append(cohen_kappa(yt, yp))
    kappas.sort()
    lower = kappas[int(0.025 * n_bootstrap)]
    upper = kappas[int(0.975 * n_bootstrap)]
    return sum(kappas)/n_bootstrap, lower, upper

def f1_binary(tp, fp, fn):
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec = tp/(tp+fn) if tp+fn>0 else 0
    return 2*prec*rec/(prec+rec) if prec+rec>0 else 0

def macro_f1(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    f1s = []
    for c in classes:
        tp = sum(1 for a,b in zip(y_true,y_pred) if a==c and b==c)
        fp = sum(1 for a,b in zip(y_true,y_pred) if a!=c and b==c)
        fn = sum(1 for a,b in zip(y_true,y_pred) if a==c and b!=c)
        f1s.append(f1_binary(tp,fp,fn))
    return sum(f1s)/len(f1s)

def micro_f1(y_true, y_pred):
    return sum(1 for a,b in zip(y_true,y_pred) if a==b)/len(y_true)

def fleiss_kappa(ratings):
    n_subj = len(ratings)
    n_raters = len(ratings[0])
    cats = sorted(set(c for subj in ratings for c in subj))
    n_cat = len(cats)
    mat = [[0]*n_cat for _ in range(n_subj)]
    for i,subj in enumerate(ratings):
        for c in subj:
            mat[i][cats.index(c)] += 1
    p_i = [sum(r*(r-1) for r in row)/(n_raters*(n_raters-1)) for row in mat]
    P_bar = sum(p_i)/n_subj
    p_c = [sum(mat[i][j] for i in range(n_subj))/(n_subj*n_raters) for j in range(n_cat)]
    P_e = sum(p**2 for p in p_c)
    return (P_bar - P_e)/(1 - P_e) if P_e != 1 else 1.0

def mcnemar_test(only_first, only_second):
    b, c = only_first, only_second
    if b+c == 0:
        return 1.0
    from math import comb
    p = sum(comb(b+c, i)*(0.5)**(b+c) for i in range(min(b,c), b+c+1))
    return 2*min(p, 1-p)

# ============================================================
# 4. АВТОМАТИЧЕСКОЕ ПОСТРОЕНИЕ СООТВЕТСТВИЙ
# ============================================================
def build_mapping_from_confusion(confusion_dict, threshold=0.8):
    mapping = {}
    all_pred = set()
    for true_counts in confusion_dict.values():
        all_pred.update(true_counts.keys())
    for pred in all_pred:
        best_true = None
        best_count = 0
        total_for_pred = sum(true_counts.get(pred,0) for true_counts in confusion_dict.values())
        if total_for_pred == 0:
            continue
        for true, counts in confusion_dict.items():
            cnt = counts.get(pred, 0)
            if cnt > best_count:
                best_count = cnt
                best_true = true
        if best_count / total_for_pred >= threshold:
            mapping[pred] = best_true
    return mapping

def print_confusion_table(confusion_dict, true_groups, pred_groups):
    true_list = sorted(true_groups)
    pred_list = sorted(pred_groups)
    print("\nТекущая матрица ошибок (reference -> алгоритм):")
    print(" " * 12, end='')
    for p in pred_list:
        print(f"{p:>8}", end='')
    print()
    for t in true_list:
        print(f"{t:<12}", end='')
        for p in pred_list:
            cnt = confusion_dict[t].get(p, 0)
            print(f"{cnt:8d}", end='')
        print()
    print()

# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ (тепловые карты, сравнение метрик)
# ============================================================
def save_confusion_matrix_heatmap(confusion_dict, true_groups, pred_groups, alg_name, output_dir="."):
    if not HAS_VIZ:
        return
    true_list = sorted(true_groups)
    pred_list = sorted(pred_groups)
    # Строим матрицу в виде списка списков
    cm = [[confusion_dict[t].get(p,0) for p in pred_list] for t in true_list]
    plt.figure(figsize=(max(6, len(pred_list)*0.5), max(5, len(true_list)*0.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=pred_list, yticklabels=true_list,
                cbar_kws={'label': 'Количество образцов'})
    plt.title(f'Confusion matrix: {alg_name} vs reference', fontsize=12)
    plt.ylabel('Reference (истина)')
    plt.xlabel(f'{alg_name} (предсказание)')
    plt.tight_layout()
    filename = os.path.join(output_dir, f'confusion_{alg_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Сохранена тепловая карта: {filename}")

def save_metrics_barplot(results, output_dir="."):
    if not HAS_VIZ or not results:
        return
    alg_names = list(results.keys())
    acc = [results[n]['accuracy'] for n in alg_names]
    kappa = [results[n]['kappa'] for n in alg_names]
    f1 = [results[n]['f1_macro'] for n in alg_names]
    
    x = range(len(alg_names))
    width = 0.25
    plt.figure(figsize=(8,5))
    plt.bar([i - width for i in x], acc, width=width, label='Accuracy', color='#2E86AB')
    plt.bar(x, kappa, width=width, label="Cohen's Kappa", color='#A23B72')
    plt.bar([i + width for i in x], f1, width=width, label='F1 macro', color='#F18F01')
    plt.xticks(x, alg_names)
    plt.ylabel('Значение метрики')
    plt.ylim(0, 1)
    plt.title('Сравнение метрик алгоритмов (относительно reference)')
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Сохранён график сравнения метрик: {filename}")

def save_error_venn_barplot(errors_per_alg, output_dir="."):
    """Столбцовая диаграмма: сколько ошибок у каждого алгоритма и их пересечения"""
    if not HAS_VIZ or not errors_per_alg:
        return
    alg_names = list(errors_per_alg.keys())
    error_counts = [len(errors_per_alg[name]) for name in alg_names]
    # Подсчёт общих ошибок
    common_all = None
    for name in alg_names:
        if common_all is None:
            common_all = set(errors_per_alg[name])
        else:
            common_all &= set(errors_per_alg[name])
    # Попарные пересечения для stacked bar не делаем, просто barplot
    plt.figure(figsize=(6,5))
    bars = plt.bar(alg_names, error_counts, color='#F18F01', alpha=0.7)
    plt.ylabel('Количество ошибочных образцов')
    plt.title('Количество ошибок на алгоритм')
    for bar, cnt in zip(bars, error_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(cnt), ha='center', va='bottom')
    if common_all:
        plt.figtext(0.5, 0.01, f"Общих ошибок у всех алгоритмов: {len(common_all)}", ha='center', fontsize=9)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'error_counts.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Сохранён график количества ошибок: {filename}")

# ============================================================
# 6. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    print("=" * 70)
    print("СРАВНЕНИЕ ПРЕДСКАЗАНИЙ ГАПЛОГРУПП (автоматический маппинг + визуализация)")
    print("=" * 70)
    
    ref_file = "reference.tsv"
    alg_files = [f for f in os.listdir('.') if f.startswith('alg_') and f.endswith('.tsv')]
    alg_files.sort()
    if not os.path.exists(ref_file) or not alg_files:
        print("Ошибка: не найдены reference.tsv или alg_*.tsv")
        return
    
    # Читаем reference
    ref_data = read_haplo_tsv(ref_file, mapping=None)
    print(f"✓ reference: {len(ref_data)} образцов")
    
    # Читаем алгоритмы без маппинга для построения confusion
    raw_alg_data = {}
    for f in alg_files:
        name = os.path.splitext(f)[0]
        raw_alg_data[name] = read_haplo_tsv(f, mapping=None)
        print(f"✓ {name}: {len(raw_alg_data[name])} образцов")
    
    # Общие образцы
    common = set(ref_data.keys())
    for data in raw_alg_data.values():
        common &= set(data.keys())
    common = sorted(common)
    print(f"\nОбщих образцов: {len(common)}")
    if not common:
        print("Ошибка: нет общих образцов")
        return
    
    y_ref = [ref_data[m] for m in common]
    true_groups = set(y_ref)
    
    # Автоматический маппинг для каждого алгоритма
    final_mappings = {}
    for name, data in raw_alg_data.items():
        y_pred_raw = [data[m] for m in common]
        confusion = defaultdict(lambda: defaultdict(int))
        for t, p in zip(y_ref, y_pred_raw):
            confusion[t][p] += 1
        print_confusion_table(confusion, true_groups, set(y_pred_raw))
        if AUTO_MAPPING:
            auto_map = build_mapping_from_confusion(confusion, threshold=MAPPING_THRESHOLD)
            if auto_map:
                print(f"Автоматически предложенные соответствия для {name}:")
                for pred, true in auto_map.items():
                    print(f"  {pred} -> {true}")
                print("Применить? (y/n): ", end='')
                answer = input().strip().lower()
                if answer == 'y':
                    final_mappings[name] = auto_map
                    print(f"✓ Соответствия применены для {name}\n")
                else:
                    print(f"✗ Соответствия не применены для {name}\n")
            else:
                print(f"Не удалось найти надёжные соответствия для {name} (порог {MAPPING_THRESHOLD*100}%)\n")
    
    # Перечитываем алгоритмы с применением маппинга
    alg_data = {}
    for f in alg_files:
        name = os.path.splitext(f)[0]
        mapping = final_mappings.get(name, None)
        alg_data[name] = read_haplo_tsv(f, mapping=mapping)
    
    # Строим y_algs после маппинга
    y_algs = {}
    for name, data in alg_data.items():
        y_algs[name] = [data[m] for m in common]
    
    # Расчёт метрик
    all_haplos_ref = sorted(set(y_ref))
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ПОСЛЕ ПРИМЕНЕНИЯ СООТВЕТСТВИЙ")
    print("="*70)
    print(f"Всего образцов: {len(common)}")
    print(f"Гаплогрупп в reference: {len(all_haplos_ref)}")
    
    results = {}
    for name, y_pred in y_algs.items():
        acc = sum(1 for a,b in zip(y_ref, y_pred) if a==b)/len(y_ref)
        k_mean, k_low, k_high = bootstrap_kappa(y_ref, y_pred)
        f_macro = macro_f1(y_ref, y_pred)
        f_micro = micro_f1(y_ref, y_pred)
        results[name] = {'accuracy':acc, 'kappa':k_mean, 'kappa_ci':(k_low,k_high),
                         'f1_macro':f_macro, 'f1_micro':f_micro, 'y_pred':y_pred}
    
    # Печать метрик
    print("\n--- МЕТРИКИ КАЖДОГО АЛГОРИТМА (vs reference) ---")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy      : {res['accuracy']:.4f}")
        print(f"  Cohen's Kappa : {res['kappa']:.4f} (95% CI: {res['kappa_ci'][0]:.4f}–{res['kappa_ci'][1]:.4f})")
        print(f"  F1 macro      : {res['f1_macro']:.4f}")
        print(f"  F1 micro      : {res['f1_micro']:.4f}")
    
    # McNemar
    alg_names = list(y_algs.keys())
    for name1,name2 in combinations(alg_names,2):
        only1 = only2 = 0
        for i in range(len(y_ref)):
            c1 = (y_algs[name1][i] == y_ref[i])
            c2 = (y_algs[name2][i] == y_ref[i])
            if c1 and not c2:
                only1 += 1
            elif not c1 and c2:
                only2 += 1
        p = mcnemar_test(only1, only2)
        print(f"\n{name1} vs {name2}: только {name1} прав = {only1}, только {name2} прав = {only2}, p = {p:.6f}")
    
    # Fleiss' Kappa
    code = {h:i for i,h in enumerate(all_haplos_ref)}
    valid = []
    for i in range(len(y_ref)):
        try:
            code[y_ref[i]]
            ok = True
            for name in alg_names:
                if y_algs[name][i] not in code:
                    ok = False
                    break
            if ok:
                valid.append(i)
        except KeyError:
            continue
    ratings = []
    for i in valid:
        row = [code[y_ref[i]]] + [code[y_algs[name][i]] for name in alg_names]
        ratings.append(row)
    fleiss_k = fleiss_kappa(ratings) if ratings else float('nan')
    print(f"\nFleiss' Kappa (reference + алгоритмы): {fleiss_k:.4f}")
    
    # ВИЗУАЛИЗАЦИЯ
    if HAS_VIZ:
        print("\n--- ГЕНЕРАЦИЯ ГРАФИКОВ ---")
        # 1. Confusion matrices для каждого алгоритма
        for name, y_pred in y_algs.items():
            # Построим confusion dict для текущего алгоритма (после маппинга)
            conf = defaultdict(lambda: defaultdict(int))
            for t, p in zip(y_ref, y_pred):
                conf[t][p] += 1
            # Определим все возможные группы (истинные и предсказанные)
            pred_set = set(p for p in y_pred)
            true_set = set(y_ref)
            # Сохраняем тепловую карту
            save_confusion_matrix_heatmap(conf, true_set, pred_set, name)
        
        # 2. Сравнительный barplot метрик
        save_metrics_barplot(results)
        
        # 3. Диаграмма количества ошибок
        errors_per_alg = {}
        for name, y_pred in y_algs.items():
            err_idx = [i for i, (t,p) in enumerate(zip(y_ref, y_pred)) if t != p]
            errors_per_alg[name] = err_idx
        save_error_venn_barplot(errors_per_alg)
    else:
        print("\n⚠️ Графики не созданы: установите matplotlib и seaborn (pip install matplotlib seaborn)")
    
    # Confusion matrix в текстовом виде для лучшего алгоритма
    best = max(results.keys(), key=lambda x: results[x]['kappa'])
    y_pred_best = results[best]['y_pred']
    classes = all_haplos_ref
    cm = [[0]*len(classes) for _ in range(len(classes))]
    for t,p in zip(y_ref, y_pred_best):
        if p in classes:
            i = classes.index(t)
            j = classes.index(p)
            cm[i][j] += 1
    print(f"\n--- ТЕПЛОВАЯ КАРТА ОШИБОК В ТЕКСТОВОМ ВИДЕ (для {best}) ---")
    max_len = max(len(c) for c in classes)
    print(" " * (max_len+2), end='')
    for c in classes:
        print(f"{c:>6}", end='')
    print()
    for i,true in enumerate(classes):
        print(f"{true:<{max_len+2}}", end='')
        for j in range(len(classes)):
            print(f"{cm[i][j]:6d}", end='')
        print()
    
    # Сохраняем текстовый отчёт
    with open('comparison_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Сравнение гаплогрупп\n")
        f.write(f"Образцов: {len(common)}\n")
        f.write(f"Гаплогрупп: {len(classes)}\n")
        for name,res in results.items():
            f.write(f"{name}: Acc={res['accuracy']:.4f}, Kappa={res['kappa']:.4f}, F1_macro={res['f1_macro']:.4f}\n")
        f.write(f"\nFleiss Kappa: {fleiss_k:.4f}\n")
    print("\n✓ Отчёт сохранён в comparison_summary.txt")
    print("="*70)

if __name__ == "__main__":
    main()