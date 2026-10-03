"""
ПОЛНАЯ ВЕРСИЯ - анализ ВСЕХ сотрудников без ограничений
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import combinations
import json
from datetime import datetime
import warnings
import platform
import logging
import heapq
import math
import matplotlib.pyplot as plt
import os
from scipy.special import comb
warnings.filterwarnings('ignore')

PROGRAM_NAME = (
    "Team Formation Pipeline"
)

PROGRAM_VERSION = "1.0.0"

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

DEFAULT_CONFIG = {
    "weights": [0.25, 0.15, 0.12, 0.15, 0.10, 0.13, 0.10],

    "project_budget": 1200000,
    "project_hours": 160,

    "min_team_size": 2,
    "max_team_size": 6,

    "alpha_base": 20000,
    "beta": 1.2,

    "generate_salaries": True,
    "base_rate_per_hour": 1800,

    "target_correlation": 0.65,
    "salary_variation": 0.5,
    "outlier_fraction": 0.1,

    "top_coalitions_count": 10,
    "required_project_value": 100000
}

if not os.path.exists("config.json"):

    with open(
        "config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            DEFAULT_CONFIG,
            f,
            ensure_ascii=False,
            indent=4
        )

with open(
    "config.json",
    "r",
    encoding="utf-8"
) as f:

    CONFIG = json.load(f)

CONFIG["weights"] = np.array(
    CONFIG["weights"]
)

logging.info("Программа запущена")

# ============================================================================
# 1. ОБРАБОТКА JIRA-ДАННЫХ (для всех сотрудников)
# ============================================================================
def process_jira_data(filepath='jira_issues.csv'):
    """
    Обрабатывает Jira-данные для ВСЕХ сотрудников
    """

    print("=" * 60)
    print("ШАГ 1: Обработка Jira-данных (ВСЕ сотрудники)")
    print("=" * 60)

    try:
        # Загружаем все данные
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python',
                        encoding='utf-8')
        print(f"Загружено {len(df)} строк из {filepath}")
        logging.info(
            f"Загружено {len(df)} строк Jira-данных из {filepath}"
            )

    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        print("Используем тестовые данные...")
        df = generate_test_data()

    # Базовые проверки
    if 'assignee_id' not in df.columns:
        print("Создаем assignee_id...")
        df['assignee_id'] = [f'EMP_{i:03d}' for i in range(1, len(df)+1)]

    # Очистка от NaN
    initial_count = len(df)
    df = df.dropna(subset=['assignee_id'])
    print(f"После очистки от NaN: {len(df)} строк (удалено {initial_count - len(df)})")

    # Агрегация метрик для всех сотрудников
    print("Агрегация метрик для всех сотрудников...")

    # Базовые метрики
    agg = pd.DataFrame()
    agg['tasks_count'] = df.groupby('assignee_id').size()

    # Если есть дополнительные колонки, используем их
    if 'priority' in df.columns:
        priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df['priority_numeric'] = df['priority'].map(priority_map).fillna(2)
        agg['avg_priority'] = df.groupby('assignee_id')['priority_numeric'].mean()
    else:
        agg['avg_priority'] = np.random.uniform(1.5, 3.5, len(agg))

    # Добавляем случайные метрики для полноты
    np.random.seed(42)
    n_employees = len(agg)

    agg['collaboration_score'] = np.random.uniform(0.3, 0.95, n_employees)
    agg['efficiency_score'] = np.random.uniform(0.4, 0.9, n_employees)
    agg['knowledge_depth'] = np.random.uniform(0.5, 1.0, n_employees)

    agg = agg.reset_index()

    print(f"Обработано {len(agg)} уникальных сотрудников")
    logging.info(
        f"Обработано {len(agg)} сотрудников"
        )

    # Расчет S_score для всех сотрудников
    print("Расчет S_score для всех сотрудников...")

    # Нормализация метрик
    metrics = ['tasks_count', 'avg_priority', 'collaboration_score',
               'efficiency_score', 'knowledge_depth']

    for metric in metrics:
        if metric in agg.columns:
            # Для priority: выше = лучше (сложнее задачи)
            if metric == 'avg_priority':
                agg[metric] = agg[metric] / agg[metric].max()

            # Z-нормализация
            mean_val = agg[metric].mean()
            std_val = agg[metric].std()
            if std_val > 0:
                agg[f'z_{metric}'] = (agg[metric] - mean_val) / std_val
            else:
                agg[f'z_{metric}'] = 0

    # Берем z-метрики
    z_cols = [col for col in agg.columns if col.startswith('z_')]

    if z_cols:
        z_matrix = agg[z_cols].values

        # Используем веса из конфигурации
        n_metrics = min(len(z_cols), len(CONFIG['weights']))
        weights = CONFIG['weights'][:n_metrics]
        weights = weights / weights.sum()  # Нормализуем

        # Расчет через нормальное распределение
        norm_z = norm.cdf(z_matrix[:, :n_metrics])
        S_score = (norm_z * weights).sum(axis=1)

        # Нормализация к [0, 1]
        S_min, S_max = S_score.min(), S_score.max()
        if S_max - S_min > 0:
            S_score = (S_score - S_min) / (S_max - S_min)
        else:
            S_score = np.ones_like(S_score) * 0.5

        agg['S_score'] = S_score
    else:
        # Случайные S_score если нет метрик
        agg['S_score'] = np.random.uniform(0.4, 0.95, n_employees)

    print(f"Диапазон S_score: [{agg['S_score'].min():.3f}, {agg['S_score'].max():.3f}]")
    print(f"Средний S_score: {agg['S_score'].mean():.3f}")

    return agg[['assignee_id', 'S_score']]

def generate_test_data():
    """Генерация тестовых данных"""
    np.random.seed(42)

    n_employees = 100
    n_tasks = 10000

    employees = [f'EMP_{i:03d}' for i in range(1, n_employees+1)]

    data = {
        'assignee_id': np.random.choice(employees, n_tasks),
        'priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_tasks,
                                    p=[0.2, 0.5, 0.2, 0.1]),
        'status': np.random.choice(['Open', 'In Progress', 'Resolved'], n_tasks),
    }

    return pd.DataFrame(data)

# ============================================================================
# 2. ГЕНЕРАЦИЯ ЗАРПЛАТ ДЛЯ ВСЕХ СОТРУДНИКОВ
# ============================================================================
def generate_salaries_for_all_employees(df, target_corr=0.65, variation=0.5):
    """
    Генерирует зарплаты для ВСЕХ сотрудников с контролируемой корреляцией
    """

    print("\n" + "=" * 60)
    print("ШАГ 2: Генерация зарплат для всех сотрудников")
    print("=" * 60)

    n = len(df)
    S_scores = df['S_score'].values

    print(f"Генерация зарплат для {n} сотрудников...")

    # Базовая зарплата линейно зависит от S_score
    base_salary = 40_000 + 160_000 * S_scores  # 40k-200k базовый диапазон

    # Добавляем шум для достижения целевой корреляции
    noise_std = (1 - target_corr) * 80_000
    noise = np.random.normal(0, noise_std, n)

    # Добавляем индивидуальную вариацию
    variation_factor = np.random.uniform(1 - variation, 1 + variation, n)

    salaries = (base_salary + noise) * variation_factor

    # Добавляем выбросы (10%)
    n_outliers = max(1, int(n * CONFIG['outlier_fraction']))
    outlier_indices = np.random.choice(n, n_outliers, replace=False)

    # Низкие выбросы
    low_outliers = outlier_indices[:n_outliers//2]
    if len(low_outliers) > 0:
        salaries[low_outliers] = np.random.uniform(30_000, 50_000, len(low_outliers))

    # Высокие выбросы
    high_outliers = outlier_indices[n_outliers//2:]
    if len(high_outliers) > 0:
        salaries[high_outliers] = np.random.uniform(250_000, 400_000, len(high_outliers))

    # Ограничения и округление
    salaries = np.clip(salaries, 25_000, 450_000)
    salaries = np.round(salaries, -3).astype(int)  # Округление до тысяч

    # Часовая ставка и минимальные гарантии
    hourly_rates = salaries / CONFIG['project_hours']
    min_salaries = np.round(salaries * 0.75, -3).astype(int)

    # Добавляем в DataFrame
    df = df.copy()
    df['salary'] = salaries
    df['hourly_rate'] = hourly_rates
    df['min_salary'] = min_salaries

    # Проверяем корреляцию
    actual_corr = np.corrcoef(S_scores, salaries)[0, 1]
    logging.info(
        f"Корреляция S_score и зарплат: {actual_corr:.3f}"
        )

    print(f"Статистика зарплат:")
    print(f"  Целевая корреляция: {target_corr:.3f}")
    print(f"  Фактическая корреляция: {actual_corr:.3f}")
    print(f"  Минимальная зарплата: {salaries.min():,} ₽")
    print(f"  Максимальная зарплата: {salaries.max():,} ₽")
    print(f"  Средняя зарплата: {salaries.mean():,.0f} ₽")
    print(f"  Медианная зарплата: {np.median(salaries):,.0f} ₽")
    print(f"  Сотрудников: {n}")

    # Анализ распределения
    print(f"\nРаспределение сотрудников по зарплатам:")
    bins = [0, 50000, 100000, 150000, 200000, 300000, 500000]
    labels = ['<50k', '50-100k', '100-150k', '150-200k', '200-300k', '>300k']

    for i in range(len(bins)-1):
        count = ((salaries >= bins[i]) & (salaries < bins[i+1])).sum()
        percentage = count / n * 100
        print(f"  {labels[i]}: {count} чел. ({percentage:.1f}%)")

    return df

# ============================================================================
# 3. УМНЫЙ ПОИСК КОАЛИЦИЙ ДЛЯ ВСЕХ СОТРУДНИКОВ
# ============================================================================
class SmartCoalitionFinder:
    """
    Умный поиск коалиций для большого количества сотрудников
    Использует эвристики для поиска без полного перебора
    """

    def __init__(self, df, alpha_base=20000, beta=1.2,
                 project_hours=160, project_budget=1_200_000,
                 min_team_size=2, max_team_size=6):

        self.df = df.copy()
        self.n = len(df)
        self.alpha_base = alpha_base
        self.beta = beta
        self.project_hours = project_hours
        self.project_budget = project_budget
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size

        # Преобразуем в numpy arrays
        self.ids = df['assignee_id'].values
        self.S_scores = df['S_score'].values
        self.salaries = df['salary'].values
        self.hourly_rates = df['hourly_rate'].values
        self.min_salaries = df['min_salary'].values

        # Сортируем сотрудников по эффективности (S_score / зарплата)
        self.efficiency = self.S_scores / (self.salaries + 1)
        self.sorted_indices = np.argsort(self.efficiency)[::-1]  # По убыванию

        print(f"\nИнициализация умного поиска коалиций:")
        print(f"Всего сотрудников: {self.n}")
        print(f"Анализ ВСЕХ сотрудников")
        print(f"Размеры команд: {min_team_size}-{max_team_size}")
        print(f"Бюджет проекта: {project_budget:,} ₽")
        print(f"Самый эффективный сотрудник: {self.ids[self.sorted_indices[0]]} "
              f"(S={self.S_scores[self.sorted_indices[0]]:.3f}, "
              f"з/п={self.salaries[self.sorted_indices[0]]:,} ₽)")

    def characteristic_function(self, indices):
        """
        Характеристическая функция v(S)
        v(S) = H(S) - α * |S|^β
        """
        if not indices:
            return 0.0

        if isinstance(indices, tuple):
            indices = list(indices)

        size = len(indices)

        # Человеческий капитал
        H_S = (
            np.sum(self.S_scores[indices])
            * 250000
            )

        # Штраф за размер команды
        penalty = self.alpha_base * (size ** self.beta)

        return H_S - penalty

    def coalition_cost(self, indices):
        """Стоимость коалиции"""
        if not indices:
            return 0

        if isinstance(indices, tuple):
            indices = list(indices)

        return np.sum(self.salaries[indices])

    def coalition_min_cost(self, indices):
        """Минимальная стоимость коалиции"""
        if not indices:
            return 0

        if isinstance(indices, tuple):
            indices = list(indices)

        return np.sum(self.min_salaries[indices])

    def smart_find_top_coalitions(self, top_k=10, max_iterations=int(CONFIG.get("search_iterations", 2500))):
        """
        Умный поиск топ-K коалиций без полного перебора
        Использует эвристики на основе эффективности сотрудников
        """

        print("\n" + "=" * 60)
        print(f"ШАГ 3: Умный поиск топ-{top_k} коалиций")
        print("=" * 60)

        heap = []
        iterations = 0

        # Стратегия 1: Берем самых эффективных сотрудников
        print("Стратегия 1: Комбинации самых эффективных сотрудников...")

        # Берем топ-N самых эффективных для комбинирования
        top_efficient = min(20, self.n)  # Берем топ-20 самых эффективных
        efficient_indices = self.sorted_indices[:top_efficient]

        for size in range(self.min_team_size, min(self.max_team_size, top_efficient) + 1):
            # Оцениваем количество комбинаций
            n_combos = comb(len(efficient_indices), size, exact=True)
            if n_combos > 10000:
                print(f"  Размер {size}: слишком много комбинаций ({n_combos:,}), пропускаем")
                continue

            print(f"  Анализ размера {size}...")

            for combo in combinations(efficient_indices, size):
                iterations += 1
                if iterations > max_iterations:
                    break

                indices = list(combo)

                # Проверяем бюджет
                total_cost = self.coalition_cost(indices)
                if total_cost > self.project_budget:
                    continue

                min_cost = self.coalition_min_cost(indices)
                if min_cost >= self.project_budget:
                    continue

                # Проверяем ценность
                value = self.characteristic_function(indices)
                if value < CONFIG['required_project_value']:
                    continue

                # Сохраняем в кучу
                if len(heap) < top_k:
                    heapq.heappush(heap, (value, indices, total_cost, min_cost))
                else:
                    min_val, _, _, _ = heap[0]
                    if value > min_val:
                        heapq.heapreplace(heap, (value, indices, total_cost, min_cost))

            if iterations > max_iterations:
                break

        # Стратегия 2: Случайные комбинации из всех сотрудников
        print("Стратегия 2: Случайные комбинации из всех сотрудников...")

        np.random.seed(42)
        remaining_iterations = max_iterations - iterations

        for _ in range(min(remaining_iterations, 5000)):
            iterations += 1

            # Случайный размер команды
            size = np.random.randint(self.min_team_size, self.max_team_size + 1)

            # Случайные сотрудники (с учетом эффективности)
            if np.random.random() < 0.7:  # 70% chance взять эффективных
                # Берем в основном эффективных
                n_effective = min(size, len(self.sorted_indices)//2)
                effective_part = list(np.random.choice(self.sorted_indices[:len(self.sorted_indices)//2],
                                                      n_effective, replace=False))

                if size > n_effective:
                    # Добавляем случайных
                    remaining = size - n_effective
                    other_indices = list(set(range(self.n)) - set(effective_part))
                    other_part = list(np.random.choice(other_indices, remaining, replace=False))
                    indices = effective_part + other_part
                else:
                    indices = effective_part
            else:
                # Полностью случайная команда
                indices = list(np.random.choice(range(self.n), size, replace=False))

            # Проверки
            total_cost = self.coalition_cost(indices)
            if total_cost > self.project_budget:
                continue

            min_cost = self.coalition_min_cost(indices)
            if min_cost >= self.project_budget:
                continue

            value = self.characteristic_function(indices)
            if value < CONFIG['required_project_value']:
                continue

            # Сохраняем
            if len(heap) < top_k:
                heapq.heappush(heap, (value, indices, total_cost, min_cost))
            else:
                min_val, _, _, _ = heap[0]
                if value > min_val:
                    heapq.heapreplace(heap, (value, indices, total_cost, min_cost))

        # Сортируем результат
        top_coalitions = sorted(heap, key=lambda x: x[0], reverse=True)

        print(f"\nСтатистика поиска:")
        print(f"Итераций выполнено: {iterations:,}")
        print(f"Найдено допустимых коалиций: {len(top_coalitions)}")

        if top_coalitions:
            print(f"\nТОП-{len(top_coalitions)} КОАЛИЦИЙ:")
            print("-" * 100)
            print(f"{'№':<3} {'Размер':<8} {'Ценность (v)':<15} {'Стоимость':<12} "
                  f"{'Min гарантии':<13} {'Остаток бюджета':<15} {'Эффективность':<12}")
            print("-" * 100)

            for i, (value, indices, cost, min_cost) in enumerate(top_coalitions, 1):
                remaining = self.project_budget - cost
                efficiency = value / cost if cost > 0 else 0

                print(f"{i:<3} {len(indices):<8} {value:,.0f} ₽{'':<2} {cost:,.0f} ₽{'':<3} "
                      f"{min_cost:,.0f} ₽{'':<3} {remaining:,.0f} ₽{'':<2} {efficiency:.3f}")

                if i <= 3:
                    # ИСПРАВЛЕНИЕ ОШИБКИ: ЯВНОЕ ПРЕОБРАЗОВАНИЕ В СТРОКУ
                    member_ids = [str(self.ids[idx]) for idx in indices[:5]]
                    print(f"     Состав: {', '.join(member_ids)}" +
                          ("..." if len(indices) > 5 else ""))

            print("-" * 100)
        else:
            print("Не найдено подходящих коалиций")

        logging.info(
            f"Найдено {len(top_coalitions)} допустимых коалиций"
            )
        return top_coalitions

    def analyze_key_coalitions(self, top_coalitions):
        """
        Анализирует 3 ключевые коалиции
        """

        print("\n" + "=" * 60)
        print("ШАГ 4: Анализ ключевых коалиций")
        print("=" * 60)

        if not top_coalitions:
            print("Нет коалиций для анализа")
            return None

        # Находим ключевые коалиции
        min_cost_coalition = min(top_coalitions, key=lambda x: x[2])
        optimal_coalition = max(top_coalitions, key=lambda x: x[0] / (x[2] + 1))
        max_cost_coalition = max(top_coalitions, key=lambda x: x[2])

        key_coalitions = {
            'minimal': min_cost_coalition,
            'optimal': optimal_coalition,
            'maximal': max_cost_coalition
        }

        # Выводим информацию
        print("\nКЛЮЧЕВЫЕ КОАЛИЦИИ:")
        print("-" * 130)
        print(f"{'Тип':<15} {'Размер':<8} {'Ценность (v)':<15} {'Стоимость':<12} "
              f"{'Min гарантии':<13} {'v/₽':<10} {'Ср.S':<8} {'Состав'}")
        print("-" * 130)

        for coal_type, (value, indices, cost, min_cost) in key_coalitions.items():
            efficiency = value / cost if cost > 0 else 0
            avg_S = np.mean(self.S_scores[indices])
            # ИСПРАВЛЕНИЕ ОШИБКИ: ЯВНОЕ ПРЕОБРАЗОВАНИЕ В СТРОКУ
            member_ids = [str(self.ids[idx]) for idx in indices[:4]]
            members_str = ', '.join(member_ids) + ("..." if len(indices) > 4 else "")

            type_name = {
                'minimal': 'Минимальная',
                'optimal': 'Оптимальная',
                'maximal': 'Максимальная'
            }[coal_type]

            print(f"{type_name:<15} {len(indices):<8} {value:,.0f} ₽{'':<2} "
                  f"{cost:,.0f} ₽{'':<3} {min_cost:,.0f} ₽{'':<3} "
                  f"{efficiency:.3f}{'':<6} {avg_S:.3f}{'':<4} {members_str}")

        print("-" * 130)

        return key_coalitions

    def calculate_shapley_for_coalition(self, indices):
        """
        Расчет вектора Шепли для коалиции
        """
        if not indices:
            return np.array([]), np.array([])

        if isinstance(indices, tuple):
            indices = list(indices)

        size = len(indices)
        if size == 0:
            return np.array([]), np.array([])

        # Значение коалиции
        v_S = self.characteristic_function(indices)

        if v_S <= 0:
            print("ВНИМАНИЕ: Ценность коалиции ≤ 0")
            shapley_values = np.ones(size) * v_S / size
            return shapley_values, indices

        # Упрощенный расчет Шепли
        shapley_values = np.zeros(size)

        for i, idx in enumerate(indices):
            # Коалиция без i-го игрока
            indices_without_i = [j for j in indices if j != idx]
            v_without_i = self.characteristic_function(indices_without_i)

            # Предельный вклад
            marginal = max(0, v_S - v_without_i)

            # Взвешиваем по S_score
            shapley_values[i] = self.S_scores[idx] * marginal

        # Нормализация
        total_shapley = shapley_values.sum()
        if total_shapley > 0:
            shapley_values = shapley_values * (v_S / total_shapley)
        else:
            shapley_values = np.ones(size) * v_S / size

        return shapley_values, indices

    def distribute_budget_with_shapley(self, coal_type, coal_data):
        """
        Распределение бюджета по Шепли
        """
        value, indices, total_cost, min_cost = coal_data

        if isinstance(indices, tuple):
            indices = list(indices)

        size = len(indices)

        print(f"\n{'='*50}")
        type_name = {
            'minimal': 'МИНИМАЛЬНАЯ КОАЛИЦИЯ (выполняющая задачу)',
            'optimal': 'ОПТИМАЛЬНАЯ КОАЛИЦИЯ (цена/качество)',
            'maximal': 'МАКСИМАЛЬНАЯ КОАЛИЦИЯ (в рамках бюджета)'
        }[coal_type]
        print(type_name)
        print('='*50)

        print(f"Размер команды: {size} человек")
        print(f"Ценность коалиции: {value:,.0f} ₽")
        print(f"Фактическая стоимость: {total_cost:,.0f} ₽")
        print(f"Минимальные гарантии: {min_cost:,.0f} ₽")
        print(f"Бюджет проекта: {self.project_budget:,} ₽")

        # Расчет вектора Шепли
        shapley_values, _ = self.calculate_shapley_for_coalition(indices)

        if len(shapley_values) == 0:
            print("Ошибка расчета вектора Шепли")
            return None

        # Шаг 1: Гарантируем минимальные выплаты
        min_salaries_coalition = self.min_salaries[indices]
        total_min = min_salaries_coalition.sum()

        print(f"\n1. Гарантированные минимумы: {total_min:,.0f} ₽")

        # Проверяем, что минимальные гарантии меньше бюджета
        if total_min >= self.project_budget:
            print(f"ВНИМАНИЕ: Минимальные гарантии превышают бюджет")
            print("Используем пропорциональное распределение бюджета")
            weights = min_salaries_coalition / total_min
            payouts = weights * self.project_budget
        else:
            # Шаг 2: Распределяем остаток по Шепли
            remainder = self.project_budget - total_min
            print(f"2. Остаток для распределения по Шепли: {remainder:,.0f} ₽")

            # Нормализуем вектор Шепли к остатку
            shapley_excess = shapley_values - min_salaries_coalition
            shapley_excess[shapley_excess < 0] = 0

            total_excess = shapley_excess.sum()

            if total_excess <= 0:
                print("Избыточная полезность = 0, распределяем остаток равномерно")
                payouts = min_salaries_coalition + remainder / size
            else:
                weights = shapley_excess / total_excess
                payouts = min_salaries_coalition + weights * remainder

        # Округление
        payouts = np.round(payouts, -3).astype(int)

        # Корректировка (если есть небольшие расхождения)
        payout_total = payouts.sum()
        discrepancy = self.project_budget - payout_total

        if abs(discrepancy) > 0:
            print(f"3. Корректировка выплат на {discrepancy:,} ₽")
            if discrepancy > 0:
                # Добавляем к наименьшим выплатам
                sorted_indices = np.argsort(payouts)
                for i in sorted_indices:
                    payouts[i] += 1000
                    discrepancy -= 1000
                    if discrepancy <= 0:
                        break
            else:
                # Уменьшаем наибольшие выплаты (но не ниже минимума)
                sorted_indices = np.argsort(payouts)[::-1]
                for i in sorted_indices:
                    if payouts[i] - 1000 >= min_salaries_coalition[i]:
                        payouts[i] -= 1000
                        discrepancy += 1000
                        if discrepancy >= 0:
                            break

        # Собираем результаты
        results = []
        for i, idx in enumerate(indices):
            employee_id = self.ids[idx]
            S_score = self.S_scores[idx]
            market_salary = self.salaries[idx]
            min_salary = min_salaries_coalition[i]
            shapley_value = shapley_values[i] if i < len(shapley_values) else 0
            final_payout = payouts[i]
            bonus = final_payout - min_salary

            results.append({
                'employee_id': str(employee_id),  # Явное преобразование в строку
                'S_score': float(S_score),
                'market_salary': int(market_salary),
                'min_salary': int(min_salary),
                'shapley_value': float(shapley_value),
                'final_payout': int(final_payout),
                'bonus': int(bonus),
                'efficiency_ratio': float(S_score / market_salary * 1000) if market_salary > 0 else 0
            })

        # Сортировка по выплатам
        results.sort(key=lambda x: x['final_payout'], reverse=True)

        # Вывод детальной таблицы
        print(f"\nДЕТАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ВЫПЛАТ:")
        print("-" * 130)
        print(f"{'Сотрудник':<12} {'S':<6} {'Рын.з/п':<10} {'Min':<10} "
              f"{'Шепли':<12} {'Итог':<12} {'Бонус':<10} {'Эфф.':<8}")
        print("-" * 130)

        total_payout = 0
        total_bonus = 0

        for r in results:
            print(f"{r['employee_id']:<12} {r['S_score']:<6.3f} "
                  f"{r['market_salary']:<10,} {r['min_salary']:<10,} "
                  f"{r['shapley_value']:<12,.0f} {r['final_payout']:<12,} "
                  f"{r['bonus']:<10,} {r['efficiency_ratio']:<8.1f}")

            total_payout += r['final_payout']
            total_bonus += r['bonus']

        print("-" * 130)
        print(f"{'ИТОГО':<12} {'':<6} {'':<10} {'':<10} "
              f"{'':<12} {total_payout:<12,} {total_bonus:<10,}")

        print(f"\nИспользовано бюджета: {total_payout:,} ₽ ({total_payout/self.project_budget*100:.1f}%)")
        print(f"Остаток бюджета: {self.project_budget - total_payout:,} ₽")

        return results

# ============================================================================
# 4. ОСНОВНОЙ ПРОЦЕСС
# ============================================================================
def save_run_metadata():

    metadata = {
    "created_at": str(datetime.now()),
    "project_budget": CONFIG["project_budget"],
    "project_hours": CONFIG["project_hours"],
    "alpha_base": CONFIG["alpha_base"],
    "beta": CONFIG["beta"],
    "min_team_size": CONFIG["min_team_size"],
    "max_team_size": CONFIG["max_team_size"],
    "program_name": PROGRAM_NAME,
    "program_version": PROGRAM_VERSION,
    "python_version": platform.python_version(),
    "platform": platform.platform()
    }

    with open(
        "run_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    logging.info("Паспорт запуска сохранен")

def create_readme():

    if os.path.exists("README.md"):
        return

    readme_text = """
# Team Formation Program

Программа анализа кадровых данных и формирования проектных команд.

Входные данные:
- jira_issues.csv

Выходные данные:
- all_employees_processed.csv
- comprehensive_coalition_analysis.csv
- shapley_*_results.json
- run_metadata.json
- pipeline.log

Основные функции:
- расчет S_score
- оценка человеческого капитала
- поиск коалиций
- распределение бюджета по Шепли
- Monte-Carlo моделирование

Язык программирования:
Python
"""

    with open(
        "README.md",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(readme_text)

    logging.info(
        "README.md создан"
    )

def main():
    """Основной процесс анализа ВСЕХ сотрудников"""

    print("\n" + "=" * 70)
    print("АЛГОРИТМ ФОРМИРОВАНИЯ ПРОЕКТНЫХ КОМАНД")
    print("Анализ ВСЕХ сотрудников компании")
    print("=" * 70)

    start_time = datetime.now()
    create_readme()
    save_run_metadata()

    try:
        # Шаг 1: Обработка данных для ВСЕХ сотрудников
        df_processed = process_jira_data('jira_issues.csv')

        # Шаг 2: Генерация зарплат для ВСЕХ сотрудников
        df_with_salaries = generate_salaries_for_all_employees(
            df_processed,
            target_corr=CONFIG['target_correlation'],
            variation=CONFIG['salary_variation']
        )

        # Сохраняем полные данные
        output_file = 'all_employees_processed.csv'
        df_with_salaries.to_csv(output_file, index=False)
        print(f"\nПолные данные сохранены в {output_file}")
        print(f"Всего обработано сотрудников: {len(df_with_salaries)}")

        # Шаг 3: Инициализация умного поиска для ВСЕХ сотрудников
        finder = SmartCoalitionFinder(
            df_with_salaries,
            alpha_base=CONFIG['alpha_base'],
            beta=CONFIG['beta'],
            project_hours=CONFIG['project_hours'],
            project_budget=CONFIG['project_budget'],
            min_team_size=CONFIG['min_team_size'],
            max_team_size=CONFIG['max_team_size']
        )

        # Шаг 4: Умный поиск топ-10 коалиций
        top_coalitions = finder.smart_find_top_coalitions(
            top_k=CONFIG['top_coalitions_count'],
            max_iterations=int(CONFIG.get("search_iterations", 2500))
        )

        if not top_coalitions:
            print("Не найдено подходящих коалиций. Попробуйте изменить параметры.")
            return

        # Шаг 5: Анализ ключевых коалиций
        key_coalitions = finder.analyze_key_coalitions(top_coalitions)

        if not key_coalitions:
            print("Не удалось выделить ключевые коалиции")
            return

        # Шаг 6: Распределение бюджета по вектору Шепли
        print("\n" + "=" * 70)
        print("ШАГ 5: РАСПРЕДЕЛЕНИЕ БЮДЖЕТА ПО ВЕКТОРУ ШЕПЛИ")
        print("Анализ для 3 ключевых коалиций")
        print("=" * 70)

        shapley_results = {}

        for coal_type, coal_data in key_coalitions.items():
            results = finder.distribute_budget_with_shapley(coal_type, coal_data)
            shapley_results[coal_type] = results

            # Сохранение результатов в JSON
            if results:
                output_file = f'shapley_{coal_type}_results.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'coalition_type': coal_type,
                        'team_size': len(coal_data[1]),
                        'coalition_value': float(coal_data[0]),
                        'total_cost': int(coal_data[2]),
                        'min_guarantees': int(coal_data[3]),
                        'budget_usage_percent': sum(r['final_payout'] for r in results) / CONFIG['project_budget'] * 100,
                        'distribution': results,
                        'summary': {
                            'total_payout': sum(r['final_payout'] for r in results),
                            'total_bonus': sum(r['bonus'] for r in results),
                            'avg_S_score': np.mean([r['S_score'] for r in results]),
                            'avg_efficiency': np.mean([r['efficiency_ratio'] for r in results])
                        }
                    }, f, ensure_ascii=False, indent=2)

                print(f"Результаты сохранены в {output_file}")

        # Шаг 7: Создание сводного отчета
        print("\n" + "=" * 70)
        print("ШАГ 6: СВОДНЫЙ ОТЧЕТ ПО КЛЮЧЕВЫМ КОАЛИЦИЯМ")
        print("=" * 70)

        summary_data = []
        for coal_type, coal_data in key_coalitions.items():
            value, indices, cost, min_cost = coal_data
            size = len(indices)

            type_name = {
                'minimal': 'Минимальная',
                'optimal': 'Оптимальная',
                'maximal': 'Максимальная'
            }[coal_type]

            efficiency = value / cost if cost > 0 else 0
            budget_usage = cost / CONFIG['project_budget'] * 100
            avg_S = np.mean(finder.S_scores[indices])
            avg_salary = np.mean(finder.salaries[indices])

            # Находим "скрытых бриллиантов" и "неоправданно дорогих" в коалиции
            efficiencies = finder.S_scores[indices] / (finder.salaries[indices] + 1)
            hidden_gems = np.sum(efficiencies > np.percentile(efficiencies, 75))
            overpaid = np.sum(efficiencies < np.percentile(efficiencies, 25))

            summary_data.append({
                'Тип коалиции': type_name,
                'Размер': size,
                'Ценность (v)': f"{value:,.0f} ₽",
                'Стоимость': f"{cost:,.0f} ₽",
                'Эффективность (v/₽)': f"{efficiency:.3f}",
                'Исп.бюджета': f"{budget_usage:.1f}%",
                'Ср.S_score': f"{avg_S:.3f}",
                'Ср.з/п': f"{avg_salary:,.0f} ₽",
                'Скрытые бриллианты': hidden_gems,
                'Неоправданно дорогие': overpaid
            })

        # Создаем и сохраняем сводную таблицу
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

        summary_df.to_csv('comprehensive_coalition_analysis.csv', index=False, encoding='utf-8')
        print(f"\nСводный отчет сохранен в comprehensive_coalition_analysis.csv")

        # Дополнительная статистика
        print("\n" + "=" * 70)
        print("ОБЩАЯ СТАТИСТИКА АНАЛИЗА")
        print("=" * 70)

        print(f"Всего проанализировано сотрудников: {finder.n}")
        print(f"Диапазон S_score: {finder.S_scores.min():.3f} - {finder.S_scores.max():.3f}")
        print(f"Диапазон зарплат: {finder.salaries.min():,} - {finder.salaries.max():,} ₽")
        print(f"Средняя корреляция S_score и зарплаты: {np.corrcoef(finder.S_scores, finder.salaries)[0,1]:.3f}")

        # Находим лучших "скрытых бриллиантов" во всей компании
        efficiency_all = finder.S_scores / (finder.salaries + 1)
        top_gem_indices = np.argsort(efficiency_all)[-5:][::-1]

        print(f"\nТОП-5 'СКРЫТЫХ БРИЛЛИАНТОВ' во всей компании:")
        print("-" * 80)
        print(f"{'Сотрудник':<12} {'S_score':<8} {'Зарплата':<12} {'Эффективность':<15}")
        print("-" * 80)
        for idx in top_gem_indices:
            print(f"{str(finder.ids[idx]):<12} {finder.S_scores[idx]:<8.3f} "
                  f"{finder.salaries[idx]:<12,} {efficiency_all[idx]:<15.3f}")
        print("-" * 80)

        # Время выполнения
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        print(f"\n" + "=" * 70)
        print(f"АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print(f"Общее время выполнения: {execution_time:.1f} секунд")
        print("=" * 70)

        print("\nСОЗДАННЫЕ ФАЙЛЫ:")
        print("1. all_employees_processed.csv - полные данные всех сотрудников")
        print("2. comprehensive_coalition_analysis.csv - сводный анализ коалиций")
        print("3. shapley_*_results.json - распределения по Шепли для 3 коалиций")
        # ============================================================
        # IEEE FIGURE 1
        # Coordination Cost Growth
        # ============================================================
        plt.figure(figsize=(7, 4))
        alpha = CONFIG['alpha_base']
        n_values = np.arange(1, 11)
        for beta in [1.2, 1.4, 1.6]:
          costs = alpha * (n_values ** beta)
          plt.plot(
              n_values,
              costs,
              linewidth=2,
              marker='o',
              label=f'β = {beta}'
              )

        plt.xlabel('Coalition size n')
        plt.ylabel('Coordination cost C(n)')
        plt.title('Coordination Cost Growth')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            'Figure1_CoordinationCost.png',
            dpi=300,
            bbox_inches='tight'
            )

        plt.savefig(
            'Figure1_CoordinationCost.svg',
            bbox_inches='tight'
            )

        plt.close()
        print("Figure 1 saved")

        # ============================================================
        # MONTE CARLO VALIDATION
        # ============================================================
        np.random.seed(42)

        print("\nRunning Monte-Carlo validation...")
        N_SIM = int(CONFIG.get("monte_carlo_simulations", 5))
        fixed_results = []
        adaptive_results = []
        oracle_results = []

        for sim in range(N_SIM):
          # ----------------------------------------------------
          # FIXED
          # ----------------------------------------------------
          finder_fixed = SmartCoalitionFinder(
                df_with_salaries,
                alpha_base=20000,
                beta=1.2,
                project_hours=CONFIG['project_hours'],
                project_budget=CONFIG['project_budget'],
                min_team_size=CONFIG['min_team_size'],
                max_team_size=CONFIG['max_team_size']
                )

          coalitions = finder_fixed.smart_find_top_coalitions(
              top_k=1,
              max_iterations=int(CONFIG.get("search_iterations", 2500))
              )

          if coalitions:
            fixed_results.append(coalitions[0][0])
          # ----------------------------------------------------
          # ADAPTIVE
          # ----------------------------------------------------
          project_types = [
              "R&D",
              "Production",
              "Safety"
              ]

          project_type = project_types[
              sim % len(project_types)
              ]

          if project_type == "R&D":
            alpha_adaptive = 18000
            beta_adaptive = 1.10

          elif project_type == "Production":
            alpha_adaptive = 20000
            beta_adaptive = 1.30

          else:  # Safety
            alpha_adaptive = 22000
            beta_adaptive = 1.50

          finder_adaptive = SmartCoalitionFinder(
              df_with_salaries,
              alpha_base=alpha_adaptive,
              beta=beta_adaptive,
              project_hours=CONFIG['project_hours'],
              project_budget=CONFIG['project_budget'],
              min_team_size=CONFIG['min_team_size'],
              max_team_size=CONFIG['max_team_size']
              )

          coalitions = finder_adaptive.smart_find_top_coalitions(
              top_k=1,
              max_iterations=int(CONFIG.get("search_iterations", 2500))
              )

          if coalitions:
              adaptive_results.append(coalitions[0][0])
          # ----------------------------------------------------
          # ORACLE
          # ----------------------------------------------------
          best_oracle = -np.inf

          for _ in range(int(CONFIG.get("oracle_trials", 3))):
            alpha_test = np.random.uniform(
                15000,
                25000
                )
            beta_test = np.random.uniform(1.00, 1.60)

            finder_oracle = SmartCoalitionFinder(
              df_with_salaries,
              alpha_base=alpha_test,
              beta=beta_test,
              project_hours=CONFIG['project_hours'],
              project_budget=CONFIG['project_budget'],
              min_team_size=CONFIG['min_team_size'],
              max_team_size=CONFIG['max_team_size']
              )
            coalitions = finder_oracle.smart_find_top_coalitions(
              top_k=1,
              max_iterations=int(CONFIG.get("oracle_search_iterations", 1500))
              )
            if coalitions:
              best_oracle = max(
                  best_oracle,
                  coalitions[0][0]
                )
          oracle_results.append(best_oracle)

        print("Monte-Carlo completed.")

        print("fixed results:", len(fixed_results))
        print("adaptive results:", len(adaptive_results))
        print("oracle results:", len(oracle_results))

        if len(fixed_results) < 5:
          raise ValueError(
              f"Too few fixed results: {len(fixed_results)}"
              )

        if len(adaptive_results) < 5:
          raise ValueError(
              f"Too few adaptive results: {len(adaptive_results)}"
              )

        if len(oracle_results) < 5:
          raise ValueError(
                f"Too few oracle results: {len(oracle_results)}"
              )

        fixed_mean = np.mean(fixed_results) if len(fixed_results) > 0 else 0
        adaptive_mean = np.mean(adaptive_results) if len(adaptive_results) > 0 else 0
        oracle_mean = np.mean(oracle_results) if len(oracle_results) > 0 else 0

        with open("MonteCarloResults.txt", "w", encoding="utf-8") as f:
          f.write(f"Fixed: {fixed_mean:.2f}\n")
          f.write(f"Adaptive: {adaptive_mean:.2f}\n")
          f.write(f"Oracle: {oracle_mean:.2f}\n")
          f.write(f"N_SIM: {N_SIM}\n")
          f.write(f"Oracle iterations: {int(CONFIG.get('oracle_trials', 3))}\n")

        import scipy.stats as st

        if len(adaptive_results) > 1 and np.std(adaptive_results) > 0:
          ci_low, ci_high = st.t.interval(
              0.95,
              len(adaptive_results)-1,
              loc=np.mean(adaptive_results),
              scale=st.sem(adaptive_results)
              )
        else:
          ci_low = adaptive_mean
          ci_high = adaptive_mean

        print(f"95% CI: [{ci_low:.0f}; {ci_high:.0f}]")

        if fixed_mean > 0:
          improvement = (
              (adaptive_mean - fixed_mean)
              / fixed_mean
              * 100
              )

        else:
          improvement = 0

        print(
            f"Improvement: {improvement:.2f}%"
            )

        print("\nMonte-Carlo summary")
        print(f"Fixed    : {fixed_mean:,.0f}")
        print(f"Adaptive : {adaptive_mean:,.0f}")
        print(f"Oracle   : {oracle_mean:,.0f}")

        # ============================================================
        # IEEE FIGURE 2
        # Performance Comparison
        # ============================================================
        strategies = [
            'Fixed',
            'Adaptive',
            'Oracle'
            ]

        values = [
            float(fixed_mean),
            float(adaptive_mean),
            float(oracle_mean)
            ]

        plt.figure(figsize=(6, 4))

        bars = plt.bar(
                strategies,
                values
                )

        plt.ylabel('Mean coalition value')
        plt.title('Comparison of Calibration Strategies')

        for bar in bars:
              height = bar.get_height()

              plt.text(
                  bar.get_x() + bar.get_width()/2,
                  height,
                  f'{int(height)}',
                  ha='center',
                  va='bottom'
              )

        plt.tight_layout()

        plt.savefig(
                'Figure2_PerformanceComparison.png',
                dpi=300,
                bbox_inches='tight'
                )

        plt.savefig(
                'Figure2_PerformanceComparison.svg',
                bbox_inches='tight'
                )

        plt.close()

        print("Figure 2 saved")

        logging.info(
            f"Программа завершена успешно. "
            f"Сотрудников обработано: {finder.n}"
            )

        return df_with_salaries, finder, shapley_results, summary_df

    except Exception as e:
      logging.exception(
          "Критическая ошибка выполнения"
          )
      print(f"\nОШИБКА: {str(e)}")

      import traceback
      traceback.print_exc()

      return None, None, None, None

# ============================================================================
# ЗАПУСК АНАЛИЗА ВСЕХ СОТРУДНИКОВ
# ============================================================================
if __name__ == "__main__":
    print("Запуск полного анализа ВСЕХ сотрудников компании...")
    print("Версия: 3.0 (Анализ без ограничений)")
    print("=" * 70)

    results = main()