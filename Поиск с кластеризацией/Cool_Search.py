import os
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import numpy as np
from collections import defaultdict, Counter
import random
from networkx.algorithms import community as nx_community
import time


SEED = 0 # None
DATA_DIR = 'facebook'
LOW_END_PC = True
I_WANT_SEE = True


def load_facebook_social_circles_data(DATA_DIR, user_id):
    """
    Загружает данные Facebook Social Circles для конкретного пользователя

    Возвращает:
    - G: граф друзей (узлы - друзья пользователя, рёбра - связи между ними)
    - circles: словарь с реальными кругами {название_круга: список узлов}
    - features: словарь с атрибутами узлов (не используется в структурной деанонимизации)
    """
    print(f"\nЗагрузка данных для пользователя {user_id}:")

    edges_file = os.path.join(DATA_DIR, f"{user_id}.edges")
    circles_file = os.path.join(DATA_DIR, f"{user_id}.circles")

    # Загружаем граф
    G = nx.read_edgelist(edges_file, nodetype=int)
    print(f"Граф загружен: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")

    # Загружаем круги
    circles = {}
    with open(circles_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            circle_name = parts[0]
            members = list(map(int, parts[1:]))
            circles[circle_name] = members

    print(f"Загружено {len(circles)} кругов дружбы")

    # Анализируем размеры кругов
    circle_sizes = [len(members) for members in circles.values()]
    print(f"Размеры кругов: min={min(circle_sizes)}, max={max(circle_sizes)}, avg={np.mean(circle_sizes):.1f}")

    return G, circles


def detect_communities_with_multiple_methods(G):
    """Обнаруживает сообщества разными методами и сравнивает результаты"""
    print("\nОБНАРУЖЕНИЕ СООБЩЕСТВ РАЗНЫМИ МЕТОДАМИ")

    results = {}

    # 1. Метод Лувена
    print("1. Метод Лувена")
    start = time.time()
    partition_louvain = community_louvain.best_partition(G)
    modularity_louvain = community_louvain.modularity(partition_louvain, G)
    results['louvain'] = {
        'partition': partition_louvain,
        'modularity': modularity_louvain,
        'time': time.time() - start
    }
    print(f"\tМодулярность: {modularity_louvain:.4f} | Время: {results['louvain']['time']:.2f}с")


    # 2. Girvan-Newman (только для небольших графов)
    if not LOW_END_PC:
        if G.number_of_nodes() <= 500:
            print("2. Girvan-Newman")
            start = time.time()
            try:
                # Girvan-Newman возвращает итератор, берем лучшее разделение
                communities_gn = list(nx_community.girvan_newman(G))
                # Берем первое разбиение (на 2 сообщества) как пример
                partition_gn = {node: i for i, comm in enumerate(communities_gn[0]) for node in comm}
                modularity_gn = community_louvain.modularity(partition_gn, G)
                results['girvan_newman'] = {
                    'partition': partition_gn,
                    'modularity': modularity_gn,
                    'time': time.time() - start
                }
                print(f"Модулярность: {modularity_gn:.4f} | Время: {results['girvan_newman']['time']:.2f}с")
            except Exception as e:
                print(f"Girvan-Newman не завершился {e}")
        else:
            print("2. Girvan-Newman пропущен (граф слишком большой)")
    else:
        print("2. Girvan-Newman пропущен (компьютер не вытянет)")

    # 3. Label Propagation
    print("3. Label Propagation")
    start = time.time()
    partition_lp = {node: comm for comm, nodes in enumerate(nx_community.label_propagation_communities(G)) for node in
                    nodes}
    modularity_lp = community_louvain.modularity(partition_lp, G)
    results['label_prop'] = {
        'partition': partition_lp,
        'modularity': modularity_lp,
        'time': time.time() - start
    }
    print(f"\tМодулярность: {modularity_lp:.4f} | Время: {results['label_prop']['time']:.2f}с")

    # Выбираем лучший метод по модулярности
    best_method = max(results.items(), key=lambda x: x[1]['modularity'])[0]
    print(f"\nЛучший метод: {best_method} с модулярностью {results[best_method]['modularity']:.4f}")

    return results, best_method


def visualize_communities(G, partition, circles=None):
    """
    Визуализация найденных сообществ с возможностью сравнения с ground-truth
    Мы не вычисляем NMI, а делаем возможность визуального анализа
    """
    plt.figure(figsize=(16, 12))

    # Цвета для сообществ
    unique_comms = set(partition.values())
    colors = plt.cm.tab20(range(len(unique_comms)))
    comm_colors = {comm: colors[i] for i, comm in enumerate(unique_comms)}

    # Позиции узлов (фиксированные для сравнения)
    pos = nx.spring_layout(G, seed=42) # "Эй братуха 42"

    # Рисуем узлы
    for node in G.nodes():
        comm_id = partition[node]
        color = comm_colors[comm_id]

        # Если есть ground-truth круги, помечаем их границей
        if circles:
            in_circles = [name for name, members in circles.items() if node in members]
            if in_circles:
                edge_color = 'red'
                edge_width = 2
            else:
                edge_color = 'gray'
                edge_width = 1
        else:
            edge_color = 'gray'
            edge_width = 1

        nx.draw_networkx_nodes(G, pos, nodelist=[node],
                               node_color=[color], node_size=300,
                               edgecolors=edge_color, linewidths=edge_width)

    # Рисуем рёбра
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')

    # Создаем легенду для сообществ
    legend_elements = []
    for i, comm_id in enumerate(list(unique_comms)[:10]):  # первые 10 для легенды
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=comm_colors[comm_id],
                                          markersize=10, label=f'Сообщество {comm_id}'))

    if circles:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='white', markeredgecolor='red',
                                          markersize=10, label='В ground-truth кругах'))

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Найденные сообщества")
    plt.axis('off')
    plt.tight_layout()
    print("ВИЗУАЛИЗАЦИЯ СООБЩЕСТВ")
    try:
        plt.savefig("communities_visualization.png", dpi=300, bbox_inches='tight')
        print("Сохранение успешно!")
    except Exception as e:
        print(f"Что-то пошло не так... {e}")
    if I_WANT_SEE:
        plt.show()


def create_anonymous_graph(G):
    """Создает анонимизированную версию графа (переименовывает узлы)"""
    mapping = {node: f"user_{i}" for i, node in enumerate(G.nodes())}
    G_anon = nx.relabel_nodes(G, mapping)
    return G_anon, mapping


def select_strategic_friends(G, target_node, num_friends=5, partition=None, clustering=None):
    """
    Стратегический выбор друзей на основе структуры, найденной алгоритмом
    Ключевой принцип: мы используем то разбиение, которое нашел алгоритм,
    независимо от того, совпадает ли оно с ground-truth
    """
    friends = list(G.neighbors(target_node))
    if not friends or num_friends <= 0:
        return []

    # Если нет данных о кластеризации - вычисляем
    if partition is None:
        partition = community_louvain.best_partition(G)
    if clustering is None:
        clustering = nx.clustering(G)

    # Группируем друзей по сообществам, найденным алгоритмом
    communities = defaultdict(list)
    for friend in friends:
        comm_id = partition.get(friend, -1)
        communities[comm_id].append(friend)

    # Критерий выбора внутри сообщества: низкая кластеризация + высокая степень
    def friend_priority(friend):
        cluster_coeff = clustering.get(friend, 0)
        degree = G.degree(friend)
        # Чем ниже кластеризация и выше степень, тем выше приоритет
        return (cluster_coeff, -degree)

    # Сортируем сообщества по размеру (от крупных к мелким)
    sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)

    selected = []
    used_communities = set()

    # 1. Сначала выбираем из разных сообществ
    for comm_id, members in sorted_comms:
        if len(selected) >= num_friends:
            break

        # Сортируем членов по приоритету
        members_sorted = sorted(members, key=friend_priority)

        # Выбираем лучшего кандидата из сообщества
        for candidate in members_sorted:
            if candidate not in selected:
                selected.append(candidate)
                used_communities.add(comm_id)
                break

    # 2. Если не набрали достаточно - выбираем из самых информативных узлов
    if len(selected) < num_friends:
        remaining_friends = [f for f in friends if f not in selected]
        # Сортируем по информативности: низкая кластеризация + высокая степень
        remaining_sorted = sorted(remaining_friends,
                                  key=lambda f: (clustering.get(f, 0), -G.degree(f)))
        selected.extend(remaining_sorted[:num_friends - len(selected)])

    return selected[:num_friends]


def structural_similarity(node1_friends, node2_friends):
    """Вычисляет структурную близость через коэффициент Жаккара"""
    if not node1_friends and not node2_friends:
        return 1.0

    intersection = len(set(node1_friends) & set(node2_friends))
    union = len(set(node1_friends) | set(node2_friends))
    return intersection / union if union > 0 else 0


def deanonymize_user(known_friends, graph, partition=None):
    """
    Деанонимизирует пользователя по известным друзьям с учетом кластеризации

    Улучшения:
    - Учитываем принадлежность к сообществам
    - Повышаем вес друзей из разных сообществ
    """
    best_match = None
    max_similarity = -1

    for node in graph.nodes():
        if node in known_friends:  # Не сравниваем с самими друзьями
            continue

        node_friends = list(graph.neighbors(node))

        # Вычисляем структурную близость
        similarity = structural_similarity(known_friends, node_friends)

        # Дополнительный бонус за совпадение сообществ (если доступно)
        if partition and similarity > 0:
            # Считаем, сколько друзей из каких сообществ
            known_communities = Counter([partition.get(f, -1) for f in known_friends])
            node_communities = Counter([partition.get(f, -1) for f in node_friends])

            # Вычисляем сходство по сообществам
            community_similarity = 0
            for comm_id, count in known_communities.items():
                if comm_id in node_communities:
                    # Вес пропорционален важности сообщества
                    community_similarity += min(count, node_communities[comm_id]) / len(known_friends)

            # Комбинируем структурное и сообщественное сходство
            similarity = 0.7 * similarity + 0.3 * community_similarity

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = node

    return best_match, max_similarity


def visualize_deanonymization_process(G, circles, target_node, strategic_friends, random_friends, strategic_match, random_match):
    """Визуализирует процесс деанонимизации с выделением кругов"""
    print("\nВИЗУАЛИЗАЦИЯ ПРОЦЕССА ДЕАНОНИМИЗАЦИИ")

    # Создаем анонимизированную версию для визуализации
    G_anon, mapping = create_anonymous_graph(G)

    # Инвертируем mapping для поиска оригинальных узлов
    reverse_mapping = {v: k for k, v in mapping.items()}

    # Преобразуем целевой узел и друзей в анонимизированные имена
    anon_target = mapping[target_node]
    anon_strategic_friends = [mapping[f] for f in strategic_friends if f in mapping]
    anon_random_friends = [mapping[f] for f in random_friends if f in mapping]
    anon_strategic_match = mapping.get(strategic_match, strategic_match)
    anon_random_match = mapping.get(random_match, random_match)

    # Создаем подграф для визуализации
    nodes_to_show = {anon_target, anon_strategic_match, anon_random_match}
    nodes_to_show.update(anon_strategic_friends)
    nodes_to_show.update(anon_random_friends)

    # Добавляем друзей друзей для лучшей визуализации
    for friend in list(nodes_to_show):
        if friend in G_anon:
            for fof in G_anon.neighbors(friend):
                nodes_to_show.add(fof)

    subgraph = G_anon.subgraph(nodes_to_show)

    # Определяем позиции узлов
    pos = nx.spring_layout(subgraph, seed=SEED)

    # Создаем цветовую палитру для кругов
    circle_colors = {}
    unique_circles = range(10)  # Ограничиваем количество цветов
    colors = plt.cm.tab10(range(len(unique_circles)))

    for i, circle_id in enumerate(unique_circles):
        circle_colors[circle_id] = colors[i]

    # Рисуем граф
    plt.figure(figsize=(16, 12))

    # Определяем принадлежность узлов к кругам (для оригинальных узлов)
    node_to_circle = defaultdict(int)
    for circle_id, (circle_name, members) in enumerate(list(circles.items())[:10]):  # первые 10 кругов
        for node in members:
            if node in G.nodes():
                node_to_circle[node] = circle_id % 10

    # Рисуем узлы с цветами по кругам
    for node in subgraph.nodes():
        orig_node = reverse_mapping.get(node, None)
        circle_id = node_to_circle.get(orig_node, -1) if orig_node else -1

        # Цвет по кругу
        color = circle_colors.get(circle_id, (0.8, 0.8, 0.8)) if circle_id >= 0 else (0.8, 0.8, 0.8)

        # Выделяем целевой узел
        if node == anon_target:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
                                   node_size=600, node_color='red', edgecolors='black', linewidths=2)
        # Выделяем стратегический результат
        elif node == anon_strategic_match:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
                                   node_size=500, node_color='green', edgecolors='black', linewidths=1.5)
        # Выделяем случайный результат
        elif node == anon_random_match:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
                                   node_size=500, node_color='orange', edgecolors='black', linewidths=1.5)
        # Стратегические друзья
        elif node in anon_strategic_friends:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
                                   node_size=400, node_color=[color], edgecolors='blue', linewidths=1.5)
        # Случайные друзья
        elif node in anon_random_friends:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
                                   node_size=350, node_color=[color], edgecolors='purple', linewidths=1)
        # Обычные узлы
        else:
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
                                   node_size=300, node_color=[color], edgecolors='gray', alpha=0.7)

    # Рисуем рёбра
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5, edge_color='gray')

    # Рисуем метки
    nx.draw_networkx_labels(subgraph, pos, font_size=8)

    # Добавляем легенду
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Целевой пользователь'),
        Patch(facecolor='green', edgecolor='black', label='Стратегический результат'),
        Patch(facecolor='orange', edgecolor='black', label='Случайный результат'),
        Patch(facecolor='blue', edgecolor='black', label='Стратегические друзья'),
        Patch(facecolor='purple', edgecolor='black', label='Случайные друзья')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.title(
        f"Процесс деанонимизации\nСтратегия: {'успех' if anon_strategic_match == anon_target else 'неудача'}, Случайно: {'успех' if anon_random_match == anon_target else 'неудача'}")
    plt.axis('off')
    plt.tight_layout()

    # Сохраняем и показываем
    try:
        plt.savefig("facebook_deanonymization.png", dpi=300, bbox_inches='tight')
        print("Сохранение успешно!")
    except Exception as e:
        print(f"Что-то пошло не так... {e}")
    if I_WANT_SEE:
        plt.show()


def analyze_community_structure(G, partition, target_node=None):
    """
    Визуальный анализ структуры сообществ без сравнения с ground-truth
    Цель: понять, насколько осмысленно разбиение, которое нашел алгоритм
    """
    print("\n=== ВИЗУАЛЬНЫЙ АНАЛИЗ СТРУКТУРЫ СООБЩЕСТВ ===")
    print("Анализируем: насколько осмысленно разбиение, найденное алгоритмом")

    # Статистика по сообществам
    comm_sizes = Counter(partition.values())
    print(f"Найдено {len(comm_sizes)} сообществ")
    print(
        f"Размеры сообществ: min={min(comm_sizes.values())}, max={max(comm_sizes.values())}, avg={sum(comm_sizes.values()) / len(comm_sizes):.1f}")

    # Анализ целевого узла (если указан)
    if target_node and target_node in partition:
        target_comm = partition[target_node]
        friends = list(G.neighbors(target_node))

        print(f"\nАнализ для целевого узла {target_node}:")
        print(f"  Принадлежит к сообществу: {target_comm}")

        # Распределение друзей по сообществам
        friend_comms = Counter()
        for friend in friends:
            if friend in partition:
                friend_comms[partition[friend]] += 1

        print("  Распределение друзей по сообществам:")
        for comm_id, count in friend_comms.most_common(5):
            percent = count / len(friends) * 100 if friends else 0
            print(f"    Сообщество {comm_id}: {count} друзей ({percent:.1f}%)")

    # Визуализация
    visualize_communities(G, partition)

# Генерируем SEED, чтобы потом была возможность повторить эксперимент
if SEED is None:
    SEED = random.randint(1, 1000)
random.seed(SEED)

# Шаг 2: Загружаем данные
G, circles = load_facebook_social_circles_data(DATA_DIR, user_id=0)

# Шаг 3: Обнаруживаем сообщества
results, best_method = detect_communities_with_multiple_methods(G)
partition = results[best_method]['partition']
clustering = nx.clustering(G)

# Шаг 4: Сравниваем сообщества с реальными кругами
nmi = visualize_communities(G, partition, circles)

# Шаг 5: Выбираем случайного пользователя как "жертву"
target_node = random.choice(list(G.nodes()))
print(f"\nДЕАНОНИМИЗАЦИЯ КОНКРЕТНОГО ПОЛЬЗОВАТЕЛЯ")
print(f"Целевой пользователь: {target_node}")

# Получаем его друзей
friends = list(G.neighbors(target_node))
print(f"Количество друзей: {len(friends)}")

# Стратегический выбор 5 друзей
strategic_friends = select_strategic_friends(
    G, target_node, num_friends=5, partition=partition, clustering=clustering
)

# Случайный выбор 5 друзей
random_friends = random.sample(friends, min(5, len(friends))) if friends else []

# Анализ выбранных друзей
print("\nСтратегически выбранные друзья (с анализом):")
for i, friend in enumerate(strategic_friends, 1):
    community = partition.get(friend, -1)
    cluster_coeff = clustering.get(friend, 0)
    degree = G.degree(friend)

    # Определяем роль в структуре
    if cluster_coeff < 0.3:
        role = "Мост между сообществами"
    else:
        role = "Член плотного сообщества"

    print(
        f"{i}. {friend} | Сообщество: {community} | Кластеризация: {cluster_coeff:.3f} | Степень: {degree} | Роль: {role}")

print("\nСлучайно выбранные друзья:")
for i, friend in enumerate(random_friends, 1):
    print(f"{i}. {friend}")

# Деанонимизируем
strategic_match, strategic_similarity = deanonymize_user(strategic_friends, G, partition)
random_match, random_similarity = deanonymize_user(random_friends, G, partition)

print(f"\nРЕЗУЛЬТАТЫ ДЕАНОНИМИЗАЦИИ:")
print(f"Стратегический выбор: {strategic_match} (коэффициент близости: {strategic_similarity:.4f})")
print(f"Случайный выбор: {random_match} (коэффициент близости: {random_similarity:.4f})")

# Проверяем результат
if strategic_match == target_node:
    print(f"УСПЕХ (стратегия): Пользователь идентифицирован верно! {strategic_match} == {target_node}")
else:
    print(f"ОШИБКА (стратегия): Предсказание {strategic_match}, а правильный ответ {target_node}")

if random_match == target_node:
    print(f"УСПЕХ (случайно): Пользователь идентифицирован верно! {random_match} == {target_node}")
else:
    print(f"ОШИБКА (случайно): Предсказание {random_match}, а правильный ответ {target_node}")

# Шаг 6: Визуализируем процесс
visualize_deanonymization_process(
    G, circles, target_node,
    strategic_friends, random_friends,
    strategic_match, random_match
)
