from logging import exception
import networkx as nx
import random
import matplotlib.pyplot as plt


# seed = None
seed = 0
number_of_friends = 5

def load_facebook_graph():
    # Загружает граф из файла
    # Возвращает: NetworkX Graph
    G = nx.read_edgelist("facebook_combined.txt")
    print(f"Граф загружен! {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    return G


def create_anonymous_graph(G):
    # Создает анонимизированную версию графа (переименовывает узлы)
    # Возвращает: NetworkX Graph с анонимными ID
    # Создаем словарь для анонимизации
    mapping = {node: f"user_{i}" for i, node in enumerate(G.nodes())}
    G_anon = nx.relabel_nodes(G, mapping)
    return G_anon


def structural_similarity(node1_friends, node2_friends):
    # Вычисляет структурную близость через коэффициент Жаккара
    # Формула: |N(v) ∩ N(u)| / |N(v) ∪ N(u)|

    intersection = len(set(node1_friends) & set(node2_friends))
    union = len(set(node1_friends) | set(node2_friends))
    return intersection / union if union > 0 else 0


def deanonymize_user(known_friends, graph):
    """
    Деанонимизирует пользователя по известным друзьям

    Алгоритм:
    1. Для каждого узла в графе вычисляем структурную близость
    2. Выбираем узел с максимальной близостью
    """

    best_match = None
    max_similarity = 0

    for node in graph.nodes():
        node_friends = list(graph.neighbors(node))
        similarity = structural_similarity(known_friends, node_friends)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = node

    return best_match, max_similarity


def visualize_deanonymization(graph, target_node, known_friends, match):
    # Визуализирует процесс деанонимизации

    # Создаем подграф для визуализации
    nodes_to_show = {target_node, match}
    for friend in known_friends:
        nodes_to_show.add(friend)
        # Добавляем друзей друзей для лучшей визуализации
        for fof in graph.neighbors(friend):
            nodes_to_show.add(fof)

    subgraph = graph.subgraph(nodes_to_show)

    # Определяем позиции узлов
    pos = nx.spring_layout(subgraph, seed=42)

    # Рисуем граф
    plt.figure(figsize=(12, 8))

    # Рисуем все узлы
    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color='lightblue')

    # Выделяем целевой узел красным
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[target_node], node_size=500, node_color='red')

    # Выделяем найденный узел зеленым (если это не целевой)
    if match != target_node:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[match], node_size=500, node_color='green')

    # Рисуем рёбра
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)

    # Рисуем метки
    nx.draw_networkx_labels(subgraph, pos, font_size=8)

    plt.title(f"Процесс деанонимизации\nЦелевой узел: {target_node}, Найденный: {match}\nКоличество друзей: {number_of_friends}")
    plt.axis('off')
    plt.tight_layout()
    try:
        plt.savefig("deanonymization_visualization.png", dpi=300)
        print("Сохранение успешно!")
        plt.show()
    except exception as e:
        print(e)


# Генерируем seed, чтобы потом была возможность повторить эксперимент
if seed is None:
    seed = random.randint(1, 1000)
random.seed(seed)

# Загружаем граф
G = load_facebook_graph()

# Создаем анонимизированную версию
G_anon = create_anonymous_graph(G)

# Показываем seed
print(f"Seed: {seed}")

# Выбираем случайного пользователя как "жертву"
target_node = random.choice(list(G_anon.nodes()))
print(f"\nЦелевой пользователь для деанонимизации: {target_node}")

# Получаем его друзей
friends = list(G_anon.neighbors(target_node))
print(f"Количество друзей: {len(friends)}")

# Симулируем знание друзей
num_known = min(number_of_friends, len(friends))
known_friends = random.sample(friends, num_known) if friends else []
print(f"Известные друзья ({num_known}): {known_friends}")

# МКД момент (деанонимизация)
match, similarity = deanonymize_user(known_friends, G_anon)
print(f"\nРЕЗУЛЬТАТ: Деанонимизирован {match} с коэффициентом близости {similarity:.4f}")

# Проверяем результат
if match == target_node:
    print("УСПЕХ: Пользователь идентифицирован верно!")
else:
    print(f"ОШИБКА: Предсказание {match}, а правильный ответ {target_node}")

# Визуализируем
visualize_deanonymization(G_anon, target_node, known_friends, match)