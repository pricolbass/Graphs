from faker import Faker
import networkx as nx
import random

fake = Faker()

G = nx.read_edgelist("facebook_combined.txt")

for node in G.nodes():
    # Генерируем User-Agent (пример: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...")
    user_agent = fake.user_agent()
    # print(user_agent)

    # Парсим ОС и браузер из User-Agent
    os = "Windows" if "Windows" in user_agent else "MacOS" if "Mac" in user_agent else "Linux"
    browser = "Chrome" if "Chrome" in user_agent else "Firefox" if "Firefox" in user_agent else "Safari"

    # Добавляем атрибуты
    G.nodes[node]["os"] = os
    G.nodes[node]["browser"] = browser
    G.nodes[node]["resolution"] = random.choice(["1920x1080", "1366x768", "2560x1440"])
    G.nodes[node]["activity_time"] = random.randint(8, 22)

# Сохраняем граф с атрибутами в формате GraphML (чтобы не терять данные)
nx.write_graphml(G, "facebook_with_attributes.graphml")