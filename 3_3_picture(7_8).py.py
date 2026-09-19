import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# ШАГ 0: Создание тестовой структуры данных (имитация датасета)
# ==========================================
# Имена классов в формате YOLO
class_names = {
    0: "Hardhat (Каска)",
    1: "Safety Vest (Жилет)",
    2: "NO-Hardhat (Без каски)",
    3: "NO-Safety Vest (Без жилета)",
    4: "Person (Человек)",
}

# Генерируем тестовые текстовые аннотации YOLO (715 рамок на 260 картинках)
np.random.seed(42)
mock_annotations = []
for img_id in range(1, 261):
    # На каждой картинке от 2 до 5 объектов СИЗ
    num_objects = np.random.randint(2, 6)
    for _ in range(num_objects):
        # Генерируем ID класса с учетом дисбаланса (касок и жилетов больше, чем нарушений)
        cls_id = np.random.choice([0, 1, 2, 3, 4], p=[0.35, 0.30, 0.08, 0.07, 0.20])
        # Генерируем нормализованные координаты YOLO [x_center, y_center, width, height]
        w = np.random.uniform(0.05, 0.25)
        h = np.random.uniform(0.10, 0.50)
        x = np.random.uniform(w / 2, 1.0 - w / 2)
        y = np.random.uniform(h / 2, 1.0 - h / 2)
        mock_annotations.append([img_id, cls_id, x, y, w, h])

df_annotations = pd.DataFrame(
    mock_annotations, columns=["img_id", "class_id", "x", "y", "w", "h"]
)
df_annotations["class_name"] = df_annotations["class_id"].map(class_names)

# ==========================================
# ШАГ 1: Построение графика баланса классов (Изумрудный стиль)
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})

plt.figure(figsize=(10, 5))
class_counts = df_annotations["class_name"].value_counts()
sns.barplot(
    x=class_counts.values,
    y=class_counts.index,
    color="#008080",
    alpha=0.85,
    edgecolor="black",
)
plt.title("Анализ количества и баланса классов в аннотациях СИЗ (YOLO)")
plt.xlabel("Количество размеченных объектов (рамок), шт.")
plt.ylabel("Название целевого класса")
plt.tight_layout()
plt.savefig("figure7.png", dpi=300)
plt.show()

# ==========================================
# ШАГ 2: Построение графиков геометрии аннотаций (Размеры рамок)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Распределение ширины рамок
sns.histplot(
    data=df_annotations,
    x="w",
    kde=True,
    color="#008080",
    ax=axes[0],
    alpha=0.5,
)
axes[0].set_title("Распределение ширины прямоугольных рамок (Width)")
axes[0].set_xlabel("Нормализованная ширина рамки")
axes[0].set_ylabel("Частота")

# Распределение высоты рамок
sns.histplot(
    data=df_annotations,
    x="h",
    kde=True,
    color="#E07A5F",
    ax=axes[1],
    alpha=0.6,
)
axes[1].set_title("Распределение высоты прямоугольных рамок (Height)")
axes[1].set_xlabel("Нормализованная высота рамки")
axes[1].set_ylabel("Частота")

plt.tight_layout()
plt.savefig("figure8.png", dpi=300)
plt.show()

# Печатаем краткую сводку для текста отчета
print(df_annotations["class_name"].value_counts())
