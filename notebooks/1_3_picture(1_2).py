import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Загрузка сохраненного датасета из локальной папки проекта
df = pd.read_csv("insurance.csv")

# Настройка общего уникального стиля графиков
sns.set_theme(style="whitegrid")  # Добавляем аккуратную сетку на фон
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})

# ==========================================
# ЭТАП 1: Визуализация распределения вещественных признаков
# Изменяем цвет на глубокий изумрудно-зеленый (teal)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Распределение возраста (age)
sns.histplot(
    data=df,
    x="age",
    kde=True,
    color="#008080",
    stat="density",
    ax=axes[0],
    alpha=0.5,
)
axes[0].set_title("Распределение age")
axes[0].set_ylabel("Density")

# Распределение индекса массы тела (bmi)
sns.histplot(
    data=df,
    x="bmi",
    kde=True,
    color="#008080",
    stat="density",
    ax=axes[1],
    alpha=0.5,
)
axes[1].set_title("Распределение bmi")
axes[1].set_ylabel("Density")

# Распределение целевой переменной (charges)
sns.histplot(
    data=df,
    x="charges",
    kde=True,
    color="#008080",
    stat="density",
    ax=axes[2],
    alpha=0.5,
)
axes[2].set_title("Распределение charges")
axes[2].set_ylabel("Density")

plt.tight_layout()
plt.savefig("figure1.png", dpi=300)
plt.show()

# ==========================================
# ЭТАП 2: Визуализация категориальных признаков
# Изменяем цвет на приглушенный терракотовый / коралловый
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Распределение по полу (sex)
sns.countplot(data=df, x="sex", color="#E07A5F", ax=axes[0], alpha=0.9)
axes[0].set_title("Распределение по полу (sex)")
axes[0].set_ylabel("Количество")

# Распределение по статусу курения (smoker)
sns.countplot(data=df, x="smoker", color="#E07A5F", ax=axes[1], alpha=0.9)
axes[1].set_title("Статус курения (smoker)")
axes[1].set_ylabel("Количество")

# Распределение по регионам (region)
sns.countplot(data=df, x="region", color="#E07A5F", ax=axes[2], alpha=0.9)
axes[2].set_title("Распределение по регионам (region)")
axes[2].set_ylabel("Количество")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=15)

plt.tight_layout()
plt.savefig("figure2.png", dpi=300)
plt.show()
