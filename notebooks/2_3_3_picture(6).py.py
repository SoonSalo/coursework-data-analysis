import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Загрузка датасета временного ряда
df = pd.read_csv("tuv_nel_data.csv")

# Настройка стиля графиков
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})

# Создаем общее окно для двух типов шкал (расходы/плотность и доли)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Диапазоны расходов
sns.boxplot(
    data=df[["mass_flow_true", "mass_flow_raw"]],
    palette=["#008080", "#E07A5F"],
    ax=axes[0],
    width=0.5,
)
axes[0].set_title("Диапазоны массовых расходов")
axes[0].set_ylabel("Расход, кг/с")
axes[0].set_xticklabels(["Истинный", "Сырой"])

# 2. Диапазон плотности
sns.boxplot(data=df[["density"]], color="#008080", ax=axes[1], width=0.3)
axes[1].set_title("Диапазон плотности смеси")
axes[1].set_ylabel("Плотность, кг/м³")
axes[1].set_xticklabels(["Плотность"])

# 3. Диапазоны безразмерных долей (GVF и WC)
sns.boxplot(
    data=df[["GVF", "WC"]],
    palette=["#008080", "#E07A5F"],
    ax=axes[2],
    width=0.5,
)
axes[2].set_title("Диапазоны GVF и WC")
axes[2].set_ylabel("Доля (от 0.0 до 1.0)")
axes[2].set_xticklabels(["GVF (газ)", "WC (вода)"])

plt.tight_layout()
# Сохраняем график для отчета
plt.savefig("figure6.png", dpi=300)
plt.show()
