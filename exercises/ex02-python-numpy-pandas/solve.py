import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# ---------- Q3: NumPy ----------
rng = np.random.default_rng(334)
x1 = rng.random(4)
x2 = rng.integers(0, 20, size=(5, 4))
x3 = rng.random((5, 4, 3))
x4 = rng.integers(0, 10, size=(5, 4))

print("=== Q3.1 ===")
print("x3.shape:", x3.shape)
print("x2.size:", x2.size)

print("=== Q3.2 ===")
print("x2 * x4 (element-wise):")
print(x2 * x4)
print("x2 @ x1 (matrix-vector product):")
print(x2 @ x1)

# ---------- Q4: iris ----------
df = pd.read_csv('iris_cs334.csv')

print("=== Q4.1 ===")
median_petal_length = df['petal_length'].median()
mean_sepal_width_virginica = df[df['variety'] == 'Virginica']['sepal_width'].mean()
std_sepal_length_setosa = df[df['variety'] == 'Setosa']['sepal_length'].std()
print("median petal_length:", round(median_petal_length, 2))
print("mean sepal_width (Virginica):", round(mean_sepal_width_virginica, 2))
print("std sepal_length (Setosa):", round(std_sepal_length_setosa, 2))

# ---------- Q4.2: scatterplot ----------
plt.figure()
for variety, group in df.groupby('variety'):
    plt.scatter(group['petal_width'], group['petal_length'], label=variety, s=20)
plt.xlabel('petal_width')
plt.ylabel('petal_length')
plt.title('Petal Length vs Petal Width by Variety')
plt.legend()
plt.savefig('figures/q4_2_scatterplot.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------- Q4.3: boxplot ----------
plt.figure()
varieties = sorted(df['variety'].unique())
data = [df[df['variety'] == v]['sepal_length'] for v in varieties]
plt.boxplot(data, labels=varieties)
plt.xlabel('variety')
plt.ylabel('sepal_length')
plt.title('Sepal Length by Variety')
plt.savefig('figures/q4_3_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plots saved.")
