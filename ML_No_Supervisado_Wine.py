# -*- coding: utf-8 -*-
# ML_No_Supervisado_Wine.py – versión script del notebook de Google Colab

# ==============================================================================
# # Informe de Trabajo – Machine Learning No Supervisado
# **Clustering (K-means y Jerárquico) y Reducción de Dimensionalidad (PCA y SVD) sobre el dataset *Wine***
# 
# Autor: Diego Yanzir Gutiérrez · Curso: Introducción a Machine Learning
# 
# Repositorio: https://github.com/DiegoYanzir/Informe-de-Trabajo-ML-no-Supervisado
# 
# Este notebook es autoejecutable en Google Colab (*Entorno de ejecución → Ejecutar todas*). No requiere archivos externos: el dataset se carga desde `scikit-learn`. Cada gráfica se muestra en pantalla y, además, se guarda como PNG en la carpeta `figuras/` para insertarla en el informe.
# ==============================================================================

# ==============================================================================
# ## 1. Importación de librerías
# ==============================================================================

# --- Librerías base ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Scikit-learn: datos, preprocesamiento, modelos y métricas ---
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

# --- SciPy: clustering jerárquico y dendrograma ---
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

# --- Configuración general ---
SEMILLA = 42                      # reproducibilidad de todos los resultados
np.random.seed(SEMILLA)
pd.set_option("display.precision", 3)
os.makedirs("figuras", exist_ok=True)

# Estilo de las gráficas: paleta categórica apta para daltonismo + marcadores distintos
COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]   # azul, naranja, aguamarina
MARCADORES = ["o", "s", "^"]
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984", "axes.titleweight": "bold",
})

def guardar_figura(nombre):
    # Guarda la figura actual en figuras/<nombre>.png y la muestra
    plt.tight_layout()
    plt.savefig(f"figuras/{nombre}.png", bbox_inches="tight")
    plt.show()

def alinear_etiquetas(etiquetas, referencia):
    # Renumera los clústeres según la clase real mayoritaria de cada uno.
    # Solo cambia el NOMBRE de cada grupo (0, 1, 2) para que colores y tablas sean
    # comparables con los cultivares; no altera la partición encontrada.
    tabla = pd.crosstab(etiquetas, referencia)
    mapa = {c: tabla.loc[c].idxmax() for c in tabla.index}
    return np.array([mapa[e] for e in etiquetas])

def dispersion_por_grupo(ax, X2d, etiquetas, prefijo="Clúster"):
    # Dibuja un diagrama de dispersión 2D con un color y un marcador por grupo
    for i, g in enumerate(np.unique(etiquetas)):
        m = etiquetas == g
        ax.scatter(X2d[m, 0], X2d[m, 1], s=34, c=COLORES[i % 3],
                   marker=MARCADORES[i % 3], edgecolors="white", linewidths=0.6,
                   alpha=0.9, label=f"{prefijo} {g}")
    ax.legend(frameon=False)

print("Librerías cargadas correctamente.")

# ==============================================================================
# ## 2. Carga, exploración y preprocesamiento de los datos
# El dataset *Wine* (UCI) contiene 178 muestras de vino de tres cultivares de la misma región de Italia, descritas por 13 variables fisicoquímicas continuas. Las etiquetas reales (`target`) **no** se usan para entrenar: solo se reservan para validar externamente los agrupamientos.
# ==============================================================================

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y_real = wine.target                     # solo para validación externa

print(f"Dimensiones: {X.shape[0]} muestras × {X.shape[1]} variables")
print("Valores faltantes:", int(X.isna().sum().sum()))
print("Distribución de clases reales:", np.bincount(y_real))
print(X.describe().T[["mean", "std", "min", "max"]])

# Las variables tienen escalas muy distintas (p. ej., 'proline' ~ 10^3 frente a 'hue' ~ 1).
# K-means, el clustering jerárquico y PCA se basan en distancias/varianzas, por lo que
# se estandariza cada variable a media 0 y desviación típica 1: z = (x - μ) / σ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

resumen = pd.DataFrame({"media": X_scaled.mean(axis=0), "desv_std": X_scaled.std(axis=0)},
                       index=X.columns)
print("Comprobación tras el escalado (media ≈ 0, desv. típica = 1):")
print(resumen.round(3))

# ==============================================================================
# ## 3. Clustering con K-means
# ### 3.1 Método del codo (inercia) y coeficiente de silueta
# Se ajusta K-means para *k* = 1…10 y se registra la inercia (WCSS). Como criterio complementario, se calcula el coeficiente de silueta para *k* ≥ 2.
# ==============================================================================

rango_k = range(1, 11)
inercias, siluetas = [], []
for k in rango_k:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=SEMILLA)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_scaled, km.labels_) if k > 1 else np.nan)

tabla_k = pd.DataFrame({"k": list(rango_k), "inercia (WCSS)": inercias, "silueta": siluetas})
print(tabla_k.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(rango_k), inercias, "-o", color=COLORES[0], lw=2, ms=7)
axes[0].axvline(3, ls="--", color="#8a8984", lw=1)
axes[0].annotate("codo (k = 3)", xy=(3, inercias[2]), xytext=(4.5, inercias[2] + 300),
                 arrowprops=dict(arrowstyle="->", color="#52514e"), color="#52514e")
axes[0].set(title="Método del codo", xlabel="Número de clústeres (k)",
            ylabel="Inercia (WCSS)", xticks=list(rango_k))
axes[1].plot(list(rango_k)[1:], siluetas[1:], "-o", color=COLORES[1], lw=2, ms=7)
axes[1].axvline(3, ls="--", color="#8a8984", lw=1)
axes[1].set(title="Coeficiente de silueta por k", xlabel="Número de clústeres (k)",
            ylabel="Silueta media", xticks=list(rango_k)[1:])
guardar_figura("01_kmeans_codo_silueta")

# ==============================================================================
# ### 3.2 Ajuste del modelo final (k = 3) y métricas
# ==============================================================================

K_OPTIMO = 3
kmeans = KMeans(n_clusters=K_OPTIMO, init="k-means++", n_init=10, random_state=SEMILLA)
etiquetas_km = alinear_etiquetas(kmeans.fit_predict(X_scaled), y_real)

def metricas(nombre, etiquetas):
    # Métricas internas (no usan etiquetas reales) y externa (ARI, usa etiquetas reales)
    return {"modelo": nombre,
            "silueta": silhouette_score(X_scaled, etiquetas),
            "Davies-Bouldin": davies_bouldin_score(X_scaled, etiquetas),
            "Calinski-Harabasz": calinski_harabasz_score(X_scaled, etiquetas),
            "ARI vs. clases reales": adjusted_rand_score(y_real, etiquetas)}

print(f"Inercia final: {kmeans.inertia_:.2f}   Iteraciones: {kmeans.n_iter_}")
print("Tamaño de los clústeres:", np.bincount(etiquetas_km))
print(pd.DataFrame([metricas("K-means (k=3)", etiquetas_km)]).round(3))

# Tabla de contingencia: clústeres encontrados frente a cultivares reales
pd.crosstab(pd.Series(y_real, name="Cultivar real"),
            pd.Series(etiquetas_km, name="Clúster K-means"))

# Perfil de cada clúster en unidades originales (interpretación de los centroides)
perfil_km = X.groupby(etiquetas_km).mean().T
perfil_km.columns = [f"Clúster {c}" for c in perfil_km.columns]
print(perfil_km.round(2))

# ==============================================================================
# ### 3.3 Visualización de los clústeres
# Los datos tienen 13 dimensiones; para visualizarlos se proyectan sobre los dos primeros componentes principales (sección 5). Los centroides se proyectan con la misma transformación.
# ==============================================================================

pca_vis = PCA(n_components=2, random_state=SEMILLA).fit(X_scaled)
X_vis = pca_vis.transform(X_scaled)
centroides_vis = pca_vis.transform(np.vstack(
    [X_scaled[etiquetas_km == g].mean(axis=0) for g in range(3)]))   # centroides ya renumerados

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
dispersion_por_grupo(axes[0], X_vis, etiquetas_km)
axes[0].scatter(centroides_vis[:, 0], centroides_vis[:, 1], s=260, marker="X",
                c="#0b0b0b", edgecolors="white", linewidths=1.5, label="Centroides")
axes[0].legend(frameon=False)
axes[0].set(title="K-means (k = 3) proyectado en PCA", xlabel="CP1", ylabel="CP2")
dispersion_por_grupo(axes[1], X_vis, y_real, prefijo="Cultivar")
axes[1].set(title="Cultivares reales (referencia)", xlabel="CP1")
guardar_figura("02_kmeans_clusters")

# ==============================================================================
# ## 4. Clustering Jerárquico Aglomerativo
# ### 4.1 Dendrograma (enlace de Ward)
# El método de Ward fusiona, en cada paso, el par de clústeres que produce el menor aumento de la varianza intraclúster.
# ==============================================================================

Z = linkage(X_scaled, method="ward")       # matriz de enlace (n-1 fusiones)
set_link_color_palette(COLORES)            # colores de las ramas bajo el corte

# Altura de corte que produce 3 clústeres: punto medio entre la 2.ª y 3.ª fusión más altas
alturas = np.sort(Z[:, 2])
corte = (alturas[-2] + alturas[-3]) / 2
print(f"Alturas de las últimas fusiones: {np.round(alturas[-5:], 2)}")
print(f"Altura de corte elegida: {corte:.2f}")

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True, leaf_rotation=90,
           leaf_font_size=8, color_threshold=corte, above_threshold_color="#8a8984", ax=ax)
ax.axhline(corte, ls="--", color="#e34948", lw=1.5)
ax.text(ax.get_xlim()[1] * 0.99, corte + 0.6, f"corte = {corte:.1f} → 3 clústeres",
        ha="right", color="#52514e")
ax.set(title="Dendrograma – enlace de Ward (últimas 30 fusiones)",
       xlabel="Muestras o (tamaño del subgrupo)", ylabel="Distancia de Ward")
ax.grid(False)
guardar_figura("03_dendrograma")

# ==============================================================================
# ### 4.2 Ajuste con `AgglomerativeClustering` y comparación de criterios de enlace
# ==============================================================================

filas = []
for enlace in ["ward", "complete", "average", "single"]:
    modelo = AgglomerativeClustering(n_clusters=3, linkage=enlace)
    filas.append(metricas(f"Jerárquico ({enlace})", modelo.fit_predict(X_scaled)))
print(pd.DataFrame(filas).round(3))

jerarquico = AgglomerativeClustering(n_clusters=3, linkage="ward")
etiquetas_hc = alinear_etiquetas(jerarquico.fit_predict(X_scaled), y_real)
print("Tamaño de los clústeres:", np.bincount(etiquetas_hc))
print("Concordancia K-means vs. Jerárquico (ARI):",
      round(adjusted_rand_score(etiquetas_km, etiquetas_hc), 3))
pd.crosstab(pd.Series(y_real, name="Cultivar real"),
            pd.Series(etiquetas_hc, name="Clúster jerárquico"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
dispersion_por_grupo(axes[0], X_vis, etiquetas_hc)
axes[0].set(title="Jerárquico aglomerativo (Ward, 3 clústeres)", xlabel="CP1", ylabel="CP2")
dispersion_por_grupo(axes[1], X_vis, etiquetas_km)
axes[1].set(title="K-means (k = 3)", xlabel="CP1")
guardar_figura("04_jerarquico_clusters")

# ==============================================================================
# ### 4.3 Comparación global de los modelos de clustering
# ==============================================================================

comparacion = pd.DataFrame([metricas("K-means (k=3)", etiquetas_km),
                            metricas("Jerárquico Ward (3)", etiquetas_hc)]).round(3)
print(comparacion)

# ==============================================================================
# ## 5. Reducción de dimensionalidad con PCA
# ### 5.1 Varianza explicada por cada componente
# ==============================================================================

pca_full = PCA(random_state=SEMILLA).fit(X_scaled)      # los 13 componentes
var_ratio = pca_full.explained_variance_ratio_
var_acum = np.cumsum(var_ratio)

tabla_pca = pd.DataFrame({"componente": [f"CP{i+1}" for i in range(len(var_ratio))],
                          "autovalor": pca_full.explained_variance_,
                          "varianza explicada": var_ratio,
                          "varianza acumulada": var_acum})
print(tabla_pca.round(3).to_string(index=False))
print(f"\nComponentes necesarios para retener ≥ 80 %: {np.argmax(var_acum >= 0.80) + 1}")
print(f"Componentes necesarios para retener ≥ 95 %: {np.argmax(var_acum >= 0.95) + 1}")

fig, ax = plt.subplots(figsize=(10, 4.2))
idx = np.arange(1, len(var_ratio) + 1)
ax.bar(idx, var_ratio * 100, color=COLORES[0], width=0.7, label="Individual")
ax.plot(idx, var_acum * 100, "-o", color=COLORES[1], lw=2, ms=6, label="Acumulada")
ax.axhline(80, ls="--", color="#8a8984", lw=1)
ax.text(13.4, 81, "80 %", color="#52514e", ha="right")
for i in range(3):
    ax.annotate(f"{var_acum[i]*100:.1f} %", (idx[i], var_acum[i] * 100),
                textcoords="offset points", xytext=(0, 8), ha="center", color="#52514e")
ax.set(title="PCA – varianza explicada (scree plot)", xlabel="Componente principal",
       ylabel="Varianza explicada (%)", xticks=idx, ylim=(0, 108))
ax.legend(frameon=False, loc="center right")
guardar_figura("05_pca_varianza")

# ==============================================================================
# ### 5.2 Reducción a 2 componentes y visualización 2D
# ==============================================================================

pca = PCA(n_components=2, random_state=SEMILLA)
X_pca = pca.fit_transform(X_scaled)
print("Varianza explicada por CP1 y CP2:", np.round(pca.explained_variance_ratio_, 4))
print(f"Varianza total retenida en 2D: {pca.explained_variance_ratio_.sum()*100:.2f} %")

# Cargas (loadings): contribución de cada variable original a cada componente
cargas = pd.DataFrame(pca.components_.T, index=X.columns, columns=["CP1", "CP2"])
print("\nVariables con mayor carga absoluta:")
print(cargas.reindex(cargas["CP1"].abs().sort_values(ascending=False).index).round(3))

fig, ax = plt.subplots(figsize=(8.5, 6.5))
dispersion_por_grupo(ax, X_pca, y_real, prefijo="Cultivar")
# Biplot: flechas de las variables con mayor peso en el plano CP1–CP2
escala = 3.2
top = (cargas**2).sum(axis=1).sort_values(ascending=False).index[:6]
for var in top:
    ax.arrow(0, 0, cargas.loc[var, "CP1"] * escala, cargas.loc[var, "CP2"] * escala,
             color="#52514e", width=0.012, head_width=0.09, alpha=0.8)
    ax.text(cargas.loc[var, "CP1"] * escala * 1.13, cargas.loc[var, "CP2"] * escala * 1.13,
            var.replace("_of_diluted_wines", ""), fontsize=8, color="#0b0b0b", ha="center")
ax.set(title=f"PCA 2D – {pca.explained_variance_ratio_.sum()*100:.1f} % de la varianza",
       xlabel=f"CP1 ({pca.explained_variance_ratio_[0]*100:.1f} %)",
       ylabel=f"CP2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
guardar_figura("06_pca_2d_biplot")

# ==============================================================================
# ## 6. Reducción de dimensionalidad con SVD (`TruncatedSVD`)
# `TruncatedSVD` factoriza la matriz de datos como **X ≈ U<sub>k</sub> Σ<sub>k</sub> V<sub>k</sub><sup>T</sup>** sin centrarla previamente, lo que la hace apta para matrices dispersas. Se aplica en dos escenarios:
# 1. **Datos estandarizados** (centrados): el resultado debe coincidir con PCA.
# 2. **Datos originales sin centrar ni escalar**: muestra el efecto de omitir el preprocesamiento.
# ==============================================================================

# Escenario 1: SVD sobre datos estandarizados (centrados)
svd = TruncatedSVD(n_components=2, algorithm="randomized", n_iter=10, random_state=SEMILLA)
X_svd = svd.fit_transform(X_scaled)
print("SVD (datos estandarizados)")
print("  Valores singulares:", np.round(svd.singular_values_, 3))
print("  Varianza explicada:", np.round(svd.explained_variance_ratio_, 4),
      f"→ total {svd.explained_variance_ratio_.sum()*100:.2f} %")

# Escenario 2: SVD sobre datos originales (sin centrar ni escalar)
svd_raw = TruncatedSVD(n_components=2, random_state=SEMILLA)
X_svd_raw = svd_raw.fit_transform(X.values)
print("\nSVD (datos originales sin escalar)")
print("  Varianza explicada:", np.round(svd_raw.explained_variance_ratio_, 4),
      f"→ total {svd_raw.explained_variance_ratio_.sum()*100:.2f} %")
peso_v1 = pd.Series(np.abs(svd_raw.components_[0]), index=X.columns).sort_values(ascending=False)
print("  Variable dominante en el 1.er vector singular:", peso_v1.index[0],
      f"(peso {peso_v1.iloc[0]:.4f})")

# Verificación numérica: con datos centrados, SVD y PCA son equivalentes.
# Relación teórica: λ_i = σ_i² / (n − 1)
n = X_scaled.shape[0]
print("\nλ (PCA):        ", np.round(pca.explained_variance_, 4))
print("σ² / (n − 1):   ", np.round(svd.singular_values_**2 / (n - 1), 4))
print("|corr| CP1–SV1: ", round(abs(np.corrcoef(X_pca[:, 0], X_svd[:, 0])[0, 1]), 6))
print("|corr| CP2–SV2: ", round(abs(np.corrcoef(X_pca[:, 1], X_svd[:, 1])[0, 1]), 6))

# Varianza explicada acumulada con SVD para 1..12 componentes (datos estandarizados)
svd_12 = TruncatedSVD(n_components=12, algorithm="arpack", random_state=SEMILLA).fit(X_scaled)
tabla_svd = pd.DataFrame({"componentes": range(1, 13),
                          "SVD acumulada": np.cumsum(svd_12.explained_variance_ratio_),
                          "PCA acumulada": var_acum[:12]})
print(tabla_svd.round(4).T)

# Gráfica comparativa: PCA vs. SVD estandarizado vs. SVD sin escalar
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
dispersion_por_grupo(axes[0], X_pca, y_real, prefijo="Cultivar")
axes[0].set(title=f"PCA ({pca.explained_variance_ratio_.sum()*100:.1f} %)",
            xlabel="CP1", ylabel="CP2")
dispersion_por_grupo(axes[1], X_svd, y_real, prefijo="Cultivar")
axes[1].set(title=f"TruncatedSVD – estandarizado ({svd.explained_variance_ratio_.sum()*100:.1f} %)",
            xlabel="SV1", ylabel="SV2")
dispersion_por_grupo(axes[2], X_svd_raw, y_real, prefijo="Cultivar")
axes[2].set(title=f"TruncatedSVD – sin escalar ({svd_raw.explained_variance_ratio_.sum()*100:.1f} %)",
            xlabel="SV1", ylabel="SV2")
guardar_figura("07_pca_vs_svd")

fig, ax = plt.subplots(figsize=(9, 4))
comp = np.arange(1, 13)
ax.plot(comp, var_acum[:12] * 100, "-o", color=COLORES[0], lw=2, ms=8, label="PCA")
ax.plot(comp, np.cumsum(svd_12.explained_variance_ratio_) * 100, "--s", color=COLORES[1],
        lw=2, ms=5, label="TruncatedSVD (estandarizado)")
ax.set(title="Varianza explicada acumulada: PCA vs. SVD", xlabel="Número de componentes",
       ylabel="Varianza acumulada (%)", xticks=comp)
ax.legend(frameon=False, loc="lower right")
guardar_figura("08_varianza_pca_vs_svd")

# ==============================================================================
# ## 7. Clustering en el espacio reducido (PCA 2D) – efecto de la reducción
# Se evalúa si agrupar sobre 2 componentes conserva la estructura encontrada en las 13 variables.
# ==============================================================================

km_pca = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(X_pca)
resumen_final = pd.DataFrame({
    "espacio": ["13 variables estandarizadas", "PCA (2 componentes)"],
    "silueta (en su espacio)": [silhouette_score(X_scaled, etiquetas_km),
                                silhouette_score(X_pca, km_pca.labels_)],
    "ARI vs. clases reales": [adjusted_rand_score(y_real, etiquetas_km),
                              adjusted_rand_score(y_real, km_pca.labels_)],
    "ARI vs. K-means 13D": [1.0, adjusted_rand_score(etiquetas_km, km_pca.labels_)]})
print(resumen_final.round(3))

# ==============================================================================
# ## 8. Conclusión del notebook
# - El método del codo y la silueta coinciden en **k = 3**, que corresponde con los tres cultivares del dataset.
# - K-means y el clustering jerárquico de Ward producen particiones muy parecidas y con alta concordancia con las clases reales, sin haberlas usado en el entrenamiento.
# - Con 2 componentes, PCA retiene aproximadamente el 55 % de la varianza y basta para separar visualmente los tres grupos.
# - `TruncatedSVD` sobre datos estandarizados reproduce PCA; sin escalar, la variable de mayor magnitud (*proline*) domina la proyección, lo que confirma la importancia del preprocesamiento.
# ==============================================================================
