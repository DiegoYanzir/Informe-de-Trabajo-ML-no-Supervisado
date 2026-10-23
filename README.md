# Informe de Trabajo – ML No Supervisado

Clustering (K-means y jerárquico aglomerativo) y reducción de dimensionalidad (PCA y TruncatedSVD) sobre el dataset *Wine* de scikit-learn.

| Archivo | Contenido |
|---|---|
| `ML_No_Supervisado_Wine.ipynb` | Notebook de Google Colab (Entorno de ejecución → Ejecutar todas) |
| `ML_No_Supervisado_Wine.py` | Mismo código como script de Python |
| `figuras/` | Gráficas generadas por el notebook (PNG) |

**Resultados principales:** k = 3 (codo y silueta); K-means ARI = 0.897 (96.6 % de coincidencia con los cultivares); Ward ARI = 0.790; PCA 2D retiene el 55.4 % de la varianza; TruncatedSVD sobre datos estandarizados reproduce exactamente a PCA.

Requisitos: numpy, pandas, matplotlib, scikit-learn, scipy (preinstalados en Colab).
