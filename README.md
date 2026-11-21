# Predicción de oportunidades de negocio ganadas o perdidas

Modelo que estima si una oportunidad de venta se gana o se pierde, a partir de datos de
oportunidades de 7 países latinoamericanos. El foco es la **integración de datos entre
países**: en vez de un único modelo global o un modelo aislado por país, se entrena cada
país con los países cuyo proceso de decisión se parece al suyo.

## Datos (`BD/`)
Registros de oportunidades de venta de una empresa de accesorios y repuestos de automóvil, con el
resultado final de cada una (ganada o perdida). El repositorio no documenta el origen del conjunto
de datos.

**`oportunidad.csv`** (entrenamiento): 31.823 oportunidades, 16 columnas, 7 países.
- **Resultado** (`Opportunity Result`): 7.246 ganadas (23 %) y 24.577 perdidas (77 %).
- **Producto:** `Supplies Group` (4 grupos: Car Accessories 20.294, Performance & Non-auto 11.168,
  Tires & Wheels 241, Car Electronics 120) y `Supplies Subgroup` (11 subgrupos).
- **Proceso de venta:** `Sales Stage Change Count` (cambios de etapa, de 1 a 20, mediana 3),
  `Total Days Identified Through Closing` (días hasta el cierre, de 0 a 192, mediana 12) y
  `Total Days Identified Through Qualified` (días hasta la calificación).
- **Valor:** `Opportunity Amount USD` (de 0 a 1.000.000, mediana 49.418) y `Deal Size Category`
  (7 categorías).
- **Cliente:** `Client Size By Revenue` y `Client Size By Employee Count` (escala 1-5; el 76 % está en
  la categoría 1) y `Revenue From Client Past Two Years` (escala 0-4; la mayoría es 0).
- **Competencia:** `Competitor Type` (Unknown 23.119, Known 4.925, None 3.779).
- **Ubicación:** `Lat`, `Lon` y `Country_Code`. Oportunidades por país: Perú 8.569, Chile 6.197,
  Brasil 3.885, Ecuador 3.765, Bolivia 3.298, Venezuela 3.123 y Colombia 2.986.
- `Opportunity Number` es el identificador; 308 filas no lo tienen y no se usa como variable.

**`NewOpportunitiesList.xlsx`:** 31.823 oportunidades con las mismas variables, sin `Opportunity Result`,
más las columnas `Country` e `ID`. Ver limitaciones.

**`Countries.xlsx`:** tabla de 9 países (`Country`, `ID`). Francia y Alemania no tienen oportunidades
en `oportunidad.csv`.

## Método (`main.py`)
1. Codificación de variables categóricas y estandarización (dentro de cada fold).
2. Clasificador AdaBoost (200 estimadores), validación cruzada estratificada de 5 folds.
3. Selección de variables con mRMR (máxima relevancia, mínima redundancia, K=10). Se conservan las
   variables elegidas en al menos 4 de los 5 folds.
4. Tres líneas base globales: todas las variables, variables relevantes, y variables relevantes sin
   el código de país.
5. Línea base por país: un modelo independiente por país.
6. Integración de datos: se entrena con cada país y se prueba en los demás; dos países se consideran
   "similares" si el modelo de uno rinde casi igual en el otro. El modelo de cada país se entrena
   entonces con los datos de sus países similares.

## Resultados (5-fold CV, ejecución de `main.py`)
Modelos globales:

| | Todas las variables | Variables relevantes | Relevantes sin país |
|---|---|---|---|
| AUC | 0,78 | 0,84 | 0,88 |
| F-score | 0,42 | 0,42 | 0,61 |
| Precisión | 0,66 | 0,68 | 0,73 |
| Recall | 0,31 | 0,31 | 0,52 |

Por país (AUC, modelo propio frente a datos integrados):

| País | Modelo propio | Integración |
|---|---|---|
| Bolivia | 0,848 | 0,869 |
| Brasil | 0,866 | 0,884 |
| Chile | 0,877 | 0,884 |
| Colombia | 0,886 | 0,904 |
| Ecuador | 0,866 | 0,883 |
| Perú | 0,883 | 0,883 |
| Venezuela | 0,857 | 0,867 |

La integración mejora el AUC en 6 de los 7 países (entre +0,001 y +0,02). En Perú no se encontraron
países similares, así que el resultado es el mismo. Con clases desbalanceadas (77 % perdidas), el AUC y
el recall son más informativos que la exactitud.

## Limitaciones
- La selección de variables y el umbral de "países similares" se calculan sobre los mismos datos que se
  evalúan; los números son optimistas. Una evaluación rigurosa pediría un conjunto de prueba aparte.
- `NewOpportunitiesList.xlsx` no es un conjunto de prueba independiente: 31.810 de sus 31.823 filas
  coinciden en todas sus variables con filas de `oportunidad.csv`. Por eso el script solo imprime el
  número de predicciones (ganadas/perdidas) y no se reporta como métrica.
- Francia y Alemania aparecen en la tabla de países pero no tienen datos de entrenamiento.
- No hay comparación con otros clasificadores ni ajuste de hiperparámetros.

## Ejecución
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py        # ~3 min; las figuras (ROC, mapas de calor) se muestran con plt.show()
```
Probado con Python 3.8. `Presentation.pdf` explica la metodología propuesta.
