# **Challenge Data Analytics Engineer – Mercado Libre**
**Segmentación de Sellers + Clasificador Semántico con GenAI**

Readme generado con IA y validado por revisión del autor.
Este repositorio contiene la solución completa del desafío técnico, incluyendo:

- Limpieza y modelado de datos del marketplace
- Construcción de features a nivel seller
- Segmentación mediante clustering (K-Means)
- Evaluación de calidad de clusters y análisis para negocio
- Extensión GenAI usando embeddings generados por LLM
- Clasificador capaz de asignar sellers nuevos a un cluster
- Pruebas con sellers nunca vistos

## 📁 Estructura del repositorio
```bash
challenge_meli/
├── datos/
│   ├── crudo/
│   │   └── data_por_producto.csv      # Datos iniciales de la prueba
│   │
│   └── procesado/
│       ├── cluster_profile.csv        # Perfil de cada cluster
│       ├── data_seller.csv            # Base transformada a nivel seller
│       └── sellers_clustered.csv      # Sellers limpios + cluster asignado
│
├── modelos/
│   ├── embeddings_train.npy           # Embeddings usados en clasificador
│   ├── kmeans_sellers_k4.pkl          # Modelo KMeans final
│   ├── modelo_logreg_embeddings.pkl   # Clasificador semántico
│   └── scaler_robust.pkl              # Scaler para clustering
│
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_proc_y_construccion_dataset_sellers.ipynb
│   ├── 03_modelado_clusterizacion.ipynb
│   └── 04_modulo_genai.ipynb
│
├── presentacion/
│   └── Presentación.pdf               # Deck resumen del challenge
│
├── README.md
└── requirements.txt
```


# Cómo ejecutar este proyecto
A continuación se describe el flujo para reproducir completamente el challenge desde cero. Todos los pasos están basados en los notebooks incluidos en este repositorio.

## 0. Requisitos Previos
- Crear un entorno virtual (recomendado)
- Instalar dependencias desde `requirements.txt`
```python
python -m venv .venv
source .venv/Scripts/activate   # En Windows
pip install -r requirements.txt
```
## 1. Exploración Inicial de Datos — Notebook 01

- Ubicado en: `notebooks/01_exploracion_datos.ipynb`
- En este notebook se debe cargar la data curda ubicada en `datos/crudo/data_por_producto.csv`

**Contenido y contexto del notebook**

### ✔ Revisión estructural del dataset
- 185.250 publicaciones  
- 14 variables originales  
- 46.586 sellers únicos  

### ✔ Distribución de sellers
- La mayoría de vendedores tiene muy pocas publicaciones: la mediana es 1 ítem y el 75% tiene ≤ 3 productos, mostrando una base amplia de sellers pequeños/ocasionales.
- La distribución de publicaciones es claramente *long-tail*: existe una cola larga de vendedores con muchas publicaciones, incluyendo outliers por encima de ~1.400 ítems.

### ✔ Precios y Stock
- Las estadísticas descriptivas de `price` muestran una distribución extremadamente sesgada: la mediana es ~568, mientras que el máximo llega a **4700 millones**, evidenciando outliers económicos enormes.
- `stock` presenta un patrón similar: la mediana es 8 unidades, pero existen ítems con **hasta 99.999 unidades**, lo que sugiere la presencia de vendedores mayoristas o catálogos artificialmente inflados. Además se detectan productos agotados o sin stock (6k aprox)

### ✔ Reputación del seller
- La mayoría de vendedores tiene reputaciones positivas: **green (29%)**, **green_silver (14%)**, **green_platinum (13%)** y **green_gold (12%)**. Esto sugiere una base importante de vendedores con trayectoria y buen desempeño

### ✔ Condition + Refurbished

- El catálogo está fuertemente dominado por productos **new**, que representan el **91.6%** de todas las publicaciones. Esto sugiere que la mayoría de los sellers ofrece inventario nuevo, típico de vendedores más formales o profesionales.

### ✔ Logistic Type

- La variable `logistic_type` muestra una fuerte predominancia de **XD**, que representa aproximadamente **63%** de las publicaciones. Esto indica que la mayoría de los productos se gestionan mediante un flujo donde el vendedor entrega al carrier, o deja en un place y la paquetería usa un HUB intermedio
- El segundo tipo logístico más frecuente es **FBM (fulfillment by Mercado Libre)**, con **17%** del catálogo
 
## 2. Construcción del Dataset a Nivel Seller — Notebook 02

- Ubicado en: `notebooks/02_proc_y_construccion_dataset_sellers.ipynb`
- En este notebook se debe cargar la data curda ubicada en `datos/crudo/data_por_producto.csv`

**Contenido y contexto del notebook**
### ✔ Detección y análisis de nulos
- `regular_price` con **73% nulos** → imputado usando `price`  
- `price` y `stock` con **0** → reglas de negocio para depuración  
- `seller_reputation` con **~1.3% nulos** → eliminados por error de data  

### ✔ Outliers
### Metodologías probadas:
| Método        | Resultado |
|---------------|-----------|
| IQR (1.5× y 3×) | Límites negativos, mala estabilidad por escala sesgada |
| Percentil 99 (P99) | **Seleccionado**: mejor balance limpieza / preservación |

- Eliminación justificada de outliers estructurales.

### ✔ Transformaciones
- `log1p` aplicado a precio y stock → estabiliza escala extremadamente sesgada.
- Todas las decisiones documentadas para modelado posterior.

### ✔ Agregaciones a nivel seller
- Nº de publicaciones  
- Diversidad de categorías  
- Categoría dominante  
- Precio medio / mediano  
- Stock total / medio  
- Flags: `new`, `used`,`not_specified` `refurbished` 
- Logística 
- IQR, P99 + log-transforms del notebook anterior  

### ✔ Enriquecimiento
- Entropía de categorías → diversificación  
- % de concentración en moda  
- Mapeo ordinal de reputación (0 a 8)  

### ✔ Resultado
Dataset robusto, limpio y listo para clustering:  
➡ **data_seller.csv**


## 3. Modelado de Clustering — Notebook 03
- Ubicado en: `notebooks/03_modelado_clusterizacion.ipynb`

**Contenido y contexto del notebook**

### ✔ Preparación
- Escalamiento robusto (RobustScaler)  
- Selección de features estructurales  

### ✔ Selección de K
Evaluado mediante:
- Elbow Method  
- Silhouette Score  

→ Ambos sugieren **K = 4**

### ✔ Entrenamiento
- `KMeans(n_clusters=4, random_state=13, n_init=20)`
- Eliminación de outliers residuales en espacio escalado  

### ✔ Análisis de clusters
1. Sellers pequeños y de baja reputación  
2. Sellers en crecimiento  
3. Sellers diversificados, formales, alta reputación  
4. Sellers especializados y de alta operación logística  

➡ **sellers_clustered.csv**  
➡ **cluster_profile.csv**


## 4. Módulo GenAI — Clasificador Semántico — Notebook 04

- Ubicado en: `notebooks/04_modulo_genai.ipynb`

Este notebook entrena un modelo de clasificación capaz de predecir el cluster de un seller nunca visto, usando embeddings de lenguaje

**Contenido y contexto del notebook**
### ✔ Construcción del texto descriptivo
"Seller con X publicaciones, Y categorías, reputación Z, % nuevos..., logística..."

### ✔ Generación de embeddings
- Modelo: **text-embedding-3-small**  
- Batching de 256  
- Se incluyen todas las variables relevantes  

⚠ API Key:
```python
os.environ["OPENAI_API_KEY"] = "insertar API key aqui" 
```

# Notas Importantes:

- Ninguna celda depende de parámetros ocultos o rutas externas. Todo es reproducible
- La API Key **NO** se incluye en el repositorio
- Los modelos ya entrenados están en /modelos por si se desea ejecutar sin entrenar

# Conlusiones Generales:

- Se construyó un pipeline sólido desde datos raw → dataset seller → clusters → clasificador.
- El modelo KMeans encontró 4 segmentos con interpretabilidad clara y acción comercial directa.
- El módulo GenAI complementa el proyecto con una solución moderna, escalable y útil para onboarding.
- Todo el trabajo está documentado en forma de notebooks replicables.
