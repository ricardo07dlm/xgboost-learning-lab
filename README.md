# Machine Learning Lab Architecture – FastAPI + Scikit-learn + XGBoost

Este proyecto explica paso a paso cómo funciona el algoritmo **XGBoost**
mediante una PoC construida con:

- Scikit-learn (ecosistema / pipelines / métricas)
- XGBoost (motor de Machine Learning)
- FastAPI (capa de exposición REST)
- Exception Handling (mecanismo de gestión y control de errores)

---
## 🧱 Requisitos

- Python 3.9+
- xgboost
- scikit-learn
- pandas
- numpy
- joblib
- fastapi
- uvicorn

---
## 🎯 Objetivo

Explorar cómo el algoritmo XGBoost puede ser integrado en una arquitectura moderna de Machine Learning, cubriendo:

1) **Fundamentos teóricos de XGBoost**
2) **Integración de XGBoost mediante la API de Scikit-learn**
3) **PoC – Arquitectura y Implementación de un Pipeline de Machine Learning con XGBoost y Scikit-Learn**
4) **Decisiones Técnica de Arquitectura** 
---

## 1. Fundamentos teóricos de XGBoost
XGBoost (eXtreme Gradient Boosting) es una librería de machine learning que implementa el algoritmo de Gradient Boosting sobre árboles de decisión, combinando múltiples modelos débiles para construir modelos predictivos altamente eficientes y precisos, optimizada para ser:

- ⚡ Eficaz
- 🎯 Precisa
- 🧱 Robusta frente al overfitting
- 📊 Excelente para datos tabulares

> 🧠  **Gradient Boosting** es un algoritmo de aprendizaje supervisado que aprende relaciones y estructuras en los datos mediante la combinación secuencial de modelos débiles (normalmente árboles de decisión). <BR>
>> Función: **f(X)→Y**

### Ciclo de entrenamiento iterativo de XGBoost

XGBoost entrena múltiples árboles de decisión poco profundos de manera secuencial, donde cada nuevo árbol se ajusta a los errores (residuos) del conjunto de árboles anteriores, y la combinación ponderada de todos ellos da lugar al modelo final de Gradient Boosting.

**> Ciclo Base::**

✅ Se entrena un árbol muy simple.<br>
✅ Ese árbol comete errores, que se calculan explícitamente.<br>
✅ El siguiente árbol se centra en corregir esos errores <br>
→ aprende a predecir el residuo (error), no el valor final. <br>
✅ El proceso se repite iterativamente muchas veces. <br>
✅ Al final, se combinan todos los árboles optimizados para formar un modelo de Gradient Boosting <br>


### Componentes estructurales de XGBoost

Componentes principales que compone la libreria XGBoost:

**1) Matrix → estructura optimizada de datos**<br>
**2) Booster → motor interno del modelo**<br>
**3) train() → API nativa de XGBoost**<br>

> 🧠  **NOTA:** En esta PoC nos apoyaremos exclusivamente en el componente Booster, responsable del mecanismo interno de aprendizaje del modelo. El acceso y la operación del modelo de XGBoost se efectuarán a través de la API compatible con Scikit-learn, facilitando su integración en pipelines de ML

#### API Nativa de XGBoost

**1.Métodos de entrenamiento (Training API)** <br>
Son los que construyen el modelo:
* fit() → Entrena el modelo
* set_params() → Ajusta hiperparámetros
* get_params() → Recupera configuración

**2. Métodos de inferencia (Inference / Prediction API)** <br>
Son los que usa tu aplicación en runtime:
* predict() → Clase / valor predicho
* predict_proba() → Probabilidades (clasificación)

**3. Métodos de evaluación (Evaluation API)** <br>
Relacionados con métricas:

* score() (wrapper sklearn)
* evals_result()
* métricas configuradas (eval_metric)

**4. Métodos de persistencia (Model Persistence API)**<br>
Gestionado el ciclo de persistencia de los modelos:

* save_model()
* load_model()
* get_booster()
---
# 2. Integración de XGBoost mediante la API de Scikit-learn

**XGBoost wrappers → sklearn API contract:**

Básicamente, scikit-learn actúa como framework de orquestación (pipelines, preprocesado, métricas), mientras que XGBoost funciona como el motor de Machine Learning responsable del entrenamiento y la optimización del modelo mediante **Gradient Boosted Decision Trees**.

| Machine Learning   | Funcion      | Significado práctico                 |
|--------------------|--------------|--------------------------------------|
| **`scikit-learn`** | Framework    | framework de orquestación            |
| **`XGBoost`**      | Motor Modelo | motor de aprendizaje especializado   |

La integración entre **Scikit-learn y XGBoost** se produce gracias a los wrappers proporcionados por XGBoost (XGBClassifier, XGBRegressor), los cuales implementan la interfaz estándar de estimadores de **sklearn**.
Este enfoque permite que **Scikit-learn** actúe como capa de orquestación del pipeline, mientras que **XGBoost** opera como motor de aprendizaje subyacente.

**Flujo de Invocacion Interna:**

![Texto alternativo](/dev-xgboost-learning-py/resources/docs/img/flow_xgboost_scikitlearn.jpg)

1️⃣ **Scikit-learn Pipeline ::** <br> 
Esta la capa de orquestación<br>

2️⃣ **Estimator Interface (fit / predict) ::**<br>
Esta capa representa el contrato estándar de sklearn. Scikit-learn no sabe si el modelo es 
* RandomForest
* LogisticRegression
* XGBoost
* LightGBM

3️⃣ **XGBClassifier Wrapper ::**<br> 
Aquí ocurre la magia de integración.
El wrapper: <br> 
* Implementa la interfaz sklearn <br> 
* Traduce llamadas → motor XGBoost <br> 
* Convierte datos → DMatrix <br> 

4️⃣ **XGBoost Booster (Core Engine) ::** <br> 
Este es el motor real de Machine Learning.

Aquí sucede:

* Gradient Boosting <br> 
* Construcción de árboles <br> 
* Optimización <br> 
* Regularización <br> 

El Booster no conoce sklearn. Solo ejecuta aprendizaje.

### ⚙️ Scikit-Learn ::  Componentes y Métodos de Machine Learning


| Método                         | Firma / API | Tipo | Capa Arquitectónica | Rol en el Flujo |
|--------------------------------|-------------|------|----------------------|------------------|
| **make_classification**        | `make_classification(n_samples, n_features, ...)` | Generador de datos | Preparación de datos | Genera un dataset sintético para experimentación y validación |
| **train_test_split**           | `train_test_split(X, y, test_size, random_state, ...)` | Utilidad | Preparación de datos | Divide el dataset en subconjuntos de entrenamiento y prueba |
| **GradientBoostingClassifier** | `GradientBoostingClassifier(**hyperparameters)` | Estimador (ML Model) | Motor de aprendizaje | Implementa Gradient Boosted Decision Trees |
| **pipeline.fit**               | `pipeline.fit(X, y)` | Método | Orquestación (Pipeline) | Ejecuta el entrenamiento completo del pipeline |
| **pipeline.predict**           | `pipeline.predict(X)` | Método | Orquestación (Pipeline) | Genera predicciones de clase |
| **pipeline.predict_proba**     | `pipeline.predict_proba(X)` | Método | Orquestación (Pipeline) | Genera probabilidades por clase |
| **predict_proba**              | `model.predict_proba(X)` | Método del estimador | Modelo ML | Estima la probabilidad de pertenencia a cada clase |
| **accuracy_score**             | `accuracy_score(y_true, y_pred)` | Métrica | Evaluación | Calcula proporción de aciertos |
| **precision_score**            | `precision_score(y_true, y_pred)` | Métrica | Evaluación | Evalúa calidad de positivos predichos |
| **recall_score**               | `recall_score(y_true, y_pred)` | Métrica | Evaluación | Evalúa detección de positivos reales |
| **f1_score**                   | `f1_score(y_true, y_pred)` | Métrica | Evaluación | Balance entre precisión y recall |


---
## 3. PoC – Arquitectura y Implementación de un Pipeline de Machine Learning con XGBoost y Scikit-Learn 

### 3.1 🧩 Contexto Funcional

Esta PoC se sitúa en el dominio de negocio de Seguros y tiene como finalidad construir un modelo de Machine Learning orientado a la predicción del riesgo de fraude. El modelo aprende patrones de comportamiento del cliente y estima probabilidades de riesgo basadas en variables de entrada representativas del contexto de negocio.

**La PoC cubre las principales fases operativas del modelo XGBoost:**

✅ Entrenamiento → model.fit(X, y) <br>
✅ Predicción → model.predict(X) <br>
✅ Probabilidades → model.predict_proba(X) (clasificación) <br>
✅ Persistencia del modelo → save_model() / joblib <br>
✅ Metadatos / Métricas → accuracy, AUC, feature  <br>

**Propósito y Salida del Modelo:**  
✅ **Modelo de Machine Learning →**  Scoring de Riesgo / Fraude
✅ **Qué predicción debe proporcionar→** Probabilidad de impago o fraude por cliente


### 3.2 🏗️ Arquitectura Técnica

La PoC sigue una arquitectura desacoplada en capas:

1️⃣ **API Layer (FastAPI) ::**
Responsable del contrato REST, validación de entradas y serialización de respuestas.

2️⃣ **Service Layer ::**
Orquestación del flujo de Machine Learning (transformación → predicción → respuesta).

3️⃣ **ML Layer (Scikit-learn + XGBoost) ::**
Entrenamiento, predicción y evaluación del modelo.

4️⃣ **Persistence Layer (Joblib) ::**
Serialización y carga del pipeline entrenado.

5️⃣ **Exception Handling Layer ::**
Gestión centralizada de excepciones, propagando errores de forma controlada desde la capa de ML hasta la capa de API.

#### Diagrama Arquitectónico ####

![Texto alternativo](/dev-xgboost-learning-py/resources/docs/img/Arch_Model-XGBoost.png)


Este diseño permite:

✅ Separación de responsabilidades  
✅ Reutilización del modelo  
✅ Testabilidad  
✅ Evolución hacia producción


### 3.3 Modelo y Pipeline de Entrenamiento 

**Tipo de Entrenamiento:** Aprendizaje supervisado 

**Donde tendremos:**<br>
X → Features (datos de entrada / variables predictoras) <br>
y → Label (verdad histórica / variable objetivo)


**Output: Label (y)**
 - LOW = 0
 - MEDIUM = 1
 - HIGH = 2 

**Input:: Features (X)**
- edad
- ingresos_mensuales
- incidentes_previos
- ratio_deuda_ingresos
- num_productos
- canal

### 3.4 Modelo de Entrenamiento

#### 📌 **Training Endpoint Specification**

| Parámetro | Definición |
|------------|--------------|
| **Operation** | Model Training |
| **Protocol** | REST |
| **Method** | POST |
| **Resource Path** | `/api/training` |
| **Output Contract** | `TrainResponse` |
| **Successful Response** | 200 |
| **Error Response** | 404 |

#### 3.4.1 Modelo Entrada de Entrenamiento

**✅ JSON Request API-Friendly – /training**
**
```json
{
      "features": [
        {
          "edad": 24,
          "ingresos_mensuales": 1500,
          "antiguedad_meses": 8,
          "incidentes_previos": 2,
          "ratio_deuda_ingresos": 0.62,
          "num_productos": 1,
          "canal": "web"
        }
      ],
      "target": [
        {
          "risk_level": "medium"
        }
      ],
      "params": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3
      }
}
```
**JSON Transformado Pydantic/Schema → Panda/DataFrame** 

```json
    {
      "X": [
        [45, 3200.0, 60, 0, 0.25, 3, 1]
      ],
      "y": [0],
      "feature_names": [
        "edad",
        "ingresos_mensuales",
        "antiguedad_meses",
        "incidentes_previos",
        "ratio_deuda_ingresos",
        "num_productos",
        "canal"
      ],
      "params": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3
      }
    }
```

**(1) Elemento Entrada Datos → features[ ] > X**

El elemento features[] define el contrato de entrada de la API REST<br>
A partir de esta estructura, los datos se normalizan y se convierten en una matriz 2D: **X → (n_samples × n_features)**

**Donde:**
* Filas → Registros de clientes (observaciones / samples)
* Columnas → Variables predictoras (features)
* Dimensiones → Ejemplo: 4 × 6 (4 clientes, 6 features por cliente)


**(2) Elemento Objetivo → y**<br>
Representa un vector 1D que contiene la verdad histórica (ground truth) que el modelo debe aprender a predecir a partir de **X**

**Donde: y → [0, 1, 0, 1]**

* Cada posición de y[i] corresponde exactamente a una fila de **X[i]**
* Cada valor representa la etiqueta (label) asociada al registro 

**Ejemplo de mapping:**
- 0 → Riesgo Bajo
- 1 → Riesgo Alto


**(3) Elemento  feature_names: []**

Define el significado semántico de cada columna de la matriz X, estableciendo:

   - **X[]** → Matriz numérica que ve el algoritmo
   - **feature_names** → el significado humano de cada columna la matriz

**Ejemplo:**

    ```
    | Columna | Posición | Feature              |
    |---------|----------|----------------------|
    | 0       | X[i][0]  | edad                 |
    | 1       | X[i][1]  | ingresos_mensuales   |
    | 2       | X[i][2]  | antiguedad_meses     |
    | 3       | X[i][3]  | incidentes_previos   |
    | 4       | X[i][4]  | ratio_deuda_ingresos |
    | 5       | X[i][5]  | num_productos        |
    | 6       | X[i][5]  | canal                |.


**(4) Elemento  params: {}:**<br>
Representa los hiperparámetros del modelo, es decir, los valores de configuración que controlan el proceso de entrenamiento.
   - **n_estimators** → Número de árboles que se entrenan secuencialmente.
   - **learning_rate** → Cuánto aporta cada árbol al modelo final.
     - Valores pequeños → Aprendizaje más lento pero más estable
     - Valores grandes → Aprendizaje más rápido pero más inestable
   - **max_depth** → Profundidad máxima de cada árbol.
     - Árboles poco profundos → modelos más generales
     - Árboles muy profundos → memorizan datos (overfitting)

**Relación hiperparametros:**<br>
🔹**learning_rate** controla la velocidad de aprendizaje<BR>
🔹**n_estimators** compensa esa velocidad mediante el número de árboles.<br>

**Reglas prácticas:**<br>
🔹 Si bajo **learning_rate**, subo **n_estimators**<br>
🔹 Si subo **learning_rate**, bajo **n_estimators**

#### 3.4.2 Modelo de Salida – Métricas y Estado del Modelo

**✅ JSON Response API-Friendly – /training**
```json

{
  "model": {
    "id": "xgb_model_20260224_222532",
    "lifecycle": "trained"
  },
  "performance": {
    "accuracy": "70.00%",
    "precision": "83.33%",
    "recall": "70.00%",
    "f1_score": "73.00%"
  },
  "training_summary": {
    "samples_used": 50,
    "feature_dimension": 7
  },
  "features": [
    "edad",
    "ingresos_mensuales",
    "antiguedad_meses",
    "incidentes_previos",
    "ratio_deuda_ingresos",
    "num_productos",
    "canal"
  ]
}
```

El objeto de respuesta resume el estado del modelo entrenado, su desempeño predictivo y la configuración estructural utilizada durante el proceso de entrenamiento.

#### **🔹 Model { }**

Contiene la información de identificación y ciclo de vida del artefacto ML.

- **id →** Identificador único del modelo persistido  
- **lifecycle →** Estado actual del modelo dentro del flujo ML  

Permite versionado, trazabilidad y recuperación del modelo.

#### **🔹 Performance { }**

Resume las métricas de evaluación obtenidas tras el entrenamiento.

- **accuracy →** Proporción global de predicciones correctas  
- **precision →** Calidad de las predicciones positivas  
- **recall →** Capacidad del modelo para detectar eventos reales  
- **f1_score →** Balance entre precision y recall  

Estas métricas permiten validar la calidad del modelo.


#### **🔹training_summary{ }**

Describe las características estructurales del entrenamiento.

- **samples_used →** Número de registros utilizados  
- **feature_dimension →** Número de variables predictoras  

Facilita auditoría y debugging del modelo.


#### **🔹 features{ }**

Lista explícita de las variables utilizadas por el pipeline ML.

Define el contrato del modelo:

✔ Orden de entrada  
✔ Dimensionalidad esperada  
✔ Interpretación semántica  

Garantiza consistencia durante la inferencia.


### 3.5 Modelo de Predicción – Inferencia

#### 📌 **Predict Endpoint Specification**

| Parámetro | Definición |
|------------|--------------|
| **Operation** | Model Inference (`predict_proba`) |
| **Protocol** | REST |
| **Method** | POST |
| **Resource Path** | `/api/predictproba` |
| **Output Contract** | `PredictResponse` |
| **Successful Response** | 200 |
| **Error Response** | 404 |


#### 3.5.1 Modelo Entrada de Predicción – Inferencia

**✅ JSON Request API-Friendly – /predictproba**

```json
{
  "model_id": "xgb_model_20260224_222532",
  "features": [
    {
      "client_id": "0001",
      "edad": 24,
      "ingresos_mensuales": 1500,
      "antiguedad_meses": 8,
      "incidentes_previos": 2,
      "ratio_deuda_ingresos": 0.62,
      "num_productos": 1,
      "canal": "web"
    },
    {
      "client_id": "0002",
      "edad": 47,
      "ingresos_mensuales": 3400,
      "antiguedad_meses": 72,
      "incidentes_previos": 0,
      "ratio_deuda_ingresos": 0.21,
      "num_productos": 4,
      "canal": "portal"
    },
    {
      "client_id": "0003",
      "edad": 35,
      "ingresos_mensuales": 2200,
      "antiguedad_meses": 30,
      "incidentes_previos": 1,
      "ratio_deuda_ingresos": 0.41,
      "num_productos": 3,
      "canal": "fisico"
    }
  ]
}
```

#### ✅ **Explicación JSON de Entrada – Proceso de Inferencia (`predict_proba`)**

El objeto de entrada define el contrato de datos requerido por la API REST para ejecutar el proceso de inferencia del modelo de Machine Learning.

Este formato sigue un diseño **API-Friendly**, permitiendo desacoplar la representación externa del modelo interno.

#### 🔹 **model_id**

Identificador único del modelo previamente entrenado y persistido.

Permite:

✅ Seleccionar dinámicamente el artefacto ML  
✅ Gestionar versionado del modelo  
✅ Garantizar trazabilidad de inferencia  

Durante la ejecución, el sistema:

✔ Recupera el pipeline serializado  
✔ Carga el modelo en memoria  


#### 🔹 **features**

Contiene la lista de registros (clientes) que serán evaluados por el modelo.

Cada elemento representa una observación independiente.

**Estructura conceptual:**

`features → (n_samples × n_features)`

Donde:

- **n_samples →** Número de clientes evaluados  
- **n_features →** Variables predictoras del modelo  

#### 🔹 **client_id**

Identificador del registro procesado.

Su función es estrictamente operativa:

✔ No participa en el modelo  
✔ Permite trazabilidad y auditoría  
✔ Vincula input ↔ output  

#### 🔹 **Variables Predictoras**

Las variables incluidas corresponden exactamente al contrato esperado por el pipeline ML:

✔ edad  
✔ ingresos_mensuales  
✔ antiguedad_meses  
✔ incidentes_previos  
✔ ratio_deuda_ingresos  
✔ num_productos  
✔ canal  

Estas variables definen:

✅ Dimensionalidad del modelo  
✅ Orden estructural del pipeline  
✅ Consistencia de inferencia  

#### 3.5.2 Modelo Salida de Predicción – Inferencia

**✅ JSON Response API-Friendly – /predictproba**

```json
{
  "predictions": [
    {
      "client_id": "0001",
      "age": 24,
      "risk": "medium",
      "score": "93.23%",
      "risk_ranking": [
        { "risk": "high", "score": "4.16%" },
        { "risk": "low", "score": "2.61%" }
      ]
    },
    {
      "client_id": "0002",
      "age": 47,
      "risk": "low",
      "score": "87.79%",
      "risk_ranking": [
        { "risk": "medium", "score": "11.53%" },
        { "risk": "high", "score": "0.67%" }
      ]
    },
    {
      "client_id": "0003",
      "age": 35,
      "risk": "low",
      "score": "94.68%",
      "risk_ranking": [
        { "risk": "medium", "score": "4.73%" },
        { "risk": "high", "score": "0.59%" }
      ]
    }
  ]
}
```

#### **Explicación del Predict Response – Inferencia del Modelo**

El objeto de respuesta representa el resultado del proceso de inferencia ejecutado por el pipeline de Machine Learning.
Cada elemento dentro de `predictions[]` corresponde a un registro evaluado por el modelo.

#### 🔹 **predictions**

Contiene la lista de clientes procesados por el modelo.

Cada predicción encapsula:

✔ Identificación del registro  
✔ Clase predicha  
✔ Confianza del modelo  
✔ Distribución probabilística completa  

#### 🔹 **client_id**

Identificador del registro evaluado.

Permite:

✅ Trazabilidad  
✅ Auditoría  
✅ Integración con sistemas externos  


#### 🔹 **risk**

Clase predicha por el modelo.

Representa la categoría con mayor probabilidad estimada:

- `low`
- `medium`
- `high`

#### 🔹 **score**

Probabilidad asociada a la clase predicha.

Indica el nivel de confianza del modelo en su decisión.

Ejemplo:
```:
✔ 93.23% → Alta certeza en la predicción
```

#### 🔹 **risk_ranking**

Distribución probabilística completa del modelo.

Representa las probabilidades de las clases alternativas:

✔ Permite interpretabilidad <br>
✔ Permite análisis de incertidumbre <br>
✔ Soporta decisiones basadas en riesgo <br>

Ejemplo:
```:
- risk = medium
- score = 93.23%
```

### 3.6 Machine Learning System Architecture – Blueprint 

El diagrama describe la arquitectura del arquetipo Python utilizado en la PoC, mostrando la interacción entre las capas de API (FastAPI), procesamiento de Machine Learning (Scikit-learn + XGBoost) y la gestión centralizada de excepciones.

![Texto alternativo](/dev-xgboost-learning-py/resources/docs/img/ml_system_arch.png)


| Capa | Clase / Elemento | Tipo / Patrón | Responsabilidad (definición) |
|------|-------------------|---------------|-------------------------------|
| API/Controller | `/api/training` | Router/Controller (FastAPI endpoint) | Expone el endpoint de entrenamiento: recibe el request, valida el contrato, delega en `TrainingServiceImpl` y devuelve la respuesta serializada. |
| API/Controller | `/api/predictproba` | Router/Controller (FastAPI endpoint) | Expone el endpoint de inferencia/probabilidad: transforma request → servicio, invoca `PredictServiceImpl`, devuelve probabilidades/score. |
| API/Controller | `/api/repository` | Router/Controller (FastAPI endpoint) | Expone operaciones de repositorio (listar/cargar/guardar modelos): delega en `RepositoryServiceImpl`. |
| Schemas | `TrainSchema` | DTO / Request-Response Schema (Pydantic) | Contrato de entrada/salida para training (features/target/params) y validación de tipos. |
| Schemas | `PredictSchema` | DTO / Request-Response Schema (Pydantic) | Contrato de entrada/salida para predicción (features/model_id, etc.) y validación de payload. |
| Schemas | `RepositorySchema` | DTO / Request-Response Schema (Pydantic) | Contrato para operaciones de repositorio (model_id, filtros, metadatos, etc.). |
| Service | `TrainingServiceImpl` | Service Layer / Orchestrator | Orquesta el flujo de entrenamiento: preparación → llamada a `Training` (ML layer) → construcción de respuesta (Builders/Mappers) → manejo de errores. |
| Service | `PredictServiceImpl` | Service Layer / Orchestrator | Orquesta el flujo de predicción: carga pipeline/modelo → `PredictProba` → mapea resultado a response y controla excepciones. |
| Service | `RepositoryServiceImpl` | Service Layer / Facade | Orquesta casos de uso de persistencia: invoca `Repository` (ML layer) para guardar/listar/cargar modelos y retorna DTOs amigables. |
| Service | `Builders` | Builder | Construye objetos de respuesta (Response DTOs) a partir de resultados internos (métricas, predicciones, metadatos). |
| Service | `Mappers` | Mapper/Assembler | Convierte entre representaciones: Schema/API ↔ DTO interno ↔ estructuras ML (X/y) ↔ Response. |
| ML (Scikit/XGB) | `Training` | Use Case / ML Component | Implementa el entrenamiento: arma pipeline, ajusta modelo XGBoost, ejecuta fit, evalúa (si aplica) y produce artefacto entrenado + métricas. |
| ML (Scikit/XGB) | `PredictProba` | Use Case / ML Component | Ejecuta inferencia con `predict_proba()` (score/probabilidades) usando pipeline entrenado. |
| ML (Scikit/XGB) | `Repository` | Repository (Persistencia ML) | Persistencia/carga del pipeline/modelo (normalmente con Joblib): versionado por `model_id`, listar modelos, recuperar artefactos. |
| ML (Scikit/XGB) | `build_preprocessor` | Factory / Builder (Pipeline) | Crea el preprocesador (ColumnTransformer, encoders, scalers) y prepara el pipeline de scikit-learn. |
| ML (Scikit/XGB) | `TrainingDto` | DTO interno | Estructura interna para transportar datos/resultado de training (model_id, métricas, features, etc.) entre capas. |
| ML (Scikit/XGB) | `MetricsDto` | DTO interno | Representa métricas calculadas (accuracy/precision/recall/f1, etc.) y las hace independientes del framework. |
| Exception Handler | `register_exception_handlers` | Global Exception Registration | Registra handlers globales en FastAPI para mapear excepciones → HTTP status + body estándar. |
| Exception Handler | `ServiceError` | Base Exception | Error base/estándar de la capa de servicio (mensaje, código, status, clave i18n, etc.). |
| Exception Handler | `TrainingError` | Domain/UseCase Exception | Error controlado para fallos en entrenamiento (fit, pipeline, validación, etc.). |
| Exception Handler | `PredictProbaError` | Domain/UseCase Exception | Error controlado para fallos en inferencia/probabilidad (modelo no cargado, shape mismatch, etc.). |
| Exception Handler | `RepositoryError` | Domain/UseCase Exception | Error controlado para fallos de persistencia (save/load/list, permisos, path, artefacto corrupto). |
| Logging | `LOGGING` | Cross-cutting concern | Observabilidad transversal: logs por capa (API/Service/ML), trazabilidad de errores, métricas de ejecución, debugging. |

### 3.7 Gradient Boosting ML Pipeline Architecture 

Este diseño representa un enfoque transaccional de Machine Learning, en el que cada capa del sistema asume responsabilidades claramente definidas, garantizando desacoplamiento, mantenibilidad y claridad estructural dentro del contexto técnico de ML

![Texto alternativo](/dev-xgboost-learning-py/resources/docs/img/arch_model.png)

La siguiente tabla resume el flujo transaccional completo del sistema, desde la interacción API-REST hasta el entrenamiento, la inferencia y la persistencia del modelo de Machine Learning.

| Fase | Capa | Componente | Operación | Descripción Transaccional |
|------|------|-------------|-------------|-----------------------------|
| 1 | API-REST (FastAPI) | Endpoint Layer | Recepción Request JSON | La API recibe un payload JSON validado mediante modelos Pydantic, garantizando contrato y tipos de datos. |
| 2 | API-REST (FastAPI) | Transform Model | Validación & Parsing | El request es validado y convertido en objetos tipados (DTOs), evitando errores estructurales o de tipado. |
| 3 | Service Layer | Service Facade | Orquestación Caso de Uso | La capa de servicio actúa como fachada, desacoplando la API del núcleo ML y dirigiendo el flujo hacia training, prediction o repository. |
| 4 | Service Layer | Transform | Adaptación de Datos | Los datos se transforman desde la representación API hacia estructuras consumibles por el pipeline ML (X / matrices / features). |
| 5 | Machine Learning Layer | Dataset Splitting | train_test_split | Se particiona el dataset en conjuntos de entrenamiento y validación para evitar sesgo y permitir evaluación objetiva. |
| 6 | Machine Learning Layer | Pipeline Orchestration | Preprocessing Pipeline | Se aplican transformaciones (scaling, encoding, feature engineering) mediante Scikit-learn. |
| 7 | Machine Learning Layer | Model Fitting | Fit (XGBoost) | El modelo XGBoost es entrenado utilizando los datos transformados, optimizando la función objetivo. |
| 8 | Machine Learning Layer | Evaluation Phase | predict(X) & metrics | Se evalúa la calidad predictiva del modelo mediante métricas (accuracy u otras). |
| 9 | Machine Learning Layer | Predict Phase | predict_proba(X) | Durante inferencia, el pipeline ejecuta scoring probabilístico para estimación de riesgo/fraude. |
| 10 | Machine Learning Layer | Storage Phase | Persist Model (.pkl) | El pipeline/modelo entrenado es serializado y almacenado para reutilización futura. |
| 11 | Machine Learning Layer | Storage Phase | Load Model (.pkl) | En predicción, el modelo persistido es cargado dinámicamente. |
| 12 | Service Layer | Builder / Mapping Object | Construcción Response | Se construyen respuestas API-friendly a partir de outputs internos (scores, métricas, metadatos). |
| 13 | API-REST (FastAPI) | Response | Serialización JSON | La respuesta es serializada a JSON garantizando contrato REST consistente. |
| 14 | Cross-Cutting | Exception Handling | Gestión de Errores | Las excepciones son interceptadas y normalizadas en errores HTTP controlados. |
| 15 | Cross-Cutting | Logging | Observabilidad | Se registran eventos operacionales, errores y métricas de ejecución. |

## 4. Decisiones Arquitectónicas

---

### 1️⃣ ¿Por qué utilizar algoritmos basados en árboles de decisión?

Existen varias razones fundamentales para utilizar modelos basados en árboles:

1. **Capturan relaciones no lineales de forma natural**  
2. **Manejan eficientemente variables heterogéneas**, tanto numéricas como categóricas  
3. **No requieren normalización ni escalado previo**  
4. **Ofrecen excelente rendimiento en datos tabulares**, especialmente en dominios como seguros, banca o fraude  

Este tipo de algoritmos resulta especialmente adecuado para escenarios de scoring y clasificación de riesgo.

---

### 2️⃣ ¿Por qué utilizar Pandas para transformar modelos Pydantic en DataFrames?

El uso de Pandas introduce ventajas estructurales y operacionales:

✅ Conservación explícita de *feature names*  
✅ Eliminación de errores por orden de columnas  
✅ Mayor trazabilidad y facilidad de debugging  
✅ Compatibilidad directa con Scikit-learn  
✅ Integración eficiente con XGBoost  

Esta decisión mejora significativamente la robustez del pipeline.

---

### **2️3️⃣ ¿Por qué utilizar Joblib para la persistencia del modelo?**

Joblib es una librería de Python diseñada para la serialización eficiente de objetos complejos, siendo especialmente adecuada para artefactos de Machine Learning.

Su adopción permite:

✅ Persistir modelos entrenados  
✅ Recuperar pipelines completos de inferencia  
✅ Minimizar el coste de reconstrucción del modelo  

```python
joblib.dump(pipeline, "model.pkl")
```
---
### **4️⃣ ¿Por qué utilizar Scikit-learn junto con XGBoost?**

La arquitectura definida para la PoC establece como criterio la implementación del algoritmo XGBoost a través de la API compatible con Scikit-learn, con el objetivo de aprovechar su ecosistema de herramientas, manteniendo XGBoost como motor de Machine Learning.

Esta combinación permite una clara separación de responsabilidades:

**XGBoost → Motor de Machine Learning**

✔ Entrenamiento del modelo  
✔ Predicción  
✔ Optimización interna  
✔ Gestión de boosting / árboles  

**Scikit-learn → Ecosistema / Framework**

✔ Pipelines (`Pipeline`)  
✔ Validación (`train_test_split`, `cross_val_score`)  
✔ Tuning (`GridSearchCV`, `RandomizedSearchCV`)  
✔ Métricas (`accuracy_score`, `roc_auc_score`)  
✔ Preprocesamiento (`StandardScaler`, `OneHotEncoder`)  

Este enfoque maximiza la modularidad, reproducibilidad y mantenibilidad del sistema.

---
### **5️⃣ ¿Por qué utilizar `sklearn.pipeline.Pipeline`?**

El uso de `Pipeline` permite definir un flujo secuencial, determinista y reproducible de procesamiento de datos y entrenamiento del modelo.

```python
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])
```
El bloque **preprocessor** representa la capa de transformación que prepara los datos antes de que el modelo aprenda o prediga.

---

### **6️⃣ ¿Por qué persistir el pipeline completo y no solo el modelo XGBoost?**

Porque el pipeline encapsula todo el conocimiento aprendido durante el entrenamiento, no solo el modelo final.

El pipeline contiene:

✔ Modelo entrenado  
✔ Encoders entrenados (`OneHotEncoder`)  
✔ Imputadores entrenados (`SimpleImputer`)  
✔ Estadísticas aprendidas  
✔ Categorías detectadas  
✔ Reglas de transformación 
