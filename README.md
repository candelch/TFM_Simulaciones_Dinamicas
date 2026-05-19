# Simulación VLBI del Jet de SS 433: Evaluación de la Red Africana

Este repositorio recoge el trabajo desarrollado para mi Trabajo de Fin de Máster (TFM). El proyecto tiene como objetivo simular la cinemática relativista del microquásar SS 433 y evaluar cuantitativamente cómo la fidelidad de las imágenes interferométricas mejora al expandir la actual European VLBI Network (EVN) con la futura red de estaciones en el continente africano. 

Para realizar estas simulaciones he utilizado la librería eht-imaging (ehtim), implementando un flujo de trabajo que abarca desde la construcción del modelo teórico hasta los algoritmos de síntesis de apertura y la extracción de gráficas para la memoria.

## Estructura del Repositorio

El código y los resultados están organizados en dos directorios principales para separar el desarrollo analítico de los productos finales:

```text
├── scripts/
│   ├── helicoidal_jet_movie_triple_2.0.py   # Código completo: toy model, imaging pipeline y cálculos de análisis.
│   ├── uvcolors.py                          # Generación de gráficos de cobertura UV para la memoria del TFM.
│   └── innercore2.0.py                      # Código de creación y prueba del toy model del jet.
├── resultados/
│   ├── Animation_02_Exagerado.gif           # Evolución temporal (Modelo vs EVN vs EVN+Afr).
│   └── (Opcional: Añade aquí tus gráficos PDF)
├── .gitignore                               
└── README.md                                
```

## Metodología y Scripts

El desarrollo teórico parte del script `innercore2.0.py`, el cual contiene el código de creación del toy model. En este módulo construí la base física y geométrica del sistema, definiendo la inyección de componentes gaussianas continuas que siguen la trayectoria de precesión del jet y estableciendo los parámetros de decaimiento de flujo a medida que el plasma relativista se aleja del núcleo central.

Una vez validado el modelo teórico, la investigación principal se ejecuta mediante `helicoidal_jet_movie_triple_2.0.py`. Este es el código maestro que integra el toy model con el pipeline de imagen (imaging pipeline) y realiza todos los cálculos para los gráficos de análisis. A través de este script genero una simulación temporal de 43 días bajo distintos escenarios de velocidad, simulo las observaciones interferométricas inyectando ruido instrumental y aplico procesos iterativos de auto-calibración por fases. Además, este código centraliza la extracción de todas las métricas cuantitativas (como el RMSE, la correlación cruzada normalizada y el rango dinámico) y genera las curvas de evolución global que demuestran la mejora de la red combinada.

Para respaldar el análisis instrumental en la memoria del TFM, desarrollé el script `uvcolors.py`, destinado exclusivamente a generar los gráficos de cobertura en el plano UV. Este código calcula la ventana de observación óptima para el tránsito de SS 433 y mapea cómo se llena el plano de frecuencias espaciales con la rotación terrestre. Mediante un sistema de gradientes de color cronológicos, estas gráficas permiten visualizar claramente cómo las líneas de base Norte-Sur que aportan las estaciones africanas completan las regiones no muestreadas por la red europea aislada.

## Resultados Destacados

La siguiente animación ilustra el resultado del proceso de reconstrucción dinámico bajo el escenario de cinemática exagerada, comparando el modelo matemático original (generado por el toy model) con las reconstrucciones sintéticas de ambas redes interferométricas:

![Simulación comparativa de SS 433](resultados/Animation_02_Exagerado.gif)

El análisis de las métricas extraídas durante las simulaciones confirma la necesidad de la expansión de la red. La incorporación de antenas en latitudes sur (como la estación de Hartebeesthoek o los futuros nodos en Ghana y Namibia) aporta líneas de base que son críticas para la resolución angular. 

Esto resuelve uno de los problemas inherentes de la EVN en estas declinaciones: la severa elongación vertical del haz de síntesis (clean beam). Al observar la elipse de resolución en los mapas generados por el pipeline, es evidente que la red conjunta EVN+África produce un haz mucho más circular, lo que se traduce en una caída directa del RMSE, una estabilización de la fase de clausura y una reducción sustancial de los artefactos numéricos introducidos por el algoritmo de limpieza.

## Ejecución del Código

Para reproducir los experimentos, es necesario disponer de un entorno de Python 3.8 o superior y las dependencias de análisis estándar:

```bash
pip install numpy matplotlib imageio scikit-image
```

Además, es imprescindible instalar el framework de simulación interferométrica:

```bash
pip install eht-imaging
```

Para generar los mapas vectoriales del plano UV utilizados en la memoria:

```bash
python scripts/uvcolors.py
```

Para ejecutar el código completo con el toy model, el pipeline de imagen y la extracción de todas las gráficas de análisis (este proceso requiere un tiempo de cómputo elevado debido a los múltiples bucles de optimización del Imager y los pasos de self-calibration):

```bash
python scripts/helicoidal_jet_movie_triple_2.0.py
```

---
Trabajo de Fin de Máster
Año 2026
