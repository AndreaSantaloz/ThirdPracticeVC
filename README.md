# VC - PRÁCTICA 3
## Autores del proyecto 
1. Luis Martín Pérez
2. Andrea Santana López
## Introdución
Se basa en la realización de dos actividades, donde la primera consiste en un clasificador de monedas y la otra actividad es un clasificador de microplásticos.
## Aspectos a tener en cuenta 
En las otras prácticas ya se tenían instalados Python, NumPy y Matplotlib, pero ahora hay que instalar Seaborn y Scikit-learn:
```
pip install scikit-learn seaborn
```

## Clasificador de Monedas

El script está diseñado para detectar, clasificar y sumar el valor de monedas de euro a partir de una imagen utilizando la librería OpenCV. Su precisión depende de una calibración manual que establece la relación entre píxeles y milímetros.

1. Preparación y Detección Inicial:
El script comienza definiendo los diámetros reales y los valores monetarios de todas las monedas de euro. La función principal, calcular_precio_moneda, carga la imagen, la convierte a escala de grises y aplica un filtro de mediana (cv2.medianBlur) para reducir el ruido. Luego, utiliza la Transformada de Hough para Círculos (cv2.HoughCircles) para identificar todas las monedas, devolviendo sus coordenadas de centro (x,y) y su radio (r) en píxeles.

2. Calibración Interactiva:
Esta es la fase de escalado crucial. El script solicita al usuario que haga click en la moneda de 1 euro para usarla como referencia. Una función anidada (evento_click) captura este click, identifica el círculo detectado más cercano y almacena su radio en píxeles (rref​). Con este dato, se calcula el factor de conversión de píxeles a milímetros (Factormm/pıˊxel​) dividiendo el diámetro real del 1 euro (23.25 mm) por el diámetro detectado (2×rref​).

3. Corrección de Sesgo y Ajuste de Rangos:
Para compensar la distorsión de la cámara (como la inclinación o la perspectiva), el script calcula la diferencia relativa (un porcentaje de error) entre el diámetro real del 1 euro y el diámetro que se acaba de detectar. A continuación, aplica esta misma corrección porcentual para ajustar dinámicamente los rangos de diámetro ideales de todas las demás monedas. Esto asegura que la clasificación se mantenga precisa aunque la imagen esté ligeramente distorsionada.

4. Clasificación Final y Resultado: El script recorre todos los círculos detectados, calcula el diámetro estimado en mm para cada uno (2×r×Factorescala/píxel​) y lo compara con los rangos ajustados para determinar su tipo (e.g., '2_euros', '50_centimos'). Finalmente, suma el valor de todas las monedas al total_dinero, dibuja un círculo verde y una etiqueta sobre cada moneda en la imagen, y muestra el resultado final en la consola.

En este fragmento de código usando el algoritmo de Hough detectamos todos los objetos circulares de la escena 
```
circulos_detectados = cv2.HoughCircles(pimg, cv2.HOUGH_GRADIENT, 1, 100,
                                       param1=100, param2=50, minRadius=50, maxRadius=150)
```
La siguiente función es un evento para que cuando toque la moneda de 1 euro esta te devuelva los demás valores de las otras monedas.

```
def evento_click(event, x, y, flags, userdata):
    # ...
    if event == cv2.EVENT_LBUTTONDOWN:
        # ...
        for cx, cy, r in circulos.astype(float):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < distancia_minima:
                distancia_minima, circulo_ref = dist, (cx, cy, r)
        if circulo_ref is not None:
            punto_referencia_1_euro["radio"] = circulo_ref[2]
            # ...
            cv2.destroyWindow("Selecciona moneda de 1 euro")
```
En este fragmento de código calculamos las mediciones en píxeles a mm y ajusta los errores.
```
# Cálculo del factor mm/píxel
radio_ref = punto_referencia_1_euro["radio"]
diametro_1_euro_mm = monedas_dimensiones['1_euro']
factor_mm_por_pixel = diametro_1_euro_mm / (2 * radio_ref)

# Cálculo de la diferencia relativa (sesgo)
diametro_detectado_1_euro = 2 * radio_ref * factor_mm_por_pixel
diferencia_relativa = (diametro_detectado_1_euro - diametro_1_euro_mm) / diametro_1_euro_mm

# Ajuste de rangos
for nombre, (min_d, max_d) in rangos_base.items():
    # ...
    min_corr = min_d * (1 + diferencia_relativa)
    max_corr = max_d * (1 + diferencia_relativa)
    rangos_ajustados[nombre] = (min_corr, max_corr)
```
Y otro fragmento relevante es como clasifica y el procedimiento es así cada radio detectado (r) se multiplica por 2 y por el factor_mm_por_pixel para obtener el diámetro real estimado (diametro_mm).
El script compara este diaˊmetro estimado con los rangos ajustados (rangos_ajustados) en un bucle simple. La primera coincidencia (if min_d <= diametro_mm <= max_d:) determina el moneda_tipo.
El valor monetario correspondiente se suma al total_dinero.
```
for (x, y, r) in circulos_ordenados:
    diametro_mm = 2 * r * factor_mm_por_pixel
    moneda_tipo = "desconocida"
    for nombre, (min_d, max_d) in rangos_ajustados.items():
        if min_d <= diametro_mm <= max_d:
            moneda_tipo = nombre
            break
    # ... suma el valor y anota en la imagen
```
Aquí se proporciona imagenes relevantes de como ha funcionado en algunos casos.

<br>

![ImagenFinalMonedas](./CoinsImages/ImagenFinalMonedas.png)

<br>

![Prueba2Final](./CoinsImages/Prueba2Final.png)

## Clasificador  de Microplásticos
El segundo ejercicio se trata de crear un clasificador de microplásticos de los siguientes tipos: fragmento, pellet y alquitrán.
Por ello, tras investigar sobre posibles algoritmos de clasificación, se seleccionó Random Forest por su capacidad para clasificar los microplásticos.

Primero, se inicializaron las variables para obtener las características geométricas escogidas, que son: área y perímetro, además de las etiquetas.
```
features_list = [] 
labels_list = []
label_map = {"FRA": 0, "PEL": 1, "TAR": 2}
reverse_label_map = {v: k for k, v in label_map.items()}
```
Segundo, declaramos las rutas de los ficheros usados para clasificar.
```
images_data = [
    {"path": "./MicroplasticImages/pellet-03-olympus-10-01-2020.jpg", "label": "PEL"},
    {"path": "./MicroplasticImages/fragment-03-olympus-10-01-2020.jpg", "label": "FRA"},
    {"path": "./MicroplasticImages/tar-03-olympus-10-01-2020.jpg", "label": "TAR"},
]

```
Tercero, asignamos colores al tipo de microplásticos.
```
color_map_bgr = {
    0: (0, 0, 255),    # FRA (Rojo)
    1: (255, 0, 0),    # PEL (Azul)
    2: (0, 255, 0)     # TAR (Verde)
}
```
Cuarto, creamos una función para procesar los contornos de los microplásticos de la imagen.
Primero, cargamos la imagen en formato BGR y comprobamos que exista. Luego la pasamos a escala de grises y eliminamos el ruido.
Después, umbralizamos la imagen usando un umbral adaptativo gaussiano.
Posteriormente, encontramos los contornos y vamos iterando sobre ellos con un bucle donde calculamos el área y comprobamos si es mayor de 50 píxeles.
Si lo es, hallamos el perímetro.
Si se están obteniendo las características para entrenar, se almacenan; si no, se usan para predecir con los datos que tiene, dibujando círculos de color en función del tipo de microplástico y mostrando las tres imágenes.        
```
def process_contours(image_path,microplastic_label,is_training):
      #Comprobamos que la imagen existe
      img_bgr = cv2.imread(image_path)
      img_with_predictions = img_bgr.copy()
      if img_bgr is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        return
      # Preprocesamiento: Escala de grises y desenfoque
      img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
      img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
      # Umbral Adaptativo Gaussiano
      thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5) 
      # Encontrar contornos: clave para objetos individuales
      contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for contour in contours:
        area = cv2.contourArea(contour)
        # Filtro de área mínima
        if area > 50:
            perimeter = cv2.arcLength(contour, True)
            
            if not is_training:
                predict_and_draw_contours(
                    [area, perimeter], 
                    rf_model, 
                    X.columns,
                    contour,         
                    img_with_predictions 
                )
            else:
               process_data_training([area,perimeter], microplastic_label)
      if not is_training:
           plt.figure(figsize=(10, 5))
           plt.title(f"Clasificación Visual Individual ")
           plt.imshow(cv2.cvtColor(img_with_predictions, cv2.COLOR_BGR2RGB))
           plt.show()
```
Aquí está la función que predice y dibuja los círculos.
Cargamos los datos para la predicción, predecimos la clase con el clasificador y usamos la función drawContours para dibujarla en función del tipo de microplástico.
```
def predict_and_draw_contours(params, classifier, features_cols,contour, img_with_predictions):
      # Crear el dato para la predicción
    new_feature = pd.DataFrame([params], columns=features_cols)   
    
    # Predecir la clase
    predicted_label_code = classifier.predict(new_feature)[0] 
    predicted_class = reverse_label_map[predicted_label_code]
    color = color_map_bgr.get(predicted_label_code, (255, 255, 255))
    
    # Dibuja el contorno del objeto
    cv2.drawContours(img_with_predictions, [contour], -1, color, 3)
      
```
Aquí está la función que almacena los datos para el entrenamiento, donde se guardan los valores en features_list y labels_list.
```
def process_data_training(params, microplastic_label):
    #Añadir los datos para el entrenamiento
    features_list.append(params)
    labels_list.append(microplastic_label) 
```
Por último, aquí está la parte donde se entrena con el algoritmo Random Forest, se genera la matriz de confusión con los datos testeados y predichos, y se muestra tanto la matriz como las tres imágenes usando los bucles que iteran sobre las imágenes para testear y predecir los tipos de microplástico.

```

for item in images_data:
    process_contours(item["path"], label_map[item["label"]],True)

# Preparación de Datos y Entrenamiento
X = pd.DataFrame(features_list, columns=['Area', 'Perimetro'])
y = pd.Series(labels_list)
print(f"\nTotal de muestras extraídas: {len(X)}")
# División y Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#Entrenamiento de los datos
rf_model.fit(X_train, y_train)

# Predicción y Evaluación
y_pred = rf_model.predict(X_test)
print('\n--- Resultados de Random Forest ---')
print(f"Precisión del clasificador (Test): {accuracy_score(y_test, y_pred):.4f}")

# --- MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_test, y_pred)
class_names = list(label_map.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión del Clasificador Random Forest')
plt.show()
for item in images_data:
    process_contours(item["path"],"",False)
```
<br>

![Matriz confusión](./MicroplasticImages/MatrizConfusion.png)

<br>

![One image](./MicroplasticImages/OneImage.png)

<br>

![Second image](./MicroplasticImages/SecondImage.png)

<br>

![Third image](./MicroplasticImages/ThirdImage.png)


## Tecnologías
1. Python
2. Matplotlib
3. Numpy
4. Searborn
5. Scikit-learn
6. Pandas

## Recursos
1. [Random Forest]( https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2. [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
3. [FindContours and DrawContours](https://www.youtube.com/watch?v=FczN93nT-dQ)