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

## Detector de Monedas

Con la función HoughCircles se detectan las monedas.

## Detector  de Microplásticos
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

![Matriz confusión](./MicroplasticImages/MatrizConfusion.png)

![One image](./MicroplasticImages/OneImage.png)

![Second image](./MicroplasticImages/SecondImage.png)

![Third image](./MicroplasticImages/ThirdImage.png)


## Tecnologías
1. Python
2. Matplotlib
3. Numpy
4. Searborn
5. Scikit-learn
6. Pandas

