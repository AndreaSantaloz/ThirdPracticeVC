# VC - PRACTICA 3
## Autores del proyecto 
1. Luis Martín Pérez
2. Andrea Santana López
## Introdución
Se basa en la realización de dos actividades donde la primera actividad consiste en un detector de monedas y la otra actividad es un detector de Microplásticos.
## Aspectos a tener en cuenta 
Para las otras práticas ya se tenía instalado python,numpy y matplotlib ,pero ahora hay que instalar searborn y scikit-learn
```
pip install scikit-learn seaborn
```

## Detector de Monedas

Con la función HoughCircles se detectan las monedas.

## Detector  de Microplásticos
El segundo ejercicio se trata de hacer un clasificador de Microplásticos de los siguientes tipos: fragmento,pellet y alquitrán.Es por eso que tras investigar sobre posibles algoritmos de clasifica
ción se selecciono Random Forest por su capacidad para clasificar los Microplásticos.

Primero se inicializó las variables para obtener las características geometricas escogidas que son:
área y perímetro y para las etiquetas.
```
features_list = [] 
labels_list = []
label_map = {"FRA": 0, "PEL": 1, "TAR": 2}
reverse_label_map = {v: k for k, v in label_map.items()}
```
Segundo declaramos las rutas de nuestros ficheros usados para clasificar.
```
images_data = [
    {"path": "./MicroplasticImages/pellet-03-olympus-10-01-2020.jpg", "label": "PEL"},
    {"path": "./MicroplasticImages/fragment-03-olympus-10-01-2020.jpg", "label": "FRA"},
    {"path": "./MicroplasticImages/tar-03-olympus-10-01-2020.jpg", "label": "TAR"},
]

```
Tercero, asignamos colores al tipo de microplasticos
```
color_map_bgr = {
    0: (0, 0, 255),    # FRA (Rojo)
    1: (255, 0, 0),    # PEL (Azul)
    2: (0, 255, 0)     # TAR (Verde)
}
```
Cuarto creamos una función para la extracción de características de la imagen donde primero declaramos la imagen de tipo BGR y comprobamos que existe,luego la pasamos a escala de grises y le quitamos el ruido,después umbralizamos la imagen usando un umbral adaptativo Gaussiano.Luego encontramos los contornos y vamos iterando sobre ellos con un bucle donde calculamos el área y comprobamos si es mayor de 50 px  si lo es pues hallamos el perimetro  y almacenamos las características y tras salir del bucle imprimos que ha sido procesado la imagen.
```
def process_image_for_training(image_path, true_label_code):
    
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        return
        
    # --- Detección OPTIMIZADA: Umbral Adaptativo (Más robusto) ---
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Umbral Adaptativo Gaussiano
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5) 
    
    
    # Encontrar contornos: clave para objetos individuales
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterar sobre contornos y extraer características
    for c in contours:
        area = cv2.contourArea(c)
        # Filtro de área mínima
        if area > 50:
            perimeter = cv2.arcLength(c, True)
            
            # Almacenar la característica y la etiqueta real
            features_list.append([area, perimeter])
            labels_list.append(true_label_code)

    print(f"Procesado (Entrenamiento): {image_path}. Muestras añadidas: {len([c for c in contours if cv2.contourArea(c) > 50])}")
```
Finalmente preparamos los datos para el Dataframe utilizando la libreria pandas y separamos los datos para entrenamiento y test donde luego los entrenamos y luego realizamos una predicción ,además que mostramos la matriz de confusión  y las tres imagenes rodeando cada microplastico clasificado en función de los colores puestos para cada microplástico.
```

# --- 3. Ejecutar Extracción y Entrenamiento ---
for item in images_data:
    process_image_for_training(item["path"], label_map[item["label"]])

# Preparación de Datos y Entrenamiento
X = pd.DataFrame(features_list, columns=['Area', 'Perimetro'])
y = pd.Series(labels_list)
print(f"\nTotal de muestras extraídas: {len(X)}")

# División y Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluación
y_pred = rf_model.predict(X_test)
print('\n--- Resultados de Random Forest ---')
print(f"Precisión del clasificador (Test): {accuracy_score(y_test, y_pred):.4f}")

# --- MATRIZ DE CONFUSIÓN ---
print('\n--- Matriz de Confusión (Random Forest) ---')
cm = confusion_matrix(y_test, y_pred)
class_names = list(label_map.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', # Cambiando el color a 'Reds' para distinguirlo
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión del Clasificador Random Forest')
plt.show()
# ----------------------------

# --- 4. Función de Predicción y Dibujo (Contorno Individual) ---
def predict_and_draw_contours(image_path, classifier, features_cols):
    """Carga, segmenta, predice la clase de CADA contorno y lo dibuja."""
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return
        
    # --- Detección: Umbral Adaptativo (Misma lógica que el entrenamiento) ---
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Umbral Adaptativo Gaussiano
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5) 
    
    # Cierre para rellenar pequeños huecos
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # ---------------------------------------------------------------------
    
    # Encontrar contornos INDIVIDUALES
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_with_predictions = img_bgr.copy()
    
    print(f"\nAnalizando y dibujando contornos en: {image_path}")

    # Iterar sobre CADA contorno, predecir y dibujar
    for c in contours:
        area = cv2.contourArea(c)
        
        # Filtro de área
        if area > 50: 
            perimeter = cv2.arcLength(c, True)
            
            # Crear el dato para la predicción
            new_feature = pd.DataFrame([[area, perimeter]], columns=features_cols)
            
            # Predecir la clase
            # Usa el clasificador pasado como argumento (rf_model)
            predicted_label_code = classifier.predict(new_feature)[0] 
            predicted_class = reverse_label_map[predicted_label_code]
            color = color_map_bgr.get(predicted_label_code, (255, 255, 255))
            
            # --- DIBUJAR CONTORNO Y CLASIFICACIÓN ---
            cv2.drawContours(img_with_predictions, [c], -1, color, 3)
            
            # Poner el texto de la clase predicha cerca del objeto
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img_with_predictions, predicted_class, (cX - 20, cY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
    # 5. Mostrar el resultado
    plt.figure(figsize=(10, 5))
    plt.title(f"Clasificación Visual Individual (RF) - {image_path.split('/')[-1]}")
    plt.imshow(cv2.cvtColor(img_with_predictions, cv2.COLOR_BGR2RGB))
    plt.show()

# --- 5. Ejecutar la Predicción y Visualización para las 3 Imágenes ---

for item in images_data:
    # Usar el nuevo modelo Random Forest
    predict_and_draw_contours(item["path"], rf_model, X.columns)
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

