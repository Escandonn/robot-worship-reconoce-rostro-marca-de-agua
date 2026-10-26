# WORSHIP — Backend ligero de detección facial
## Especificación técnica para desarrollo asistido por IA

**Versión:** 1.1  
**Fase:** 1 — Validación de rostro y anotación aproximada de la frente  
**Arquitectura obligatoria:** MVC con capa de servicio  
**Objetivo principal:** minimizar dependencias, tamaño de instalación, consumo de memoria y complejidad.

---

## 1. Objetivo

Desarrollar un backend Python que reciba **una sola imagen por solicitud** y determine si OpenCV detecta al menos un rostro frontal. Además del endpoint JSON de validación, la API ofrece un endpoint que devuelve la imagen con un círculo y un triángulo centrados en la zona estimada de la frente, únicamente cuando se detecta un rostro.

La respuesta debe incluir un booleano:

- `true`: se detectó al menos un rostro.
- `false`: no se detectaron rostros.

Esta fase no entrena un modelo personalizado ni intenta localizar landmarks anatómicos: utiliza el clasificador frontal preentrenado incluido con OpenCV y estima la frente a partir de la caja del rostro. No debe describir atributos personales, identificar individuos ni llamar a un modelo de lenguaje.

> Las imágenes de `backend/assets/frente/` son recortes de frente, no rostros completos con etiquetas. Las 54 imágenes disponibles contienen solo 9 imágenes distintas repetidas y no incluyen negativos ni anotaciones; no son suficientes para entrenar ni validar un detector propio. El endpoint de anotación espera imágenes donde pueda detectarse el rostro completo.

> `false` significa que el detector no encontró un rostro bajo sus parámetros; no demuestra que no haya una persona en la imagen. Haar Cascade puede fallar con rostros pequeños, de perfil, ocultos, inclinados o con iluminación deficiente.

## 2. Principios obligatorios

1. Mantener una arquitectura MVC explícita y verificable.
2. Separar controlador, modelo de datos, vista de respuesta y servicio de detección.
3. Usar el mínimo número razonable de dependencias.
4. No entrenar un modelo desde cero en esta fase.
5. No descargar modelos desde internet durante la ejecución.
6. No usar base de datos, colas, MQTT, frontend, LLM ni almacenamiento de imágenes.
7. Procesar la imagen en memoria y no guardarla en disco.
8. No registrar el contenido de la imagen ni sus bytes en logs.
9. Validar tipo, tamaño y decodificación de la imagen.
10. Mantener la lógica de negocio fuera del controlador.
11. Escribir pruebas automatizadas para imágenes positivas, negativas e inválidas.
12. No añadir abstracciones, interfaces o dependencias que no tengan una función concreta.

## 3. Stack mínimo

### Dependencias de ejecución

| Paquete | Uso | Obligatorio |
|---|---|---|
| `opencv-python-headless` | Decodificación de imágenes y Haar Cascade sin dependencias de GUI | Sí |
| `fastapi` | Endpoint HTTP y validación de solicitudes | Sí |
| `uvicorn` | Servidor ASGI | Sí |
| `python-multipart` | Recepción de archivos mediante `multipart/form-data` | Sí |

### Dependencias de desarrollo

| Paquete | Uso |
|---|---|
| `pytest` | Pruebas automatizadas |
| `httpx` | Pruebas HTTP de FastAPI |

### Restricciones

- No instalar `opencv-python` junto con `opencv-python-headless`.
- No instalar NumPy directamente si ya se instala como dependencia transitiva de OpenCV, salvo que el código o el entorno necesiten declararlo explícitamente.
- No usar TensorFlow, PyTorch, Ultralytics, MediaPipe, Transformers, scikit-learn ni modelos generativos en esta fase.
- No usar Pillow si OpenCV ya cubre la decodificación necesaria.
- No añadir `pydantic-settings`, ORM, Docker SDK u otras bibliotecas sin necesidad demostrable.
- Fijar versiones compatibles y probadas antes de desplegar. No inventar números de versión sin verificarlos en el entorno objetivo.

## 4. Detector

Usar el clasificador Haar Cascade frontal que incluye OpenCV:

`haarcascade_frontalface_default.xml`

Ruta preferida, sin copiar ni descargar el archivo manualmente:

`cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`

El servicio debe comprobar al inicializarse que el clasificador se cargó correctamente. Si falla, la aplicación debe producir un error de inicio claro; no debe responder falsamente que no hay rostros.

Parámetros iniciales configurables dentro del servicio:

- `scaleFactor = 1.1`
- `minNeighbors = 5`
- `minSize = (40, 40)`

Estos valores son un punto de partida, no una garantía de precisión. Ajustarlos con el conjunto de pruebas real. No agregar varios clasificadores hasta demostrar que son necesarios.

## 5. Estructura del proyecto

Crear esta estructura desde la carpeta raíz `backend/`:

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── face_controller.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── face_result.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── face_detection_service.py
│   └── views/
│       ├── __init__.py
│       └── face_view.py
├── assets/
│   └── rostros/
│       └── README.md
├── tests/
│   ├── test_face_service.py
│   └── test_face_endpoint.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

La carpeta `assets/rostros/` es para fotografías de prueba administradas por el desarrollador. El endpoint no debe depender de esa carpeta ni cargar todas las imágenes en memoria al iniciar. No incluir rostros privados en el repositorio sin autorización.

## 6. Responsabilidades MVC

### 6.1 Model — `app/models/face_result.py`

Responsabilidad: representar el resultado de la operación, sin ejecutar OpenCV ni acceder a HTTP.

Definir una estructura de datos pequeña, preferiblemente `dataclass`:

- `is_face: bool`
- `faces_count: int`
- `message: str`
- `face_boxes`: coordenadas de los rostros detectados para dibujar la respuesta, sin acoplar el modelo a HTTP.

El modelo no debe abrir archivos, conocer `UploadFile`, gestionar rutas ni crear respuestas HTTP.

### 6.2 Service — `app/services/face_detection_service.py`

Responsabilidad: contener toda la lógica de detección y anotación de imagen.

Debe:

1. Inicializar una única instancia del clasificador Haar Cascade.
2. Comprobar que el clasificador no está vacío.
3. Recibir los bytes de una imagen ya leída por el controlador.
4. Rechazar bytes vacíos o datos que no se puedan decodificar como imagen.
5. Decodificar en memoria con OpenCV.
6. Convertir la imagen a escala de grises.
7. Ejecutar `detectMultiScale` una sola vez con los parámetros configurados.
8. Contar las detecciones.
9. Devolver un `FaceResult` con el resultado y las cajas detectadas.
10. Para la operación de anotación, calcular el centro aproximado de la frente dentro de cada caja facial, dibujar un círculo con un triángulo en su interior y codificar el resultado como PNG.
11. Si no se detectan rostros, no producir una imagen anotada.
12. No guardar la imagen en disco ni imprimir bytes, imágenes ni datos sensibles en logs.
13. No crear un clasificador nuevo por cada solicitud.

Regla de negocio:

`is_face = faces_count > 0`

Distinguir los errores de entrada de una detección negativa. Una imagen inválida no debe tratarse silenciosamente como una imagen válida sin rostro.

### 6.3 Controller — `app/controllers/face_controller.py`

Responsabilidad: capa HTTP.

Crear los endpoints:

- `POST /api/v1/face/validate`: conserva la respuesta JSON de detección.
- `POST /api/v1/face/annotate`: devuelve una imagen PNG anotada solo si se detecta al menos un rostro.

Debe:

1. Recibir un archivo con `UploadFile` y `File(...)`.
2. Validar que se haya enviado un archivo y que no esté vacío.
3. Aplicar un límite de tamaño configurable, inicialmente 5 MiB.
4. Validar el tipo de archivo permitido (`image/jpeg`, `image/png` y `image/webp`) como comprobación inicial, sin confiar únicamente en el encabezado MIME.
5. Leer el contenido una sola vez y pasarlo al servicio.
6. Traducir errores de entrada a códigos HTTP apropiados.
7. Entregar el resultado a la vista para serialización.
8. No ejecutar directamente funciones de OpenCV ni contener la lógica de detección.

Para evitar leer archivos enormes en memoria, implementar lectura limitada por bloques o una estrategia equivalente que detecte si se supera el máximo permitido antes de acumular todo el archivo. No confiar solo en `Content-Length`.

### 6.4 View — `app/views/face_view.py`

Responsabilidad: convertir `FaceResult` en el contrato JSON público. La respuesta binaria PNG del endpoint de anotación debe ser creada por la capa HTTP a partir de los bytes producidos por el servicio, sin incorporar detección en la vista.

La vista debe ser una función pequeña y pura. No debe detectar rostros, leer imágenes ni acceder a FastAPI.

Contrato para una imagen válida con rostro:

```json
{
  "success": true,
  "is_face": true,
  "faces_count": 1,
  "message": "Se detectó al menos un rostro."
}
```

Contrato para una imagen válida sin rostro:

```json
{
  "success": true,
  "is_face": false,
  "faces_count": 0,
  "message": "No se detectaron rostros."
}
```

`success` indica que la solicitud se procesó correctamente. `is_face` representa el resultado del detector. El número de rostros puede ser mayor que uno.

### 6.5 Aplicación — `app/main.py`

Responsabilidad: crear la aplicación FastAPI, registrar el router y exponer un endpoint de salud.

- Nombre: `WORSHIP Face Detection API`
- Versión inicial: `1.0.0`
- Incluir el router de `face_controller.py`.
- Endpoint `GET /health` que devuelva estado de servicio.
- No crear lógica de detección aquí.
- No activar CORS abierto por defecto; no es necesario para probar desde Swagger. Añadir orígenes explícitos solo cuando un frontend lo requiera.

## 7. Contrato HTTP

### Solicitud

- Método: `POST`
- Ruta: `/api/v1/face/validate`
- Content-Type: `multipart/form-data`
- Campo del archivo: `file`
- Número de imágenes por solicitud: exactamente una
- Tamaño máximo inicial: 5 MiB, configurable
- Formatos permitidos: JPEG, PNG y WebP

El endpoint `POST /api/v1/face/annotate` acepta la misma solicitud y devuelve `image/png` si detecta al menos un rostro. Debe responder HTTP 404 con `FACE_NOT_DETECTED` si no encuentra rostro; en ese caso no debe devolver una imagen.

### Respuestas

| Situación | HTTP | Comportamiento |
|---|---:|---|
| Imagen válida, rostro detectado | 200 | `success: true`, `is_face: true` |
| Imagen válida, sin rostro detectado | 200 | `success: true`, `is_face: false` |
| Falta el archivo | 422 o 400 | Error de solicitud claramente documentado |
| Archivo vacío | 400 | Error de entrada |
| Formato no admitido | 415 | Tipo no admitido |
| Imagen corrupta o no decodificable | 400 | Imagen inválida |
| Archivo mayor al límite | 413 | Payload demasiado grande |
| Error al cargar Haar Cascade | Error de inicio | No iniciar como si el detector funcionara |
| Anotación solicitada y rostro detectado | 200 | Imagen anotada con tipo `image/png` |
| Anotación solicitada y sin rostro | 404 | Error `FACE_NOT_DETECTED`; no se genera imagen |

Los errores deben usar un formato JSON consistente, por ejemplo:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "El archivo no contiene una imagen válida."
  }
}
```

No devolver trazas internas, rutas del sistema ni detalles del entorno al cliente.

## 8. requirements.txt

Incluir únicamente las dependencias de ejecución:

```text
fastapi
uvicorn
opencv-python-headless
python-multipart
```

En `requirements-dev.txt`, referenciar `requirements.txt` y añadir:

```text
-r requirements.txt
pytest
httpx
```

Después de validar el proyecto, fijar versiones exactas compatibles para obtener despliegues reproducibles. Evitar mantener simultáneamente versiones sin límite en producción.

## 9. Pruebas obligatorias

### Pruebas del servicio

Crear `tests/test_face_service.py` para comprobar:

1. Fotografía frontal con un rostro: `is_face is True`.
2. Fotografía con varios rostros: `is_face is True` y `faces_count >= 2`, si el conjunto de prueba permite esperar ese resultado.
3. Fotografía sin rostro: `is_face is False`.
4. Bytes vacíos: error controlado o resultado de entrada inválida, según el contrato elegido.
5. Bytes que no son una imagen: error controlado.
6. Clasificador no disponible: fallo explícito de inicialización.

### Pruebas del endpoint

Crear `tests/test_face_endpoint.py` para comprobar:

1. Imagen válida con rostro devuelve HTTP 200.
2. Imagen válida sin rostro devuelve HTTP 200.
3. Archivo vacío devuelve error 4xx.
4. Imagen corrupta devuelve HTTP 400.
5. Tipo de archivo no admitido devuelve HTTP 415.
6. Archivo que supera el límite devuelve HTTP 413.
7. La respuesta mantiene siempre el contrato JSON documentado.
8. El endpoint de anotación devuelve PNG con marcadores cuando detecta rostro y 404 sin imagen cuando no lo detecta.

Las pruebas deben usar imágenes pequeñas de prueba, con licencia adecuada o creadas expresamente para el proyecto. No depender de una ruta absoluta del equipo del desarrollador. Si faltan los archivos de prueba, informar claramente y no marcar esas pruebas como aprobadas.

## 10. Seguridad y eficiencia

- No guardar imágenes recibidas.
- No registrar imágenes ni su contenido Base64.
- Limitar el tamaño de la solicitud.
- Validar la decodificación real; el MIME por sí solo no demuestra que sea una imagen.
- No exponer rutas locales ni excepciones internas.
- Mantener una sola instancia del clasificador por proceso.
- Evitar copias de imagen innecesarias.
- No ejecutar inferencia en servicios externos.
- No cargar frameworks de deep learning.
- Documentar que Haar Cascade puede generar falsos positivos y falsos negativos.
- Configurar límites de concurrencia y recursos en el servidor de despliegue si la carga lo requiere.

## 11. Ejecución local

Desde la carpeta `backend/`:

```bash
python -m venv .venv
```

Activar el entorno virtual en Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements-dev.txt
```

Iniciar el servidor:

```bash
uvicorn app.main:app --reload
```

Abrir la documentación interactiva:

`http://127.0.0.1:8000/docs`

Ejecutar pruebas:

```bash
pytest -q
```

## 12. Criterios de aceptación

La IA desarrolladora debe entregar el proyecto solo cuando:

- [ ] La estructura MVC coincide con esta especificación.
- [ ] El endpoint recibe una única imagen.
- [ ] Una imagen válida con rostro devuelve `is_face: true`.
- [ ] Una imagen válida sin rostro devuelve `is_face: false`.
- [ ] Las imágenes inválidas no se confunden con resultados negativos.
- [ ] El archivo no se persiste en disco.
- [ ] El clasificador se carga una sola vez por proceso.
- [ ] Existen límites de tamaño y validación de entrada.
- [ ] Las pruebas automatizadas se ejecutan y sus resultados se reportan honestamente.
- [ ] La imagen anotada solo se devuelve cuando el detector encuentra un rostro.
- [ ] El círculo y el triángulo se dibujan en la zona aproximada de la frente, sin persistir la imagen.
- [ ] No se añadieron dependencias o componentes fuera del alcance sin justificación.
- [ ] README explica instalación, ejecución, endpoint, respuestas y limitaciones.

## 13. Fuera del alcance de esta fase

No implementar todavía:

- Localización anatómica precisa de la frente mediante *facial landmarks*.
- Entrenamiento de un detector propio hasta disponer de datos positivos y negativos etiquetados adecuados.
- Detección de color del cabello, edad, sexo o identidad.
- Reconocimiento de objetos generales.
- Entrenamiento de modelos propios.
- LLM, Groq u otros servicios de IA.
- Base de datos, autenticación, colas o almacenamiento persistente.
- Cámara, vídeo o procesamiento continuo.

Estas funciones se evaluarán en fases posteriores sin romper el contrato de la API.

## 14. Instrucción final para la IA desarrolladora

Implementa el proyecto completo siguiendo esta especificación. Antes de escribir código, verifica las versiones y compatibilidad de las dependencias elegidas. Después, crea los archivos, ejecuta las pruebas disponibles y revisa que cada capa tenga una única responsabilidad.

No afirmes que las pruebas pasaron si no las ejecutaste. Si una prueba no puede ejecutarse por falta de imágenes, indícalo expresamente. Prioriza código sencillo, mantenible, eficiente y reproducible frente a abstracciones innecesarias.
