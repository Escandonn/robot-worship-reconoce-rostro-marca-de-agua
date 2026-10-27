# WORSHIP Face Detection API

API Python que detecta rostros frontales con el clasificador preentrenado de
OpenCV y puede devolver una imagen con un círculo y un triángulo marcados en la
zona estimada de la frente. No identifica personas, no describe atributos y no
conserva las imágenes.

## Requisitos

- Python 3.10 o posterior.
- Las dependencias de ejecución están en `requirements.txt`; para pruebas,
  instala `requirements-dev.txt`.
- `opencv-python-headless` carga el Haar Cascade distribuido con OpenCV. No se
  descargan modelos durante la ejecución.

## Instalación y ejecución

Abre PowerShell en la carpeta `backend` (no dentro de `app`). Si la terminal
está en la raíz `backend-vision-api`, entra en la carpeta del proyecto:

```powershell
Set-Location .\backend
```

Comprueba que Python esté disponible y crea el entorno virtual:

```powershell
python --version
python -m venv .venv
```

Instala las dependencias y arranca Uvicorn con el Python del entorno virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Así no hace falta activar el entorno ni depender de que el comando `uvicorn`
esté en el `PATH`. Como alternativa, puedes activar el entorno primero con
`.\.venv\Scripts\Activate.ps1` y luego ejecutar
`python -m uvicorn app.main:app --reload`. Si PowerShell bloquea la activación,
usa los comandos directos de arriba.

La documentación interactiva está disponible en `http://127.0.0.1:8000/docs`;
el estado del servicio se consulta en `GET /health`.

## Endpoint

`POST /api/v1/face/validate` recibe exactamente un archivo en el campo
`file`, mediante `multipart/form-data`. Se aceptan JPEG, PNG y WebP, con un
límite inicial de 5 MiB. El límite puede cambiarse mediante la variable de
entorno positiva `MAX_UPLOAD_SIZE_BYTES`.

`POST /api/v1/face/annotate` acepta el mismo archivo y devuelve `image/png` con
el logotipo `WORSHIP` en rojo (`#FF3038`): el nombre centrado sobre un círculo
con un triángulo delineado dentro, colocado en la zona estimada de la frente de
cada rostro detectado. Si no encuentra rostros, devuelve HTTP 404 con el error
`FACE_NOT_DETECTED` y no devuelve una imagen.

Ejemplo desde PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/face/annotate" `
  -F "file=@C:\ruta\a\rostro.jpg" `
  --output rostro-anotado.png
```

Si no se detecta rostro, el endpoint responde 404 con JSON en vez del PNG.

Respuesta cuando detecta un rostro:

```json
{
  "success": true,
  "is_face": true,
  "faces_count": 1,
  "message": "Se detectó al menos un rostro."
}
```

Una imagen válida sin rostros devuelve HTTP 200, `success: true`,
`is_face: false` y `faces_count: 0`. Los errores usan HTTP 4xx y este formato:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "El archivo no contiene una imagen válida."
  }
}
```

Los tipos no admitidos devuelven 415; imágenes vacías o inválidas, 400; archivos
que superan el límite, 413; y solicitudes sin el campo `file`, 422. No se
devuelven trazas ni rutas locales.

## Arquitectura

- `app/controllers/face_controller.py`: validación HTTP, límite de carga y
  coordinación del servicio.
- `app/models/face_result.py`: resultado de dominio inmutable.
- `app/services/face_detection_service.py`: decodificación OpenCV, detección
  Haar Cascade, localización aproximada de la frente y anotación en memoria con
  el logotipo `WORSHIP`.
- `app/views/face_view.py`: serialización del contrato JSON.
- `app/main.py`: configuración FastAPI, rutas y errores HTTP.

El servicio carga el clasificador una sola vez por proceso. Las imágenes no se
escriben en archivos por la aplicación; se procesan en memoria. El middleware
limita el cuerpo HTTP antes de que FastAPI lo analice y eleva el umbral de
archivos temporales en memoria para evitar que las cargas se vuelquen al disco.
Los registros no incluyen contenido de archivos.

## Pruebas

```powershell
pytest -q
```

Las pruebas crean imágenes sintéticas y usan un clasificador controlado para
verificar detección, anotación, conteo y coordenadas. También se verifica el
clasificador real con una imagen vacía. Para validar la precisión real con fotografías,
añade muestras autorizadas a `assets/rostros/` con los nombres `rostro.jpg`,
`varios-rostros.jpg` y `sin-rostros.jpg`. Las pruebas de esas muestras se
marcarán como omitidas mientras no estén presentes; por ello el resultado no
certifica precisión sobre fotografías reales.

## Limitaciones

La posición marcada como frente se estima a partir de la caja del rostro; no es
un punto anatómico predicho por un modelo de landmarks. Haar Cascade puede
fallar con rostros pequeños, de perfil, ocultos, inclinados o con iluminación
deficiente, y puede producir falsos positivos. `is_face: false` solo indica que
el detector no encontró un rostro con sus parámetros; no prueba que no haya una
persona. Para acelerar cargas grandes, la detección reduce la imagen a un
máximo de 800 píxeles en su dimensión mayor. Los parámetros iniciales son
`scaleFactor=1.1`, `minNeighbors=5` y `minSize=(40, 40)`.

## Modelo y datos de `assets/frente`

La API no entrena modelos al arrancar ni descarga pesos al desplegarse: usa el
clasificador Haar frontal preentrenado que viene con `opencv-python-headless`.
Las imágenes de `assets/frente` son recortes de frente, no imágenes completas
con rostro y anotaciones. En el conjunto recibido hay 54 archivos, pero solo 9
imágenes distintas repetidas; no hay ejemplos negativos ni cajas etiquetadas.
Por ello no bastan para entrenar ni medir de forma fiable un detector propio de
rostro o frente. El clasificador busca el rostro completo y la región de frente
se aproxima geométricamente. Para un detector anatómico entrenado hace falta un
conjunto autorizado más amplio con cajas o puntos de frente etiquetados,
imágenes negativas y una evaluación independiente.
