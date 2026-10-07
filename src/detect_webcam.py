"""Deteccion en tiempo real con la webcam usando el modelo entrenado con el dataset propio.

Uso (en PowerShell de Windows, con el entorno del proyecto):

    python src/detect_webcam.py
    python src/detect_webcam.py --conf 0.5 --camara 1

Controles de la ventana:
    S  guarda una captura en reports/
    Q  sale

Por cada cuadro de la webcam se dibujan las cajas de las clases detectadas, con su nombre y confianza,
y un panel con el contador por clase y los cuadros por segundo.
"""
import argparse
import os
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "runs" / "custom_yolo_model" / "weights" / "best.pt"

# Un color BGR por clase (se repiten si hay mas clases que colores)
COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 160, 0), (0, 140, 255), (255, 255, 0)]


def procesar(frame, model, conf=0.5, imgsz=640):
    """Detecta objetos en un cuadro y devuelve (imagen con cajas, contador por clase)."""
    result = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = frame.copy()
    counts = Counter()
    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        score = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(y1, h + 10)                                          # la etiqueta no se sale de la imagen
        cv2.rectangle(out, (x1, top - h - 10), (x1 + w + 8, top), color, -1)
        cv2.putText(out, label, (x1 + 4, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        counts[name] += 1
    return out, counts


def dibujar_panel(frame, model, counts, fps):
    """Panel superior izquierdo con el contador por clase y los cuadros por segundo."""
    names = [model.names[i] for i in sorted(model.names)]
    height = 40 + 28 * len(names)
    cv2.rectangle(frame, (10, 10), (260, 10 + height), (30, 30, 30), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for i, name in enumerate(names):
        color = COLORS[i % len(COLORS)]
        cv2.putText(frame, f"{name}: {counts.get(name, 0)}", (20, 66 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def run(modelo=DEFAULT_MODEL, camara=0, conf=0.5, imgsz=640, carpeta_capturas=ROOT / "reports"):
    """Abre la webcam y detecta en tiempo real hasta que se pulsa Q."""
    modelo = Path(modelo)
    if not modelo.exists():
        raise SystemExit(f"No se encontro el modelo entrenado en {modelo}. Ejecuta antes el notebook de entrenamiento.")
    model = YOLO(str(modelo))

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY       # DSHOW abre mas rapido en Windows
    cap = cv2.VideoCapture(camara, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara. Prueba con camara=1 o cierra otras apps que la usen.")

    carpeta_capturas = Path(carpeta_capturas)
    carpeta_capturas.mkdir(exist_ok=True)
    print("Detectando en tiempo real. S = captura | Q = salir")

    fps, last = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer la camara.")
            break

        annotated, counts = procesar(frame, model, conf=conf, imgsz=imgsz)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))          # promedio suave de los cuadros por segundo
        last = now
        annotated = dibujar_panel(annotated, model, counts, fps)
        cv2.imshow("Deteccion en tiempo real", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = carpeta_capturas / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), annotated)
            print("captura guardada:", path.name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteccion en tiempo real con la webcam.")
    parser.add_argument("--modelo", default=str(DEFAULT_MODEL), help="ruta del modelo entrenado (best.pt)")
    parser.add_argument("--camara", type=int, default=0, help="indice de la camara (0 es la predeterminada)")
    parser.add_argument("--conf", type=float, default=0.5, help="confianza minima para mostrar una deteccion")
    parser.add_argument("--imgsz", type=int, default=640, help="tamano de analisis")
    args = parser.parse_args()
    run(args.modelo, args.camara, args.conf, args.imgsz)
