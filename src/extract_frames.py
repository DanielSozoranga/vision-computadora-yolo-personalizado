"""Extrae cuadros utiles de un video para armar el dataset.

Uso (en PowerShell, con el entorno activado o con la ruta completa a python.exe):

    python src/extract_frames.py data/videos/termo.mp4 --clase termo --max 20

Para usar solo un tramo del video, indica segundos con --inicio y --fin:

    python src/extract_frames.py data/videos/juntos.mp4 --clase juntos --inicio 21 --max 20

El script no guarda todos los cuadros, porque en un video los cuadros seguidos son casi iguales.
Guarda solo los que son:
  1. nitidos        (descarta los movidos o borrosos)
  2. distintos      (descarta los que se parecen a uno ya guardado)
  3. bien repartidos en el tiempo del video

Las fotos se guardan en data/fotos/<clase>/ con nombres consecutivos.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def sharpness(gray):
    """Nitidez: varianza del laplaciano. Un valor bajo significa imagen borrosa."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def thumb(gray):
    """Miniatura pequena para comparar parecido entre cuadros de forma rapida."""
    return cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extrae cuadros nitidos y variados de un video.")
    parser.add_argument("video", help="ruta del video, por ejemplo data/videos/termo.mp4")
    parser.add_argument("--clase", required=True, help="nombre de la clase, por ejemplo: termo")
    parser.add_argument("--max", type=int, default=20, help="maximo de fotos a guardar")
    parser.add_argument("--paso", type=float, default=0.4, help="segundos entre cuadros candidatos")
    parser.add_argument("--diferencia", type=float, default=8.0,
                        help="diferencia minima con los cuadros ya guardados (mas alto = mas variedad)")
    parser.add_argument("--ancho", type=int, default=1280, help="ancho maximo de la foto guardada")
    parser.add_argument("--salida", default=None, help="carpeta de salida (por defecto data/fotos/<clase>)")
    parser.add_argument("--inicio", type=float, default=0.0, help="segundo del video donde empezar")
    parser.add_argument("--fin", type=float, default=None, help="segundo del video donde terminar (por defecto, el final)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.salida) if args.salida else root / "data" / "fotos" / args.clase
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(args.paso * fps)))
    first = max(0, int(args.inicio * fps))
    last = total if args.fin is None else min(total, int(args.fin * fps))
    print(f"Video: {total} cuadros, {fps:.0f} fps, {total / fps:.1f} s | candidato cada {step} cuadros"
          f" | tramo usado: {first / fps:.1f} s a {last / fps:.1f} s")

    # 1) Cuadros candidatos, uno cada 'paso' segundos, con su nitidez
    candidatos = []   # (indice, nitidez, miniatura)
    for idx in range(first, last, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidatos.append((idx, sharpness(gray), thumb(gray)))
    print("Candidatos:", len(candidatos))
    if not candidatos:
        raise SystemExit("El video no tiene cuadros legibles.")

    # 2) Descartar los borrosos (menos de la mitad de la nitidez mediana)
    mediana = float(np.median([c[1] for c in candidatos]))
    nitidos = [c for c in candidatos if c[1] >= 0.5 * mediana]
    print(f"Nitidos: {len(nitidos)} (descartados por borrosos: {len(candidatos) - len(nitidos)})")

    # 3) Descartar los muy parecidos a uno ya elegido
    elegidos = []
    for idx, _, th in nitidos:
        if all(np.abs(th - e[2]).mean() >= args.diferencia for e in elegidos):
            elegidos.append((idx, _, th))
    print(f"Distintos: {len(elegidos)}")

    # 4) Si sobran, quedarse con los mejor repartidos en el tiempo
    if len(elegidos) > args.max:
        pos = np.linspace(0, len(elegidos) - 1, args.max).round().astype(int)
        elegidos = [elegidos[i] for i in pos]

    # 5) Guardar
    count = len(list(out_dir.glob("*.jpg")))
    for idx, _, _ in elegidos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > args.ancho:
            frame = cv2.resize(frame, (args.ancho, int(h * args.ancho / w)), interpolation=cv2.INTER_AREA)
        count += 1
        cv2.imwrite(str(out_dir / f"{args.clase}_{count:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cap.release()
    print(f"Guardadas {len(elegidos)} fotos nuevas en {out_dir} (total en la carpeta: {count})")
    if len(elegidos) < args.max:
        print("Aviso: salieron menos fotos de las pedidas. Graba un video mas largo o mueve mas el objeto,"
              " o baja --diferencia (por ejemplo 6).")


if __name__ == "__main__":
    main()
