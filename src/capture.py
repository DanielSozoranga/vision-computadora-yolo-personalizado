"""Captura fotos de un objeto con la webcam para armar el dataset.

Uso (en PowerShell, con el entorno activado):

    python src/capture.py --clase termo

Controles de la ventana:
    ESPACIO  guarda una foto
    A        activa o desactiva la captura automatica (una foto cada --intervalo segundos)
    Q        sale

Las fotos se guardan en data/fotos/<clase>/ con nombres consecutivos.
Consejo: cambia el angulo, la distancia, la luz y el fondo entre foto y foto.
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Captura fotos con la webcam para un dataset.")
    parser.add_argument("--clase", required=True, help="nombre de la clase, por ejemplo: termo")
    parser.add_argument("--camara", type=int, default=0, help="indice de la camara (0 es la predeterminada)")
    parser.add_argument("--intervalo", type=float, default=1.0, help="segundos entre fotos en modo automatico")
    parser.add_argument("--meta", type=int, default=25, help="cantidad de fotos que quieres tomar")
    args = parser.parse_args()

    # Carpeta de salida: data/fotos/<clase>/ (relativa a la raiz del proyecto)
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / "fotos" / args.clase
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(out_dir.glob("*.jpg")))  # sigue la numeracion si ya habia fotos

    # CAP_DSHOW abre la camara mas rapido en Windows; en otros sistemas se ignora
    cap = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara. Prueba con --camara 1 o cierra otras apps que la usen.")

    auto, last = False, 0.0
    print(f"Clase: {args.clase} | fotos existentes: {count} | meta: {args.meta}")
    print("ESPACIO = foto | A = automatico | Q = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer la camara.")
            break

        now = time.time()
        save = False
        if auto and now - last >= args.intervalo:
            save, last = True, now

        # Texto de ayuda sobre una copia: la foto que se guarda queda limpia
        view = frame.copy()
        estado = "AUTO" if auto else "MANUAL"
        color = (0, 200, 0) if count >= args.meta else (0, 200, 255)
        cv2.putText(view, f"{args.clase}: {count}/{args.meta}  [{estado}]", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(view, "ESPACIO foto | A auto | Q salir", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Captura de dataset", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto, last = not auto, now
        if key == ord(" "):
            save = True

        if save:
            count += 1
            path = out_dir / f"{args.clase}_{count:03d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print("guardada:", path.name)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Listo. Fotos de '{args.clase}' en total: {count}")


if __name__ == "__main__":
    main()
