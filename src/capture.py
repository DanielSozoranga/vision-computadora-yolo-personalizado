"""Captura fotos con la webcam para armar o ampliar el dataset.

Uso (en PowerShell, con el entorno activado):

    python src/capture.py --clase gato --carpeta fotos_webcam --auto --meta 50

Con --auto la captura empieza sola despues de --espera segundos y se detiene al llegar a --meta, mientras mueves
el objeto frente a la camara. Sin --auto, cada foto se toma con la tecla ESPACIO.

Controles de la ventana:
    ESPACIO  guarda una foto
    A        activa o desactiva la captura automatica (una foto cada --intervalo segundos)
    Q        sale

Las fotos se guardan en data/<carpeta>/<clase>/ con nombres consecutivos. Usa la clase "fondo" para escenas sin
ningun objeto, y "juntos" para varios objetos a la vez.
Consejo: cambia el angulo, la distancia, la luz y el fondo entre foto y foto, y deja que el objeto quede a veces
parcialmente tapado por tu mano o cortado por el borde de la imagen.
"""
import argparse
import os
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Captura fotos con la webcam para un dataset.")
    parser.add_argument("--clase", required=True, help="nombre de la clase, por ejemplo: gato")
    parser.add_argument("--carpeta", default="fotos", help="carpeta dentro de data/ donde se guardan las fotos")
    parser.add_argument("--camara", type=int, default=0, help="indice de la camara (0 es la predeterminada)")
    parser.add_argument("--intervalo", type=float, default=0.8, help="segundos entre fotos en modo automatico")
    parser.add_argument("--meta", type=int, default=25, help="cantidad de fotos nuevas que quieres tomar")
    parser.add_argument("--auto", action="store_true", help="empieza en modo automatico")
    parser.add_argument("--espera", type=float, default=5.0, help="segundos de cuenta regresiva antes del modo automatico")
    args = parser.parse_args()

    # Carpeta de salida: data/<carpeta>/<clase>/ (relativa a la raiz del proyecto)
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / args.carpeta / args.clase
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(out_dir.glob("*.jpg")))      # sigue la numeracion si ya habia fotos
    inicio = count

    # CAP_DSHOW abre la camara mas rapido en Windows; en otros sistemas se usa el backend por defecto
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camara, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara. Prueba con --camara 1 o cierra otras apps que la usen.")

    auto = args.auto
    start_time = time.time()
    last = 0.0
    print(f"Clase: {args.clase} | carpeta: {out_dir} | fotos existentes: {count} | meta nueva: {args.meta}")
    print("ESPACIO = foto | A = automatico | Q = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer la camara.")
            break

        now = time.time()
        nuevas = count - inicio
        countdown = args.espera - (now - start_time)
        save = False
        if auto and countdown <= 0 and now - last >= args.intervalo:
            save, last = True, now

        # Texto de ayuda sobre una copia: la foto que se guarda queda limpia
        view = frame.copy()
        if auto and countdown > 0:
            estado = f"EMPIEZA EN {int(countdown) + 1}"
        else:
            estado = "AUTO" if auto else "MANUAL"
        color = (0, 200, 0) if nuevas >= args.meta else (0, 200, 255)
        cv2.putText(view, f"{args.clase}: {nuevas}/{args.meta}  [{estado}]", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(view, "ESPACIO foto | A auto | Q salir", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Captura de dataset", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto, last = not auto, now
            start_time = now
        if key == ord(" "):
            save = True

        if save:
            count += 1
            path = out_dir / f"{args.clase}_{count:03d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print("guardada:", path.name)
            if auto and count - inicio >= args.meta:
                print("Meta alcanzada.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Listo. Fotos nuevas de '{args.clase}': {count - inicio} | total en la carpeta: {count}")


if __name__ == "__main__":
    main()
