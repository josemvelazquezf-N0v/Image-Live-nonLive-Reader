import cv2
import numpy as np
import time
import easyocr
from collections import Counter


# Ordena 4 puntos como: top-left, top-right, bottom-right, bottom-left
def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # esquina superior izquierda
    rect[2] = pts[np.argmax(s)]      # esquina inferior derecha

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # esquina superior derecha
    rect[3] = pts[np.argmax(diff)]   # esquina inferior izquierda

    return rect


# Detecta la hoja en el frame usando HSV y devuelve la hoja enderezada
def detectar_hoja_blanca(frame):
    altura, ancho = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango base (calibrado) y margen
    base_lower = np.array([81, 0, 125], dtype=np.uint8)
    base_upper = np.array([121, 69, 245], dtype=np.uint8)
    delta = np.array([5, 15, 30], dtype=np.int32)

    lower = base_lower.astype(np.int32) - delta
    upper = base_upper.astype(np.int32) + delta

    lower = np.clip(lower, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    upper = np.clip(upper, [0, 0, 0], [179, 255, 255]).astype(np.uint8)

    mask_color = cv2.inRange(hsv, lower, upper)

    # Limpiar máscara
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, mask_clean

    mejor_contorno = None
    mejor_area = 0
    area_minima = ancho * altura * 0.20  # 20% del frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Filtrar contornos por tamaño y brillo interno
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_minima:
            continue

        hull = cv2.convexHull(cnt)

        mask_hull = np.zeros_like(gray)
        cv2.drawContours(mask_hull, [hull], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask_hull)[0]

        if mean_val < 90:
            continue

        if area > mejor_area:
            mejor_area = area
            mejor_contorno = hull

    if mejor_contorno is None:
        return None, None, mask_clean

    rect = cv2.minAreaRect(mejor_contorno)
    box = cv2.boxPoints(rect)
    pts = np.array(box, dtype="float32")

    rect_pts = ordenar_puntos(pts)

    # Tamaño de salida de la hoja enderezada
    ancho_dst = 1000
    alto_dst = 1400
    dst = np.array([
        [0, 0],
        [ancho_dst - 1, 0],
        [ancho_dst - 1, alto_dst - 1],
        [0, alto_dst - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(frame, M, (ancho_dst, alto_dst))

    contour_4pts = rect_pts.reshape(-1, 1, 2).astype(int)

    return warped, contour_4pts, mask_clean


# Aplica filtros básicos y ejecuta EasyOCR sobre una imagen BGR
def leer_texto_imagen(img_bgr, reader):
    h, w = img_bgr.shape[:2]
    if max(h, w) < 800:
        escala = 800 / max(h, w)
        img_bgr = cv2.resize(
            img_bgr,
            None,
            fx=escala,
            fy=escala,
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    resultados = reader.readtext(
        img_rgb,
        detail=0,
        paragraph=True,
        text_threshold=0.4,
        low_text=0.3,
        link_threshold=0.4,
    )

    if isinstance(resultados, list):
        texto = " ".join([r.strip() for r in resultados if isinstance(r, str)])
    else:
        texto = str(resultados)

    return texto.strip()


# Combina varios textos (por ejemplo, varias capturas de la misma hoja)
def combinar_textos(lista_textos):
    textos = [t.strip() for t in lista_textos if t and t.strip()]
    if not textos:
        return ""

    conteo = Counter(textos)
    texto_mas_comun, freq = conteo.most_common(1)[0]
    total = len(textos)

    # Si el texto más común aparece en al menos el 60% de las capturas, lo usamos
    if freq / total >= 0.6:
        return texto_mas_comun

    # Si no hay mayoría clara, se elige el texto más largo
    return max(textos, key=len)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    estado = "buscando"   # estados: buscando -> esperando -> capturando -> listo
    capturas = []
    tiempo_ultima_captura = None
    num_capturas_deseadas = 5
    delay_entre_capturas = 0.1
    pre_captura_delay = 0.0
    tiempo_inicio_conteo = None

    print("Presiona 'q' para salir en cualquier momento.")
    print("Apunta la cámara hacia una hoja del color calibrado con texto.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        frame_mostrar = frame.copy()

        hoja_warped, contorno_hoja, mask_blanco = detectar_hoja_blanca(frame)

        cv2.imshow("Mascara hoja (debug HSV)", mask_blanco)

        if contorno_hoja is not None:
            cv2.drawContours(frame_mostrar, [contorno_hoja], -1, (0, 255, 0), 2)
            cv2.putText(frame_mostrar, "Hoja detectada", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if estado == "buscando":
                estado = "esperando"
                tiempo_inicio_conteo = time.time()
                print(f"Hoja detectada. Captura iniciará en {pre_captura_delay} segundos...")

        # Control del tiempo de espera antes de empezar a capturar
        if estado == "esperando" and tiempo_inicio_conteo is not None:
            restante = pre_captura_delay - (time.time() - tiempo_inicio_conteo)
            if restante <= 0:
                estado = "capturando"
                tiempo_inicio_conteo = None
                capturas = []
                tiempo_ultima_captura = time.time()
                print(f"Iniciando captura de {num_capturas_deseadas} imagenes...")
            else:
                texto_conteo = f"Captura en {restante:.1f}s"
                cv2.putText(frame_mostrar, texto_conteo, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Captura de varias imágenes de la misma hoja
        if estado == "capturando" and hoja_warped is not None:
            tiempo_actual = time.time()
            if tiempo_actual - tiempo_ultima_captura >= delay_entre_capturas:
                capturas.append(hoja_warped.copy())
                tiempo_ultima_captura = tiempo_actual
                print(f"Captura {len(capturas)} realizada.")

                cv2.imshow("Hoja recortada (ultima captura)", hoja_warped)

                if len(capturas) >= num_capturas_deseadas:
                    estado = "listo"
                    print(f"Se han capturado las {num_capturas_deseadas} imagenes de la hoja.")
                    break

        cv2.imshow("Camara - Deteccion de hoja", frame_mostrar)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Salida manual con 'q'.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Numero de capturas obtenidas: {len(capturas)}")
    for i, img in enumerate(capturas):
        filename = f"captura_hoja_{i+1}.png"
        cv2.imwrite(filename, img)
        print(f"Imagen guardada: {filename}")

    # OCR sobre las capturas y combinación
    if capturas:
        print("\nIniciando OCR sobre las capturas...")
        reader = easyocr.Reader(['es', 'en'])

        textos_capturas = []

        for i, img in enumerate(capturas):
            print(f"\n===== Captura {i+1} =====")
            texto = leer_texto_imagen(img, reader)
            textos_capturas.append(texto)
            print("Texto detectado:")
            print(texto if texto else "[Sin texto detectado]")

        texto_final = combinar_textos(textos_capturas)

        print("\n================ TEXTO FINAL ================")
        if texto_final:
            print(texto_final)
        else:
            print("[No se pudo reconstruir texto a partir de las capturas]")
        print("=======================================================")
    else:
        print("No hay capturas disponibles para OCR.")


if __name__ == "__main__":
    main()
