import cv2
import numpy as np
import easyocr
import os
import glob
import fitz  # PyMuPDF para PDF -> imagen
from collections import Counter


# Combina varios textos (por ejemplo, de distintos preprocesados)
def combinar_textos(lista_textos):
    textos = [t.strip() for t in lista_textos if t and t.strip()]
    if not textos:
        return ""

    conteo = Counter(textos)
    texto_mas_comun, freq = conteo.most_common(1)[0]
    if freq >= 2 or len(textos) == 1:
        return texto_mas_comun

    listas_palabras = [t.split() for t in textos]
    max_len = max(len(lp) for lp in listas_palabras)
    resultado = []

    for i in range(max_len):
        palabras_pos = [lp[i] for lp in listas_palabras if i < len(lp)]
        if not palabras_pos:
            continue
        palabra = Counter(palabras_pos).most_common(1)[0][0]
        resultado.append(palabra)

    return " ".join(resultado)


def ocr_ensemble_en_imagen(img_bgr, reader):
    """
    Aplica varios preprocesados sobre la imagen completa y combina el texto.
    """

    # Asegurar tamaño mínimo para que las letras no sean muy pequeñas
    h, w = img_bgr.shape[:2]
    if max(h, w) < 800:
        esc = 800 / max(h, w)
        img_bgr = cv2.resize(
            img_bgr, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC
        )

    # Escala de grises y reducción de ruido
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Versión 1: binarización global (Otsu)
    _, thresh_otsu = cv2.threshold(
        gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Versión 2: binarización adaptativa
    thresh_adapt = cv2.adaptiveThreshold(
        gray_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        10,
    )

    # Versión 3: gris mejorado (sin binarizar)
    gray_mejorado = gray_eq

    def ocr_desde_gray(gray_img):
        img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        res = reader.readtext(img_rgb, detail=0, paragraph=True)
        if isinstance(res, list):
            return " ".join(r.strip() for r in res if isinstance(r, str)).strip()
        return str(res).strip()

    textos = [
        ocr_desde_gray(thresh_otsu),
        ocr_desde_gray(thresh_adapt),
        ocr_desde_gray(gray_mejorado),
    ]

    return combinar_textos(textos)


def leer_texto_imagen(img_bgr, reader):
    """
    Aplica OCR (ensemble de filtros) sobre la imagen completa.
    """
    texto = ocr_ensemble_en_imagen(img_bgr, reader)
    # Eliminar espacios y saltos de línea repetidos
    return " ".join(texto.split())


def procesar_imagen(ruta_imagen, reader):
    print("\n" + "=" * 60)
    print(f"IMAGEN: {os.path.basename(ruta_imagen)}")
    print("=" * 60)

    img = cv2.imread(ruta_imagen)
    if img is None:
        print("No se pudo leer la imagen.")
        return

    texto = leer_texto_imagen(img, reader)

    print("\nTEXTO DETECTADO:")
    print(texto if texto else "[Sin texto detectado]")


def procesar_pdf_rango(ruta_pdf, reader, pagina_inicio, pagina_fin):
    nombre_pdf = os.path.basename(ruta_pdf)

    print("\n" + "#" * 60)
    print(f"PDF: {nombre_pdf}")
    print("#" * 60)

    try:
        doc = fitz.open(ruta_pdf)
    except Exception as e:
        print(f"No se pudo abrir el PDF: {e}")
        return

    total_paginas = len(doc)

    pagina_inicio = max(1, pagina_inicio)
    pagina_fin = min(pagina_fin, total_paginas)

    if pagina_inicio > pagina_fin:
        print("Rango de páginas inválido.")
        doc.close()
        return

    print(
        f"Procesando páginas de {pagina_inicio} a {pagina_fin} "
        f"(total en PDF: {total_paginas})."
    )

    for num_pagina in range(pagina_inicio, pagina_fin + 1):
        print("\n" + "-" * 60)
        print(f"Página {num_pagina}")
        print("-" * 60)

        page = doc.load_page(num_pagina - 1)
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        texto = leer_texto_imagen(img, reader)

        print("\nTEXTO DETECTADO:")
        print(texto if texto else "[Sin texto detectado en esta página]")

    doc.close()
    print("\nFin del rango de páginas.")


def main():
    # Carpeta donde se buscan imágenes y PDFs
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMG")
    print("Buscando archivos en:", carpeta)

    # Buscar imágenes
    patrones_img = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    rutas_imagenes = [
        ruta
        for patron in patrones_img
        for ruta in glob.glob(os.path.join(carpeta, patron))
    ]

    # Buscar PDFs
    rutas_pdfs = glob.glob(os.path.join(carpeta, "*.pdf"))

    if not rutas_imagenes and not rutas_pdfs:
        print("No se encontraron imágenes ni PDFs en la carpeta IMG.")
        return

    documentos = [
        {"tipo": "IMG", "ruta": r, "nombre": os.path.basename(r), "info": ""}
        for r in rutas_imagenes
    ]

    for r in rutas_pdfs:
        try:
            num_pag = len(fitz.open(r))
        except Exception:
            num_pag = "desconocido"

        documentos.append(
            {
                "tipo": "PDF",
                "ruta": r,
                "nombre": os.path.basename(r),
                "info": f"{num_pag} páginas",
            }
        )

    print("Cargando modelo de EasyOCR (es/en)...")
    reader = easyocr.Reader(["es", "en"])

    while True:
        print("\n" + "=" * 60)
        print("MENÚ DE DOCUMENTOS (IMG / PDF)")
        print("=" * 60)

        for i, doc in enumerate(documentos, start=1):
            etiqueta = "[IMG]" if doc["tipo"] == "IMG" else "[PDF]"
            extra = f"  ({doc['info']})" if doc["tipo"] == "PDF" else ""
            print(f"{i:2d}) {etiqueta} {doc['nombre']}{extra}")

        print("\n 0) Salir")
        opcion = input("\nSelecciona un número de archivo (o 0 para salir): ").strip()

        if opcion in ("0", "q", "Q", "salir", "exit"):
            print("Saliendo del programa...")
            break

        if not opcion.isdigit():
            print("Entrada no válida. Escribe un número.")
            continue

        idx = int(opcion)
        if not (1 <= idx <= len(documentos)):
            print("Número fuera de rango.")
            continue

        doc_sel = documentos[idx - 1]

        if doc_sel["tipo"] == "IMG":
            procesar_imagen(doc_sel["ruta"], reader)
            continue

        print(f"\nHas seleccionado el PDF: {doc_sel['nombre']}")
        print("Indica el rango de páginas que quieres procesar.")
        print("Ejemplos: 3-10   o   5   (solo la página 5)")

        rango = input("Rango de páginas: ").strip()
        if not rango:
            print("Rango vacío, se cancela operación.")
            continue

        if "-" in rango:
            partes = rango.split("-")
            if len(partes) != 2:
                print("Formato de rango inválido. Usa ej. 3-10")
                continue
            try:
                inicio = int(partes[0])
                fin = int(partes[1])
            except ValueError:
                print("Debes escribir números en el rango.")
                continue
        else:
            try:
                inicio = fin = int(rango)
            except ValueError:
                print("Debes escribir un número de página válido.")
                continue

        procesar_pdf_rango(doc_sel["ruta"], reader, inicio, fin)


if __name__ == "__main__":
    main()
