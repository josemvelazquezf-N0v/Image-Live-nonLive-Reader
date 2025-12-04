import subprocess
import sys

paquetes = ["opencv-python", "numpy", "easyocr", "pymupdf"]

def instalar(paquete):
    print(f"Instalando {paquete}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])

if __name__ == "__main__":
    for p in paquetes:
        instalar(p)
    print("Todas las dependencias han sido instaladas.")
