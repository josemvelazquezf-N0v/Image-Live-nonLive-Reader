# Image-Live-nonLive-Reader

![Demo](Assets/Gifs/Demo.gif)

A EasyOCR aplication that Reads Images, with the Camera from the computer and also another aplication that reads the text from PDF's and Images, it works in EN/ES both languajes

# Lectura de texto con cámara y EasyOCR

Este proyecto usa la cámara de tu computadora para:

1. Detectar una hoja de papel de un color específico (calibrado en HSV).
2. Enderezar la hoja mediante una transformación de perspectiva.
3. Tomar varias fotos de la misma hoja.
4. Aplicar filtros de imagen para mejorar las letras.
5. Usar **EasyOCR** para leer el texto.
6. Combinar los resultados de varias capturas para obtener un texto final más estable.

El archivo principal se llama:

```text
CapturaVideo.py
```     

# Instalacion de depencencias

Si se quieren usar los codigos simplemente ejecuta el archivo

```text
Instalacion.py
```
Se recomienda un entorno virtual y visual studio para poder tener las ventanas emergentes de OpenCv 