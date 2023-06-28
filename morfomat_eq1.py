from tkinter import * #Libreria tkinter para la interfaz
import tkinter as tki
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import imutils
import cv2
from cv2 import bitwise_or           #Libreria OpenCV para el procesamiento de imagenes
import numpy as np   #Libreria numpy para cálculo y análisis de datos (matrices, arreglos)
from matplotlib import pyplot as plt
import math          #Modulo math proporciona funciones que son útiles en teoría de números
from unittest import result
from matplotlib import pyplot as plt #matplotlib es para crear visualizaciones (como el histograma)
import random

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def grises(img):
    imagen = img
    
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    img_convertida = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2RGB)
    cv2.imshow('Gris convertida', img_convertida)
    
    filenameGris = file+"_gris.jpg"                
    cv2.imwrite(filenameGris, img_convertida)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Grises Realizada")

def binarizacion(img):
    imagen = img
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    #_, binary_img = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
    _, binary_img = cv2.threshold(img_gris, 170, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("Imagen Binarizada", binary_img)

    filenameBinarizada = file+"_binarizada.jpg"                
    cv2.imwrite(filenameBinarizada, binary_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Binarización Realizada")
    
#Función Filtro Promedio
def promedio(img):
    #Crea el kernel
    kernel3x3 = np.ones((3,3),np.float32)/9.0
    kernel5x5 = np.ones((5,5),np.float32)/25.0

    #Filtra la imagen utilizando el kernel anterior
    salida3 = cv2.filter2D(img,-1,kernel3x3)
    salida5 = cv2.filter2D(img,-1,kernel5x5)

    cv2.imshow("salida3",salida3)
    
    cv2.imshow("salida5",salida5)
    
    filename3x3 = file+"_kernel3x3.jpg"                
    cv2.imwrite(filename3x3, salida3) #Se guarda la imagen obtenida con extensión ´jpg´ y el nombre de la técnica aplicada

    cv2.waitKey(0) #Mostrará la ventana infinitamente hasta que se presione cualquier tecla.
    cv2.destroyAllWindows()
    result.config(text="Filtro Promedio Realizado")

def roberts(img):
    imagen = img
    #nombre = img.split(".")
    gray_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    result = np.zeros(gray_img.shape, dtype=np.float32)
    mask1 = np.array([[1, 0], [0, -1]], dtype=np.float32)
    mask2 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    for i in range(1, gray_img.shape[0] - 1):
        for j in range(1, gray_img.shape[1] - 1):
            pixel1 = np.sum(gray_img[i - 1:i + 1, j - 1:j + 1] * mask1)
            pixel2 = np.sum(gray_img[i - 1:i + 1, j - 1:j + 1] * mask2)
            result[i, j] = np.sqrt(pixel1 ** 2 + pixel2 ** 2)
    result = (result / np.max(result)) * 255
    
    cv2.imshow('Filtro Robert', result.astype(np.uint8))
    #cv2.imwrite(nombre[0] + "_ro.jpg", result.astype(np.uint8))
    filenameRoberts = file+"_roberts.png"                
    cv2.imwrite(filenameRoberts, result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Roberts Realizado")

def contornos(img):
    imagen = img
    
    gauss = cv2.GaussianBlur(imagen, (3,3), 0)
    canny = cv2.Canny(gauss, 120, 240)
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    canny = cv2.dilate(canny,kernel,iterations=1)
    
    #BINARIZACIÓN PARA CONTORNOS
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img_gris, 150, 255, cv2.THRESH_BINARY)
    
    contours,_ =cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()

    cv2.drawContours(image_copy, contours, -20, (255, 0, 200), 1)
    
    contornos1,hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    cv2.drawContours(imagen, contornos1, -1, (0,255,0), 3)
      
    cv2.imshow('Binarizada',th)
    cv2.imshow("Bordes", canny)
    cv2.imshow("Contornos RETR_CCOPM", image_copy)
    cv2.imshow('Contornos RETR_TREE',imagen)
    
    filename = file+"_bordes.jpg"                
    cv2.imwrite(filename, canny)    
    
    filename = file+"_contornos-CCOMP.jpg"                
    cv2.imwrite(filename, image_copy)
    
    filename = file+"contornos-TREE.jpg"                
    cv2.imwrite(filename, imagen)
    
    filename = file+"_segmentacion.jpg"                
    cv2.imwrite(filename, th)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Detección de Bordes Realizada") #Permite a los usuarios destruir o cerrar todas las ventanas en cualquier momento después de salir del script.

def erosion(img):
    imagen = img
    # Agregue este para reajustar el tamaño
    #resized = ResizeWithAspectRatio(imagen, width=300) 
    # Erosión
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(imagen,kernel,iterations = 1)
    
    cv2.imshow('Original',imagen)
    cv2.imshow('Erosion',erosion)
    
    filenameErosion = file+"_erosion.jpg"                
    cv2.imwrite(filenameErosion, erosion)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Erosión Realizada")

def dilatacion(img):
    imagen = img
    kernel = np.ones((5,5),np.uint8)
    dilatacion = cv2.dilate(imagen,kernel,iterations = 1)

    cv2.imshow("Original", imagen)
    cv2.imshow("Dilatacion", dilatacion)

    filenameDilatada = file+"_dilatacion.png"                
    cv2.imwrite(filenameDilatada, dilatacion)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Dilatación Realizada")

def apertura(img):
    imagen = img
    
    #Filtro Apertura
    opening = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, np.ones((15,15),np.uint8))

    #Filtro Top Hat
    tophat = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, np.ones((15,15),np.uint8))

    cv2.imshow('Original',imagen)
    cv2.imshow('Apertura',opening)
    cv2.imshow('Top Hat',tophat)
    
    filenameApertura = file+"_apertura.png"                
    cv2.imwrite(filenameApertura, opening)
    
    filenameTH = file+"_tophat.png"                
    cv2.imwrite(filenameTH, tophat)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Apertura Realizada")

#Negativo de imagen
def cerradura(img):
    imagen = img
    
    closing = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))

    #Filtro Black Hat
    blackhat = cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, np.ones((15,15),np.uint8))
    
    cv2.imshow('Original',imagen)
    cv2.imshow('Cerradura', closing)
    cv2.imshow('Black Hat', blackhat)

    filenameCerradura = file+"_cerradura.png"                
    cv2.imwrite(filenameCerradura, closing)
    
    filenameBH = file+"_blackhat.png"                
    cv2.imwrite(filenameBH, blackhat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.config(text="Cerradura Realizada") #Permite a los usuarios destruir o cerrar todas las ventanas en cualquier momento después de salir del script.

#Funcion para la ecualización hiperbolica
def gradiente(img):
    imagen = img
    kernel = np.ones((5,5),np.uint8)
    
    gradiente = cv2.morphologyEx(imagen, cv2.MORPH_GRADIENT, kernel)
    
    cv2.imshow("Imagen Original",imagen)
    cv2.imshow("Gradiente Morfologico",gradiente)
    
    filenameGradiente = file+"_gradiente.png"                
    cv2.imwrite(filenameGradiente, gradiente)
    
    cv2.waitKey(0) #Mostrará la ventana infinitamente hasta que se presione cualquier tecla.
    cv2.destroyAllWindows()
    result.config(text="Gradiente Morfologico Realizado") 

def OR():
    imagen1= input("Introduce la imagen 1: ")
    imagen2 = input("Introduce la imagen 2: ")
    img1 = cv2.imread(imagen1)
    img2 = cv2.imread(imagen2)
    result = cv2.bitwise_or(img1, img2)
    cv2.imshow('Imagen OR', result)
    
    filenameOR = file+"_or.png"                
    cv2.imwrite(filenameOR, result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def start(): #Menu de opciones
    if (file == "No se ha seleccionado el archivo") or (file2 == "No se ha seleccionado el archivo"):
        print("No se ha seleccionado el archivo")
        messagebox.showerror(title="Error", message="Inserte una imagen valida")
    else:
        
            try:
                if modo.get() == "Grises":
                    grises(image)
                if modo.get() == "Binarización":
                    binarizacion(image)
                if modo.get() == "Promedio":
                    promedio(image)
                if modo.get() == "Roberts":
                    roberts(image)
                if modo.get () == "Contornos":
                    contornos(image)
                if modo.get() == "Erosión":
                    erosion(image)
                if modo.get() == "Dilatación":
                    dilatacion(image)
                if modo.get() == "Apertura":
                    apertura(image)
                if modo.get() == "Cerradura":
                    cerradura(image)
                if modo.get() == "Gradiente Morfológico":
                    gradiente(image)     
                if modo.get() == "OR":
                    OR()       
            except:
                result.config(text="No se puede realizar esa operación")
                print("No se puede realizar esa operación")

#Función que elige el tipo de imagen y abre un archivo de diálogo
file = ""
def choose():
    global file
    file = filedialog.askopenfilename(filetypes = [
        ("image", ".jfif"),
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg"),
    	("image", ".bmp")]) #Lista de tipos de archivos admitidos por este programa
    if len(file) > 0:
        global image
        image = cv2.imread(file)
    fileLabel.configure(text=file)

file2 = ""
def choose2():
    global file2
    file2 = filedialog.askopenfilename(filetypes = [
        ("image", ".jfif"),
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg"),
    	("image", ".bmp")]) #Lista de tipos de archivos admitidos por este programa
    if len(file2) > 0:
        global image2
        image2 = cv2.imread(file2)

#Ventana principal
window = Tk()
window.title("Morfología Matemática Comunidad 1")

window.config(bg = "SteelBlue2")
window.iconbitmap("./root/morfoimg.ico")

image = tki.PhotoImage(file="IPN.png")
imageS = image.subsample(6)
widget = tki.Label(image=imageS, bg = "SteelBlue2")
widget.place(x=-45,y=-4)

image2 = tki.PhotoImage(file="ESCOM.png")
imageS2 = image2.subsample(14)
widget2 = tki.Label(image=imageS2, bg = "SteelBlue2")
widget2.place(x=440,y=5)
window.geometry("600x500")

lbl = Label(window, text="ESCOM - IPN\n\n Morfología Matemática\n\n Operaciones Disponibles\n", font=("Arial", 15), bg = "SteelBlue2",  anchor="nw")
lbl.place(x=160, y=20)

s = ttk.Style()
s.configure("Peligro.TCombobox", foreground="black", width=20)
s.map("Peligro.TCombobox", foreground=[("active", "#FFA500")])

#Menú para seleccionar la operacion
modo = ttk.Combobox(window, style="Peligro.TCombobox",values=["Grises", "Binarización", "Promedio", "Roberts", "Contornos", "Erosión", "Dilatación", "Apertura", "Cerradura", "Gradiente Morfológico", "OR"],state="readonly")
modo.current(0)
modo.place(x =235,y = 150)

#Boton para cargar imagen
fileButton = Button(window, text="Cargar archivo", command=choose)
fileButton.place(x=255, y=220)
fileButton["bg"] = "#96C7F0"
fileLabel = Label(window, text=file, font=("Arial", 9), bg="light goldenrod", fg="black", width=70)
fileLabel.place(x=55, y=320)
fileLabel.config(anchor=CENTER)


#Boton de accion
btn = Button(window, text="Realizar operacion", command=start)
btn["bg"] = "#96C7F0"
btn.place(x=250, y=440)

result = Label(window, text="", font=("Arial", 12), bg="light goldenrod", fg="black",width=60)
result.place(x=30, y=385)
result.config(anchor=CENTER)

window.mainloop()