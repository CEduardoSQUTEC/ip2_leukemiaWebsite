import cv2
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
import math
import statistics as stats

# imagen = cv2.imread('FOTO2.jpg') #Se lee la imagen
# img = cv2.imread('FOTO2.jpg',0)

def countingImage(imagen):
    hsv=  cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV) #Convertimos a HSV para eliminar las tinturaciones
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY) #Imagen original convertida a escala de grises

    #Las matrices fueron escogidas con antelacion para realizar una mascara a la imagen original y así
    #tan solo obtener las tinturaciones resaltadas
    #l_b=np.array([0, 213, 0])
    #u_b=np.array([138, 255, 100])
    l_b=np.array([0, 229, 0])
    u_b=np.array([255, 255, 255])
    #aplicacion de la máscara
    mask=cv2.inRange(hsv,l_b,u_b)
    res=cv2.bitwise_and(hsv,hsv,mask=mask)
    #Se aplica un filtro de promedio para eliminar el ruido
    filtrada=cv2.fastNlMeansDenoisingColored(res,None,60,60,7,21)

    filtrada2=cv2.medianBlur(res, 13)

    #Pasamos a escala de grises la imagen filtrada
    filtrada_gray = cv2.cvtColor(filtrada2, cv2.COLOR_RGB2GRAY)
    #plt.plot(),plt.imshow(filtrada_gray),plt.title('Imagen aplicada la máscara')
    #plt.xticks([]), plt.yticks([])
    #plt.show()

    ret2,th2 = cv2.threshold(filtrada_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    iix = cv2.bitwise_not(th2)

    #Se binariza la imagen original
    ret3,th3 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ii3 = cv2.bitwise_not(th3)

    #Se dibuja la imagen de las tinturaciones binarizada

    #creamos un kernel para la dilatación
    kernel=np.ones((12,12),np.uint8)
    #Aca se aplica la dilatación a la imagen
    dilatar=cv2.dilate(th2, kernel)
    dilatar_not = cv2.bitwise_not(dilatar)

    img2_fg = cv2.bitwise_and(ii3,dilatar_not)

    label_imgx = label(img2_fg)
    regionsx = regionprops(label_imgx)

    propsx = regionprops_table(label_imgx, properties=('area','centroid',
                                                     'orientation',
                                                     'major_axis_length',
                                                     'minor_axis_length'))

    dOrderx=sorted(propsx["area"])

    dOrderx=np.array(dOrderx, dtype=float)

    mediaa=stats.median(dOrderx)
    ELI=mediaa/3.29

    # lA DIFERENCIA ENTRE LA MEDIA Y EL VALOR MIN ES DE 3.29

    if mediaa <= 123:
        label_objects, nb_labels = ndi.label(img2_fg)
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes > ELI
        mask_sizes[0] = 0
        coins_cleaned = mask_sizes[label_objects]
    elif mediaa <= 496:
        label_objects, nb_labels = ndi.label(img2_fg)
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes > ELI
        mask_sizes[0] = 0
        coins_cleaned = mask_sizes[label_objects]        

    # Detectar bordes con Canny
    canny = cv2.Canny(img2_fg, 50, 150)

    # Contornos
    (contornos, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edge_roberts = roberts(coins_cleaned)

    label_img = label(coins_cleaned)
    coins_cleaned2=coins_cleaned.astype(int)
    regions = regionprops(label_img)

    fig, ax = plt.subplots()
    ax.imshow(imagen, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=1)

    ax.axis((0, 700,504, 0))
    plt.title("")
    plt.savefig("static/img/processedImage.png")
    # plt.show()

    props = regionprops_table(label_img, properties=('area','centroid',
                                                     'orientation',
                                                     'major_axis_length',
                                                     'minor_axis_length'))

    CENTROy=props["centroid-0"]
    Centrox=props["centroid-1"]

    centroyx=np.vstack((CENTROy,Centrox))
    centroyx=np.transpose(centroyx)

    valores=[]
    valores=np.zeros((211,1),dtype=int)

    GRojo=0
    Leuco=0
    for cor in centroyx:
       if coins_cleaned2[round(cor[0])][round(cor[1])] == 1:
           GRojo += 1
       else:
           Leuco += 1

    Orden=dOrderx.astype(int)
    for i in range(len(Orden)):
        if Orden[i]/mediaa >= 1.5:
            GRojo +=1
        elif Orden[i]/mediaa >= 2.5:
            GRojo +=2
        elif Orden[i]/mediaa >= 3.5:
            Grojo =+3
        elif Orden[i]/mediaa >= 4:
            GRojo =+5

    print("La cantidad de Globulos Rojos total es: ", GRojo)
    print("La cantidad de Leucocitos es: ", Leuco) 
    return (GRojo, Leuco)