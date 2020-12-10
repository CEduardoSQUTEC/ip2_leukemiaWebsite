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
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV) # Convertimos a HSV para eliminar las tinturaciones
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY) # Imagen original convertida a escala de grises

    # Las matrices fueron escogidas con antelacion para realizar una mascara a la imagen original y así tan solo obtener las tinturaciones resaltadas
    l_b = np.array([0, 229, 0])
    u_b = np.array([255, 255, 255])

    # Aplicacion de la máscara
    mask = cv2.inRange(hsv,l_b,u_b)
    res = cv2.bitwise_and(hsv,hsv,mask=mask)

    #Se aplica un filtro de promedio para eliminar el ruido
    filtrada = cv2.fastNlMeansDenoisingColored(res,None,60,60,7,21)
    filtrada2 = cv2.medianBlur(res, 13);


    # Aca ploteamos la imagen de las tinturaciones antes de un filtrado para eliminar el ruido
    plt.plot(1),plt.imshow(res),plt.title('Aplicación de máscara para resaltar las tinturaciones')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    # Pasamos a escala de grises la imagen filtrada
    filtrada_gray = cv2.cvtColor(filtrada2, cv2.COLOR_RGB2GRAY)

    ret2,th2 = cv2.threshold(filtrada_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    iix = cv2.bitwise_not(th2)

    # Se binariza la imagen original
    ret3,th3 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ii3 = cv2.bitwise_not(th3)

    images = [iix]
    titles = ["Imagen Binarizada"]

    # Creamos un kernel para la dilatación
    kernel = np.ones((12,12),np.uint8)
    # Aca se aplica la dilatación a la imagen
    dilatar = cv2.dilate(th2, kernel)
    dilatar_not = cv2.bitwise_not(dilatar)
    titleas = ["Imagen dilatada"]

    plt.subplot(121),plt.imshow(th2),plt.title('Tinturaciones filtradas')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dilatar),plt.title('Tinturaciones filtradas dilatadas')
    plt.xticks([]), plt.yticks([])
    plt.show()

    img2_fg = cv2.bitwise_and(ii3,dilatar_not) # Apliación de la multiplicación para eliminar las tinturaciones.
    titles = ["Imagen TOTAL"]

    label_imgx = label(img2_fg)
    regionsx = regionprops(label_imgx)

    fig, ax = plt.subplots()
    ax.imshow(img2_fg, cmap=plt.cm.gray)

    for propsx in regionsx:
        y00, x00 = propsx.centroid
        orientation = propsx.orientation
        x10 = x00 + math.cos(orientation) * 0.5 * propsx.minor_axis_length
        y10 = y00 - math.sin(orientation) * 0.5 * propsx.minor_axis_length
        x20 = x00 - math.sin(orientation) * 0.5 * propsx.major_axis_length
        y20 = y00 - math.cos(orientation) * 0.5 * propsx.major_axis_length

        ax.plot((x00, x10), (y00, y10), '-r', linewidth=0.5)
        ax.plot((x00, x20), (y00, y20), '-r', linewidth=0.5)
        ax.plot(x00, y00, '.g', markersize=1)

        minrx, mincx, maxrx, maxcx = propsx.bbox
        bxx = (mincx, maxcx, maxcx, mincx, mincx)
        byx = (minrx, minrx, maxrx, maxrx, minrx)
        ax.plot(bxx, byx, '-b', linewidth=0.5)

    ax.axis((0, 700,504, 0))
    plt.show()

    propsx = regionprops_table(label_imgx, properties=('area','centroid',
                                                     'orientation',
                                                     'major_axis_length', # longitud maxima.
                                                     'minor_axis_length')) # longitud minima.

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

    plt.plot(),plt.imshow(img2_fg,'gray')
    plt.title("Imagen binarizada con ruido"), plt.xticks(), plt.yticks()
    plt.show()
    plt.plot(),plt.imshow(coins_cleaned,'gray')
    plt.title("Imagen binarizada limpia"), plt.xticks(), plt.yticks()
    plt.show()

    # Detectar bordes con Canny
    canny = cv2.Canny(img2_fg, 50, 150)

    #cv2.imshow("canny", canny)

    # Contornos
    (contornos, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Número de células
    print("Hay {} células".format(len(contornos)))

    #cv2.drawContours(coins_cleaned, contornos, -1, (0, 0, 255), 2)
    #cv2.imshow("contornos", coins_cleaned)

    #cv2.waitKey(0)

    edge_roberts = roberts(coins_cleaned)
    label_img = label(coins_cleaned)
    coins_cleaned2=coins_cleaned.astype(int)
    regions = regionprops(label_img)

    fig, ax = plt.subplots()
    ax.imshow(coins_cleaned, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=0.5)
        ax.plot(x0, y0, '.g', markersize=1)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=0.5)

    ax.axis((0, 700,504, 0))
    plt.show()

    props = regionprops_table(label_img, properties=('area','centroid',
                                                     'orientation',
                                                     'major_axis_length',
                                                     'minor_axis_length'))

    CENTROy = props["centroid-0"]
    Centrox = props["centroid-1"]

    centroyx = np.vstack((CENTROy,Centrox))
    centroyx = np.transpose(centroyx)

    valuepixel = coins_cleaned2[499,180]
    print(valuepixel)

    valores = []
    valores = np.zeros((211,1),dtype=int)

    valores=[] 
    valores=np.zeros((211,1),dtype=int) 
    # for cor in centroyx: 
    #     for i in valores: 
    #         valores[i]=coins_cleaned2[cor[0], cor[1]] 
    return coins_cleaned