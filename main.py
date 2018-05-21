import numpy as np
import cv2
import time
import xlsxwriter


warped = None,
roiPts = []
startTime = 0
endTime = 0
fgbg = None
last =None
warpedAux = None
row = 0
col = 0
worksheet = None


countFrames = 0
countFramesTrain = 0
countAux = 0

countFramesLeftArmAuX = 0
countFramesRightArmAux = 0
countFramesTopArmAux = 0
countFramesBottomArmAux = 0
countFramesCenterArmAux = 0
countFramesTracking = 0

countFramesTotalLeftArm = 0
countFramesTotalRightArm = 0
countFramesTotalTopArm = 0
countFramesTotalBottomArm = 0
countFramesTotalCenterArm = 0

yMinLeftArm = 0
yMaxLeftArm = 999

yMinRightArm = 0
yMaxRightArm = 999

xMinTopArm = 0
xMaxTopArm = 999

xMinBottomArm = 0
xMaxBottomArm = 999

last_flag = 0
masking = True

video = None

leftArmEntry = 0
rightArmEntry = 0
topArmEntry = 0
bottomArmEntry = 0
centerArmEntry = 0

BLUR_KSIZE = 5
MORPH_KSIZE = 15
NUM_FRAMES = 30


FLAG_LEFT_ARM = 1
FLAG_RIGHT_ARM = 2
FLAG_TOP_ARM = 3
FLAG_BOTTOM_ARM = 4
FLAG_CENTER = 5
FLAG_AUX = 999

NUM_INTERATION_ERODE = 1
NUM_INTERATION_DILATE = 1
MIN_AREA = 500
warpPts = []
maskPts = []

NUM_FRAMES_TRACK = 269
NUM_FRAMES_TRAIN = 269


THRES_VAL = 20
PERCENT_INSIDE = 90
PERCENT_INSIDE_CENTER = 85

def selectPointsWarp(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, warpPts

    # cv2.line(frame, (x,y), (x+50,y), (0, 255, 0), 2)
    # cv2.imshow("warp", frame)
    # print("(", x, " , ", y, ")")
    if event == cv2.EVENT_LBUTTONDOWN:
        warpPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("warp", frame)
        # print("--------------------->(", x, " , ", y, ")")

def selectPointsMask(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global warped, warpPts


    # print("(", x, " , ", y, ")")
    if event == cv2.EVENT_LBUTTONDOWN:
        maskPts.append((x, y))
        cv2.circle(warped, (x, y), 4, (0, 0, 255), 2)
        cv2.imshow("mask", warped)
        # print("--------------------->(", x, " , ", y, ")")


def order_points(pts):
    # Iniciando array com os pontos de warp
    # ordem: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # o ponto no top-left tem a menor soma
    # enquanto o bottom-right tem maior soma
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # top-right tem a menor diferença
    # enquanto que bottom left tem maior diferença
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # retorna as coordenadas ordenadas
    return rect


def four_point_transform(image, points):
    global warpedAux

    # pts = np.array([(159., 82.), (457., 87.), (474., 372.), (142., 371.)], dtype="float32")
    pts = points
    if (len(image.shape)) < 3:
        height, width = image.shape
    else:
        height, width, channels = image.shape

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # calcula a largura da nova imagem
    # como sendo maior distanância entre bottom-right e bottom-left
    # widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # maxWidth = max(int(widthA), int(widthB))
    # calculando a nova altura da imagem
    # distancia máxima entre top-right e bottom-right
    # heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # maxHeight = max(int(heightA), int(heightB))\

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    # definindo a matriz de transformação
    # e realizando o warp
    M = cv2.getPerspectiveTransform(rect, dst)
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.warpPerspective(image, M, (width, height))
    # retorna a imagem tranformada
    cv2.imwrite("image/Warped.png", warped)


    return warped


def maskImage(image, points):

    pts = points
    # pts = np.array(
    #    [(1, 240), (313, 241), (323, 55), (372, 53), (372, 236), (702, 239), (702, 274), (372, 277), (372, 468),
    #    (316, 473), (316, 280), (2, 278)], dtype=np.int32)
    # criando imagem com todos pixeis pretos
    mask = np.zeros(image.shape, dtype=np.uint8)
    # definindo os pontos da mascara
    roi_corners = pts
    # pegando o numero de canais
    if len(image.shape) > 2:
        channel_count = image.shape[2]
    else:
        channel_count = 1
    # defenindo a cor do poligono
    # que vai ser usado como mascara para branco
    # de acordo com numero de canais da imagem
    poly_color = (255,) * channel_count
    # criando um poligono preenchido a partir da mascara
    cv2.fillConvexPoly(mask, roi_corners, poly_color)
    # aplicando mascara na imagem
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def calFPS():

    global NUM_FRAMES, endTime, startTime

    endTime = time.time()
    seconds = endTime - startTime
    fpsEstimated = NUM_FRAMES / seconds
    # print("Estimated frames per second : {0}".format(fpsEstimated))
    startTime = time.time()
    endTime = 0


def saveFrame(fileName, frame, frameNum):
    fileName = fileName
    fileName += str(frameNum)
    fileName += '.png'
    cv2.imwrite(fileName, frame)


def trackMice(image):
    global warped, video, last, warpedAux, worksheet, row, col

    im2, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
            # print("Contours area", area)
            # cv2.drawContours(warped, contours, -1, (0, 255, 0), 3)
            c = max(contours, key=cv2.contourArea)
            # print("Contour Area", cv2.contourArea(c))
            if cv2.contourArea(c) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(c)
                # print(c)
                # draw the book contour (in green)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 2)
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # print("Contour Area", cv2.contourArea(c))
                # draw the contour and center of the shape on the image
                # cv2.drawContours(warped, contours, -1, (0, 255, 0), 2)
                cv2.circle(video, (cX, cY), 7, (0, 255, 0), -1)
                # cv2.putText(warped, "center", (cX - 20, cY - 20),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if last is not None:
                    cv2.line(warpedAux, last, (cX,cY), (0, 0, 255), 2)
                    last = (cX,cY)
                else:
                    last = (cX, cY)
                # print("Gray iamge shape", image.shape)
                # print("Coordenates (", cX, ", ", cY, ")")
                # applyColorMap(cX, cY)
                # regionRecognition(cX, cY, c)
                worksheet.write(row, col, cX)
                worksheet.write(row, col + 1, cY)
                row += 1

                saveFrame('warpeLines', warpedAux, 1)
                mapMazeArm(c)

def checkAllContourPoints(contour, xMin, xMax, yMin, yMax):

    global PERCENT_INSIDE
    pointInside = 0
    for points in contour:
        for point in points:
            if xMin < point[0] < xMax and yMin < point[1] < yMax:
                pointInside += 1
            # print("Pointx", point[0], "Pointy", point[1])
    # print("Points inside", pointInside)
    # print("Num points", len(contour))
    percentInside = (pointInside/len(contour)) * 100
    # print("Percent inside", percentInside)
    if percentInside >= PERCENT_INSIDE:
        return True
    else:
        return False


def checkAllContourPointsCenter(contour, xMin, xMax, yMin, yMax):

    global PERCENT_INSIDE_CENTER
    pointInside = 0
    for points in contour:
        for point in points:
            if xMin < point[0] < xMax and yMin < point[1] < yMax:
                pointInside += 1
            # print("Pointx", point[0], "Pointy", point[1])
    # print("Points inside", pointInside)
    # print("Num points", len(contour))
    percentInside = (pointInside/len(contour)) * 100
    # print("Percent inside", percentInside)
    if percentInside >= PERCENT_INSIDE_CENTER:
        return True
    else:
        return False

def calcTimeInsideArm():
    global FLAG_AUX, countFramesTotalBottomArm, countFramesTotalTopArm, countFramesTotalRightArm, \
        countFramesTotalLeftArm, countFramesTotalCenterArm, last_flag, leftArmEntry, rightArmEntry, bottomArmEntry, topArmEntry, centerArmEntry, maskPts
    var = 10
    if FLAG_AUX == FLAG_CENTER:
        if last_flag != FLAG_AUX:
            # print("Transitou para o centro")
            centerArmEntry += 1
        points = np.array(
            [(maskPts[1][0]-var, maskPts[1][1]-var), (maskPts[4][0]+var, maskPts[4][1]-var),(maskPts[7][0]+var, maskPts[7][1]+var), (maskPts[10][0]-var, maskPts[10][1]+var)], dtype=np.int32)
        # print("Center points draw", points)
        # print("mask points",maskPts)
        draw(points, (0, 255, 255))
        # print("Centro")
        countFramesTotalCenterArm += 1
        last_flag = FLAG_CENTER
    elif FLAG_AUX == FLAG_LEFT_ARM:
        if last_flag != FLAG_AUX:
            # print("Transitou para o braço esquerdo")
            leftArmEntry += 1
        # print("Braço esquerdo")
        points = np.array(
            [(maskPts[0][0], maskPts[0][1]), (maskPts[1][0], maskPts[1][1]),(maskPts[10][0], maskPts[10][1]), (maskPts[11][0], maskPts[11][1])], dtype=np.int32)
        draw(points, (0, 0, 255))
        countFramesTotalLeftArm += 1
        last_flag = FLAG_LEFT_ARM
    elif FLAG_AUX == FLAG_RIGHT_ARM:
        if last_flag != FLAG_AUX:
            # print("Transitou para o braço direito")
            rightArmEntry += 1
        # print("Braço direito")
        points = np.array(
            [(maskPts[4][0], maskPts[4][1]), (maskPts[5][0], maskPts[5][1]),(maskPts[6][0], maskPts[6][1]), (maskPts[7][0], maskPts[7][1])], dtype=np.int32)
        draw(points, (0, 255, 0))
        countFramesTotalRightArm += 1
        last_flag = FLAG_RIGHT_ARM
    elif FLAG_AUX == FLAG_TOP_ARM:
        if last_flag != FLAG_AUX:
            # print("Transitou para o braço superior")
            topArmEntry += 1
        # print("Braço superior")
        points = np.array(
            [(maskPts[1][0], maskPts[1][1]), (maskPts[2][0], maskPts[2][1]),(maskPts[3][0], maskPts[3][1]), (maskPts[4][0], maskPts[4][1])], dtype=np.int32)
        draw(points, (255, 0, 0))
        countFramesTotalTopArm += 1
        last_flag = FLAG_TOP_ARM
    elif FLAG_AUX == FLAG_BOTTOM_ARM:
        if last_flag != FLAG_AUX:
            # print("Transitou para o braço inferior")
            bottomArmEntry += 1
        # print("Braço inferior")
        points = np.array(
            [(maskPts[7][0], maskPts[7][1]), (maskPts[8][0], maskPts[8][1]),(maskPts[9][0], maskPts[9][1]), (maskPts[10][0], maskPts[10][1])], dtype=np.int32)
        draw(points, (255, 255, 0))
        countFramesTotalBottomArm += 1
        last_flag = FLAG_BOTTOM_ARM


def mapMazeArm(contour):
    global xMinBottomArm, xMaxBottomArm, yMinLeftArm, yMaxLeftArm, yMinRightArm, yMaxRightArm, \
        xMinTopArm, xMaxTopArm, FLAG_AUX, maskPts
    var = 20
    # print(maskPts)
    # print("Mapeando")
    # verifica se tá no braço esquerdo
    if checkAllContourPoints(contour, maskPts[0][0], maskPts[1][0], yMinLeftArm, yMaxLeftArm):
        FLAG_AUX = FLAG_LEFT_ARM
    # verifica se tá no braço direito
    elif checkAllContourPoints(contour, maskPts[4][0], maskPts[5][0], yMinRightArm, yMaxRightArm):
        FLAG_AUX = FLAG_RIGHT_ARM
    # verifica se tá no braço superior
    elif checkAllContourPoints(contour, xMinTopArm, xMaxTopArm, maskPts[3][1], maskPts[1][1]):
        FLAG_AUX = FLAG_TOP_ARM
    # verifica se tá no braço inferior
    elif checkAllContourPoints(contour, xMinBottomArm, xMaxBottomArm, maskPts[7][1], maskPts[9][1]):
        FLAG_AUX = FLAG_BOTTOM_ARM
    # verifica se tá no centro
    elif checkAllContourPointsCenter(contour, maskPts[1][0]-var, maskPts[4][0]+var, maskPts[1][1]-var, maskPts[7][1]+var):
        FLAG_AUX = FLAG_CENTER

    calcTimeInsideArm()


def preProcessing(frame):
    # suavizando a imagem para eliminar ruidos
    frame = cv2.GaussianBlur(frame, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # convertendo a imagem em escala de cnza
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


def imageProcessing(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE))
    # print(kernel)
    # Aplicando as filtros morfologicos
    # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, 3)

    # erodeFrameOpen = cv2.erode(frame, kernel)
    # dilateFrameOpen = cv2.dilate(erodeFrameOpen, kernel)

    dilateFrameClose = cv2.dilate(frame, kernel, NUM_INTERATION_DILATE)
    erodeFrameClose = cv2.erode(dilateFrameClose, kernel, NUM_INTERATION_ERODE)
    # frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, 4)
    # morphOpen = cv2.morphologyEx(morphClose, cv2.MORPH_OPEN, kernel)
    return erodeFrameClose


def imageTransform(frame, points):

    global video, masking, maskPts, warped,warpedAux
    # fazendo warp da imagem usando
    # transformaçap perspectiva
    warped = four_point_transform(frame, points)
    video = warped.copy()
    if masking:
        while len(maskPts) < 12:
            # print("Aplicando mascara")
            cv2.imshow("mask", warped)
            k = cv2.waitKey(0)
            if k == 27:
                break
        masking = False
        print("dentro")
        warpedAux = video

        cv2.destroyWindow("mask")
    pts = np.array(maskPts, dtype=np.int32)
    # aplicando mascara para definir o labirinto como
    # regiao de tracking
    mask = maskImage(warped, pts)

    return mask


def draw(points, color):
    global video
    alpha = 0.5
    overlay = video.copy()
    output = video.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, video)


def main():

    global warped, NUM_FRAMES, \
        endTime, countFrames, countFramesTrain, countAux, video,\
        countFramesCenterArmAux, countFramesBottomArmAux, \
        countFramesTopArmAux, countFramesRightArmAux, countFramesLeftArmAuX, countFramesTracking, \
        countFramesTotalBottomArm, countFramesTotalTopArm, countFramesTotalRightArm, countFramesTotalLeftArm, \
        countFramesTotalCenterArm, FLAG_AUX, last_flag, rightArmEntry, leftArmEntry, topArmEntry, bottomArmEntry, frame, warpPts, worksheet
    track = False
    warping = True

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('Expenses01.xlsx')
    worksheet = workbook.add_worksheet()


    cap = cv2.VideoCapture('videos/rato1.avi')

    cv2.namedWindow('warp')

    cv2.setMouseCallback("warp", selectPointsWarp)

    fgbg = cv2.createBackgroundSubtractorMOG2(NUM_FRAMES_TRAIN, THRES_VAL, False)
    fgbg.setBackgroundRatio(600)
    # iniciando algumas variaveis
    startTime = time.time()  # tempo pra calcular fps
    startTime2 = time.time()  # tempo para calcumar tempo total do video
    # É suposto o rato iniciar o teste no centro do labirinto
    last_flag = FLAG_AUX = FLAG_CENTER

    # warpedAux = cv2.imread("image/Warped.png",0)
    while cap.isOpened():

        # contadores auxiliares
        countAux += 1
        countFrames += 1
        countFramesTrain += 1

        ret, frame = cap.read()
        # print("Original frame:", frame.shape)

        # verifica se deu certo
        if not ret:
            workbook.close()
            # pra contabiilizar o ultimo frame
            # calcTime(999)
            hours, rem = divmod(countAux/15, 3600)
            minutes, seconds = divmod(rem, 60)
            totalTime = countAux/15
            totalFrames = countFramesCenterArmAux+countFramesBottomArmAux+countFramesTopArmAux + \
                          countFramesRightArmAux+countFramesLeftArmAuX
            totalTimeAnalysis = totalFrames/15
            totalFramesArms = countFramesTotalCenterArm+countFramesTotalBottomArm+countFramesTotalTopArm +\
                              countFramesTotalRightArm+countFramesTotalLeftArm
            timeOpenArms = (countFramesTotalLeftArm + countFramesTotalRightArm) / 15
            timeTotal = totalFramesArms/15
            totalEntry = rightArmEntry+leftArmEntry+topArmEntry+bottomArmEntry
            print("Tempo total do vídeo: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            print("-------------------------------------------------------------")
            print("Total sum: ", countFramesTotalCenterArm+countFramesTotalBottomArm+countFramesTotalTopArm+countFramesTotalRightArm+countFramesTotalLeftArm)
            print("Total time: ", timeTotal)
            print("Tmp center: ", countFramesTotalCenterArm/15)
            print("P T A/T", (timeOpenArms/300)*100)
            print("P Entr A/T", (rightArmEntry/totalEntry)*100)
            print("Entr F", topArmEntry+bottomArmEntry)
            break

        if warping:
            while len(warpPts) < 4:
                print("Realizando warp")
                cv2.imshow("warp", frame)
                k = cv2.waitKey(0)
                if k == 27:
                    break
            print("Terminou warping")
            cv2.destroyWindow("warp")

        pts = np.array(warpPts, dtype="float32")
        warping = False
        cv2.namedWindow('mask')
        cv2.setMouseCallback("mask", selectPointsMask)

        # extraindo somente a imagem do labirinto
        # para realizar o tracking
        imageROI = imageTransform(frame,pts)

        # saveFrame('maskImage', imageROI, 1)
        # calculo do fds
        if countFrames == NUM_FRAMES:
            calFPS()
            countFrames = 0

        # condiçao pra terminar o treinamento
        if countAux == NUM_FRAMES_TRAIN:
            fgbg.setBackgroundRatio(0.0000000000000000001)
            # fgbg.setHistory(0)
            # print("Terminou treinamento")

        # condiçao pra começar o tracking
        if countAux == NUM_FRAMES_TRACK:
            # print("Começou tracking")
            track = True
            # fgbg.setVarThreshold(15)

        # print("frame ", countAux)

        # realizando pre-processamento da imagem
        imagePreProcessed = preProcessing(imageROI)

        # aplicando o metodo para subtraçao do fundo
        # apresentado por Zivkovic
        fgmask = fgbg.apply(imagePreProcessed)

        if countFramesTrain < 500:
            nome = "Frame" + str(countFramesTrain)
            file = nome + ".png"

        # saveFrame('imageSegmented', fgmask, countFramesTracking)

        if countFramesTrain < NUM_FRAMES_TRAIN:
            fileName = 'frame'
            fileName += str(countFramesTrain)
            fileName += '.png'
            # saveFrame(fileName, imageProcessed)

        if track:
            # melhorando imagem binarizada
            imageProcessed = imageProcessing(fgmask)
            countFramesTracking += 1
            trackMice(imageProcessed)
            cv2.imshow('ImageProcessed', imageProcessed)
            # saveFrame('imageProcessed',imageProcessed,countFramesTracking)

            # saveFrame('imageTraCKING', video, countFramesTracking)

        # exibindo imagens

        cv2.imshow('BackGroundSubtractor', fgmask)
        cv2.imshow('Maze', imageROI)

        # cv2.imshow('Close', morphClose)
        cv2.imshow('video', video)

        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
