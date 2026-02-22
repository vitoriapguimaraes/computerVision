"""
Nesta etapa, utilizaremos as máscaras preestabelecidas pelo OpenCV. São elas:
KNN: muito comum nos processamentos de machine learning, trata-se de um método de cluster onde os grupos são formados com base na sua semelhança. Essa máscara utiliza as distâncias dos pixels para definir o plano de fundo e o objeto de primeiro plano;
GMG: utiliza o Teorema de Bayes e o aplica nos 5 primeiros segundos do vídeo. Através dessa teoria, atualiza os números de pixels e atribui pesos maiores aos novos pixels, o que permite melhor identificar os possíveis objetos de primeiro plano;
CNT: é um algoritmo count que verifica os valores dos pixels nos frames anteriores a fim de tentar identificar se esses pixels correspondem ao plano de fundo ou à objetos em movimento;
MOG: mixture-of-gaussians ou mistura de fundo adaptativa, utiliza a curva gaussiana comum na qual cada pixel é caracterizado por sua intensidade no espaço de cores RGB;
MOG2: versão melhorada do MOG.
"""

import cv2
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(BASE_DIR, "dataset", "Ponte.mp4")

# KNN = 5.06
# GMG = 11.04
# CNT = 3.53
# MOG = 7.49
# MOG2 = 4.93

algorithm_types = ["KNN", "GMG", "CNT", "MOG", "MOG2"]


def Subtractor(algorithm_type):
    if algorithm_type == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    print("Detector inválido")
    sys.exit(1)


cap = cv2.VideoCapture(VIDEO)
# e1 = cv2.getTickCount()

background_subtractor = []

for i, a in enumerate(algorithm_types):
    print(i, a)
    background_subtractor.append(Subtractor(a))


def main():
    # frame_number = -1
    while cap.isOpened:
        ok, frame = cap.read()

        if not ok:
            print("Frames acabaram!")
            break

        # frame_number += 1
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)

        knn = background_subtractor[0].apply(frame)
        gmg = background_subtractor[1].apply(frame)
        cnt = background_subtractor[2].apply(frame)
        mog = background_subtractor[3].apply(frame)
        mog2 = background_subtractor[4].apply(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("KNN", knn)
        cv2.imshow("GMG", gmg)
        cv2.imshow("CNT", cnt)
        cv2.imshow("MOG", mog)
        cv2.imshow("MOG2", mog2)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            # or frame_number > 300
            break

        # e2 = cv2.getTickCount()
        # t = (e2 - e1) / cv2.getTickFrequency()
        # print(t)


main()
