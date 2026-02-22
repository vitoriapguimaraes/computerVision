import cv2
import mediapipe as mp
import os

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

maos = mp_maos.Hands()

camera = cv2.VideoCapture(0)
resolucao_x = 1280
resolucao_y = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)
word_app = False
firefox_app = False
excel_app = False


def encotra_coordenadas_maos(img, lado_invertido=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = maos.process(img_rgb)
    todas_maos = []
    if resultado.multi_hand_landmarks:
        for lado_mao, marcacoes_maos in zip(
            resultado.multi_handedness, resultado.multi_hand_landmarks
        ):
            info_mao = {}
            coordenadas = []
            for marcacao in marcacoes_maos.landmark:
                coord_x, coord_y, coord_z = (
                    int(marcacao.x * resolucao_x),
                    int(marcacao.y * resolucao_y),
                    int(marcacao.z * resolucao_x),
                )
                coordenadas.append((coord_x, coord_y, coord_z))
            info_mao["coordenadas"] = coordenadas
            if lado_invertido:
                if lado_mao.classification[0].label == "Left":
                    info_mao["lado"] = "Right"
                else:
                    info_mao["lado"] = "Left"
            else:
                info_mao["lado"] = lado_mao.classification[0].label

            todas_maos.append(info_mao)
            mp_desenho.draw_landmarks(img, marcacoes_maos, mp_maos.HAND_CONNECTIONS)
    return img, todas_maos


def dedos_levantados(mao):
    dedos = []
    if mao["lado"] == "Right":
        if mao["coordenadas"][4][0] < mao["coordenadas"][3][0]:
            dedos.append(True)
        else:
            dedos.append(False)
    else:
        if mao["coordenadas"][4][0] > mao["coordenadas"][3][0]:
            dedos.append(True)
        else:
            dedos.append(False)
    for ponta_dedo in [8, 12, 16, 20]:
        if mao["coordenadas"][ponta_dedo][1] < mao["coordenadas"][ponta_dedo - 2][1]:
            dedos.append(True)
        else:
            dedos.append(False)
    return dedos


while True:
    sucesso, img = camera.read()
    img = cv2.flip(img, 1)

    img, todas_maos = encotra_coordenadas_maos(img)

    if len(todas_maos) == 1:
        info_dedos_mao1 = dedos_levantados(todas_maos[0])
        if todas_maos[0]["lado"] == "Right":
            if info_dedos_mao1 == [False, True, False, False, False] and not word_app:
                word_app = True
                os.startfile(
                    r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"
                )
            if info_dedos_mao1 == [False, True, True, False, False] and not excel_app:
                excel_app = True
                os.startfile(
                    r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"
                )
            if info_dedos_mao1 == [False, True, True, True, False] and not firefox_app:
                firefox_app = True
                os.startfile(r"C:\Program Files\Mozilla Firefox\firefox.exe")
            if info_dedos_mao1 == [False, False, False, False, False] and firefox_app:
                firefox_app = False
                os.system("TASKKILL /IM firefox.exe")
            if info_dedos_mao1 == [False, True, False, False, True]:
                break
            # fazer indicações da forma das maos como variaveis do que uma lista de false,...

    cv2.imshow("Imagem", img)

    tecla = cv2.waitKey(1)
    if tecla == 27:
        break
