import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    if dir_ == ".DS_Store":  # Skip macOS metadata file
        continue

    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # ✅ Skip files like .DS_Store and .gitignore
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):#[:1]: # itere les images dans chaque dossier
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_full_path = os.path.join(dir_path, img_path)
        if img is None:
            print(f"⚠️ Warning: Could not read {img_full_path} (Skipping)")
            continue  # Skip this image

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #rgb pour importer dans mediapap


        #plt.show()


        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:


            for hand_landmarks in results.multi_hand_landmarks: # itere landmark
                """
                    mp_drawing.draw_landmarks(
                        img_rgb,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                """ # affiche les landmarks image


                for i in range(len(hand_landmarks.landmark)):

                    print(hand_landmarks.landmark[i])

                    x = hand_landmarks.landmark[i].x # positions de chaque landmark
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)



            plt.figure()
            plt.imshow(img_rgb)

    plt.show()


f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
