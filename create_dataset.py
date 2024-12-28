# import os
# import cv2
# import mediapipe as mp
# import matplotlib.pyplot as plt
# import pickle


# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []

# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # mp_drawing.draw_landmarks(
#                 #     img_rgb, #image to draw
#                 #     hand_landmarks, #model output
#                 #     mp_hands.HAND_CONNECTIONS, #hand connections
#                 #     mp_drawing_styles.get_default_hand_landmarks_style(),
#                 #     mp_drawing_styles.get_default_hand_connections_style(),
#                 # )
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x)
#                     data_aux.append(y)

#             data.append(data_aux)
#             labels.append(dir_)

# f = open('data.pickle', 'wb')
# pickle.dump({'data':data, 'labels':labels}, f)
# f.close()



# #         plt.figure()
# #         plt.imshow(img_rgb)
# #         plt.axis('off')  # Optional: turn off axis if you just want to see the image
        
# # plt.show()
# # # plt.close()  # Close the figure to free memory




import os
import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ensure exactly 21 landmarks per hand, each with x and y
                if len(hand_landmarks.landmark) == 21:
                    for landmark in hand_landmarks.landmark:
                        data_aux.extend([landmark.x, landmark.y])  # Adding x and y for each landmark

            if len(data_aux) == 42:  # Only add if the full set of 42 features is present
                data.append(data_aux)
                labels.append(dir_)

hands.close()  # Release Mediapipe resources

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
