# import cv2
# import mediapipe as mp
# import pickle
# import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# while True:

#     data_aux = []
#     x_ = []
#     y_ = []
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     predicted_character = ""  # Initialize predicted_character for display

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style(),
#             )
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x)
#                 data_aux.append(y)
#                 x_.append(x)
#                 y_.append(y)

#         # Calculate bounding box coordinates
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         # Make a prediction with the model
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = labels_dict[int(prediction[0])]

#         # Draw rectangle and predicted character only if a hand is detected
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     # Show the frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit the loop
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

# Labels for each class in the model (update as needed)
labels_dict = {
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
    'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'
}

while True:
    data_aux = []
    x_coords = []
    y_coords = []
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Reset prediction character for each frame
    predicted_character = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Collect the x, y coordinates for each landmark
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
                data_aux.extend([landmark.x, landmark.y])

        # Check if the data_aux feature count matches the model's expected input
        if len(data_aux) == 42:  # 21 landmarks * 2 (x, y) = 42 features
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = str(prediction[0])  # Convert to standard Python string

            # Draw bounding box around the hand
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x1, y1 = int(x_min * frame.shape[1]) - 10, int(y_min * frame.shape[0]) - 10
            x2, y2 = int(x_max * frame.shape[1]) + 10, int(y_max * frame.shape[0]) + 10

            # Draw rectangle and put text for prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, labels_dict.get(predicted_character, "Unknown"), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            print("Warning: Feature count mismatch with model input requirements.")

    # Show the frame with annotations
    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
