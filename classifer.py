import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import pygame
import os
import threading
import queue

# Initialize pygame for audio playback
pygame.mixer.init()

# Audio queue and worker
audio_queue = queue.Queue()

def play_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(100000)
    os.remove(audio_file)

def audio_worker():
    while True:
        text = audio_queue.get()
        if text is None:
            break
        play_audio(text)
        audio_queue.task_done()

audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Labels dictionary
# labels_dict = {0: 'Lancy', 1: 'Tina', 2: 'Ada', 3: 'My', 4: 'Name', 5: 'is', 6: 'I', 7: 'need', 8: 'to', 9: 'go', 10: 'the', 11: 'safe', 12: 'hospital'}
labels_dict = {0: 'Lancy', 1: 'Tina', 2: 'Ada', 3: 'My', 4: 'Name', 5: 'is', 6: 'I', 7: 'need', 8: 'to', 9: 'is', 10: 'the', 11: 'safe', 12: 'hospital'}

# Variables for debouncing and frame control
last_prediction = None
last_speak_time = 0
speak_interval = 0.5  # 1 second between speech
frame_counter = 0
prediction_interval = 5  # Process every 10th frame

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_counter += 1
    if frame_counter % prediction_interval == 0:
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            current_time = time.time()
            if predicted_character != last_prediction and (current_time - last_speak_time) > speak_interval:
                print(f"Predicted: {predicted_character}")
                audio_queue.put(predicted_character)
                last_prediction = predicted_character
                last_speak_time = current_time

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    print(f"Loop time: {time.time() - start_time:.3f} seconds")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
audio_queue.put(None)
audio_thread.join()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()