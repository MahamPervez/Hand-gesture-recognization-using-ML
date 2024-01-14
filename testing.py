import tkinter as tk
from tkinter import filedialog
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image, ImageTk

# Load the hand gesture model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Create the main window
window = tk.Tk()
window.title("Hand Gesture Detection")

# Create a frame for the video feed
video_frame = tk.Frame(window)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label to display the video feed
video_label = tk.Label(video_frame)
video_label.pack()

# Create a frame for the image
image_frame = tk.Frame(window)
image_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create a label to display the image
image_label = tk.Label(image_frame)
image_label.pack()

# Initialize the hand tracking module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labes_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'LIKE',27:'DISLIKE',28:'SPIDERMAN!!!'}


# Create a function for real-time gesture detection
window_open = True

def detect_gesture_realtime():
    global window_open
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        H, W, _ = frame.shape
        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = labes_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        # Display the frame in the Tkinter window
        video_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_image = Image.fromarray(video_image)
        video_image = ImageTk.PhotoImage(video_image)
        video_label.config(image=video_image)
        video_label.image = video_image

        # Update the Tkinter window
        window.update_idletasks()
        window.update()

        # Check for the '0' key to exit
        if cv2.waitKey(50) & 0xFF == ord('0'):
            window.destroy()
            break

    cap.release()
    cv2.destroyAllWindows()
    window.protocol("WM_DELETE_WINDOW", lambda: (window.iconify(), window_open := False))


# Create a button for real-time gesture detection
realtime_button = tk.Button(window, text="Real-time Gesture Detection", command=detect_gesture_realtime)
realtime_button.pack(pady=10)

# Create a function for uploading an image and detecting gestures
def detect_gesture_from_image():
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )
    if file_path:
        image = cv2.imread(file_path)
        detect_gesture_in_image(image)

# Create a button for uploading an image
image_button = tk.Button(window, text="Upload Image", command=detect_gesture_from_image)
image_button.pack(pady=10)

# Function to detect gestures in a given image
def detect_gesture_in_image(image):
    H, W, _ = image.shape
    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_char = labes_dict[int(prediction[0])]
        print("Predicted Class Index:", int(prediction[0]))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(image, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the image in the Tkinter window
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.image = image
    if cv2.waitKey(50) & 0xFF == ord('0'):
        window.destroy()

# Start the Tkinter event loop
window.mainloop()


