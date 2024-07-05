import cv2
import mediapipe as mp
import pyautogui
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
screen_w, screen_h = pyautogui.size()


# Function to process the frames and update mouse position
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_frame)
    hand_landmarks = output.multi_hand_landmarks
    frame_h, frame_w, _ = frame.shape

    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        # Index finger tip (landmark 8) for controlling the cursor
        index_finger_tip = landmarks[8]
        x = int(index_finger_tip.x * frame_w)
        y = int(index_finger_tip.y * frame_h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        screen_x = screen_w * index_finger_tip.x
        screen_y = screen_h * index_finger_tip.y
        pyautogui.moveTo(screen_x, screen_y)

        # Middle finger tip (landmark 12) for click detection
        middle_finger_tip = landmarks[12]
        cv2.circle(frame, (int(middle_finger_tip.x * frame_w), int(middle_finger_tip.y * frame_h)), 5, (255, 0, 0), -1)

        # Detecting click when index finger and middle finger tips are close together
        if abs(index_finger_tip.x - middle_finger_tip.x) < 0.03 and abs(
                index_finger_tip.y - middle_finger_tip.y) < 0.03:
            pyautogui.click()
            pyautogui.sleep(1)

    return frame


# Function to capture and process webcam feed in a separate thread
def webcam_feed():
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        processed_frame = process_frame(frame)
        cv2.imshow('Finger Controlled Mouse', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


# Start the webcam feed thread
webcam_thread = threading.Thread(target=webcam_feed)
webcam_thread.start()
