import cv2
import time
import numpy as np

# Load the pre-trained model for human detection here (MobileNet SSD)
model_path = 'mobilenet_iter_73000.caffemodel'
config_path = 'deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Define the classes that model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

fan_areas = {}
fan_states = {}
presence_timers = {}
no_presence_timers = {}
FAN_ON_THRESHOLD = 5  # Time threshold in seconds to turn the fan ON
FAN_OFF_THRESHOLD = 3  # Time threshold in seconds to turn the fan OFF
current_fan_id = None

# Function to draw rectangles and assign fan IDs
def mark_areas(event, x, y, flags, param):
    global current_fan_id

    if event == cv2.EVENT_LBUTTONDOWN:
        current_fan_id = len(fan_areas) + 1
        print(f"Marking area for Fan ID: {current_fan_id}")
        fan_areas[current_fan_id] = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP and current_fan_id is not None:

        if current_fan_id in fan_areas and len(fan_areas[current_fan_id]) == 1:
            fan_areas[current_fan_id].append((x, y))
            fan_states[current_fan_id] = False
            presence_timers[current_fan_id] = None
            no_presence_timers[current_fan_id] = None
        current_fan_id = None

# Function to detect humans
def detect_human(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    humans = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        
        if confidence > 0.5 and CLASSES[class_id] == "person":
            box = detections[0, 0, i, 3:7] * \
                  np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            humans.append((startX, startY, endX, endY))

    return humans

cv2.namedWindow("FanFinity")
cv2.setMouseCallback("FanFinity", mark_areas)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    humans = detect_human(frame)

    # Highlight detected humans with rectangles
    for (hx1, hy1, hx2, hy2) in humans:
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
        cv2.putText(frame, "Human", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw the marked areas and check for human presence in them
    for fan_id, area in fan_areas.items():
        # Ensure both points (top-left and bottom-right) are defined
        if len(area) == 2:
            x1, y1 = area[0]
            x2, y2 = area[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Fan {fan_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Check if any human is within the marked area
            human_in_area = False
            for (hx1, hy1, hx2, hy2) in humans:
                if hx1 >= x1 and hy1 >= y1 and hx2 <= x2 and hy2 <= y2:
                    human_in_area = True
                    break

            if human_in_area:
                # Reset the timer for turning the fan OFF
                no_presence_timers[fan_id] = None

                if presence_timers[fan_id] is None:
                    presence_timers[fan_id] = time.time()
                elif time.time() - presence_timers[fan_id] >= FAN_ON_THRESHOLD:
                    fan_states[fan_id] = True  # Turn the fan ON

                cv2.putText(frame, "Human Detected", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # If human leaves the area, start the no-presence timer
                if no_presence_timers[fan_id] is None:
                    no_presence_timers[fan_id] = time.time()

                if no_presence_timers[fan_id] and time.time() - no_presence_timers[fan_id] >= FAN_OFF_THRESHOLD:
                    presence_timers[fan_id] = None
                    fan_states[fan_id] = False  # Turn the fan OFF

            status = "ON" if fan_states[fan_id] else "OFF"
            cv2.putText(frame, f"Fan {fan_id}: {status}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if fan_states[fan_id] else (255, 255, 255), 2)

    cv2.imshow("FanFinity", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
