import os
import cv2

# Create data directory if it doesn't exist
DATA_DIR = r'D:\TIH\hand'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 50

# Function to find a valid camera index
def find_valid_camera_index(max_index=5):
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return None

# Find a valid camera index
camera_index = find_valid_camera_index()
if camera_index is None:
    print("Error: No valid camera found.")
else:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Failed to open camera.")
    else:
        for j in range(number_of_classes):
            class_dir = os.path.join(DATA_DIR, str(j))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            print('Collecting data for class {}'.format(j))

            # Wait for user to be ready
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break

                cv2.putText(frame, 'Ready? Press "Q" to capture!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) == ord('q'):
                    break

            # Collect images for the dataset
            counter = 0
            while counter < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break

                cv2.imshow('frame', frame)
                # Confirm that frame has valid content
                if frame is not None and frame.size != 0:
                    img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
                    cv2.imwrite(img_path, frame)
                    print(f"Saved: {img_path}")
                    counter += 1
                else:
                    print("Error: Invalid frame.")

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
