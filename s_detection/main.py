import cv2
import os

# Define the directory for saving images
base_dir = 'dataset'
categories = ['spoons', 'forks', 'knives']

# Ensure the directories exist
for category in categories:
    os.makedirs(os.path.join(base_dir, category), exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

current_category = None
image_count = {category: len(os.listdir(os.path.join(base_dir, category))) for category in categories}

print("Press '1' for spoons, '2' for forks, '3' for knives.")
print("Press 'c' to capture an image and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam', frame)
    
    key = cv2.waitKey(1) & 0xFF

    # Select category
    if key == ord('1'):
        current_category = 'spoons'
    elif key == ord('2'):
        current_category = 'forks'
    elif key == ord('3'):
        current_category = 'knives'

    # Capture image
    if key == ord('c') and current_category:
        image_count[current_category] += 1
        image_path = os.path.join(base_dir, current_category, f'{current_category}_{image_count[current_category]}.jpg')
        cv2.imwrite(image_path, frame)
        print(f'Captured {image_path}')
    
    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()