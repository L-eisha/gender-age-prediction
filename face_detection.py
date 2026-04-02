

import cv2

# ── 1. Load image 
image = cv2.imread(r"C:\Users\leish\OneDrive\Desktop\Gender Prediction\test_images\test.jpg")

if image is None:
    print("ERROR: Could not load image. Check the file path!")
    exit()

print("Image loaded successfully!")
print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

# ── 2. Load DNN face detector 
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

# ── 3. Prepare image blob 
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0)
)

# ── 4. Run face detection ──────────────────────────────────────
net.setInput(blob)
detections = net.forward()

face_count = 0

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Only process faces with confidence above 60%
    if confidence > 0.6:
        face_count += 1

        # ── 5. Get face coordinates ────────────────────────────
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype("int")

        # ── 6. Draw box and confidence score ───────────────────
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Face: {confidence * 100:.1f}%"
        cv2.putText(
            image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        print(f"Face {face_count}: confidence {confidence * 100:.1f}%")

print(f"\nTotal faces detected: {face_count}")

# ── 7. Save and show result ────────────────────────────────────
cv2.imwrite(r"C:\Users\leish\OneDrive\Desktop\Gender Prediction\test_images\output_day1.jpg", image)
print("Result saved as output_day1.jpg")

cv2.imshow("Day 1 - Face Detection (DNN)", image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()