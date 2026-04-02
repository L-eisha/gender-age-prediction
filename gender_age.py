

import cv2
from deepface import DeepFace

# ── 1. Load image ─────────────────────────────────────────────
image = cv2.imread(r"C:\Users\leish\OneDrive\Desktop\Gender Prediction\test_images\test.jpg")

if image is None:
    print("ERROR: Could not load image!")
    exit()

print("Image loaded successfully!")

h, w = image.shape[:2]

# ── 2. Load DNN face detector ──────────────────────────────────
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0, (300, 300),
    (104.0, 177.0, 123.0)
)

net.setInput(blob)
detections = net.forward()

print(f"Processing detections...")
face_count = 0

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.6:
        face_count += 1

        # ── 3. Get face coordinates ────────────────────────────
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype("int")

        # ── 4. Add padding around face for better DeepFace accuracy
        pad = 20
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad)
        y2p = min(h, y2 + pad)

        face_crop = image[y1p:y2p, x1p:x2p]

        try:
            # ── 5. Predict gender and age ──────────────────────────
            result = DeepFace.analyze(
                face_crop,
                actions=["age", "gender"],
                enforce_detection=False
            )

            age    = result[0]["age"]
            gender = result[0]["dominant_gender"]

            # ── 6. Get gender confidence score ─────────────────────
            gender_conf = result[0]["gender"][gender]

            print(f"Face {face_count} → Gender: {gender} ({gender_conf:.1f}%), Age: {age}")

            # ── 7. Draw box and label ──────────────────────────────
            label = f"{gender} ({gender_conf:.0f}%)  Age: {age}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2
            )

            # ── 8. Also show face detection confidence ─────────────
            conf_label = f"Det: {confidence * 100:.1f}%"
            cv2.putText(
                image, conf_label, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
            )

        except Exception as e:
            print(f"Error on face {face_count}: {e}")

print(f"\nTotal faces processed: {face_count}")

# ── 9. Save and show result ────────────────────────────────────
cv2.imwrite(
    r"C:\Users\leish\OneDrive\Desktop\Gender Prediction\test_images\output_day2.jpg",
    image
)
print("Result saved as output_day2.jpg")

cv2.imshow("Day 2 - Gender & Age Prediction", image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()