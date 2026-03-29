import cv2
from deepface import DeepFace

# ── 1. Load the image ──────────────────────────────────────────
image = cv2.imread(r"C:\Users\leish\OneDrive\Desktop\Gender Prediction\test_images\test.jpg")

if image is None:
    print("ERROR: Could not load image!")
else:
    print("Image loaded successfully!")

# ── 2. Detect faces ────────────────────────────────────────────
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f"Faces found: {len(faces)}")

# ── 3. Predict gender and age for each face ────────────────────
for (x, y, w, h) in faces:
    face_crop = image[y:y+h, x:x+w]

    try:
        result = DeepFace.analyze(
            face_crop,
            actions=["age", "gender"],
            enforce_detection=False
        )

        age = result[0]["age"]
        gender = result[0]["dominant_gender"]

        print(f"Predicted → Gender: {gender}, Age: {age}")

        # ── 4. Draw box and label on image ─────────────────────
        label = f"{gender}, Age: {age}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 100, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    except Exception as e:
        print(f"Error analyzing face: {e}")

# ── 5. Save result ─────────────────────────────────────────────
cv2.imwrite(r"C:\Users\leish\OneDrive\Desktop\Gender Prediction\test_images\output_day2.jpg", image)
print("Result saved as output_day2.jpg")

# ── 6. Show result window ──────────────────────────────────────
cv2.imshow("Day 2 - Gender & Age Prediction", image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()