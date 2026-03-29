import cv2

# ── 1. Load the image ──────────────────────────────────────────
image = cv2.imread(r"C:\Users\leish\OneDrive\Desktop\Gender Prediction\test_images\test.jpg")

if image is None:
    print("ERROR: Could not load image. Check the file path!")
else:
    print("Image loaded successfully!")
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

# ── 2. Convert to grayscale (face detector needs this) ─────────
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ── 3. Load OpenCV's built-in face detector ────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── 4. Detect faces ────────────────────────────────────────────
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Faces found: {len(faces)}")

# ── 5. Draw green box around each face ─────────────────────────
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )
    cv2.putText(
        image,
        "Face Detected",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# ── 6. Save result ─────────────────────────────────────────────
cv2.imwrite("test_images/output_day1.jpg", image)
print("Result saved as: test_images/output_day1.jpg")

# ── 7. Show result window ──────────────────────────────────────
cv2.imshow("Day 1 - Face Detection", image)
print("Press any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()