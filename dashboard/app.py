import streamlit as st
import joblib
import cv2
import numpy as np
import tempfile

# Load the KMeans model
model = joblib.load('brain_tumor_kmeans.sav')
tumorLabel = 4

# Initialize contours outside the function
contours = []

# Function to detect and highlight tumors in an image
def detect_tumors(image_path):
    imgOriginal = cv2.imread(image_path)
    img = cv2.imread(image_path, 0)
    height, width = img.shape
    imgFlatten = img.reshape(height * width, 1)
    labels = model.predict(imgFlatten)
    labels2D = labels.reshape(height, width)
    mask = (labels2D == tumorLabel)
    tumorExtracted = np.bitwise_and(mask, img)

    global contours  # Use the global variable
    contours, hierarchy = cv2.findContours(tumorExtracted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOriginal, (x, y), (x + 120, y - 40), (0, 255, 0), -1)
            cv2.putText(imgOriginal, "TUMOR", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return imgOriginal

# Streamlit UI
st.title('Brain Tumor Detection App')

uploaded_image = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_image.read())
        st.image(temp_file.name, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting tumors in the uploaded image...")

        # Perform tumor detection
        result_image = detect_tumors(temp_file.name)

        st.subheader("Result:")
        st.image(result_image, caption="Tumor Detection Result", use_column_width=True)
        st.write("Number of detected tumors:", len(contours))
else:
    # Provide inbuilt example images for selection
    example_images = {
        "Example 1": "Example1.jpg",
        "Example 2": "Example2.jpg",
        "Example 3": "Example3.jpg",
        "Example 4": "Example4.jpg",
    }

    selected_example = st.selectbox("Select an example image:", list(example_images.keys()))
    example_image_path = example_images[selected_example]
    example_image = cv2.imread(example_image_path)

    st.image(example_image, caption=f"Selected Example: {selected_example}", use_column_width=True)
    st.write("Detecting tumors in the selected example image...")

    # Perform tumor detection on the example image
    example_result_image = detect_tumors(example_image_path)

    st.subheader("Result:")
    st.image(example_result_image, caption="Tumor Detection Result", use_column_width=True)
    st.write("Number of detected tumors:", len(contours))

st.sidebar.title("About")
st.sidebar.info(
    "This is a Streamlit web app for detecting brain tumors in MRI images using K-Means clustering."
)
