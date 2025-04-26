import streamlit as st
from ultralytics import YOLO
import json
from PIL import Image

# Load disease info JSON file
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# Load YOLO model
model = YOLO("model.pt")  # Load your trained model

# Manually define class names (because model.names sometimes missing on cloud)
class_names = ['Leaf Spot', 'Powdery Mildew', 'Rust', 'Blight', 'Healthy']

# Streamlit UI
st.set_page_config(page_title="AI Plant Doctor", page_icon="🌿", layout="centered")
st.title("🌱 AI Plant Doctor")
st.caption("Upload a leaf image to detect disease, symptoms, and treatments!")

# Upload image
uploaded_image = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼️ Uploaded Leaf", use_container_width=True)

    # Run inference
    results = model.predict(image)
    boxes = results[0].boxes

    if boxes:
        class_id = int(boxes.cls[0].item())
        class_name = class_names[class_id]

        st.success(f"🔍 **Prediction:** {class_name}")

        # Show info
        if class_name in disease_info:
            info = disease_info[class_name]
            st.markdown("---")
            st.markdown("### 🩺 Symptoms")
            st.info(f"**{info['symptoms']}**")

            st.markdown("### 💊 Treatment")
            st.success(f"**{info['treatment']}**")
        else:
            st.warning("❗ No detailed information found for this disease.")
    else:
        st.error("❌ No disease detected in the image.")
