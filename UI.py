import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="AI Style Transfer", layout="centered")

st.title("🎨 AI-Enhanced Style Transfer")
st.write("Upload a **content image** and a **style image**, and let the AI create a new artistic version!")

# --- Helper Functions ---
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    return tf.convert_to_tensor(img, dtype=tf.float32)

@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# --- File Upload ---
content_file = st.file_uploader("Choose Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Choose Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_img = Image.open(content_file)
    style_img = Image.open(style_file)

    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(content_img, caption="Content Image", use_column_width=True)
    with col2:
        st.image(style_img, caption="Style Image", use_column_width=True)

    if st.button("✨ Stylize"):
        with st.spinner("Applying style..."):
            content_tensor = load_image(content_file)
            style_tensor = load_image(style_file)

            model = load_model()
            stylized_tensor = model(content_tensor, style_tensor)[0]
            stylized_image = tf.image.convert_image_dtype(stylized_tensor, dtype=tf.uint8)[0].numpy()

            st.success("Done!")
            st.image(stylized_image, caption="Stylized Output", use_column_width=True)

            # Optionally provide a download button
            buf = io.BytesIO()
            Image.fromarray(stylized_image).save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button("💾 Download Stylized Image", data=byte_im, file_name="stylized.png", mime="image/png")
