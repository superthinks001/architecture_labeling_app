import streamlit as st
import pandas as pd
import os
from PIL import Image
import io
import google.generativeai as genai

# Set up page
st.set_page_config(page_title="🏡 AI House Style Classifier", layout="wide")
st.title("🏡 AI Architectural Style Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a house image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show preview
    st.image(uploaded_file, use_container_width=True)
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    image = Image.open(io.BytesIO(file_bytes))

    # Configure Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("models/gemini-2.5-pro-preview-03-25")

    # Prompt
    prompt = """
You are an expert in architectural classification. Identify the primary and secondary style, roof, porch, windows, and door using these label choices:

{
  "primary_style": "",
  "secondary_style": "",
  "roof": "",
  "porch": "",
  "windows": "",
  "door": "",
  "additional_notes": ""
}
"""

    # Send to Gemini and get response
    with st.spinner("🔍 Analyzing architectural style..."):
        try:
            response = model.generate_content([prompt, image])
            result = response.text
        except Exception as e:
            st.error("❌ Gemini API call failed.")
            st.exception(e)
            st.stop()

    # Parse result
    st.markdown("### 🧠 Predicted Style (Editable)")
    try:
        default = eval(result) if isinstance(result, str) else result
    except:
        default = {
            "primary_style": "",
            "secondary_style": "",
            "roof": "",
            "porch": "",
            "windows": "",
            "door": "",
            "additional_notes": ""
        }

    primary = st.selectbox("Primary Style", options=["Modern", "Craftsman", "Victorian", "Colonial", "Ranch", "Spanish", "Contemporary", "Traditional", "Other"], index=0)
    secondary = st.selectbox("Secondary Style", options=["Modern", "Craftsman", "Victorian", "Colonial", "Ranch", "Spanish", "Contemporary", "Traditional", "Other"], index=0)
    roof = st.text_input("Roof", default.get("roof", ""))
    porch = st.text_input("Porch", default.get("porch", ""))
    windows = st.text_input("Windows", default.get("windows", ""))
    door = st.text_input("Door", default.get("door", ""))
    notes = st.text_area("Additional Notes", default.get("additional_notes", ""))

    # Save to CSV
    if st.button("✅ Save to Dataset"):
        df = pd.DataFrame([{
            "image_file": file_name,
            "primary_style": primary,
            "secondary_style": secondary,
            "roof": roof,
            "porch": porch,
            "windows": windows,
            "door": door,
            "additional_notes": notes
        }])
        if not os.path.exists("style_feedback_dataset.csv"):
            df.to_csv("style_feedback_dataset.csv", index=False)
        else:
            df.to_csv("style_feedback_dataset.csv", mode='a', header=False, index=False)
        st.success("✅ Saved to dataset!")

    # Show full dataset
    if os.path.exists("style_feedback_dataset.csv"):
        with st.expander("📊 View Full Dataset"):
            df_view = pd.read_csv("style_feedback_dataset.csv")
            st.dataframe(df_view)
