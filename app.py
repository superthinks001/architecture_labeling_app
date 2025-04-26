import streamlit as st
import requests
import base64
import pandas as pd
import os

st.set_page_config(page_title="AI House Style Classifier", layout="wide")
st.title("🏡 AI Architectural Style Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a house image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image preview
    st.image(uploaded_file, use_column_width=True)
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name

    # Gemini Vision API call (live)
    st.markdown("### 🔍 Analyzing architectural style...")
    headers = {
        "Authorization": f"Bearer {st.secrets['GEMINI_API_KEY']}"
    }
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

    # Encode image to base64
    b64_image = base64.b64encode(file_bytes).decode("utf-8")
    prompt = {
        "contents": [{
            "parts": [
                {"text": """You are an expert in architectural classification. Identify the primary and secondary style, roof, porch, windows, and door using these label choices:
{
  "primary_style": "",
  "secondary_style": "",
  "roof": "",
  "porch": "",
  "windows": "",
  "door": "",
  "additional_notes": ""
}"""},
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": b64_image
                    }
                }
            ]
        }]
    }

    # Send to Gemini
    response = requests.post(api_url, headers=headers, json=prompt)
    result = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    # Extract as editable form
    st.markdown("### 🧠 Predicted Style (Editable)")
    default = eval(result) if isinstance(result, str) else result
    primary = st.selectbox("Primary Style", options=["Modern", "Craftsman", "Victorian", "Colonial", "Ranch", "Spanish", "Contemporary", "Traditional", "Other"], index=0)
    secondary = st.selectbox("Secondary Style", options=["Modern", "Craftsman", "Victorian", "Colonial", "Ranch", "Spanish", "Contemporary", "Traditional", "Other"], index=0)
    roof = st.text_input("Roof", default.get("roof", ""))
    porch = st.text_input("Porch", default.get("porch", ""))
    windows = st.text_input("Windows", default.get("windows", ""))
    door = st.text_input("Door", default.get("door", ""))
    notes = st.text_area("Additional Notes", default.get("additional_notes", ""))

    # Save to dataset
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

    # Preview dataset
    if os.path.exists("style_feedback_dataset.csv"):
        with st.expander("📊 View Full Dataset"):
            df_view = pd.read_csv("style_feedback_dataset.csv")
            st.dataframe(df_view)
