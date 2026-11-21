import hashlib
import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from pipeline import DocumentPipeline


BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading the quality model...")
def load_pipeline():
    return DocumentPipeline(BASE_DIR / "config.yaml")


def show_result(result):
    status = result["status"]
    if status == "accepted":
        st.success("Accepted: good image quality")
        st.write("The photo passed the quality check.")
    elif status == "rejected":
        st.warning("Rejected: please retake the photo")
        st.write("Keep the whole document in the frame. Avoid blur, glare and fingers covering the text.")
    else:
        messages = {
            "unsupported_file": "This file is not a supported image. Upload a JPG or PNG photo.",
            "corrupted_image": "The image could not be read. Try another JPG or PNG file.",
            "image_too_large": "The image exceeds the size limit. Use a file up to 20 MB and 25 million pixels.",
            "model_unavailable": "The quality model is unavailable. The app owner needs to check the model file and dependencies.",
            "config_error": "The app configuration is unavailable or invalid.",
            "inference_error": "The photo could not be processed. Please try another file.",
        }
        st.error(messages.get(result["reason"], "The file could not be processed."))
    if status != "error":
        confidence, latency = st.columns(2)
        confidence.metric("Model confidence", f"{result['quality_confidence']:.1%}")
        latency.metric("Processing time", f"{result['latency_ms']:.1f} ms")
        st.caption("Confidence is the model probability for the returned quality label. It is not a guarantee of correctness.")
    with st.expander("JSON response"):
        st.json(result)
    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        file_name="quality_result.json",
        mime="application/json",
    )


def main():
    st.set_page_config(page_title="Document Quality Gate", page_icon="📄", layout="wide")
    st.title("Document Quality Gate")
    st.write("Upload a document photo to check its image quality.")
    st.caption("Machine Learning Lab 7 · Ekaterina Shigarova · IIKb-24")
    pipeline = load_pipeline()
    if pipeline.startup_error:
        st.error("The model could not be initialized. File requests will return an error response.")
        with st.expander("Startup response", expanded=True):
            st.json(pipeline.check_model())
    upload_column, result_column = st.columns([1, 1], gap="large")
    with upload_column:
        st.subheader("Your photo")
        upload = st.file_uploader(
            "Choose a file",
            accept_multiple_files=False,
            help="Supported images: JPG and PNG. Maximum 20 MB and 25 million pixels. Other files return an error response.",
        )
        if upload is None:
            st.session_state.pop("quality_result", None)
            st.session_state.pop("quality_upload_key", None)
            data = None
        else:
            data = upload.getvalue()
            key = (upload.name, hashlib.sha256(data).hexdigest())
            if st.session_state.get("quality_upload_key") != key:
                st.session_state.pop("quality_result", None)
                st.session_state["quality_upload_key"] = key
        clicked = st.button("Check photo", type="primary", disabled=upload is None)
        if clicked and upload is not None:
            with st.spinner("Checking image quality..."):
                st.session_state["quality_result"] = pipeline.run_bytes(data, upload.name)
        result = st.session_state.get("quality_result")
        if upload is not None and result is not None and result["status"] in {"accepted", "rejected"}:
            try:
                with Image.open(io.BytesIO(data)) as original:
                    preview = original.convert("RGB")
                    preview.thumbnail((900, 900))
                st.image(preview, caption=upload.name, width="stretch")
            except Exception:
                st.info("The prediction is available; the preview could not be displayed.")
        st.caption("Uploaded photos are processed in memory and are not saved to disk by this app.")
    with result_column:
        st.subheader("Result")
        if result is None:
            st.info("Choose a photo and click Check photo.")
        else:
            st.caption(f"File: {result['file']}")
            show_result(result)
    st.divider()
    st.caption("This semester's app checks good/bad image quality. Document type classification is deferred to the next semester, as clarified by the instructor.")
    with st.expander("Model information"):
        st.write("EfficientNet-B0 · ONNX FP32 · CPU inference")
        st.write(f"Version: {pipeline.model_version}")
        if pipeline.threshold is not None:
            st.write(f"Decision: reject when P(bad) > {pipeline.threshold:.2f}.")
        st.write("The model was trained on synthetic documents. Real-photo performance has not been established.")


if __name__ == "__main__":
    main()
