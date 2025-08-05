import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

# Configure page
st.set_page_config(
    page_title="InspectPCB",
    page_icon="🔍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    """Load the trained YOLOv5 model"""
    try:
        # Try to load custom trained model first
        if os.path.exists('pcb_defect.pt'):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='pcb_defect.pt')
        else:
            # Fallback to pretrained model (you'll need to train your own)
            st.warning("Custom model not found. Please train your model first using train.py")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        model.conf = 0.25  # confidence threshold
        model.iou = 0.45   # IoU threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def draw_detections(image, results):
    """Draw bounding boxes and labels on the image"""
    
    # Convert PIL to cv2
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Define class names and colors
    class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Extract detections
    detections = results.pandas().xyxy[0]
    
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        cls = int(detection['class'])
        
        # Draw bounding box
        color = colors[cls % len(colors)]
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[cls]}: {conf:.2f}"
        cv2.putText(img_cv2, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def main():
    st.title("🔍 InspectPCB")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Update model thresholds
    model.conf = confidence_threshold
    model.iou = iou_threshold
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload PCB Image")
        uploaded_file = st.file_uploader(
            "Choose a PCB image", 
            type=["jpg", "png", "jpeg"],
            help="Upload a clear image of a PCB for defect detection"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded PCB Image", use_column_width=True)
    
    with col2:
        st.header("Detection Results")
        
        if uploaded_file and st.button("🔍 Detect Defects", type="primary"):
            with st.spinner("Analyzing PCB for defects..."):
                try:
                    # Run inference
                    results = model(image)
                    
                    # Get detection count
                    detections = results.pandas().xyxy[0]
                    num_detections = len(detections)
                    
                    if num_detections > 0:
                        # Draw detections
                        result_image = draw_detections(image, results)
                        st.image(result_image, caption=f"Detected {num_detections} defect(s)", use_column_width=True)
                        
                        # Show detection details
                        st.subheader("Defect Details")
                        for i, (_, detection) in enumerate(detections.iterrows()):
                            class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
                            defect_type = class_names[int(detection['class'])]
                            confidence = detection['confidence']
                            
                            st.write(f"**Defect {i+1}:** {defect_type} (Confidence: {confidence:.2%})")
                    else:
                        st.success("✅ No defects detected in this PCB!")
                        st.image(image, caption="No defects found", use_column_width=True)
                        
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.header("About InspectPCB")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Detectable Defects")
        st.write("""
        - Missing Hole
        - Mouse Bite
        - Open Circuit
        - Short Circuit
        - Spur
        - Spurious Copper
        """)
    
    with col2:
        st.subheader("🔧 How it Works")
        st.write("""
        1. Upload PCB image
        2. YOLOv5 model analyzes the image
        3. Defects are detected and highlighted
        4. Results show defect type and confidence
        """)
    
    with col3:
        st.subheader("💡 Tips")
        st.write("""
        - Use high-quality, well-lit images
        - Ensure PCB is clearly visible
        - Adjust confidence threshold if needed
        - Lower values detect more potential defects
        """)

if __name__ == "__main__":
    main()