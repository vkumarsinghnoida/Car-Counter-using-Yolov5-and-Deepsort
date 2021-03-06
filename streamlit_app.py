import streamlit as st
import cv2
from yolo import findObjects
import tempfile
import time

st.title('Car Detector')
st.sidebar.title('Parameters')
conf = st.sidebar.slider('Confidence', 0.0, 1.0, 0.4, step=0.1)
nms = st.sidebar.slider('NMS Threshold', 0.0, 1.0, 0.2, step=0.1)
stframe = st.empty()
DEMO_VIDEO = 'video.mp4'
st.set_option('deprecation.showfileUploaderEncoding', False)
video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

tfflie = tempfile.NamedTemporaryFile(delete=False)

if not video_file_buffer:

    cap = cv2.VideoCapture(DEMO_VIDEO)
    tfflie.name = DEMO_VIDEO
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    
else:
    tfflie.write(video_file_buffer.read())
    cap = cv2.VideoCapture(tfflie.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    
kpi1, kpi2 = st.beta_columns(2)

with kpi1:
    st.markdown("**FrameRate**")
    kpi1_text = st.markdown("0")

with kpi2:
    st.markdown("**Detected Cars**")
    kpi2_text = st.markdown("0")
    

st.markdown("<hr/>", unsafe_allow_html=True)

st.sidebar.text('Input Video')
st.sidebar.video(tfflie.name)
    
while cap.isOpened:
    
    ret, img = cap.read()    
    det, fps = findObjects(img, conf, nms)
    kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
    kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{int(det)}</h1>", unsafe_allow_html=True)
    stframe.image(img,channels = 'BGR',use_column_width=True)
    time.sleep(0.001)
    
cap.release()
cv2.destroyAllWindows()
video_file = open('output.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)


st.stop()
