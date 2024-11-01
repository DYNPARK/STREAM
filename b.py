import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

# YOLOv8 모델 로드 (사전 학습된 'yolov8n' 모델 사용)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' 대신 적절한 모델 경로를 입력하세요

st.title("Real-Time Object Detection with YOLOv8 and WebRTC")

# 프레임 처리 콜백 함수
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # YOLO 모델을 사용하여 객체 감지 수행
    results = model(img)
    
    # 결과에서 각 객체의 바운딩 박스를 그립니다
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # 바운딩 박스 좌표
        label = result.cls[0]  # 클래스 라벨 (사람 등)
        confidence = result.conf[0]  # 신뢰도 점수
        
        # 클래스 이름 얻기
        class_name = model.names[int(label)]
        
        # 사람일 경우 또는 특정 조건 추가 가능
        if class_name == "person" or confidence > 0.5:
            # 바운딩 박스와 라벨 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # BGR에서 RGB로 변환 (Streamlit에서 이미지를 표시하기 위해)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return av.VideoFrame.from_ndarray(img, format="rgb24")

# WebRTC 스트리밍 시작
webrtc_streamer(key="example", video_frame_callback=video_frame_callback)