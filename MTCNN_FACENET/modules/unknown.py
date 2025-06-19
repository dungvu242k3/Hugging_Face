import cv2

cap = cv2.VideoCapture(0)  # Mở webcam
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Độ phân giải hiện tại: {int(width)} x {int(height)}")
cap.release()
