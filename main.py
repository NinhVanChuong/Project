import cv2
cap=cv2.VideoCapture(0)
frame_width=640
frame_height=480
out = cv2.VideoWriter('dat.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

while True:
    ret,frame = cap.read()
    frame=cv2.resize(frame,(640,480))
    out.write(frame)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break
