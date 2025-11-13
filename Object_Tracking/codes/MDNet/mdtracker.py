import cv2
import torch

# 1) Open the video and grab the first frame
cap = cv2.VideoCapture("video.mp4")
ok, frame_bgr = cap.read()
if not ok:
    raise RuntimeError("Couldn't read first frame")

# 2) Let the user draw the initial ROI (x, y, w, h) in *BGR* space
init_bbox = cv2.selectROI("Select target", frame_bgr, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select target")

# 3) Create your tracker and initialize with that bbox
tracker = MDNetTracker(model_path="mdnet_backbone.pth")  # your pretrained convs
tracker.init(frame_bgr[:, :, ::-1], init_bbox)  # convert BGR->RGB for your PyTorch pipeline
# (if your MDNetTracker already handles BGR->RGB internally, skip the channel flip)

# 4) Track frame by frame
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    # MDNetTracker.update should: sample candidates around last box, score, pick best, (optionally) update head
    bbox = tracker.update(frame_bgr[:, :, ::-1])  # RGB again if needed

    # 5) Visualize
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("MDNet Tracking", frame_bgr)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
