
import cv2
import numpy as np
import requests

#assume connected with jetson via usb
URL = "http://192.168.55.1:5000/api/darknet/in_image_get_image/"

def frameProcessing(img, thresh=0.25):
    url = URL + str(thresh)
    result = np.copy(img)
    _, imgencoded = cv2.imencode('.jpg', result)
    data = imgencoded.tostring()
    res = requests.post(url, files=None, data=data, timeout=5)
    nparr = np.frombuffer(res.content, np.uint8)
    result = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return result

def Video(openpath, savepath = None, thresh = 0.25):
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if savepath is not None:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            output = frameProcessing(frame, thresh)
            if out is not None:
                # Write frame-by-frame
                out.write(output)
            # Display the resulting frame
            cv2.imshow("Input", frame)
            cv2.imshow("Output", output)
        else:
            break
        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    Video(0)