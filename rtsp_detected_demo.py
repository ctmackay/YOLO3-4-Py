import time

from pydarknet import Detector, Image
import cv2

print (cv2.getBuildInformation())
import sys

webcam = 0
sample_vid = '/mnt/wd1tb/videos/nvidia/vlc-record-2017-08-04-21h10m29s-walsh_santomas_20170602_029.mp4-.mp4'
nsd_46_stream = "rtsp://12.8.0.110:1935/birdseye/nsd_46.stream"
fps = 5.0
monitor_pipeline = 'appsrc ! queue ! autovideoconvert ! autovideosink'
# wowza_pipeline = 'appsrc ! autovideoconvert ! queue ! x264enc bitrate=500000 byte-stream=TRUE bframes=0 threads=7 ! mpegtsmux ! queue ! udpsink host=12.8.0.110 port=17046'
wowza_pipeline = 'appsrc ! queue ! autovideoconvert ! queue !  x264enc bitrate=50000 byte-stream=TRUE bframes=0 threads=7 ! queue !  rtph264pay !  queue !  udpsink host=12.8.0.110 port=17046'
# wowza_pipeline = 'appsrc ! queue ! autovideoconvert ! queue !  x264enc  bitrate=500000 byte-stream=TRUE bframes=0 threads=7 ! queue !  mp4mux !  queue !  filesink location=videotest.mp4'
# wowza_pipeline = 'appsrc ! queue !  x264enc  noise-reduction=10000 speed-preset=fast tune=zerolatency byte-stream=true threads=7 key-int-max=15 intra-refresh=true bitrate=50000 bframes=0 ! queue !  rtph264pay !  queue !  udpsink host=12.8.0.110 port=17046'
# wowza_pipeline = 'appsrc ! queue !  x264enc speed-preset=fast tune=zerolatency byte-stream=true threads=7 key-int-max=0 bitrate=50000 bframes=0 ! queue !  rtph264pay !  queue !  udpsink host=12.8.0.110 port=17046'

if __name__ == "__main__":

    data_file = sys.argv[1]
    cfg_file = sys.argv[2]
    weights_file = sys.argv[3]

    # net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
    #                bytes("cfg/coco.data", encoding="utf-8"))

    net = Detector(bytes(cfg_file, encoding="utf-8"), bytes(weights_file, encoding="utf-8"), 0,
                   bytes(data_file, encoding="utf-8"))

    cap = cv2.VideoCapture(webcam)
    out = cv2.VideoWriter(monitor_pipeline, 0, 30, (640, 480))
    time.sleep(5)

    if (not cap.isOpened()):
        print("failed to open video")
        exit(1)

    if not out.isOpened():
        print ("could not open video writer!")
        exit(1)


    while cap.isOpened():
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            elapsed_time = end_time - start_time
            # print("Elapsed Time:",elapsed_time)
            fps = 1 / elapsed_time
            print("FPS %.2f" % (fps))

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 5)
                cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 0))

            #cv2.imshow("preview", frame)
            out.write(frame)
            time.sleep(0.001)



        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
    cap.release()
    out.release()
