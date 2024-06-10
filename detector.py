import numpy as np
import cv2
import click
import time

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from random import randrange

from SFSORT import SFSORT


@click.command()
@click.argument('input_name', type=click.Path(exists=True), required=True)
@click.argument('output_name', type=click.Path(), default='./output.mp4')
@click.argument('mode', type=click.Choice(['cars', 'tanks']), default='cars')
@click.option('--imshow', type=click.BOOL, default=False)
def main(input_name:str, output_name:str, mode:str, imshow:bool):
    if mode=='cars':
        model = YOLO('weights/yolov8m.pt', 'detect')
        classnum = 2
    else:
        model = YOLO('weights/yolov8s_tanks.pt', 'detect')
        classnum = 0

    try:
        device = select_device('0')
    except:
        device = select_device('cpu')
    model.to(device)

    cap = cv2.VideoCapture(input_name)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, 30.0, (frame_width, frame_height))

    tracker_arguments = {"dynamic_tuning": True, "cth": 0.2,
                        "high_th": 0.82, "high_th_m": 0.1,
                        "match_th_first": 0.5, "match_th_first_m": 0.05,
                        "match_th_second": 0.1, "low_th": 0.1,
                        "new_track_th": 0.2, "new_track_th_m": 0.1,
                        "marginal_timeout": (7 * frame_rate // 10),
                        "central_timeout": frame_rate,
                        "horizontal_margin": frame_width // 10,
                        "vertical_margin": frame_height // 10,
                        "frame_width": frame_width,
                        "frame_height": frame_height}

    tracker = SFSORT(tracker_arguments)
    colors = {}
    frame_count = 0
    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        start_time = time.time()
        print(f"predict frame # {frame_count}/{frame_length}")
        prediction = model.predict(frame, imgsz=(640,640), conf=0.1, iou=0.45,
                                    half=False, device=device, max_det=99, classes=classnum,
                                    verbose=False)
        prediction_results = prediction[0].boxes.cpu().numpy()
        tracks = tracker.update(prediction_results.xyxy, prediction_results.conf)
        if len(tracks) == 0:
            out.write(frame)
            continue

        bbox_list = tracks[:, 0]
        track_id_list = tracks[:, 1]

        for idx, (track_id, bbox) in enumerate(zip(track_id_list, bbox_list)):
            if track_id not in colors:
                colors[track_id] = (randrange(255), randrange(255), randrange(255))

            color = colors[track_id]
            x0, y0, x1, y1 = map(int, bbox)
            annotated_frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(annotated_frame, str(track_id), (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if imshow:
            cv2.imshow("result", annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
        print(f'inf time: {round(time.time()-start_time, 2)} seconds\n')
        out.write(annotated_frame)
    cap.release()
    out.release()


if __name__=="__main__":
    main()
