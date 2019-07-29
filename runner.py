import sys
sys.path.append('efrcnn')

from efrcnn import infer
from deepsort import only_deepsort

def Detector(batch_q, result_q):
    while True:
        new_batch = batch_q.get()
        if new_batch == "Done":
            result_q.put("Done")
            break
        else:
            new_result = _infer(batch)
            result_q.put(new_result)

init = []

import cv2
vid = cv2.VideoCapture()
vid.open('/home/piyush/tracking/Videos/pass_by/test/vid_1/1.mp4')

count = 0

while vid.isOpened():
    _, frame = vid.read()
    init.append(frame)
    count += 1
    if count == 20:
        break

det = only_deepsort.Detector()
bb = infer._infer(init)

deepsort_results = []
for item in bb:
	new_ids = det.update_deepsort(item)
    deepsort_results.append(new_ids)

