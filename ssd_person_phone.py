import cv2
import numpy as np
import tensorflow as tf
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
classes = ["background","person",   "bicycle",   "car",   "motorcycle",   "airplane",
    "bus",   "train",   "truck",   "boat",   "traffic light",   "fire hydrant","unknown", "stop_sign",
    "parking meter",   "bench",   "bird",   "cat",   "dog",   "horse",   "sheep",   "cow","elephant", "bear",
    "zebra", "giraffe","unknown", "backpack", "umbrella","unknown","unknown",   "handbag",   "tie",   "suitcase",   "frisbee","skis", "snowboard",
    "sports ball",   "kite",   "baseball bat",   "baseball glove",   "skateboard",   "surfboard", "tennis racket",
    "bottle","unknown","wine glass",   "cup",  "fork",   "knife",   "spoon",   "bowl",   "banana",   "apple",   "sandwich",   "orange",
    "broccoli",   "carrot",   "hot dog",   "pizza",   "donot",   "cake",   "chair",   "couch",   "potted plant",   "bed",
    "unknown","dining table","unknown","unknown",  "toilet","unknown", "tv",   "laptop",   "mouse",   "remote",   "keyboard",   "cell phone",   "microwave",
    "oven",   "toaster",   "sink",   "refrigerator","unknown",   "book",   "clock",   "vase",   "scissors",   "teddy bear",   "hair dryer",
    "toothbrush"]
colors = np.random.uniform(0,255,size=(len(classes),3))
with tf.io.gfile.GFile('frozen_inference_graph.pb','rb') as f:
	graph_def=tf.compat.v1.GraphDef()
	graph_def.ParseFromString(f.read())
with tf.compat.v1.Session() as sess:
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
	while (True):
		_, img = cap.read()
		rows=img.shape[0]
		cols=img.shape[1]
		inp=cv2.resize(img,(220,220))
		inp=inp[:,:,[2,1,0]]
		out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
				sess.graph.get_tensor_by_name('detection_scores:0'),
                      		sess.graph.get_tensor_by_name('detection_boxes:0'),
                      		sess.graph.get_tensor_by_name('detection_classes:0')],
                     		feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
		num_detections=int(out[0][0])
		for i in range(num_detections):
			classId = int(out[3][0][i])
			score=float(out[1][0][i])
			bbox=[float(v) for v in out[2][0][i]]
			label=classes[classId]
			if (score>0.3):
				x=bbox[1]*cols
				y=bbox[0]*rows
				right=bbox[3]*cols
				bottom=bbox[2]*rows
				color=colors[classId]
				cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=2)
				cv2.rectangle(img, (int(x), int(y)), (int(right),int(y+30)),color, -1)
				cv2.putText(img, str(label),(int(x), int(y+25)),1,2,(255,255,255),2)
		cv2.imshow('XXX',img)
		key=cv2.waitKey(1)
		if (key == 27):
			break
cap.release()
cv2.destroyAllWindows()