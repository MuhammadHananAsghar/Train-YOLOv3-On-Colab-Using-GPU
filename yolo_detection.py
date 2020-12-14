import cv2
import numpy as np
import urllib.request

#Load YOLO Model
net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')
classes = ['palm', 'fist']

address = "http://10.29.206.6:8080/shot.jpg"





while True:
	imgResp = urllib.request.urlopen(address)  
	imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	img = cv2.imdecode(imgNp,-1)
	# Load Image
	height, width, _ = img.shape

	# Preprocessing for Detection using YOLO
	blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

	# After Normalization
	# Input to model
	net.setInput(blob)

	# To Make Detections and get bounded boxes and get classes names
	output_layer_names = net.getUnconnectedOutLayersNames()
	layerOutputs = net.forward(output_layer_names)

	# Extract the bounding boxes
	boxes = []
	confidences = []
	class_ids = []
	for ouput in layerOutputs:
		for detection in ouput:
			# To get highest score of class
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				center_x = int(detection[0]*width)
				center_y = int(detection[1]*height)
				w = int(detection[2]*width)
				h = int(detection[3]*height)

				# To get Upper corners of bounding boxes
				x = int(center_x - w/2)
				y = int(center_y - h/2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)


	# in the case if we have more than bounding boxes for a single object then we want to select bounding box having highest confidence using this function
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

	font = cv2.FONT_HERSHEY_PLAIN
	# To get random color values to make more beautiful
	colors = np.random.uniform(0, 255, size=(len(boxes), 3))

	if len(indexes) > 0:
		# iterate over all boxes found
		for i in indexes.flatten():
			# getting postion of bounding boxes
			x, y, w, h = boxes[i]
			# get label name from classes
			label = str(classes[class_ids[i]])
			# getting confidence value
			confidence = str(round(confidences[i], 2))
			# assigning color
			color = colors[i]
			#creating bounding box
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			#Displaying label class name
			cv2.putText(img, label.capitalize() + " " + confidence, (x+5, y+25), font, 2, (255, 255, 255), 2)

	cv2.imshow("Image", img)

	interrupt = cv2.waitKey(1)
	if interrupt & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
