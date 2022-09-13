import cv2
import numpy
import time

def gather_cv2_cameras(max_ind=20):
	all_cv2_cameras = []
	for i in range(max_ind):
		cap = cv2.VideoCapture(i)
		if cap.read()[0]:
			camera = CV2Camera(cap)
			all_cv2_cameras.append(camera)
	return all_cv2_cameras

class CV2Camera:
	def __init__(self, cap):
		self._cap = cap
		self._serial_number = 'cv2_camera' #temporary

	def read_camera(self, enforce_same_dim=False):
		# Get a new frame from camera
		retval, frame = self._cap.read()
		if not retval: return None

		# Extract left and right images from side-by-side
		read_time = time.time()
		left_img, right_img = numpy.split(frame, 2, axis=1)
		left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
		right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

		left_img = cv2.resize(left_img, dsize=(128, 96), interpolation=cv2.INTER_AREA)
		right_img = cv2.resize(right_img, dsize=(128, 96), interpolation=cv2.INTER_AREA)

		dict_1 = {'array': left_img, 'shape': left_img.shape, 'type': 'rgb',
			'read_time': read_time, 'serial_number': self._serial_number + '/left'}
		dict_2 = {'array': right_img,  'shape': right_img.shape, 'type': 'rgb',
			'read_time': read_time, 'serial_number': self._serial_number + '/right'}
		
		return [dict_1, dict_2]

	def disable_camera(self):
		self._cap.release()
