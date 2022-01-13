import cv2
import numpy as np
import math
import Polygon as plg
import random
import config
THRESH_INTERSEC = 0.5

def watershed1(image, viz=False):
	boxes = []
	if len(image.shape) == 3:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image
	if viz:
		cv2.imshow("gray", gray)
		cv2.waitKey()
	ret, binary = cv2.threshold(gray, 0.6 * np.max(gray), 255, cv2.THRESH_BINARY)
	if viz:
		cv2.imshow("binary", binary)
		cv2.waitKey()
	# 形态学操作，进一步消除图像中噪点
	kernel = np.ones((3, 3), np.uint8)
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # iterations连续两次开操作
	sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
	if viz:
		cv2.imshow("sure_bg", sure_bg)
		cv2.waitKey()
	
	# 距离变换
	dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
	if viz:
		cv2.imshow("dist", dist)
		cv2.waitKey()
	ret, sure_fg = cv2.threshold(dist, 0.2 * np.max(dist), 255, cv2.THRESH_BINARY)
	surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
	if viz:
		cv2.imshow("surface_fg", surface_fg)
		cv2.waitKey()
	unknown = cv2.subtract(sure_bg, surface_fg)
	# 获取maskers,在markers中含有种子区域
	ret, markers = cv2.connectedComponents(surface_fg)
	
	# 分水岭变换
	markers = markers + 1
	markers[unknown == 255] = 0
	
	if viz:
		color_markers = np.uint8(markers)
		color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
		cv2.imshow("color_markers", color_markers)
		cv2.waitKey()
	
	markers = cv2.watershed(image, markers=markers)
	image[markers == -1] = [0, 0, 255]
	if viz:
		cv2.imshow("image", image)
		cv2.waitKey()
	for i in range(2, np.max(markers) + 1):
		np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
		# print(np_contours.shape)
		rectangle = cv2.minAreaRect(np_contours)
		box = cv2.boxPoints(rectangle)
		w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
		box_ratio = max(w, h) / (min(w, h) + 1e-5)
		if abs(1 - box_ratio) <= 0.1:
			l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
			t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
			box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
		
		# make clock-wise order
		startidx = box.sum(axis=1).argmin()
		box = np.roll(box, 4 - startidx, 0)
		box = np.array(box)
		boxes.append(box)
	return np.array(boxes)


def getDetCharBoxes_core(textmap, text_threshold=0.5, low_text=0.4):
	# prepare data
	textmap = textmap.copy()
	img_h, img_w = textmap.shape
	
	""" labeling method """
	ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
	nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8),
	                                                                     connectivity=4)
	
	det = []
	mapper = []
	for k in range(1, nLabels):
		# size filtering
		size = stats[k, cv2.CC_STAT_AREA]
		if size < 10: continue
		
		# thresholding
		if np.max(textmap[labels == k]) < text_threshold: continue
		
		# make segmentation map
		segmap = np.zeros(textmap.shape, dtype=np.uint8)
		segmap[labels == k] = 255
		# segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
		x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
		w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
		niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
		sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
		# boundary check
		if sx < 0: sx = 0
		if sy < 0: sy = 0
		if ex >= img_w: ex = img_w
		if ey >= img_h: ey = img_h
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
		segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
		
		# make box
		np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
		rectangle = cv2.minAreaRect(np_contours)
		box = cv2.boxPoints(rectangle)
		
		# align diamond-shape
		w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
		box_ratio = max(w, h) / (min(w, h) + 1e-5)
		if abs(1 - box_ratio) <= 0.1:
			l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
			t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
			box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
		
		# make clock-wise order
		startidx = box.sum(axis=1).argmin()
		box = np.roll(box, 4 - startidx, 0)
		box = np.array(box)
		
		det.append(box)
		mapper.append(k)
	
	return det, labels, mapper


def watershed2(image, viz=False):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray) / 255.0
	boxes, _, _ = getDetCharBoxes_core(gray)
	return np.array(boxes)


cnt = 0

def showmat(name, mat):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, mat)
    cv2.waitKey(0)


def watershed(oriimage, image, viz=True, idx=-1):
	global cnt
	# idx = random.randint(0, 10000)
	# cv2.imwrite("debug/img_" + str(idx) + ".jpg", image)
	# cv2.imwrite("debug/ori_" + str(idx) + ".jpg", oriimage)
	# viz = True
	boxes = []
	if len(image.shape) == 3:
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	else:
		gray = image
	if viz:
		cv2.imshow("gray", gray)
		cv2.waitKey()

	# ret: threshold = 0.2 * np.max(gray)
	# binary image: bi_image[pt] = 255 if bi_image[pt] > threshold else 0
	ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
	if viz:
		cv2.imshow("binary", binary)
		cv2.waitKey()
	# 形态学操作，进一步消除图像中噪点
	kernel = np.ones((3, 3), np.uint8)
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # iterations连续两次开操作
	sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
	sure_bg = mb
	if viz:
		cv2.imshow("sure_bg", mb)
		cv2.waitKey()
	# 距离变换
	# dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
	# if viz:
	#     cv2.imshow("dist", dist)
	#     cv2.waitKey()
	ret, sure_fg = cv2.threshold(gray, 0.4 * gray.max(), 255, cv2.THRESH_BINARY)
	surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
	if viz:
		cv2.imshow("surface_fg", surface_fg)
		cv2.waitKey()
	unknown = cv2.subtract(sure_bg, surface_fg)

	if viz:
		cv2.imshow("unknown", unknown)
		cv2.waitKey()
	# 获取maskers,在markers中含有种子区域
	# ret, markers = cv2.connectedComponents(surface_fg)
	
	nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg,
	                                                                     connectivity=4)
	# 分水岭变换
	markers = labels.copy() + 1
	# markers = markers+1
	markers[unknown == 255] = 0
	
	# if viz:
	#     color_markers = np.uint8(markers)
	#     color_markers = color_markers / (color_markers.max() / 255)
	#     color_markers = np.uint8(color_markers)
	#     color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
	#     cv2.imshow("color_markers", color_markers)
	#     cv2.waitKey()
	# a = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
	markers = cv2.watershed(oriimage, markers=markers)
	oriimage[markers == -1] = [0, 0, 255]

	# if viz:
	#     color_markers = np.uint8(markers + 1)
	#     color_markers = color_markers / (color_markers.max() / 255)
	#     color_markers = np.uint8(color_markers)
	#     color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
	#     cv2.imshow("color_markers1", color_markers)
	#     cv2.waitKey()
	
	if viz:
		cv2.imshow("image", oriimage)
		cv2.waitKey()
	for i in range(2, np.max(markers) + 1):
		np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
		# cv2.drawContours(image,[np_contours],0,(0,255,0),2)
		# showmat("cnt",image)
		rectangle = cv2.minAreaRect(np_contours)
		box = cv2.boxPoints(rectangle)
		startidx = box.sum(axis=1).argmin()
		box = np.roll(box, 4 - startidx, 0)
		poly = plg.Polygon(box)
		area = poly.area()
		if area < 10:
			continue
		
		x, y, w, h = cv2.boundingRect(np_contours)
		box = [x, y, w, h]
		boxes.append(box)
	
	# box = np.array(box)
	# boxes.append(box)
	boxes = sorted(boxes, key=lambda kv: kv[0])
	boxes = remove_intersec(boxes, True)
	for box in boxes:
		x, y, w, h = box
		cv2.rectangle(oriimage, (x, y), (x + w, y + h),
		              (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
	if viz:
		cv2.imshow("image", oriimage)
		cv2.waitKey()
	# idx = random.randint(0, 10000)
	# cv2.imwrite("debug/" + str(idx) + ".jpg", oriimage)
	# cnt += 1
	for i in range(len(boxes)):
		x, y, w, h = boxes[i]
		boxes[i] = [[x, y*config.expand_scale], [x + w, y*config.expand_scale], [x + w, (y + h)*config.expand_scale], [x, (y + h)*config.expand_scale]]
	return np.array(boxes)


def check_intersec(x1, x1_e, x2, x2_e, intersec):
	if x1_e - x1 < x2_e - x2:
		w = x2_e - x2
	else:
		w = x1_e - x1
	if intersec / w > THRESH_INTERSEC:
		return True
	return False


def remove_intersec(list_boxes, flag):
	if not flag:
		return list_boxes
	
	checked = []
	del_box = []
	for i in range(len(list_boxes)):
		if i in checked:
			continue
		checked.append(i)
		for j in range(i + 1, len(list_boxes)):
			box1 = list_boxes[i]
			x1, y1, w1, h1 = box1
			x1_e = x1 + w1
			y1_e = y1 + h1
			
			box2 = list_boxes[j]
			x2, y2, w2, h2 = box2
			x2_e = x2 + w2
			y2_e = y2 + h2
			type = -1
			if x2 < x1 < x2_e and x2_e <= x1_e:
				intersec = x2_e - x1
				merge = check_intersec(x1, x1_e, x2, x2_e, intersec)
				type = 1
			elif (x1 <= x2 < x2_e and x2_e <= x1_e) or (x2 <= x1 < x1_e and x1_e <= x2_e):
				merge = True
			elif x1 < x2 <= x1_e and x1_e < x2_e:
				intersec = x1_e - x2
				merge = check_intersec(x1, x1_e, x2, x2_e, intersec)
				type = 2
			else:
				merge = False
			if merge:
				xs = min(x1, x2)
				ys = min(y1, y2)
				xe = max(x1_e, x2_e)
				ye = max(y1_e, y2_e)
				bb = [xs, ys, xe - xs, ye - ys]
				list_boxes[i] = bb
				checked.append(j)
				del_box.append(j)
			else:
				if type == 1:
					mid = (x1 + x2_e) // 2
					box1 = [mid + 1, y1, x1_e - mid - 1, h1]
					box2 = [x2, y2, mid - x2, h2]
					list_boxes[i] = box1
					list_boxes[j] = box2
				elif type == 2:
					mid = (x1_e + x2) // 2
					box1 = [x1, y1, mid - x1 - 1, h1]
					box2 = [mid, y1, x2_e - mid, h2]
					list_boxes[i] = box1
					list_boxes[j] = box2
	result = []
	for idx, box in enumerate(list_boxes):
		if idx in del_box:
			continue
		result.append(box)
	return result


if __name__ == '__main__':
	image = cv2.imread('test_folder/transform/im1201.jpg', cv2.IMREAD_COLOR)
	oriimage = cv2.imread('test_folder/image/im1201.jpg')
	boxes = watershed(oriimage, image, False, idx=0).astype(int)
	for i in range(boxes.shape[0]):
		print(i)
		bbox = boxes[i,:, :]
		bbox = np.expand_dims(bbox, axis = 0)
		print(bbox.shape)
		bbox = bbox.reshape((-1,1,2))
		vis_img = cv2.polylines(oriimage,[bbox],False,(0,255,255))
	cv2.imwrite('./vis/vis.png', vis_img)
