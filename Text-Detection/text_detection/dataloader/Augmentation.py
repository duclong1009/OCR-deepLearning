import os
import cv2
import random
from torchvision.transforms import transforms
import numpy as np

def showmat(name, mat):
	cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
	cv2.imshow(name, mat)
	cv2.waitKey(0)


def check_intersec(y1, y4, yj1, yj4, intersec):
	if yj4 - yj1 < y4 - y1:
		h = yj4 - y1
	else:
		h = y4 - y1
	if intersec / h > 0.3:
		return True
	return False


def color_jitter_image():
	color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.2)
	# transform = transforms.ColorJitter()
	# transform.get_params(
	# 	color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
	# 	color_jitter.hue)
	return color_jitter


def random_scale(img, bboxes, target_size=768, max_scale=3, min_scale=0.3, rnd_scale=0.5):
	if random.random() < rnd_scale:
		h, w = img.shape[0:2]
		if h > target_size or w > target_size:
			scale = random.uniform(min_scale, max_scale / 2)
		else:
			scale = random.uniform(min_scale * 2, max_scale)
		bboxes *= scale
		# print("scale: ", scale, h, w)
		img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
	return img


def padding_image(images, target_h, target_w):
	padded_imgs = []
	actual_h, actual_w = images[0].shape[:2]
	target_h = max(target_h, actual_h)
	target_w = max(target_w, actual_w)
	pad_h = target_h - actual_h
	pad_w = target_w - actual_w
	rnd_pad_top, rnd_pad_bot, rnd_pad_right, rnd_pad_left = 0, 0, 0, 0
	if pad_h > 0:
		rnd_pad_top = random.randint(0, pad_h)
		rnd_pad_bot = pad_h - rnd_pad_top
	if pad_w > 0:
		rnd_pad_left = random.randint(0, pad_w)
		rnd_pad_right = pad_w - rnd_pad_left
	for idx, image in enumerate(images):
		mean = int(np.mean(image))
		input_dimension = len(image.shape)
		target_shape = (target_h, target_w) if input_dimension == 2 else (target_h, target_w, 3)
		big_img = np.ones(target_shape, dtype=np.uint8) * mean if idx != (len(images) - 1) else np.ones(target_shape, dtype=np.float32)
		big_img[rnd_pad_top:target_h - rnd_pad_bot, rnd_pad_right:target_w - rnd_pad_left] = image
		padded_imgs.append(big_img)
	return padded_imgs


def random_crop(imgs, img_size, scale=1.5):
	target_h, target_w = img_size
	padded_h, padded_w = int(target_h * scale), int(target_w * scale)
	imgs = padding_image(imgs, padded_h, padded_w)
	x = random.randint(0, imgs[0].shape[1] - target_w)
	y = random.randint(0, imgs[0].shape[0] - target_h)
	for idx, img in enumerate(imgs):
		img = img[y:y + target_h, x:x + target_w]
		imgs[idx] = img
	return imgs


def random_horizontal_flip(imgs, rnd_flip=0.4):
	if random.random() < rnd_flip:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1).copy()
	return imgs


def random_rotate(imgs, rnd_rotate=0.3):
	max_angle = 10
	if random.random() < rnd_rotate:
		angle = random.uniform(-max_angle, max_angle)
		# print("rnd angle: ",angle)
		for i in range(len(imgs)):
			img = imgs[i]
			w, h = img.shape[:2]
			rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
			img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
			imgs[i] = img_rotation
	return imgs


if __name__ == '__main__':
	path = "test_folder/"
	img_dir = os.path.join(path, 'image')
	hor_dir = os.path.join(path, 'transform/horizontal_flip_image')
	rot_dir = os.path.join(path, 'transform/rotate_image')

	for img_name in os.listdir(img_dir):
		img = cv2.imread(os.path.join(img_dir, img_name))
		hor_flip_img = random_horizontal_flip([img], rnd_flip=1.0)
		if not os.path.exists(hor_dir):
			os.mkdir(hor_dir)
		cv2.imwrite(os.path.join(hor_dir, img_name), hor_flip_img[0])

	for img_name in os.listdir(img_dir):
		img = cv2.imread(os.path.join(img_dir, img_name))
		rot_img = random_rotate([img], rnd_rotate=1.0)
		if not os.path.exists(rot_dir):
			os.mkdir(rot_dir)
		cv2.imwrite(os.path.join(rot_dir, img_name), rot_img[0])




