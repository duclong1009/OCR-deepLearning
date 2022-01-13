import numpy as np
import cv2
import os
from shapely.geometry import LineString, Polygon
import config


def showmat(name, mat):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, mat)
    cv2.waitKey(0)


class GaussianTransformer(object):
    def __init__(self, imgSize=512):
        # distanceRatio = 3.34
        self.imgSize = imgSize
        self.standardGaussianHeat = self._gen_gaussian_heatmap(imgSize)

        start_position_region = config.start_position_region # 0.15
        start_position_affinity = config.start_position_affinity # 0.15

        size_of_heatmap_region = config.size_of_heatmap_region # 1 - start_position_region * 2
        size_of_heatmap_affinity = config.size_of_heatmap_affinity # 1 - start_position_affinity * 2

        # We only take region box in the inside part of heatmap, exclude some surface part outside
        # regionbox: coordinate [4, 2]
        x_r, y_r, w_r, h_r = int(start_position_region * imgSize), int(start_position_region * imgSize), \
                             int(size_of_heatmap_region * imgSize), int(size_of_heatmap_region * imgSize)
        self.regionbox = np.array([[x_r, y_r], [x_r + w_r, y_r], [x_r + w_r, y_r + h_r], [x_r, y_r + h_r]],
                                  dtype=np.int32)

        # similar to affinity box
        x_a, y_a, w_a, h_a = int(start_position_affinity * imgSize), int(start_position_affinity * imgSize), int(
            size_of_heatmap_affinity * imgSize), int(
            size_of_heatmap_affinity * imgSize)
        self.affinitybox = np.array([[x_a, y_a], [x_a + w_a, y_a], [x_a + w_a, y_a + h_a], [x_a, y_a + h_a]],
                                    dtype=np.int32)

        self.oribox = np.array([[0, 0, 1], [imgSize - 1, 0, 1], [imgSize - 1, imgSize - 1, 1], [0, imgSize - 1, 1]],
                               dtype=np.int32)

    def _gen_gaussian_heatmap(self, imgSize):
        """
        Make a heat map with shape == imgSize, the heat score will be more increasing if pixel is nearer the center
        :param imgSize:
        :return: heat map with pixel in [0, 255]
        """
        def scaled_gaussian(x):
            return np.exp(-(1 / 2) * (x ** 2))  # not pdf

        # Make a grid heat map
        x, y = np.meshgrid(np.linspace(-2.5, 2.5, imgSize),
                           np.linspace(-2.5, 2.5, imgSize))
        distance_from_center = np.linalg.norm(np.stack([x, y], axis=0), axis=0, keepdims=False)
        scaled_gaussian_prob = scaled_gaussian(distance_from_center)
        heat = np.clip(np.round(scaled_gaussian_prob * 255), 0, 255).astype(np.uint8)
        return heat

    def order_points(self, box, image_shape):
        """
        :param box: character box [4, 2], it may be not a rectangle
        :param image_shape: [H, W]
        :return if box isn't in image or (h_box and w_box are differential) keep stable, else convert to a rectangle char box
        """
        blank_img = np.ones((image_shape), dtype=np.uint8) * 255
        contour = np.array(box, dtype=np.int32).reshape(4, 2)
        # Fill 0 in blank_img if pixel in contour char box
        cv2.polylines(blank_img, [contour], True, 0)

        # cnts: (h: (num_pixel, ), w: (num_pixel, )) - index of pixel == 0
        cnts = np.where(blank_img == 0)
        # cnts: [num_pixel, 2]
        cnts = np.array([[x, y] for (y, x) in zip(cnts[0], cnts[1])])


        # rect_box = cv2.minAreaRect(cnts)
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if len(cnts) == 0:
            # Error if point not in boundary of image
            return box
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(cnts[:, 0]), max(cnts[:, 0])
            t, b = min(cnts[:, 1]), max(cnts[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        return box

    def add_region_character(self, image, target_bbox, regionbox=None):
        """
        :param image: region score map up to now, on begin - zero map
        :param target_bbox: character box [4, 2]
        :param regionbox: coordinate of region box in heat map [4, 2]
        :return: image: region score map after update
        """
        image_shape = image.shape[:2]
        # convert target_bbox to a rectangle box if width and height are almost the same, otherwise keep stable
        target_bbox = self.order_points(target_bbox, image_shape)
        # print(target_bbox)
        # not update score map if char box is not in boundary of image
        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True
        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        transformed = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
        image = np.where(transformed > image, transformed, image)
        return image

    def add_affinity_character(self, image, target_bbox):
        return self.add_region_character(image, target_bbox, self.affinitybox)

    def add_affinity(self, image, bbox_1, bbox_2):
        """
        Calculate affinity of two adjacent character box
        :param image:
        :param bbox_1: [4, 2]
        :param bbox_2: [4, 2]
        :return:
        """
        try:
            # diagonal11: diagonal from bbox_1[0]: top left, bbox_1[2]]: bottom right
            diagonal11 = LineString([bbox_1[0], bbox_1[2]])
            # diagonal11: diagonal from bbox_1[0]: top right, bbox_1[2]]: bottom left
            diagonal12 = LineString([bbox_1[1], bbox_1[3]])
            # center_1: the center of character box 1
            center_1 = diagonal11.intersection(diagonal12).coords[:][0]

            diagonal21 = LineString([bbox_2[0], bbox_2[2]])
            diagonal22 = LineString([bbox_2[1], bbox_2[3]])
            # center_2: the center of character box 2
            center_2 = diagonal21.intersection(diagonal22).coords[:][0]

            tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
            bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
            tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
            br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)
            affinity_box = np.array([tl, tr, br, bl])
        except:
            pass
        return self.add_affinity_character(image, affinity_box.copy()), np.expand_dims(affinity_box, axis=0)

    def generate_region(self, image_size, bboxes):
        """
        Generate region score map of image
        :param image_size: [H, W]
        :param bboxes: [num_word, num_char, 4, 2): (4, 2): (tl, tr, br, bl)
        :return:
        """
        height, width = image_size[0], image_size[1]
        # initial: zero map
        target = np.zeros([height, width], dtype=np.uint8)
        # for each word:
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i].copy())
            # For each character box of each word:
            for j in range(bboxes[i].shape[0]):
                target = self.add_region_character(target, character_bbox[j])

        return target

    def generate_affinity(self, image_size, bboxes, words):
        """
        Generate affinity score map of image
        :param image_size: [H, W]
        :param bboxes: [num_word, num_char, 4, 2): (4, 2): (tl, tr, br, bl)
        :param words: (num_word, )
        :return: target: affinity score, shape = [H, W]
        :return: affinities: affinity boxes, shape = ?
        """
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        for i in range(len(words)):
            try:
                # character_bbox = [num_char, 4, 2]
                character_bbox = np.array(bboxes[i])
                total_letters = 0
                for char_num in range(character_bbox.shape[0] - 1):
                    # Calculate affinity of two adjacent character box
                    target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                         character_bbox[total_letters + 1])
                    affinities.append(affinity)
                    total_letters += 1
            except:
                pass
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities

    def saveGaussianHeat(self):
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        cv2.polylines(warped_color, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standardGaussianHeat1 = self.standardGaussianHeat.copy()
        standardGaussianHeat1[standardGaussianHeat1 > 0] = 255
        threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1, cv2.COLORMAP_JET)
        cv2.polylines(threshhold_guassian, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)


def test_diagon():
    diagonal1 = LineString([[0, 0], [2, 2]])
    diagonal2 = LineString([[0, 1], [1, 0]])
    center = diagonal1.intersection(diagonal2).coords[:]
    print(center)

def test_polyline():
    blank_img = np.ones((10, 5), dtype=np.uint8) * 255
    # showmat("Before polyline", blank_img)
    box = [1, 2, 2, 2, 2, 3, 1, 3]
    contour = np.array(box, dtype=np.int32).reshape(4, 2)
    cv2.polylines(blank_img, [contour], True, 0)
    # showmat("After polyline", blank_img)
    print(blank_img)

    cnts = np.where(blank_img == 0)
    print(cnts)
    print('Update')
    cnts = np.array([[x, y] for (y, x) in zip(cnts[0], cnts[1])])
    print(cnts)
    print(cnts.shape)

def test_add_region_score_box(gaussian):
    score_map = gaussian.add_region_character(np.zeros((100, 100), dtype=np.uint8), target_bbox=
                                  np.array([[30, 10], [60, 10], [60, 30], [30, 30]]))
    showmat("score_map", score_map)

if __name__ == '__main__':
    gaussian = GaussianTransformer(1024)
    test_add_region_score_box(gaussian)
    gaussian.saveGaussianHeat()


