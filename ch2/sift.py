import cv2
import numpy as np
import random

def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def sift_image_alignment(img1, img2):
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    good_match = get_good_match(des1, des2)
    if len(good_match) > 4:
        pts_a = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        pts_b = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
        h, status = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=4)
        img_out = cv2.warpPerspective(img2, h, (img1.shape[1], img1.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    macth_img = cv2.drawMatches(img1, kp1, img2, kp2, random.sample(good_match, 200), None, flags=2)
    return img_out, macth_img, h, status


img1 = cv2.imread('data/Image A.jpg')
img2 = cv2.imread('data/Image B.jpg')

result, img_kp_conn, H, _ = sift_image_alignment(img1, img2)
cv2.imwrite('result/sift/result.jpg', result)
cv2.imwrite('result/sift/img_kp_conn.jpg', img_kp_conn)
print(H)
