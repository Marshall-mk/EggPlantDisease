import cv2
import numpy as np
import random
import glob

def rotate_image(image_path, angle=10):
    img = cv2.imread(image_path)
    (height, width) = img.shape[:2]
    center = (width/2, height/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (width, height))

def translate_image(image_path):
    img = cv2.imread(image_path)
    img_size = img.shape[0]

    # Randomly flip the image horizontally
    if random.choice([True, False]):
        img = cv2.flip(img, 1)

    # Define the range for random x-translation
    x_translation_range = (int(-0.2 * img_size), int(0.2 * img_size))
    x_translation = random.randint(x_translation_range[0], x_translation_range[1])

    # Define the range for random y-translation
    y_translation_range = (int(-0.2 * img_size), int(0.2 * img_size))
    y_translation = random.randint(y_translation_range[0], y_translation_range[1])

    # Define the affine transformation matrix
    M = np.float32([[1, 0, x_translation],
                    [0, 1, y_translation]])

    # Apply the affine transformation matrix to the image
    img = cv2.warpAffine(img, M, (img_size, img_size))
    return img

def brightness(image_path, alpha=1.0, beta=0):
    img = cv2.imread(image_path)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def zoom(image_path, zoom_factor=0.5):
    img = cv2.imread(image_path)
    (height, width) = img.shape[:2]
    center = (width/2, height/2)
    zoom_matrix = cv2.getRotationMatrix2D(center, 0, zoom_factor)
    return cv2.warpAffine(img, zoom_matrix, (width, height))

def horizontal_flip(image_path):
    img = cv2.imread(image_path)
    img = cv2.flip(img, 1)
    return img

if __name__ == '__main__':
    image_list = glob.glob('../data/train/*/*.jpg')

    for i in range(len(image_list)):
        orig_image_name = image_list[i].split('.jpg')[0]
        
        hor = horizontal_flip(image_path=str(image_list[i]))
        cv2.imwrite(f'{orig_image_name}_hor.jpg', hor)

        rot = rotate_image(image_list[i])
        cv2.imwrite(f'{orig_image_name}_rot.jpg', rot)  
        
        trans = translate_image(image_list[i])
        cv2.imwrite(f'{orig_image_name}_trans.jpg', trans)

        bright = brightness(image_list[i], alpha=1.5, beta=0)
        cv2.imwrite(f'{orig_image_name}_bright.jpg', bright)

        zoomed = zoom(image_list[i], zoom_factor=1.5)
        cv2.imwrite(f'{orig_image_name}_zoom.jpg', zoomed)
    print('Data augmentation complete')