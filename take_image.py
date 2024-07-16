from utils import *
import os


def save_image(image, name):
    path = "/Users/simoneroman/Desktop/DL/DeepLearning_project/images_aug/"
    image_np = np.array(image)
    cv2.imwrite(path + name + ".jpg", image_np)



num_aug = 19

current_dir = os.path.dirname(os.path.abspath(__file__))
pathDatasetImagenetA = os.path.join(current_dir, "datasets/imagenet-a")

_, test_data = get_WholeDataset(batch_size=1, img_root=pathDatasetImagenetA)

image, target = test_data[0]
centroid = (100, 100)
images, names = apply_augmentations(image, num_aug, centroid)




for i in range(num_aug):
    save_image(images[i], names[i])