import torch
import os
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import PIL.ImageOps
from model3 import SiameseNetwork
from config import Config
from torch.utils.data import DataLoader, Dataset
import sys
from train3 import image_size

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# threshold = 0.76
model_path = "./trained/DogSiamese.pkl"
use_gpu = False

#
class TestDataset(Dataset):
    def __init__(self, left_image, right_images):
        self.left_image = left_image
        self.right_images = right_images
        self.transform = transforms.Compose([
            # transforms.Resize((int(image_size*1.2), int(image_size*1.2))),
            transforms.CenterCrop(image_size),
            # transforms.Resize((image_size, image_size)),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, idx):
        left_img = self.left_image
        right_img = self.right_images[idx]
        # print("TestDataset[%d] %s - %s" % (idx, left_img, right_img))

        img0 = Image.open(left_img)
        img1 = Image.open(right_img)
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img0 = PIL.ImageOps.equalize(img0)
        img1 = PIL.ImageOps.equalize(img1)
        # img0 = ImageEnhance.Sharpness(img0).enhance(100.0)
        # img1 = ImageEnhance.Sharpness(img1).enhance(100.0)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # img0 = torch.as_tensor(np.reshape(img0, (3, image_size, image_size)), dtype=torch.float32)
        # img1 = torch.as_tensor(np.reshape(img1, (3, image_size, image_size)), dtype=torch.float32)

        return img0, img1

    def __len__(self):
        return len(self.right_images)


def calculate_far_frr(inferences, group_size, threshold):
    fa, fr = 0, 0
    dogs = len(inferences)
    group_count = int(dogs / group_size)
    for dog in range(dogs):
        dog_group = int(dog / group_size)
        for g in range(group_count):
            if g == dog_group:
                for i in range(group_size):
                    if not (inferences[dog][g * group_size + i] > threshold):
                        fr += 1
            else:
                for i in range(group_size):
                    if inferences[dog][g * group_size + i] > threshold:
                        fa += 1

    # print("dogs:%d, group_size:%d" % (dogs, group_size))
    fa_base = dogs * dogs - dogs * group_size
    fr_base = dogs * group_size
    return fa / fa_base, fr / fr_base


def verify_dogs(test_model, left_dogs, right_dogs):
    inferences = []

    for left_dog in left_dogs:
        inf = []
        test_set = TestDataset(left_dog, right_dogs)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=0)
        for i, data in enumerate(test_dataloader):
            img0, img1 = data
            if use_gpu:
                similarity = test_model.evaluate(img0.cuda(), img1.cuda())
            else:
                similarity = test_model.evaluate(img0.cpu(), img1.cpu())
            inf.append(similarity)
            print("\r%s | %s = %f" % (left_dog, right_dogs[i], similarity), end="")
        print()
        inferences.append(inf)

    return inferences


if __name__ == '__main__':
    model_path = Config.Evaluate.model_path
    for a in sys.argv[1:]:
        if a.lower() == 'gpu':
            use_gpu = True
        else:
            model_path = a.split("=")[1]
    print('use gpu：', use_gpu)
    print('model path：', model_path)

    inference_output_path = Config.Evaluate.inference_output_path
    eer_output_path = Config.Evaluate.eer_output_path
    dog_input_root = Config.Evaluate.test_dir
    dog_count = Config.Evaluate.dog_count
    group_size = Config.Evaluate.group_size

    print("Model path: %s" % (model_path,))
    print("Input path: %s" % (dog_input_root,))
    print("dog_count: %s" % (dog_count,))
    print("group_size: %s" % (group_size,))

    if use_gpu:
        siam_test = SiameseNetwork(image_size).cuda()
        siam_test.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
    else:
        siam_test = SiameseNetwork(image_size).cpu()
        siam_test.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    siam_test.eval()

    dog_paths = [x for x in os.walk(dog_input_root)]
    dog_paths.sort()
    img_paths = []
    for dog_path in dog_paths[1: dog_count + 1]:
        img_paths.extend(os.path.join(dog_path[0], x) for x in dog_path[2][:group_size])

    inferences = verify_dogs(siam_test, img_paths, img_paths)
    dogs = len(img_paths)
    group_count = int(dogs / group_size)
    header = [""]
    for g in range(group_count):
        for i in range(group_size):
            header.append("{}_{}".format(g+1, i+1))
    lines = ["\t".join(header) + "\n"]
    for dd in range(dogs):
        line = header[dd + 1] + "\t"
        d = 0
        for g in range(group_count):
            for i in range(group_size):
                line += "{}\t".format(inferences[dd][d])
                d += 1
        lines.append(line[:-1] + "\n")
    with open(inference_output_path, "w") as fp:
        fp.writelines(lines)

    lines = ["Threshold\tFAR\tFRR\n"]
    for i in range(101):
        far, frr = calculate_far_frr(inferences, group_size, threshold=i)
        lines.append("%d\t%f\t%f\n" % (i, far, frr))

    with open(eer_output_path, "w") as fp:
        fp.writelines(lines)
