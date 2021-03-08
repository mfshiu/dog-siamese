import os
import sys
import torch
from config import Config
from train3 import image_size
from model import SiameseNetwork
from evaluate3 import TestDataset
from torch.utils.data import DataLoader

use_gpu = False
register_dir = "./data/ct0202a/"
threshold = 65
siam_model = None
log_lines = []


def exam_dog(dog_id, img_path):
    exam_count = Config.exam_count
    dog_dir = os.path.join(register_dir, dog_id)
    walked = [x for x in os.walk(dog_dir)][0]
    dog_paths = [os.path.join(walked[0], x) for x in walked[2]]
    dog_paths.sort()
    similarities = []

    test_set = TestDataset(img_path, dog_paths[0: min(len(dog_paths), exam_count + 1)])
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=0)
    for i, data in enumerate(test_dataloader):
        img0, img1 = data
        if use_gpu:
            similarity = siam_model.evaluate(img0.cuda(), img1.cuda())
        else:
            similarity = siam_model.evaluate(img0.cpu(), img1.cpu())
        similarities.append(similarity)

    is_same = len([x for x in similarities if x > threshold]) > exam_count // 2
    return is_same, similarities


def find_dog(img_path):
    walked = [x for x in os.walk(register_dir)][0]
    dog_ids = walked[1]
    max_avg = 0
    hit_dog = None

    for dog_id in dog_ids:
        similarities = exam_dog(dog_id, img_path)[1]
        avg = sum(similarities) / len(similarities)
        log_lines.append("%s->%s(avg:%s,%s)\n" % (img_path, dog_id, avg, similarities))
        # avg = max(similarities)
        if avg > threshold and max_avg < avg:
            max_avg = avg
            hit_dog = dog_id

    return hit_dog, max_avg


if __name__ == '__main__':
    register_dir = Config.register_dir
    dog_id = None
    dog_img = None
    exam_dir = None
    model_path = "./trained/DogSiamese-2.pkl"
    for a in sys.argv[1:]:
        if a.lower() == 'gpu':
            use_gpu = True
        else:
            aa = a.split("=")
            if "dog" == aa[0]:
                dog_id = aa[1]
            elif "img" == aa[0]:
                dog_img = aa[1]
            elif "exam_dir" == aa[0]:
                exam_dir = aa[1]
            elif "model" == aa[0]:
                model_path = aa[1]
            else:
                register_dir = a
    print('Use gpu：', use_gpu)
    print('Register dir：', register_dir)
    print('Dog ID to be checked：', dog_id)
    print('Dog image to check：', dog_img)

    if use_gpu:
        siam_model = SiameseNetwork(image_size).cuda()
        siam_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
    else:
        siam_model = SiameseNetwork(image_size).cpu()
        siam_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    siam_model.eval()

    if exam_dir:
        img_paths = []
        for path, subdirs, files in os.walk(exam_dir):
            for name in files:
                img_paths.append(os.path.join(path, name))
        img_paths.sort()
        for i, img in enumerate(img_paths):
            find_id, similarity = find_dog(img)
            if find_id:
                print("%s = %s (%s)" % (img_paths[i], find_id, similarity))
            else:
                print("%s = None" % (img_paths[i],))
    elif dog_id:
        is_same = exam_dog(dog_id, dog_img)[0]
        if is_same:
            print("Yes, The dog is %s." % (dog_id,))
        else:
            print("No, The dog is not %s." % (dog_id,))
    else:
        find_id, similarity = find_dog(dog_img)
        if find_id:
            print("The dog is %s, similarity is %s" % (find_id, similarity))
        else:
            print("Cannot find the dog.")

    with open("exam.log", "w") as fp:
        fp.writelines(log_lines)



