import os

if __name__ == '__main__':
    root = "data/dog40x4_seg"
    paths = [x for x in os.walk(root)]
    for file in paths[0][2]:
        dire = os.path.join(root, file[:3])
        if not os.path.exists(dire):
            os.makedirs(dire)
        path = os.path.join(root, file)
        os.rename(path, os.path.join(dire, file))
