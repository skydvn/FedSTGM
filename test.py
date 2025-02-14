import numpy as np
import torch

train_images_per_class = 600
test_images_per_class = 100

def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = 'dataset/imagenet1k-classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test

train, train_label, test, test_leabl = load_imagenet([1])

def check_image_shapes(tensor_images):
    shapes = [img.shape for img in tensor_images]
    print(f"Co {len(shapes)} anh")
    unique_shapes = set(shapes)
    if len(unique_shapes) == 1:
        print(f"Tất cả ảnh có cùng kích thước: {unique_shapes.pop()}")
    else:
        print("Ảnh có các kích thước khác nhau:")
        for shape in unique_shapes:
            print(shape)

check_image_shapes(train)