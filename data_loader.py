from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import random
import os
import numpy as np
from PIL import Image

class Base_Dataset(data.Dataset):
    def __init__(self, root, partition, target_ratio=0.0):
        super(Base_Dataset, self).__init__()
        # set dataset info
        self.root = root
        self.partition = partition
        self.target_ratio = target_ratio
        # self.target_ratio=0 no mixup
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])

    def __len__(self):

        if self.partition == 'train':
            if self.label_flag is None:
                return int(min(self.alpha))
            else:
                return int(self.num_labeled_target)
                # return int((self.num_labeled_target) / (self.num_class - 1))
        elif self.partition == 'test':
            return int(len(self.target_image) / (self.num_class - 1))


    def __getitem__(self, item):

        image_data = []
        label_data = []

        # have a labeled part and a random sampled part
        random_real_label = []
        class_index_target = []

        # domain_label = []
        ST_split = [] # Mask of targets to be evaluated
        # select index for support class

        # Phase 1
        if self.label_flag is None:
            class_index_source = list(range(self.num_class - 1))
            random.shuffle(class_index_source)
            if self.partition == 'train':
                for classes in class_index_source:
                    # select support samples from source domain or target domain
                    image = Image.open(random.choice(self.source_image[classes])).convert('RGB')
                    if self.transformer is not None:
                        image = self.transformer(image)
                    image_data.append(image)
                    label_data.append(classes)
                    ST_split.append(0)
                # random load source


                for i in range(self.num_class - 1):
                    index = random.choice(list(range(len(self.source_image))))
                    # index = random.choice(list(range(len(self.label_flag))))
                    source_image = Image.open(self.all_source_images[index]).convert('RGB')
                    if self.transformer is not None:
                        source_image = self.transformer(source_image)
                    image_data.append(source_image)
                    label_data.append(self.all_source_labels[index])
                    random_real_label.append(self.all_source_labels[index])
                    # domain_label.append(0)
                    ST_split.append(0)

            if self.partition == 'test':
                for i in range(self.num_class - 1):
                    index = random.choice(list(range(len(self.target_image))))
                    # index = random.choice(list(range(len(self.label_flag))))
                    target_image = Image.open(self.target_image[index]).convert('RGB')
                    if self.transformer is not None:
                        target_image = self.transformer(target_image)
                    image_data.append(target_image)
                    label_data.append(0) # just to ignore
                    # random_real_label.append(self.target_label[index])
                    ST_split.append(1)
                for i in range(self.num_class - 1):
                    target_image = Image.open(self.target_image[item * (self.num_class - 1) + i]).convert('RGB')
                    if self.transformer is not None:
                        target_image = self.transformer(target_image)
                    image_data.append(target_image)
                    label_data.append(self.num_class)
                    random_real_label.append(self.target_label[item * (self.num_class - 1) + i])
                    # domain_label.append(0)
                    ST_split.append(1)


        # Phase 2
        else:
            # num_class_index_target = int(self.target_ratio * (self.num_class - 1))

            if self.target_ratio > 0:

                class_index_target = self.available_index

                # class_index_target = random.sample(available_index, min(num_class_index_target, len(available_index)))

            for classes in class_index_target:
                # select support samples from source domain or target domain
                image = Image.open(random.choice(self.target_image_list[classes])).convert('RGB')

                if self.transformer is not None:
                    image = self.transformer(image)
                image_data.append(image)
                label_data.append(classes)
                # domain_label.append(0)
                ST_split.append(0)
                # target_real_label.append(classes)

            # add target samples
            for i in range(self.num_class - 1):

                if self.partition == 'train':
                    index = random.choice(list(range(len(self.label_flag))))
                    # index = random.choice(list(range(len(self.target_image))))
                    target_image = Image.open(self.target_image[index]).convert('RGB')
                    if self.transformer is not None:
                        target_image = self.transformer(target_image)
                    image_data.append(target_image)
                    label_data.append(self.label_flag[index])
                    random_real_label.append(self.target_label[index])
                    # domain_label.append(0)
                    ST_split.append(0)

                elif self.partition == 'test':
                    target_image = Image.open(self.target_image[item * (self.num_class - 1) + i]).convert('RGB')
                    if self.transformer is not None:
                        target_image = self.transformer(target_image)
                    image_data.append(target_image)
                    label_data.append(self.num_class)
                    random_real_label.append(self.target_label[item * (self.num_class - 1) + i])
                    # domain_label.append(0)
                    ST_split.append(1)

        image_data = torch.stack(image_data)
        label_data = torch.tensor(label_data).type(torch.LongTensor)
        real_label_data = torch.tensor(random_real_label)
        # domain_label = torch.tensor(domain_label)
        ST_split = torch.tensor(ST_split)
        return image_data, label_data, real_label_data, ST_split

    def load_dataset(self):
        source_image_list = {key: [] for key in range(self.num_class - 1)}
        target_image_list = []
        target_label_list = []
        all_source_images = []
        all_source_labels = []

        with open(self.source_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                if label == str(self.num_class-1):
                    continue
                source_image_list[int(label)].append(image_dir)
                # source_image_list.append(image_dir)
                all_source_images.append(image_dir)
                all_source_labels.append(int(label))


        with open(self.target_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = line.split(' ')
                label = label.strip()
                # target_image_list[int(label)].append(image_dir)
                target_image_list.append(image_dir)
                target_label_list.append(int(label))

        return source_image_list, target_image_list, target_label_list, all_source_images, all_source_labels


class Office_Dataset(Base_Dataset):

    def __init__(self, root, partition, label_flag=None, source='A', target='W', target_ratio=0.0):
        super(Office_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join(root, src_name)
        self.target_path = os.path.join(root, tar_name)
        self.class_name = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle",
                           "calculator", "desk_chair", "desk_lamp", "desktop_computer", "file_cabinet", "unk"]
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label, self.all_source_images, self.all_source_labels = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # source only
        # if self.label_flag is None:
            # self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        if self.label_flag is not None:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
            self.available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                                   and key < self.num_class - 1]
            self.num_labeled_target = sum([len(self.target_image_list[key]) for key in self.target_image_list.keys()[:-2]])

        if self.target_ratio > 0:
            self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in self.source_image.keys()]
        else:
            self.alpha_value = self.alpha

        self.alpha_value = np.array(self.alpha_value)
        self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

    def getFilePath(self, source, target):

        if source == 'A':
            src_name = 'amazon_src_list.txt'
        elif source == 'W':
            src_name = 'webcam_src_list.txt'
        elif source == 'D':
            src_name = 'dslr_src_list.txt'
        else:
            print("Unknown Source Type, only supports A W D.")

        if target == 'A':
            tar_name = 'amazon_tar_list.txt'
        elif target == 'W':
            tar_name = 'webcam_tar_list.txt'
        elif target == 'D':
            tar_name = 'dslr_tar_list.txt'
        else:
            print("Unknown Target Type, only supports A W D.")

        return src_name, tar_name




class Home_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, source='A', target='R', target_ratio=0.0):
        super(Home_Dataset, self).__init__(root, partition, target_ratio)
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join(root, src_name)
        self.target_path = os.path.join(root, tar_name)
        self.class_name = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                           'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                           'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
                           'Fork', 'unk']
        self.num_class = len(self.class_name)

        self.source_image, self.target_image, self.target_label, self.all_source_images, self.all_source_labels = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag

            # self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        if self.label_flag is not None:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
            self.available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                               and key < self.num_class - 1]
            self.num_labeled_target = sum(
                [len(self.target_image_list[key]) for key in list(self.target_image_list.keys())[:-2]])


        # if self.target_ratio > 0:
        #     self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in
        #                         self.source_image.keys()]
        # else:
        #     self.alpha_value = self.alpha
        #
        # self.alpha_value = np.array(self.alpha_value)
        # self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        # self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

    def getFilePath(self, source, target):

        if source == 'A':
            src_name = 'art_source.txt'
        elif source == 'C':
            src_name = 'clip_source.txt'
        elif source == 'P':
            src_name = 'product_source.txt'
        elif source == 'R':
            src_name = 'real_source.txt'
        else:
            print("Unknown Source Type, only supports A C P R.")

        if target == 'A':
            tar_name = 'art_tar.txt'
        elif target == 'C':
            tar_name = 'clip_tar.txt'
        elif target == 'P':
            tar_name = 'product_tar.txt'
        elif target == 'R':
            tar_name = 'real_tar.txt'
        else:
            print("Unknown Target Type, only supports A C P R.")

        return src_name, tar_name


class Visda_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
        super(Visda_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = os.path.join(root, 'source_list.txt')
        self.target_path = os.path.join(root, 'target_list.txt')
        self.class_name = ["bicycle", "bus", "car", "motorcycle", "train", "truck", 'unk']
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label, self.all_source_images, self.all_source_labels = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # source-only stage
        # if self.label_flag is None:
        #     self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        if self.label_flag is not None:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

            self.available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                                   and key < self.num_class - 1]
            self.num_labeled_target = sum(
                [len(self.target_image_list[key]) for key in self.target_image_list.keys()][:-2])


class Visda18_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
        super(Visda18_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = os.path.join(root, 'source_list_k.txt')
        self.target_path = os.path.join(root, 'target_list.txt')
        self.class_name = ["areoplane","bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant",
                           "skateboard", "train", "truck", 'unk']
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label, self.all_source_images, self.all_source_labels = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag

            # self.label_flag = torch.ones(len(self.target_image)) * self.num_class


        if self.label_flag is not None:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
            self.available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                               and key < self.num_class - 1]
            self.num_labeled_target = sum(
                [len(self.target_image_list[key]) for key in list(self.target_image_list.keys())[:-2]])

