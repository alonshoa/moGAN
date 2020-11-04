from torchvision import datasets, transforms
from moGAN_git.base import BaseDataLoader
from moGAN_git.data.FB_dataset import FB_Dataset, FB_Dataset_test


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FB_Dataloader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,max_files = 10):

        self.data_dir = data_dir
        self.dataset = FB_Dataset(data_dir,max_files=max_files)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FB_Dataloader_test(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,max_files = 10):

        self.data_dir = data_dir
        self.dataset = FB_Dataset_test("D:\\idc\\0_Thises\\FB_DATA\\TalkingWithHands32M\\mocap_data\\session1\\take6\\take6\\take6_noFingers_deep2_scale_local_worldpos.csv")
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)