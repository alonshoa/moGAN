import os
from torch.utils.data import Dataset
import pandas as pd


class FB_Dataset_test(Dataset):
    def __init__(self, file_name, use_time=False):
        df = pd.read_csv(file_name)
        self.size = len(df)
        self.df = df.sample(frac=1).reset_index(drop=True)
        if not use_time:
            self.df = self.df.drop('Time',axis=1)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.df.iloc[index].values,1


class FB_Dataset(Dataset):
    def __init__(self, data_dir,max_files = 10,use_time = False):
        super().__init__()
        self.data_dir = data_dir
        print(data_dir)
        self.calc_len(self.data_dir,max_files)

        # df = pd.read_csv(file_name)
        # self.size = len(df)
        # self.df = df.sample(frac=1).reset_index(drop=True)
        # if not use_time:
        #     self.df = self.df.drop('Time',axis=1)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        print("first", index)
        start,stop, df = self.get_file_from_idx(index)
        l = len(df)
        print("second:", index,"df ", l, "start", start, "stop", stop, "index - start",index - start)

        return df.iloc[index - start].values,1 if l < index - start else df.iloc[l-1].values

    def calc_len(self,data_dir,max_files):
        cumsum = 0
        self.file_len = {}
        for i,file in zip(range(max_files),os.listdir(data_dir)):
            df = pd.read_csv(os.path.join(data_dir,file))
            df = df.drop('Time',axis=1)
            self.file_len[file] = (cumsum,cumsum + len(df)-1,df)
            cumsum += len(self.file_len[file][2])

        self.size = cumsum
        print(f"used {i} files")

    def get_file_from_idx(self, index):
        for k,v in self.file_len.items():
            if index > v[0] and index < v[1]: ## index in file
                return v
        print(index)
        return v