import os
import torch
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DownscalingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, low_res_data, high_res_data, low_var_name=None, high_var_name=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            low_res_path (string): Path to the low resolution data.
            high_res_path (string): Path to the high resolution data.
        """
        self.low_res_data = low_res_data
        self.high_res_data = high_res_data
        self.low_var_name = low_var_name
        self.high_var_name = high_var_name
        
        if len(self.low_res_data) != len(self.high_res_data):
            raise ValueError("Low res and high res data must have the same length")


    def __len__(self):
        return len(self.low_res_data)

    def get_var_name(self):
        if self.low_var_name is None or self.high_var_name is None:
            warnings.warn("Some variable names are not set")
        print("Low res variable name: ", self.low_var_name)
        print("High res variable name: ", self.high_var_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        low_res = np.array(self.low_res_data[idx])
        high_res = self.high_res_data[idx]

        sample = {'low_res': low_res, 'high_res': high_res}

        return sample

    

def make_clean_data(out_var, in_var, year):
    if year < 2010 or year > 2020:
        print("year must be greater than 2010 and less than 2020")
        return
    in_path = 'download/era5/'+in_var+'-2010_2020.nc'
    in_data = nc.Dataset(in_path)
    # get the 3-hour of the start day from Unix timestamp
    start_hour = datetime(2010, 1, 1, 0).timestamp()/3600/3
    end_hour = datetime(2021, 1, 1, 0).timestamp()/3600/3
    low_hour = datetime(year, 1, 1, 0).timestamp()/3600/3
    high_hour = datetime(year+1, 1, 1, 0).timestamp()/3600/3
    start_index = int(low_hour - start_hour)
    end_index = int(high_hour - start_hour)
    in_data = in_data[in_var][start_index:end_index]
    out_data = np.load('download/cerra/' + out_var + '-' + str(year) + '.npy')

    return in_data, out_data


if __name__ == "__main__":
    in_data, out_data = make_clean_data('si10', 'u10', 2019)
    dataset = DownscalingDataset(in_data, out_data, low_var_name='u10', high_var_name='si10')
    dataset.get_var_name()
    print(len(dataset))
    print(dataset[0]["low_res"].shape)
    print(dataset[0]["high_res"].shape)
    
    

