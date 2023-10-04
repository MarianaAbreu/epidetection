# Python script with useful functions to call on notebooks
# Make sure this script is in the same directory as the notebook you are working on
# Import this script in your notebook with: from utils import *

# Created by: Mariana Abreu
# Date: 2023-09-25

# Import libraries
# built-in
import os

# third-party
import pandas as pd
from mne.io import read_raw_edf


def read_edf(file):
    """
    Function to read edf file
    Parameters:
        file (string): file path
    Returns:
        data (dataframe): raw data points with timestamp column
    """
    # raw edf data
    raw = read_raw_edf(file)
    # raw data points saved in a dataframe
    data = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)
    # start date
    start_date = raw.info['meas_date']
    # sampling frequency
    fs = raw.info['sfreq']
    # get timestamp column
    data['timestamp'] = pd.date_range(start=start_date, periods=len(data), freq=f'{1/fs}S')
    return data


class Patient():
    # class to handle single patient data from Seer dataset

    def __init__(self, id, dir) -> None:
        
        self.id = id # patient id (string)
        self.pat_dir = self.getdir(dir) # patient directory (string)
        self.annotations = None
        pass
    
    def getdir(self, dir):
        """
        Function to get patient directory
        Parameters:
            dir (string): directory where patient folder is located
        Returns:
            pat_dir (string): patient directory
        """
        # check if directory exists
        # if not, raise error
        if os.path.isdir(os.path.join(dir, self.id)):
            print(f'Patient directory found "{os.path.join(dir, self.id)}"')
            return os.path.join(dir, self.id)
        else:
            raise FileExistsError(f"Directory {dir} does not exist")

    def get_seizures_annotations(self):
        """
        Function to get seizures annotations
        Parameters:
            None
        Returns:
            None
        """
        # check if annotations file exists
        annot_files = [file for file in os.listdir(self.pat_dir) if 'annotations' in file.lower()]
        if len(annot_files) > 1:
            raise FileExistsError(f"More than one annotations file found in {self.pat_dir}: \n {annot_files}")
        elif len(annot_files) == 0:
            raise FileExistsError(f"No annotations file found in {self.pat_dir}")
        else:
            annot_file = annot_files[0]
            read_annotations = pd.read_csv(os.path.join(self.pat_dir, annot_file))
            # create a timestamp column for start time and end time based on the utc time (number of seconds since 1970-01-01 00:00:00)
            # this timestamp must be offset by the utc offset (in seconds) to get the local time - which is in column timezone
            offset = read_annotations['timezone'] * pd.Timedelta(hours=1) # convert offset to timedelta
            read_annotations['timestamp_start'] = read_annotations['start_time'].astype('datetime64[ms]') + offset
            read_annotations['timestamp_end'] = read_annotations['end_time'].astype('datetime64[ms]') + offset
            
            print(f"Annotations file found: {annot_file}")
            self.annotations = read_annotations
            return read_annotations

    def get_sensor_data_raw(self, sensor, save=True, savedir=None):
        """
        Function to get raw sensor data
        Parameters:
            sensor (string): sensor name
        Returns:
            None
        """
        if sensor.lower() in ['bvp', 'temp', 'acc', 'eda', 'hr']:
            sensor = sensor.upper()
        else:
            raise ValueError(f"Sensor {sensor} not recognized, please use one of the following: bvp, temp, acc, eda, hr")
        # get all files names that exist inside patient's directory and contain the sensor name in uppercase
        # these files should be sorted by date (since the date is part of the file name, this should be ok)
        sensor_files = sorted([file for file in os.listdir(self.pat_dir) if (sensor in file.upper() and file.endswith('.edf'))])
        if len(sensor_files) > 0:
            # this line extract the data from all sensor files and stacks all together in a dataframe table
            sensor_data = pd.concat(list(map(lambda file: read_edf(os.path.join(self.pat_dir, file)), sensor_files)))
        else:
            raise FileExistsError(f"No {sensor} files found in {self.pat_dir}")
        if save:
            if savedir is None:
                savedir = f'raw_data_{self.id}_{sensor}.parquet'
            sensor_data.to_parquet(savedir, engine = 'fastparquet')
        return sensor_data
    
    def get_sensor_data(self, sensor, savedir = None):
        """
        Function to get sensor data from all files
        If the parquet file already exists, it will read the file
        If not, it will call the function get_sensor_data_raw to get the data and save it

        Parameters:
            sensor (string): sensor name
        Returns:
            sensor data (dataframe): sensor data
        """
        if savedir is None:
            savedir = f'raw_data_{self.id}_{sensor}.parquet'
        if os.path.isfile(savedir):
            sensor_data = pd.read_parquet(savedir, engine='fastparquet')
        else:
            sensor_data = self.get_sensor_data_raw(sensor=sensor, save=True, savedir=savedir)
        return sensor_data
        


if __name__ == "__main__":

    id = 'MSEL_01110'
    dir = 'data'
    patient_class = Patient(id=id, dir=dir)
    annotations = patient_class.get_seizures_annotations()
    data = patient_class.get_sensor_data(sensor='hr')

    pass