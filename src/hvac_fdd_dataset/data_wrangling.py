import os
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from scipy.io import loadmat
from tqdm import tqdm

from config import config

logger.configure(**config.logging)
while len(list(Path(".").glob("pyproject.toml"))) == 0:
    os.chdir("..")
print(f"Changed working dir to {os.getcwd()}")


def wrangle_data():
    """
    Reads in original Matlab .mat files and converts to Pandas DataFrames and saves to parquet files.
    For internal CSIRO use only.
    """

    data_dir = Path("//fs1-cbr.nexus.csiro.au/{en-energy-sys}/source/OD-202088 HVAC FDD Trials/Newcastle_HVAC_Fault_Generation/NewcastleSiteDatasets/")

    # Read Ground-Truth Data
    gt: pd.DataFrame = pd.read_excel(f"{data_dir}/ground truth files/Fault Experiments 2013 2014.xlsx").iloc[:, 0:10]

    # Add Date and time columns together
    gt = gt.dropna(subset=['Start Date', 'Start Time', 'End Date', 'End Time'])
    gt['Start Datetime'] = pd.to_datetime(gt['Start Date'].astype(str) + ' ' + gt['Start Time'].astype(str), format='%Y-%m-%d %H:%M:%S')
    gt['End Datetime'] = pd.to_datetime(gt['End Date'].astype(str) + ' ' + gt['End Time'].astype(str), format='%Y-%m-%d %H:%M:%S')
    gt = gt.drop(columns=['Start Date', 'Start Time', 'End Date', 'End Time'])
    gt.to_parquet("data/fault-experiments.parquet")

    # Read AHU time-series data
    ahus = list(data_dir.glob("2013_2015_AHU/AHU*/Daily"))
    for ahu_dir in ahus:
        name = ahu_dir.parent.name
        mat_files = list(ahu_dir.glob("*.mat"))
        ahu_dfs = []

        for mat_file in tqdm(mat_files, f"Reading {name} .mat files."):
            m = loadmat(mat_file)
            if 'sensorName' in m and 'sensorData' in m:
                df = pd.DataFrame(data=m['sensorData'].T,
                                  columns=[c[0] for c in m['sensorName'][0]],
                                  index=pd.to_datetime(m['Time'].ravel() - 719529, unit='D'))  # 719529 is the datenum value of the Unix epoch start (1970-01-01)
                ahu_dfs.append(df)

        df = pd.concat(ahu_dfs, axis='rows').sort_index()
        df.index = df.index.round('1s')
        df.columns = pd.io.parsers.base_parser.ParserBase({'names': df.columns, 'usecols': None})._maybe_dedup_names(df.columns)
        df.to_parquet(f"data/{name}.parquet")


def join_ground_truth():
    """
    Joins the ground truth data with the AHU time-series data.
    """
    gt = pd.read_parquet("data/fault-experiments.parquet")
    gt.rename(columns={'Start Datetime': 'start', 'End Datetime': 'end'}, inplace=True)

    ahus = ["AHU9", "AHU10"]

    for ahu_name in ahus:
        ahu_num = int(re.search(r'\d+', ahu_name).group())
        df = pd.read_parquet(f"data/{ahu_name}.parquet").reset_index()
        df.rename(columns={'index': 'Datetime'}, inplace=True)
        df["AHU"] = ahu_num

        # Join ground truth intervals to sensor data from the same intervals, see https://stackoverflow.com/a/45510620/4771628
        import pandasql as ps
        sqlcode = '''
            select *
            from df
            inner join gt on df.AHU=gt.AHU
            where df.Datetime >= gt.start and df.Datetime < gt.end
        '''
        joined = ps.sqldf(sqlcode, locals())
        joined.columns = pd.io.parsers.base_parser.ParserBase({'names': joined.columns, 'usecols': None})._maybe_dedup_names(joined.columns)
        joined.to_parquet(f"data/{ahu_name}+ground_truth.parquet")


if __name__ == "__main__":
    # wrangle_data()
    join_ground_truth()
