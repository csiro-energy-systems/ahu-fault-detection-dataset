import os
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from scipy.io import loadmat
from tqdm import tqdm

from config import config
from hvac_fdd_dataset.fault_generator import experiments_missing_from, read_fault_log, read_proxy_runs

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


def wrangle_proxy_points(xyc_dir: Path):
    """
    Saves the proxy-point values recorded in the fault generator's chart files to ``data/proxy_points.parquet``.
    For internal CSIRO use only.

    :param xyc_dir: folder of the fault generator's ``FaultGen *.xyc`` chart files.
    """
    read_proxy_runs(xyc_dir).to_parquet("data/proxy_points.parquet", index=False)


def add_logged_experiments(log_files: list, start_year: int):
    """
    Appends fault experiments recorded in fault-generator logs but missing from ``data/fault-experiments.parquet``.
    Run after ``wrangle_data()``, which rewrites that file from the spreadsheet. For internal CSIRO use only.

    :param log_files: the fault generator's ``log4jOutput.txt`` files.
    :param start_year: year in which the logs start, for events logged before their first full date.
    """
    gt = pd.read_parquet("data/fault-experiments.parquet")
    logged = pd.concat([read_fault_log(f, start_year) for f in log_files], ignore_index=True)
    missing = experiments_missing_from(gt, logged.drop_duplicates(["AHU", "Fault Type", "Start Datetime"]))
    missing.index = range(gt.index.max() + 1, gt.index.max() + 1 + len(missing))
    logger.info(f"Appending {len(missing)} logged experiments missing from the ground truth")
    pd.concat([gt, missing]).to_parquet("data/fault-experiments.parquet", coerce_timestamps="us", allow_truncated_timestamps=True)


#: Hand-entered fan belt slip experiments dated 17 Sep 2013 ran on 16 Sep 2013, according to the fault generator's log.
MISDATED_FAN_SLIP = (pd.Timestamp("2013-09-17"), pd.Timestamp("2013-09-16"))

#: The fault generator set AHU 9's RA damper proxy to 100% at this time, then a hand-entered override of the same proxy replaced it.
AHU9_RA_DAMPER_SET = pd.Timestamp("2013-09-18 09:04:35")


def correct_hand_entered_experiments():
    """
    Corrects hand-entered rows of ``data/fault-experiments.parquet`` that the fault generator's log contradicts or leaves incomplete.
    Run after ``add_logged_experiments()``. For internal CSIRO use only.
    """
    gt = pd.read_parquet("data/fault-experiments.parquet")
    recovered = "(recovered from fault-generator log)"
    slip = gt[gt["Fault Type"] == "SA Fan Belt Slip"]
    is_logged = slip["Fault Description"].str.endswith(recovered, na=False)
    hand = slip[~is_logged & (slip["Start Datetime"].dt.normalize() == MISDATED_FAN_SLIP[0])]
    logged = slip[is_logged & (slip["Start Datetime"].dt.normalize() == MISDATED_FAN_SLIP[1])]
    for i, row in hand.iterrows():
        match = logged[(logged.AHU == row.AHU) & (logged.Magnitude.abs() == abs(row.Magnitude))]
        if match.empty:  # set by hand in the BMS rather than by the fault generator, so the log can't date it
            continue
        if len(match) > 1:
            raise ValueError(f"Several logged fan belt slips match hand-entered row {i}")
        times = ["Start Datetime", "End Datetime", "Duration (h:mm)"]
        gt.loc[i, times] = match[times].iloc[0].values
        gt.loc[i, "Fault Description"] = match["Fault Description"].iloc[0].replace(recovered, "(times from fault-generator log)")
        gt = gt.drop(match.index)

    if not (gt["Start Datetime"] == AHU9_RA_DAMPER_SET).any():
        same_proxy = gt[(gt.AHU == 9) & (gt["Fault Type"] == "RA Damper Stuck Open")]
        end = same_proxy.loc[same_proxy["Start Datetime"] > AHU9_RA_DAMPER_SET, "Start Datetime"].min()
        if pd.isna(end) or end.normalize() != AHU9_RA_DAMPER_SET.normalize():
            raise ValueError(f"No hand-entered AHU 9 RA damper fault follows {AHU9_RA_DAMPER_SET} on the same day")
        description = f"{AHU9_RA_DAMPER_SET:%a %b %d %H:%M:%S EST %Y} - set - 100.0, ended by the next hand-entered fault {recovered}"
        gt.loc[gt.index.max() + 1] = [9, (pd.Timestamp(0) + (end - AHU9_RA_DAMPER_SET)).time(), 100.0, "RA Damper Stuck Open",
                                      "RA Damper Stuck Open 100.0", description, AHU9_RA_DAMPER_SET, end]
    gt.to_parquet("data/fault-experiments.parquet", coerce_timestamps="us", allow_truncated_timestamps=True)


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
