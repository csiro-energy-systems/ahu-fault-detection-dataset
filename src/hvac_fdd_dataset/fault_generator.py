"""
Readers for the records kept by CSIRO's fault-generation software: its chart files (``*.xyc``) and its log file (``log4jOutput.txt``).
The source files are internal to CSIRO; these readers document how the published files were derived from them.
"""

import bz2
import re
import xml.etree.ElementTree as ET  # nosec B405 - only parses CSIRO's own archived chart files
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

LOCAL_TZ = "Australia/Sydney"

#: Ground-truth fault type for each overridden BMS point (suffix after ``CS.EC.OFF.AHU nn.``), as already used in ``fault-experiments.parquet``.
FAULT_TYPES = {
    "OA DAMP PRX": "OA Damper Stuck Open",
    "RA DAMP PRX": "RA Damper Stuck Open",
    "HW VALVE PRX": "HW Valve Proxy Stuck Open",
    "HW VALVE": "HW Valve Stuck Open",  # direct override of the controller's command, before proxies were added
    "CHW VALVE PRX": "CHW Valve Proxy Stuck Open",
    "Z2 TEMP PRX": "Zone Temp Drift",
    "SA FANSPD OFS": "SA Fan Belt Slip",
    "ZONE TEMP OFS": "Zone Average Temp Drift",
    "LEAK.SIM.DMP": "SA Duct Leakage",
}

#: Points the BMS doesn't control, so the fault generator ended their faults by writing 0 rather than releasing the override.
ZERO_ENDED_POINTS = {"SA FANSPD OFS", "ZONE TEMP OFS", "LEAK.SIM.DMP"}

#: Shorter overrides in the log were connectivity tests, not fault experiments.
MIN_FAULT_DURATION = pd.Timedelta("1min")

#: The fault generator scheduled faults of up to 2 hours; longer intervals mean the log lacks the release (e.g. it was done by hand in the BMS).
MAX_FAULT_DURATION = pd.Timedelta("3h")

GROUND_TRUTH_COLUMNS = ["AHU", "Duration (h:mm)", "Magnitude", "Fault Type", "Fault Name", "Fault Description", "Start Datetime", "End Datetime"]

# The chart files have no XML declaration and contain raw Windows-1252 bytes and control characters in their number-format metadata.
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_POINT_PREFIX = re.compile(r"^config/Drivers/\w+/(?P<controller>[^/]+)/points/(?:FDD\.BMS/)?")
_RUN_FILE = re.compile(r"\.points\.(?P<point>.+?) (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) ")
_LOG_EVENT = re.compile(
    r"^\w+ +- (?P<day>\d\d)/(?P<month>\d\d) (?P<time>\d\d:\d\d:\d\d)\.\d+ - (?P<action>Setting|Resetting) "
    r"config/Drivers/\w+/[^/]+/points/(?:FDD\.BMS/)?CS\.EC\.OFF\.AHU (?P<ahu>\d+)\.(?P<point>.+?)(?: to (?P<value>-?[\d.]+))? \("
)
_LOG_YEAR = re.compile(r"\w{3} \w{3} \d\d \d\d:\d\d:\d\d \w+ (?P<year>\d{4})")
_JAVA_DATE = "%a %b %d %H:%M:%S EST %Y"  # the fault generator's timestamp format, used in existing ``Fault Description`` values


def read_chart(xyc_file: Path) -> pd.DataFrame:
    """
    Reads every data series in a fault-generator chart file.

    :param xyc_file: bzip2-compressed XStream XML of an ``au.csiro.ie.vis.jfreechart.XYChart``.
    :return: ``time_utc``, ``time_local`` (Sydney time, tz-naive, as in the AHU data), ``controller``, ``point`` and ``value`` columns.
    """
    text = _XML_ILLEGAL.sub("", bz2.decompress(xyc_file.read_bytes()).decode("latin-1"))
    rows = []
    for series in ET.fromstring(text).iter("org.jfree.data.xy.XYSeries"):  # nosec B314
        key = series.findtext("key")
        if key is None:  # XStream back-reference to a series already read
            continue
        m = _POINT_PREFIX.match(key)
        controller, point = (m["controller"], key[m.end() :]) if m else ("", key)
        for item in series.iterfind("data/org.jfree.data.xy.XYDataItem"):
            y = item.findtext("y")
            rows.append((int(float(item.findtext("x"))), controller, point, float(y) if y else None))
    df = pd.DataFrame(rows, columns=["epoch_ms", "controller", "point", "value"])
    utc = pd.to_datetime(df.pop("epoch_ms"), unit="ms", utc=True)
    df.insert(0, "time_local", utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None))
    df.insert(0, "time_utc", utc)
    return df


def read_proxy_runs(xyc_dir: Path) -> pd.DataFrame:
    """
    Reads the proxy-point (``... PRX``) series from every fault-generator chart file in a folder.

    :param xyc_dir: folder of ``FaultGen *.xyc`` chart files.
    :return: :func:`read_chart` columns plus ``run_file`` and ``overridden_point`` (the point the run overrode).
    """
    frames = []
    for f in sorted(xyc_dir.glob("*.xyc")):
        if f.stat().st_size == 0:  # aborted runs saved an empty chart
            continue
        df = read_chart(f)
        frames.append(df[df.point.str.endswith("PRX")].assign(run_file=f.name, overridden_point=_RUN_FILE.search(f.name)["point"]))
    return pd.concat(frames, ignore_index=True)


def read_fault_log(log_file: Path, start_year: Optional[int] = None) -> pd.DataFrame:
    """
    Reconstructs fault experiments from the set and reset events in a fault-generator log.

    :param log_file: the fault generator's ``log4jOutput.txt``.
    :param start_year: year of the log's first events, needed when they precede its first full date.
    :return: one row per fault experiment, in the same schema as ``data/fault-experiments.parquet``.
    """
    events = _read_log_events(log_file, start_year)
    rows = []
    for (ahu, point), ev in events.groupby(["AHU", "Point"], sort=False):
        if point not in FAULT_TYPES:
            continue
        ev = ev.reset_index(drop=True)
        for i, start in ev.iloc[:-1].iterrows():
            if start.Action != "Setting" or (point in ZERO_ENDED_POINTS and start.Value == 0):
                continue
            end = ev.iloc[i + 1]
            duration = end.Datetime - start.Datetime
            if duration > MAX_FAULT_DURATION:
                logger.warning(f"Skipping AHU {ahu} {point} set to {start.Value} at {start.Datetime}: no release logged until {end.Datetime}")
            elif duration >= MIN_FAULT_DURATION:
                rows.append(_ground_truth_row(ahu, FAULT_TYPES[point], start, end))
    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS).sort_values("Start Datetime", ignore_index=True)


def experiments_missing_from(ground_truth: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    """Returns the ``experiments`` that overlap no ``ground_truth`` experiment of the same AHU and fault type."""

    def is_recorded(r: pd.Series) -> bool:
        g = ground_truth[(ground_truth.AHU == r.AHU) & (ground_truth["Fault Type"] == r["Fault Type"])]
        return bool(((g["Start Datetime"] < r["End Datetime"]) & (g["End Datetime"] > r["Start Datetime"])).any())

    return experiments[~experiments.apply(is_recorded, axis=1)]


def _read_log_events(log_file: Path, start_year: Optional[int]) -> pd.DataFrame:
    """Reads ``Setting`` and ``Resetting`` events from a fault-generator log, dating each from the latest full date logged before it."""
    events, year = [], None if start_year is None else str(start_year)
    for line in log_file.read_text(encoding="latin-1").splitlines():
        if m := _LOG_YEAR.search(line):
            year = m["year"]
        if m := _LOG_EVENT.match(line):
            if year is None:
                raise ValueError(f"No full date logged before event: {line}")
            when = pd.Timestamp(f"{year}-{m['month']}-{m['day']} {m['time']}")
            events.append((when, int(m["ahu"]), m["point"].strip(), m["action"], float(m["value"]) if m["value"] else None))
    return pd.DataFrame(events, columns=["Datetime", "AHU", "Point", "Action", "Value"])


def _ground_truth_row(ahu: int, fault_type: str, start: pd.Series, end: pd.Series) -> list:
    """Builds a ground-truth row for a fault set by ``start`` and ended by ``end``, matching the fault generator's own naming."""
    end_text = f"reset - {start.Value}" if end.Action == "Resetting" else f"set - {end.Value}"
    events = f"{start.Datetime.strftime(_JAVA_DATE)} - set - {start.Value} | {end.Datetime.strftime(_JAVA_DATE)} - {end_text}"
    return [
        ahu,
        (pd.Timestamp(0) + (end.Datetime - start.Datetime)).time(),
        start.Value,
        fault_type,
        f"{fault_type} {start.Value}",
        f"{events} (recovered from fault-generator log)",
        start.Datetime,
        end.Datetime,
    ]
