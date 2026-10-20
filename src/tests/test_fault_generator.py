"""Tests for reconstructing fault experiments and proxy values from fault-generator records."""

import bz2
from datetime import time
from pathlib import Path

import pandas as pd

from hvac_fdd_dataset.fault_generator import experiments_missing_from, read_chart, read_fault_log

POINT = "config/Drivers/Bacnet/CS.EC.PXM22/points/CS.EC.OFF.AHU"


def _event(day_time: str, action: str) -> str:
    """Formats a fault-generator log line for a set or reset event."""
    return f"INFO  - {day_time}.000 - {action} (FaultGen.java:275) main"


def _fault(ahu: int, fault_type: str, start: str, end: str) -> dict:
    """Builds a minimal ground-truth row."""
    return {"AHU": ahu, "Fault Type": fault_type, "Start Datetime": pd.Timestamp(start), "End Datetime": pd.Timestamp(end)}


def test_read_fault_log_keeps_completed_faults_only(tmp_path: Path):
    """Logged faults become ground-truth rows; seconds-long tests and overrides with no logged release are dropped."""
    log = tmp_path / "log4jOutput.txt"
    log.write_text("\n".join([
        "INFO  - 08/01 10:36:22.760 - Sleeping for 15 minutes (until Thu Jan 08 10:51:22 EST 2015) [FaultGen.java:253]",
        _event("08/01 10:40:00", f"Setting {POINT} 09.CHW VALVE PRX to 90.0"),
        _event("08/01 10:40:02", f"Resetting {POINT} 09.CHW VALVE PRX"),
        _event("08/01 10:51:23", f"Setting {POINT} 10.RA DAMP PRX to 90.0"),
        _event("08/01 11:21:25", f"Resetting {POINT} 10.RA DAMP PRX"),
        _event("08/01 13:57:58", f"Setting {POINT} 09.ZONE TEMP OFS to 0.5"),
        _event("08/01 14:28:00", f"Setting {POINT} 09.ZONE TEMP OFS to 0"),
        _event("08/01 15:00:00", f"Setting {POINT} 09.OA DAMP PRX to 50.0"),
        _event("09/01 09:00:00", f"Resetting {POINT} 09.OA DAMP PRX"),
    ]))

    gt = read_fault_log(log)

    assert gt[["AHU", "Fault Type", "Magnitude"]].values.tolist() == [[10, "RA Damper Stuck Open", 90.0], [9, "Zone Average Temp Drift", 0.5]]
    assert gt["Start Datetime"].tolist() == [pd.Timestamp("2015-01-08 10:51:23"), pd.Timestamp("2015-01-08 13:57:58")]
    assert gt["Duration (h:mm)"].tolist() == [time(0, 30, 2), time(0, 30, 2)]


def test_experiments_missing_from_ignores_recorded_faults():
    """A logged fault overlapping a ground-truth fault of the same AHU and fault type counts as recorded."""
    gt = pd.DataFrame([_fault(9, "OA Damper Stuck Open", "2013-09-13 08:08", "2013-09-13 10:09")])
    logged = pd.DataFrame([
        _fault(9, "OA Damper Stuck Open", "2013-09-13 08:09", "2013-09-13 10:09"),
        _fault(10, "OA Damper Stuck Open", "2013-09-13 08:09", "2013-09-13 10:09"),
        _fault(9, "RA Damper Stuck Open", "2013-09-13 08:09", "2013-09-13 10:09"),
    ])

    assert experiments_missing_from(gt, logged)[["AHU", "Fault Type"]].values.tolist() == [[10, "OA Damper Stuck Open"], [9, "RA Damper Stuck Open"]]


def test_read_chart_reads_series_in_local_time(tmp_path: Path):
    """Chart series are read with UTC and Sydney local times, despite raw Windows-1252 bytes in the chart's XML."""
    xml = (
        "<au.csiro.ie.vis.jfreechart.XYChart><perMill>\x89</perMill><org.jfree.data.xy.XYSeries>"
        "<key>config/Drivers/Bacnet/CS.EC.MBC25/points/CS.EC.OFF.AHU 09.OA DAMP PRX</key><data><org.jfree.data.xy.XYDataItem>"
        '<x class="long">1379023738618</x><y class="double">100.0</y></org.jfree.data.xy.XYDataItem></data>'
        "</org.jfree.data.xy.XYSeries></au.csiro.ie.vis.jfreechart.XYChart>"
    )
    chart = tmp_path / "chart.xyc"
    chart.write_bytes(bz2.compress(xml.encode("latin-1")))

    df = read_chart(chart)

    assert df[["controller", "point", "value"]].values.tolist() == [["CS.EC.MBC25", "CS.EC.OFF.AHU 09.OA DAMP PRX", 100.0]]
    assert df.time_local.iloc[0] == pd.Timestamp("2013-09-13 08:08:58.618")
