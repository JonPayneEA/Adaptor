"""
pi_xml.py - FEWS Published Interface (PI) XML reader/writer utilities.

Handles:
  - Reading PI-XML timeseries into pandas DataFrames
  - Writing pandas DataFrames back to PI-XML timeseries
  - Reading PI run-info XML (run_info.xml)
  - Writing PI diagnostics XML (diag.xml)

Conforms to the PI-XML schema:
  http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# PI XML namespace
PI_NS = "http://www.wldelft.nl/fews/PI"
PI_NSMAP = {"pi": PI_NS}

# Diagnostics log levels (FEWS convention)
DIAG_LEVELS = {
    "fatal": 0,
    "error": 1,
    "warning": 2,
    "info": 3,
    "debug": 4,
}


# =============================================================================
# Run Info
# =============================================================================
class RunInfo:
    """Parsed contents of a FEWS run_info.xml / pi_run.xml file."""

    def __init__(self, work_dir: str, start_time: datetime, end_time: datetime,
                 time_zero: datetime, time_step_seconds: int,
                 input_dir: str, output_dir: str, state_dir: str = ""):
        self.work_dir = work_dir
        self.start_time = start_time
        self.end_time = end_time
        self.time_zero = time_zero
        self.time_step_seconds = time_step_seconds
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.state_dir = state_dir


def parse_run_info(run_info_path: str) -> RunInfo:
    """Parse FEWS pi_run.xml / run_info.xml file.

    The General Adapter exports a run file that tells the adapter
    about the run period, directories, and time step.
    """
    tree = ET.parse(run_info_path)
    root = tree.getroot()

    # Handle namespaced or non-namespaced elements
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def _find(tag):
        el = root.find(f"{ns}{tag}")
        if el is None:
            el = root.find(tag)
        return el

    def _parse_dt(el):
        d = el.get("date")
        t = el.get("time", "00:00:00")
        return datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")

    start_el = _find("startDateTime")
    end_el = _find("endDateTime")
    t0_el = _find("time0")

    time_step_el = _find("timeStep")
    if time_step_el is not None:
        unit = time_step_el.get("unit", "second")
        mult = int(time_step_el.get("multiplier", "86400"))
        if unit == "hour":
            ts_seconds = mult * 3600
        elif unit == "minute":
            ts_seconds = mult * 60
        else:
            ts_seconds = mult
    else:
        ts_seconds = 86400  # default daily

    work_dir_el = _find("workDir")
    input_dir_el = _find("inputDir")  # not always present
    output_dir_el = _find("outputDir")  # not always present

    return RunInfo(
        work_dir=work_dir_el.text if work_dir_el is not None else ".",
        start_time=_parse_dt(start_el),
        end_time=_parse_dt(end_el),
        time_zero=_parse_dt(t0_el) if t0_el is not None else _parse_dt(start_el),
        time_step_seconds=ts_seconds,
        input_dir=input_dir_el.text if input_dir_el is not None else "input",
        output_dir=output_dir_el.text if output_dir_el is not None else "output",
    )


# =============================================================================
# Read PI-XML Timeseries
# =============================================================================
def read_pi_timeseries(xml_path: str, missing_value: float = -999.0) -> dict:
    """Read a FEWS PI-XML timeseries file into a dict of DataFrames.

    Returns
    -------
    dict
        Keys are (locationId, parameterId) tuples.
        Values are pd.DataFrame with DatetimeIndex and a 'value' column.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Read timezone offset
    tz_el = root.find(f"{ns}timeZone")
    tz_offset = float(tz_el.text) if tz_el is not None else 0.0

    result = {}

    for series in root.findall(f"{ns}series"):
        header = series.find(f"{ns}header")
        loc_id = header.find(f"{ns}locationId").text
        param_id = header.find(f"{ns}parameterId").text

        miss_el = header.find(f"{ns}missVal")
        miss_val = float(miss_el.text) if miss_el is not None else missing_value

        dates = []
        values = []
        flags = []

        for event in series.findall(f"{ns}event"):
            d = event.get("date")
            t = event.get("time", "00:00:00")
            dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
            # Adjust for timezone to get UTC
            dt = dt - timedelta(hours=tz_offset)

            val = float(event.get("value"))
            flag = int(event.get("flag", "0"))

            dates.append(dt)
            values.append(val if val != miss_val else float("nan"))
            flags.append(flag)

        df = pd.DataFrame({"value": values, "flag": flags}, index=pd.DatetimeIndex(dates))
        df.index.name = "datetime"
        result[(loc_id, param_id)] = df

    return result


# =============================================================================
# Write PI-XML Timeseries
# =============================================================================
def write_pi_timeseries(
    data: dict,
    output_path: str,
    missing_value: float = -999.0,
    time_zone: float = 0.0,
    time_step_unit: str = "second",
    time_step_multiplier: int = 86400,
    forecast_time: Optional[datetime] = None,
):
    """Write a dict of DataFrames to FEWS PI-XML timeseries file.

    Parameters
    ----------
    data : dict
        Keys are (locationId, parameterId) tuples.
        Values are pd.DataFrame with DatetimeIndex and 'value' column.
    output_path : str
        Path for the output XML file.
    missing_value : float
        Missing value placeholder.
    time_zone : float
        Timezone offset from UTC.
    time_step_unit : str
        e.g. 'second', 'hour'.
    time_step_multiplier : int
        Multiplier for time step unit.
    forecast_time : datetime, optional
        Forecast T0 timestamp.
    """
    root = ET.Element("TimeSeries")
    root.set("xmlns", PI_NS)
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:schemaLocation",
             f"{PI_NS} http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd")
    root.set("version", "1.2")

    tz_el = ET.SubElement(root, "timeZone")
    tz_el.text = str(time_zone)

    for (loc_id, param_id), df in data.items():
        series = ET.SubElement(root, "series")
        header = ET.SubElement(series, "header")

        ET.SubElement(header, "type").text = "instantaneous"
        ET.SubElement(header, "locationId").text = str(loc_id)
        ET.SubElement(header, "parameterId").text = str(param_id)

        ts_el = ET.SubElement(header, "timeStep")
        ts_el.set("unit", time_step_unit)
        ts_el.set("multiplier", str(time_step_multiplier))

        if len(df) > 0:
            idx = df.index
            start_dt = idx[0] + timedelta(hours=time_zone)
            end_dt = idx[-1] + timedelta(hours=time_zone)

            start_el = ET.SubElement(header, "startDate")
            start_el.set("date", start_dt.strftime("%Y-%m-%d"))
            start_el.set("time", start_dt.strftime("%H:%M:%S"))

            end_el = ET.SubElement(header, "endDate")
            end_el.set("date", end_dt.strftime("%Y-%m-%d"))
            end_el.set("time", end_dt.strftime("%H:%M:%S"))
        else:
            # Empty series
            now = datetime.utcnow()
            for tag in ("startDate", "endDate"):
                el = ET.SubElement(header, tag)
                el.set("date", now.strftime("%Y-%m-%d"))
                el.set("time", now.strftime("%H:%M:%S"))

        if forecast_time is not None:
            fc_el = ET.SubElement(header, "forecastDate")
            fc_dt = forecast_time + timedelta(hours=time_zone)
            fc_el.set("date", fc_dt.strftime("%Y-%m-%d"))
            fc_el.set("time", fc_dt.strftime("%H:%M:%S"))

        ET.SubElement(header, "missVal").text = str(missing_value)

        for dt, row in df.iterrows():
            val = row["value"]
            event = ET.SubElement(series, "event")
            local_dt = dt + timedelta(hours=time_zone)
            event.set("date", local_dt.strftime("%Y-%m-%d"))
            event.set("time", local_dt.strftime("%H:%M:%S"))

            if pd.isna(val):
                event.set("value", str(missing_value))
            else:
                event.set("value", f"{val:.6f}")

            if "flag" in row and not pd.isna(row.get("flag")):
                event.set("flag", str(int(row["flag"])))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


# =============================================================================
# Diagnostics
# =============================================================================
class DiagnosticsWriter:
    """Accumulates log messages and writes FEWS PI diagnostics XML.

    FEWS expects one diagnostics file per adapter phase (pre/run/post).
    Levels: 0=fatal, 1=error, 2=warning, 3=info, 4=debug.
    """

    def __init__(self):
        self.messages = []

    def log(self, level: int, message: str):
        self.messages.append((level, message))

    def fatal(self, msg):
        self.log(0, msg)

    def error(self, msg):
        self.log(1, msg)

    def warn(self, msg):
        self.log(2, msg)

    def info(self, msg):
        self.log(3, msg)

    def debug(self, msg):
        self.log(4, msg)

    def has_errors(self) -> bool:
        return any(lvl <= 1 for lvl, _ in self.messages)

    def write(self, output_path: str):
        """Write accumulated messages to PI diagnostics XML."""
        root = ET.Element("Diag")
        root.set("xmlns", PI_NS)
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:schemaLocation",
                 f"{PI_NS} http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_diag.xsd")
        root.set("version", "1.2")

        for level, message in self.messages:
            line = ET.SubElement(root, "line")
            line.set("level", str(level))
            line.set("description", message)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)
