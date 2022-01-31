import glob
import os

def find_newest_report():
    home = os.getenv("MLONMCU_HOME")
    list_of_files = glob.glob(os.path.join(home, "out", "*")) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

import pandas as pd

def tabularize_latest_report():
    report_file = find_newest_report()
    df = pd.read_csv(report_file, sep=",")
    return df
