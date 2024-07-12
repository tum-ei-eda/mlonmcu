import sys
import subprocess
from pathlib import Path
import configparser

config = configparser.ConfigParser()


ini_path = Path("benchmarks.ini")

config.read(ini_path)

benchmarks = config.sections()

assert len(sys.argv) >= 3
name = sys.argv[1]
out_dir = sys.argv[2]
extra_args = sys.argv[3:]

assert name in benchmarks
data = config[name]

script = data["script"]
args = data["args"].split(" ")

# Path(out_dir).mkdir(exist_ok=True)

args = ["python", script, *args, "--out", out_dir, *extra_args]
print("Executing:", " ".join(args))
subprocess.run(args, check=True)
