#!/usr/bin/env python3
"""Print the software/hardware facts that must be logged with every run (paste the output into the run record)."""
import os, platform, sys
import numpy, pandas
print("python   :", sys.version.replace("\n", " "))
print("numpy    :", numpy.__version__)
print("pandas   :", pandas.__version__)
try:
    import matplotlib; print("matplotlib:", matplotlib.__version__)
except Exception as e:
    print("matplotlib: not installed", e)
print("platform :", platform.platform())
print("machine  :", platform.machine(), "| processor:", platform.processor())
print("cpu count:", os.cpu_count(), "(logical)")
