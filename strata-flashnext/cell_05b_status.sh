#!/bin/sh
# CELL 5b — setup progress (re-run freely). Success = the log shows the server
# starting / serving on :8080.
tail -15 /kaggle/working/strata_setup.log
echo "---"
pgrep -af "strata|setup.py" | head -3 || echo "(no setup process running)"
