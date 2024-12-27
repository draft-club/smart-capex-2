# crontab -e
SHELL=/bin/bash

(crontab -l 2>/dev/null; echo "0  6  *  *  2 root python3 $(realpath "$0")/validations_request.py") | crontab -