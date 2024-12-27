from datetime import datetime
import requests
import scripts_config
import time
import os

# get the timestamp from last week
a_week = datetime.timedelta(days=7).total_seconds()
today = time.time()
a_week_ago = today - a_week

if __name__ == '__main__':
    for file in os.listdir(scripts_config.data_folder):
        ctime = os.path.getctime(scripts_config.data_folder+"/"+file)
        # validate files created no more than a week ago
        if ctime > a_week_ago:
            # get file type from file name
            file_type = file.split('hourly')[0][:-1].replace('_', '').replace('loss', '').replace('counter', '').lower()
            requests.get(scripts_config.api_url,
                         json={'path': '/data/{}'.format(file),
                               'file_type': file_type,
                               'delimiter': ',',
                               'n_examples': 5,
                               'init': False
                               }
                         )
