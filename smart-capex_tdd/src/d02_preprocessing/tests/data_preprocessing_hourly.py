import os
import pandas as pd

from src.d00_conf.conf import conf, conf_loader
conf_loader('fake_country_e2e.json')
path_to_df = {
    os.path.join(conf["PATH"]["RAW_DATA"], 'OSS_4G', 'OMA_EXTRACT_COMPLET.csv'):
        pd.DataFrame({
            'Cell FDD TDD Indication': ["CELL_FDD", "CELL_FDD"]
        })
}
