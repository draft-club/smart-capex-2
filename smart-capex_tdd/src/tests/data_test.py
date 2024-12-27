"""
Module containing fake data for preprocessing test
"""
import os
import pandas as pd
from src.d00_conf.conf import conf, conf_loader
from src.d01_utils.utils import parse_arguments

#args = parse_arguments()
conf_loader('fake_country_e2e.json')

path_to_df = {
    os.path.join(conf['PATH']['RAW_DATA'],
                 'df_sites.csv'):
    pd.DataFrame({
        'site_id': ['AFO001', 'AFO001', 'AFO001', 'AFO001', 'AFO001'],
        'latitude': [32.2072, 32.2072, 32.2072, 32.2072, 32.2072],
        'longitude': [-6.54033, -6.54033, -6.54033, -6.54033, -6.54033],
        'commune': ['Afourar', 'Afourar', 'Afourar', 'Afourar', 'Afourar'],
        'ville': ['Afourar', 'Afourar', 'Afourar', 'Afourar', 'Afourar'],
        'province': ['Azilal', 'Azilal', 'Azilal', 'Azilal', 'Azilal'],
        'region': ['Béni Mellal-Khénifra', 'Béni Mellal-Khénifra', 'Béni Mellal-Khénifra',
                   'Béni Mellal-Khénifra', 'Béni Mellal-Khénifra'],
        'site_name': ['AFO001', 'AFO001', 'AFO001', 'AFO001', 'AFO001'],
        'sector': ['R', 'S', 'T', 'U', 'V'],
        'cell_name': ['AFO001R', 'AFO001S', 'AFO001T', 'AFO001U', 'AFO001V'],
        'cell_id': [30016.0, 30017.0, 30018.0, 30010.0, 30011.0],
        'cell_lac': [50700.0, 50700.0, 50700.0, 50700.0, 50700.0],
        'cell_tech': ['3G', '3G', '3G', '3G', '3G'],
        'azimuth': [60.0, 180.0, 300.0, 60.0, 180.0],
        'horizantal_beam_width': [65.0, 65.0, 65.0, 65.0, 65.0],
        'vertical_beam_width': [7.0, 7.0, 7.0, 7.0, 7.0],
        'downtilt': [2.0, 2.0, 2.0, 2.0, 2.0],
        'cell_band': ['U2100-F3', 'U2100-F3', 'U2100-F3', 'U2100-F1', 'U2100-F1']}
)}
