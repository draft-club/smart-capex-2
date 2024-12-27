"""
Module containing fake data for capacity test
TODO: add data for 3G and 4G
"""

import pandas as pd

df_time_advance_2g = pd.DataFrame({
    'PERIOD_START_TIME': ['01.25.2021 00:00:00', '01.25.2021 00:00:00', '01.25.2021 00:00:00',
                          '01.25.2021 00:00:00','01.25.2021 00:00:00', '01.25.2021 00:00:00',
                          '01.25.2021 00:00:00', '01.25.2021 00:00:00','01.25.2021 00:00:00'],
    'BSC name': ['BSC_AAA_1', 'BSC_AAA_1', 'BSC_AAA_1', 'BSC_AAA_1', 'BSC_AAA_1', 'BSC_AAA_1',
                 'BSC_AAA_1', 'BSC_AAA_1','BSC_AAA_1'],
    'BCF name': ['OCI0005_AAA', 'OCI0005_AAA', 'OCI0005_AAA', 'OCI0005_AAA', 'OCI0005_AAA',
                 'OCI0005_AAA','OCI0006_S_AAA_CIT_G9G18U9L8L26', 'OCI0006_S_AAA_CIT_G9G18U9L8L26',
                 'OCI0006_S_AAA_CIT_G9G18U9L8L26'],
    'BTS name': ['OCI0005_AAA_G18-1', 'OCI0005_AAA_G18-2', 'OCI0005_AAA_G18-3', 'OCI0005_AAA_G9-1',
                 'OCI0005_AAA_G9-2','OCI0005_AAA_G9-3', 'OCI0006_S_AAA_CIT_G18-1',
                 'OCI0006_S_AAA_CIT_G18-2', 'OCI0006_S_AAA_CIT_G18-3'],
    '550m': [10000] * 9,
    '1100m': [10000] * 9,
    '1650m': [10000] * 9,
    '2200m': [10000] * 9,
    '2750m': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    '3300m': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    '3850m': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    '4400m': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    '4950m': [10000] * 9,
    '5500m': [0, 0, 0, 0, 0, 0, 0, 0, 0]
})
