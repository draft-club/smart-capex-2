from kfp.dsl import (Dataset,
                     Input,
                     component)
from utils.config import pipeline_config


@component(base_image=pipeline_config["base_image"])
def compute_npvs(project_id: str,
                 location: str,
                 wacc: int,
                 npv_of_the_upgrade_table_id: str,
                 df_cash_flow_data_input: Input[Dataset]):
    """It computes the Net Present Value (NPV) of cash flows resulting from an upgrade 

    Args:
        project_id (str): It holds the project_id of GCP
        location (str): It holds the location assigned to the project on GCP
        wacc (int): It holds the Weighted Average Cost of Capital (WACC) used for discounting cash flows
        npv_of_the_upgrade_table_id (str): It holds the resource name on BigQuery
        df_cash_flow_data_input (Input[Dataset]): It holds cash flow data
    """
    # imports
    import pandas as pd
    import numpy as np
    import numpy_financial as npf
    import pandas_gbq

    def compute_npv_aux(df, wacc):
        df = df.reset_index()
        df = df[['year', 'increase_cash_flow_due_to_the_upgrade']].sort_values(
            by='year',
            ascending=True)
        return npv_since_2nd_years(values=df['increase_cash_flow_due_to_the_upgrade'].values,
                                   rate=wacc / 100)

    def npv_since_2nd_years(rate, values):
        """
        Returns the NPV (Net Present Value) of a cash flow series discount since the first year

        """
        values = np.asarray(values)
        values = np.nan_to_num(values)
        return (values / (1 + rate) ** np.arange(1, len(values) + 1)).sum(axis=0)

    # Load Data
    df_cash_flow = pd.read_parquet(df_cash_flow_data_input.path)

    nb_year = df_cash_flow['cash_flow_year'].unique()
    nb_year = np.sort(nb_year)

    df_cash_flow_discount = df_cash_flow[~(df_cash_flow['cash_flow_year'].isin([nb_year[0]]))]
    df_cash_flow_no_discount = df_cash_flow[df_cash_flow['cash_flow_year'].isin([nb_year[0]])]

    df_cash_flow_no_discount_npv = (df_cash_flow_no_discount.groupby(['site_id', 'cell_band'])
                                    [['increase_cash_flow_due_to_the_upgrade']].sum().reset_index())
    df_cash_flow_no_discount_npv.columns = ['site_id', 'cell_band', 'capex_cf_y1']

    df_cash_flow_discount_npv = (df_cash_flow_discount.groupby(['site_id', 'cell_band'])
                                 .apply(lambda x: compute_npv_aux(x, wacc=wacc)).reset_index())
    df_cash_flow_discount_npv.columns = ['site_id', 'cell_band', 'NPV_cf_y2']

    df_npv = df_cash_flow_discount_npv.merge(df_cash_flow_no_discount_npv, on=['site_id', 'cell_band'], how='left')

    df_npv['NPV'] = df_npv['capex_cf_y1'] + df_npv['NPV_cf_y2']
    df_npv.drop(['capex_cf_y1', 'NPV_cf_y2'], axis=1, inplace=True)
    df_npv.columns = ['site_id', 'cell_band', 'NPV']

    # Reorganize the final dataset
    df_cash_flow_pv = pd.pivot_table(df_cash_flow, values='increase_cash_flow_due_to_the_upgrade',
                                     index=['site_id', 'cell_band'],
                                     columns=['cash_flow_year'], aggfunc=np.sum)

    new_columns_names = []
    for i in df_cash_flow_pv.columns:
        new_columns_names.append('cash_flow_year_' + str(int(i)))

    df_cash_flow_pv.columns = new_columns_names
    df_cash_flow_pv.reset_index(inplace=True)

    df_opex_cost_year_pv = pd.pivot_table(df_cash_flow, values='opex_costs',
                                          index=['site_id', 'cell_band'],
                                          columns=['cash_flow_year'], aggfunc=np.sum)

    opex_columns_names = []
    for i in df_opex_cost_year_pv.columns:
        opex_columns_names.append('opex_cost_year_' + str(int(i)))

    df_opex_cost_year_pv.columns = opex_columns_names
    df_opex_cost_year_pv.reset_index(inplace=True)

    df = df_cash_flow[['site_id', 'cell_band']].drop_duplicates()

    df = df.merge(df_cash_flow_pv, on=['site_id', 'cell_band'], how='left')

    df = df.merge(df_opex_cost_year_pv, on=['site_id', 'cell_band'], how='left')

    # Merge with the npv
    df = df.merge(df_npv, on=['site_id', 'cell_band'],how='left')
    df['total_opex'] = df[[col for col in df.columns if col.startswith('opex')]].sum(axis=1)
    df['total_revenue'] = (df[[col for col in df.columns if col.startswith('cash_flow')]].sum(axis=1) +
            df[[col for col in df.columns if col.startswith('opex')]].sum(axis=1) - df['cash_flow_year_0'])

    df['EBITDA_Value'] = df['total_revenue'] - df['total_opex']
    df['EBITDA'] = df['EBITDA_Value'] / df['total_revenue']

    df_irr_columns = ['site_id', 'cell_band']
    df_irr_columns = df_irr_columns + new_columns_names
    df_irr_0_1 = df[df_irr_columns]
    df_irr_0_1['cash_flow_years_0_1'] = df_irr_0_1['cash_flow_year_0'] + df_irr_0_1['cash_flow_year_1']
    df_irr_0_1.drop(['cash_flow_year_0', 'cash_flow_year_1'], axis=1, inplace=True)
    df_irr_0_1 = df_irr_0_1.reindex(columns=['site_id', 'cell_band', 'cash_flow_years_0_1'] + df_irr_columns[4:])

    irr_columns = df_irr_0_1.columns[2:]
    df_irr_0_1 = df_irr_0_1[df_irr_0_1.cash_flow_years_0_1.notna()]
    df_irr_0_1 = df_irr_0_1.fillna(0)
    df_irr_0_1['IRR'] = df_irr_0_1[irr_columns].apply(npf.irr, axis=1)

    df_irr = df_irr_0_1[['site_id', 'cell_band', 'IRR']]

    df = df.merge(df_irr, on=['site_id', 'cell_band'], how='left')

    print("df shape: ", df.shape)
    print("columns", df.info())
    pandas_gbq.to_gbq(df, npv_of_the_upgrade_table_id, project_id=project_id,
                      location=location, if_exists='replace')
