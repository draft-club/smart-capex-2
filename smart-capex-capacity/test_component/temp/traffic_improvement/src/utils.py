## List of functions used during the project
import pandas as pd
from sqlalchemy.orm import sessionmaker
import time
import unicodedata
from contextlib import contextmanager
import os
#from src.d00_conf.conf import conf
from src.conf import conf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dateutil import rrule
import sqlalchemy
import getpass
import json

def saveSample(df, name, nrow = 1000):
    df.sample(n = nrow).to_csv(name, sep = ";")
    return None


""" General functions """
#function to count execution time of a function
@contextmanager
def timer(title):
    """Gives execution time of a function"""
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, cols=None, nan_as_category=True):
    original_columns = list(df.columns)

    if cols == None:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    else:
        categorical_columns = cols
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

#removes accent from a text variable
def remove_accent(text):
    """Remove tildes from a text variable"""
    try:
        text = np.unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
    return str(text)

## Include a prefix to columns
def insert_prefix_to_columns(df, prefix, columns_exclude):
    df_dict = df.columns.values
    for name in df_dict:
        if name in columns_exclude:
            continue
        df.rename(columns={name: prefix +  "_" + name}, inplace=True)
    return df


def get_week_period(year, week):
    year = str(year)
    week = str(week)
    if len(week)==1:
        week = '0'+ week
    return year + week

def get_yearmonth_from_week_period(week_period, week_day=0):
    week_period = str(week_period) + str(week_day)
    dd = datetime.strptime(week_period, "%Y%W%w")
    return str(dd.year) + str(dd.month).zfill(2)

def createDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return None

def getLastFile(path, pattern = ""):
    files = os.listdir(path)
    files = [f for f in files if pattern in f]
    files.sort()
    return files[-1]

def get_band(x):
    if len(x)==1:
        return x
    else:
        try:
            return x.split('-')[-2].split('_')[-1]
        except:
            return x

def check_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv(df, path, name, separator = conf['CSV']['SEP']):
    if ('.csv' in name) == False:
        name = name +'.csv'
    df.to_csv(os.path.join(path, name), sep = separator, index = False)
    return None

def chdir_path(directory):
    os.chdir(directory)

def write_model(model, path, name):
    if('.sav' in name) == False:
        name = name + '.sav'
    chdir_path(os.path.join(path, conf['EXEC_TIME']))
    pickle.dump(model, open(name, 'wb'))
    return None

def read_model(path, name):
    if ('.sav' in name) == False:
        name = name + '.sav'

    list_subfolders_with_paths = [f.name for f in os.scandir(os.path.join(conf['PATH']['MODELS'], 'activation_model'))
                                  if f.is_dir()]

    last_model = max(list_subfolders_with_paths)
    print("The last model is from : " + last_model)

    chdir_path(os.path.join(path, last_model))

    loaded_model = pickle.load(open(name, 'rb'))

    return loaded_model

def correct_bands_upgraded(row):
    if row == 'G9-L8':
        return 'L8'
    elif row == 'G9-L26':
        return 'L26'
    elif row == 'L26-U9':
        return 'L26'
    else:
        return row

def correct_tech_upgraded(row):
    if row == '2G-4G':
        return '4G'
    elif row == '4G-3G':
        return '4G'
    else:
        return row



def plot_prb(df_predicted, df_traffic_forecasting, df_affected_cells, site_id, cell_band):

    if (df_affected_cells[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1).any():
        if (df_traffic_forecasting[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1).any():
            if (df_predicted[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1).any():

                week_upgrade = df_affected_cells[(df_affected_cells[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1)]['week_of_the_upgrade'].unique().astype(int)

                if week_upgrade.shape[0]==1:

                    date_index = list(df_affected_cells[(df_affected_cells[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1)]['week_period']).index(week_upgrade)

                    y_pred = df_predicted[(df_predicted[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1)]['y_test']

                    list_date = list(df_affected_cells[(df_affected_cells[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1)]['week_period'])#[:date_index+9]
                    list_prb = list(df_affected_cells[(df_affected_cells[['site_id','cell_band']].values == [site_id,cell_band]).all(axis=1)]['cell_occupation_dl_percentage'])#[:date_index+9]
                    # list_prb.append(y_pred)

                    list_date_bis = [str(x) for x in list_date]

                    if len(list_date) == len(list_prb):
                        # df_plot = pd.DataFrame({'date' : list_date, 'cell_congestion' : list_prb})
                        # df_plot['cell_congestion'] = df_plot['cell_congestion'].astype(float)
                        # df_plot.plot(x='date', y='cell_congestion', kind='line')
                        fig, ax = plt.subplots()
                        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                        plt.plot(list_date_bis, list_prb, ls = 'solid')
                        plt.ylim([0,100])
                        plt.locator_params(axis='x', nbins=10)
                        plt.plot(list_date_bis, np.repeat(y_pred,len(list_date))) ##y_pred line
                        plt.plot(np.repeat(week_upgrade.astype(str), 2),[0,100])
                        plt.xticks(rotation=45)
                        # plt.plot(list_date[date_index:date_index+1], list_prb[date_index:date_index+1], "g*")
                        plt.title(site_id + " " + cell_band)
                        plt.xlabel('week_period')
                        plt.ylabel('congestion')
                        plt.show()
                    else:
                        print('Not working cell : {}'.format(site_id))
                        print('Len list_date : {}'.format(len(list_date)))
                        print('Len list_prb  : {}'.format(len(list_prb)))
                        print('Len date_index+8 : {}'.format(date_index+8))
                else :
                    print('Not working cell : {}'.format(site_id))
                    print("there is 2 upgrade week : {}".format(week_upgrade))
            else:
                print("Not in df_predicted")
        else:
            print("Not in df_traffic_forecasting")
    else:
        print("Not in df_affected_cells")
    return None


def weeks_between(start_date, end_date):
    weeks = rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date)
    return weeks.count()


def get_lag_between_two_week_periods(week_period_1, week_period_2):
    week_period_1, week_period_2 = str(week_period_1), str(week_period_2)

    year1 = int(week_period_1[:4])
    week1 = int(week_period_1[-2:])
    year2 = int(week_period_2[:4])
    week2 = int(week_period_2[-2:])
    return - (53*year1 + week1-(53*year2 +week2))

def convert_float_string(aux):
    if pd.isnull(aux):
        return aux
    else:
        try:
            aux1 = str(int(aux))
        except:
            aux1 = str(aux)
    return aux1

def remove_columns_started_with_unnamed(df):
    df = df.drop(
        df.columns[
            df.columns.str.startswith('Unnamed')], axis='columns')
    return df

def get_month_year_period(d):
    if pd.isnull(d):
        return d
    month = '{:02d}'.format(d.month)
    year = '{:04d}'.format(d.year)
    return year + month

def truncate_table(engine, table_name):
  Session = sessionmaker(bind=engine)
  session = Session()
  session.execute('''TRUNCATE TABLE ''' + table_name)
  session.commit()
  session.close()

def write_sql_database(df, table_name, if_exists='replace', delete_planif=False):
    user = getpass.getuser()
    path = os.path.join('home',user,'my.cnf')
    engine = sqlalchemy.create_engine('mysql+pymysql://',
                                      connect_args={'read_default_file': "/"+ path})
    if delete_planif:
        planification_id = df.planification_id.unique()[0]
        req = "delete from {} where planification_id={}".format(table_name, str(planification_id))
        with engine.connect() as con:
            con.execute(req)

    if if_exists == 'truncate':
        truncate_table(engine, table_name)
        if_exists = 'append'

    df.to_sql(table_name,
              con=engine,
              if_exists=if_exists,
              index = False)
    return None



def write_sql_database_without_plannif(df, table_name, if_exists='replace'):
  """
  Write pandas data frame in the given table in the data base
  if 'if_exists'=replace, the table is delete and a new one with same name is created for insertion
  if 'if_exists'=truncate, old data are deleted before insertion in the table
  if 'if_exists'=append, new data are inserted without delete any existing data
  """
  user = getpass.getuser()
  path = os.path.join('home', user, 'my.cnf')
  engine = sqlalchemy.create_engine('mysql+pymysql://',
                                    connect_args={'read_default_file': "/"+ path})

  if if_exists == 'truncate':
    truncate_table(engine, table_name)
    if_exists = 'append'

  df.to_sql(table_name,
            con=engine,
            if_exists=if_exists,
            index = False)
  return None














def read_from_sql_database(table_name, chunksize=None):
    user = getpass.getuser()
    path = os.path.join('home',user,'my.cnf')
    engine = sqlalchemy.create_engine('mysql+pymysql://',
                                      connect_args={'read_default_file': "/" + path})
    df = pd.read_sql(table_name,
                     con=engine,
                     chunksize=chunksize)
    return df


def write_csv_sql(df, name, path="", sql=True, csv=True, if_exists='replace', delete_planif=True):
    if sql:
        if '.csv' in name:
            name = name.split(".csv")[0]
        write_sql_database(df, name, if_exists, delete_planif)
    if csv:
        write_csv(df, path, name)
    return  None



def calculate_npv(df, wacc):
    df['NPV_DISCOUNT'] = 0  # Crée une nouvelle colonne "npv" initialisée à zéro

    # Parcourt chaque colonne commençant par "cash_flow_year_"
    for column in df.columns:
        if column.startswith('cash_flow_year_'):
            year = int(column.split('_')[-1])
            discount_factor = (1 + wacc / 100) ** year
            df['cash_flow_discount_' + str(year)] = df[column] / discount_factor
            df['NPV_DISCOUNT'] += df[column] / discount_factor

    return df



def npv_since_2nd_years(rate, values):
    """
    Returns the NPV (Net Present Value) of a cash flow series discount since the first year

    """

    #Todo : REMOVE
    values = np.asarray(values)
    values = np.nan_to_num(values)
    return (values / (1+rate)**np.arange(1, len(values)+1)).sum(axis=0)

def irr(values):
    """
    Return the Internal Rate of Return (IRR).

    .. deprecated:: 1.18

       `irr` is deprecated; for details, see NEP 32 [1]_.
       Use the corresponding function in the numpy-financial library,
       https://pypi.org/project/numpy-financial.

    This is the "average" periodically compounded rate of return
    that gives a net present value of 0.0; for a more complete explanation,
    see Notes below.

    :class:`decimal.Decimal` type is not supported.

    Parameters
    ----------
    values : array_like, shape(N,)
        Input cash flows per time period.  By convention, net "deposits"
        are negative and net "withdrawals" are positive.  Thus, for
        example, at least the first element of `values`, which represents
        the initial investment, will typically be negative.

    Returns
    -------
    out : float
        Internal Rate of Return for periodic input values.

    Notes
    -----
    The IRR is perhaps best understood through an example (illustrated
    using np.irr in the Examples section below).  Suppose one invests 100
    units and then makes the following withdrawals at regular (fixed)
    intervals: 39, 59, 55, 20.  Assuming the ending value is 0, one's 100
    unit investment yields 173 units; however, due to the combination of
    compounding and the periodic withdrawals, the "average" rate of return
    is neither simply 0.73/4 nor (1.73)^0.25-1.  Rather, it is the solution
    (for :math:`r`) of the equation:

    .. math:: -100 + \\frac{39}{1+r} + \\frac{59}{(1+r)^2}
     + \\frac{55}{(1+r)^3} + \\frac{20}{(1+r)^4} = 0

    In general, for `values` :math:`= [v_0, v_1, ... v_M]`,
    irr is the solution of the equation: [2]_

    .. math:: \\sum_{t=0}^M{\\frac{v_t}{(1+irr)^{t}}} = 0

    References
    ----------
    .. [1] NumPy Enhancement Proposal (NEP) 32,
       https://numpy.org/neps/nep-0032-remove-financial-functions.html
    .. [2] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
       Addison-Wesley, 2003, pg. 348.

    Examples
    --------
    >>> round(np.irr([-100, 39, 59, 55, 20]), 5)
    0.28095
    >>> round(np.irr([-100, 0, 0, 74]), 5)
    -0.0955
    >>> round(np.irr([-100, 100, 0, -7]), 5)
    -0.0833
    >>> round(np.irr([-100, 100, 0, 7]), 5)
    0.06206
    >>> round(np.irr([-5, 10.5, 1, -8, 1]), 5)
    0.0886

    """
    # `np.roots` call is why this function does not support Decimal type.
    #
    # Ultimately Decimal support needs to be added to np.roots, which has
    # greater implications on the entire linear algebra module and how it does
    # eigenvalue computations.
    res = np.roots(values[::-1])
    mask = (res.imag == 0) & (res.real > -1.5)
    if not mask.any():
        return np.nan
    res = res[mask].real
    # NPV(rate) = 0 can have more than one solution so we return
    # only the solution closest to zero.
    rate = 1/res - 1
    rate = rate.item(np.argmin(np.abs(rate)))
    return rate


def payback_of_investment(investment, cashflows):
    """The payback period refers to the length of time required
       for an investment to have its initial cost recovered.

       payback_of_investment(200.0, [60.0, 60.0, 70.0, 90.0])
       3.1111111111111112
    """
    total, years, cumulative = 0.0, 0, []
    if not cashflows or (sum(cashflows) < investment):
        raise Exception("insufficient cashflows")
    for cashflow in cashflows:
        total += cashflow
        if total < investment:
            years += 1
        cumulative.append(total)
    A = years
    B = investment - cumulative[years - 1]
    C = cumulative[years] - cumulative[years - 1]
    return A + (B / C)


def payback(cashflows):
    """The payback period refers to the length of time required
       for an investment to have its initial cost recovered.

       (This version accepts a list of cashflows)

       payback([-200.0, 60.0, 60.0, 70.0, 90.0])
       3.1111111111111112
    """
    investment, cashflows = cashflows[0], cashflows[1:]
    if investment < 0: investment = -investment
    return payback_of_investment(investment, cashflows)


#Write dictionary object on disk
def saveJSON(dictObj, name, space=4):
    try:
        with open(name, 'w') as outfile:
            json.dump(dictObj, outfile, indent=space)
    except:
        return "Une erreur est survenue! Enregistrement du fichier de configuration"

#Read json file from disk
def loadJSON(name):
    try:
        with open(name) as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(e)
        return "Une erreur est survenue!: Lecture du fichier de configuration"


# Repartition of CPU depending on the user
def CPU_repartition():
    if os.getlogin() == 'ivan.sidorenko':
        os.system("taskset -p -c 15 %d" % os.getpid())
    elif os.getlogin() == 'alaa.khaldi':
        os.system("taskset -p -c 14 %d" % os.getpid())
    elif os.getlogin() == 'ange.tape':
        os.system("taskset -p -c 13 %d" % os.getpid())
    elif os.getlogin() == 'arnold.chuenfo':
        os.system("taskset -p -c 12 %d" % os.getpid())
