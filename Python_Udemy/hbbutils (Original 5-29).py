# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 12:42:08 2018

@author: pullelam
"""

import pyodbc as po
import numpy as np   
import pandas as pd
from sqlalchemy import create_engine
import datetime
import dateutil.relativedelta
import smtplib
import email
import fnmatch
import os
import sys
import glob
import json
from os import listdir
from os.path import isfile
import pickle
import pdb
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
import shutil
import zipfile
# import pandleau
#from tableausdk import *
#from tableausdk.Server import * 
#from tableausdk.Extract import *
import yaml
import logging
import logging.config
import functools
# sys.path.append(os.path.abspath("R:\_Department\Analytics\pythonlib"))

# Add location of own packages to path
sys.path.append(
        os.path.abspath("R:\_Department\Analytics\pythonlib")
        )


# basedir = os.path.abspath(os.path.dirname(__file__)) # For old config


def create_log_path(path):
    """
    Create filename for log file. To base log file name on fn of calling python file,
    set "path=__file__". 
    (This cannot be made default argument, because this would return path to hbbutils!)
    """
    py_fn = os.path.basename(path)
    log_root = os.path.splitext(py_fn)[0] # Remove file extension
    log_fn = f'{log_root}.log'
    py_dir = os.path.dirname(path) # Dir w py files
    
    # Create log directory if it doesn't exist
    if not os.path.isdir(
        os.path.join(py_dir, 'logs')
    ):
        os.makedirs(
            os.path.join(py_dir, 'logs')
        )
    
    log_path = os.path.join(py_dir, 'logs', log_fn)
    return log_path
    

def load_log_config(
    conf_fn='config.yaml',
    conf_dir=r'R:\_Department\Analytics\logging'
):
    """
    Reads a yaml config and configures logging. Note that names of args are 
    different from setup_logging() for backward compatibility. 
    Now this function usually won't need to be called directly; use 
    setup_logging() instead.
    """
    conf_path = os.path.join(conf_dir, conf_fn)
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

def setup_logging(
    logger_name, 
    config_fn='config.yaml', 
    config_dir=r'R:\_Department\Analytics\logging'
):
    """
    Usage:
    logger = setup_logging(logger_name=__name__)
    """

    load_log_config(config_fn, config_dir)
    logger = logging.getLogger(logger_name)

    # Visually separate new log from previous log
    if __name__ == '__main__':
        logger.info(
            f'========================================\n'
        )

    return logger

# Set up logging for this file    
logger = setup_logging


def first_func():
    pd.set_option('display.max_row', 1000)
    # Set iPython's max column width to 50
    pd.set_option('display.max_columns', 60)
    print ("Function Calls Start")
    
def load_config(verbose=True):
    """
    Loads shared config and - if exists in home directory - private config.
    Usage: 
    config = load_config()
    """

    # Shared config
    shared_config_path = 'R:\_Department\Analytics\config_shared.json'
                           
    # Load shared config       
    with open(shared_config_path, 'r') as f:
        config = json.load(f)

    # Load private config, if exists
    try:
        # Expected location: Home directory (only works on Windows)
        private_config_path = os.path.join(
            os.environ['USERPROFILE'], # Home directory
            'config_private.json'
            )
        # Load private config       
        with open(private_config_path, 'r') as f:
            config_private = json.load(f)
            # Add to shared config (note theses are both dictionaries)
            config.update(config_private) # Overlapping keys, if any, will be taken from private config.

    except FileNotFoundError:
        if verbose==False:
            pass
        else:
            print('Only the PUBLIC config was loaded because the file "config_private.json" does ' 
                  'not exist in the home directory.')

    return config
    
    
def mongodb_conn_str(auth_db='admin'):
    """
    Get connection string for mongoDB, using credentials from config.
    
    Parameters
    ----------
    auth_db str: authentication database (where credentials are stored).
    
    """
    UID = load_config()['MongoDB']['UID']
    PWD = load_config()['MongoDB']['PWD']
    conn_str = f'mongodb://{UID}:{PWD}@riccop-display:27017/{auth_db}'
    return conn_str    


#class Config(object):
#    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
#    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
#        'sqlite:///' + os.path.join(basedir, 'app.db')
#    SQLALCHEMY_TRACK_MODIFICATIONS = False
## Function to Send email using GMAIL API
## Required : 
##              uname               - Username
##              app_password        - App_password 
##              email_from          - From Email Address 
##              email_to            - to Email Address (list) 
##              email_Subject       - Email Subject 
##              email_text          - Email Body 
def login_email(uname, app_password):
    #email_to = ['Mallik.Pullela@Hamiltonbeach.com', 'Mallik.Pullela@Hamiltonbeach.com']  
    try:  
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(uname, app_password)
        return (server)
    except:  
        print('Something went wrong logging into the email account...')

# def send_email(uname, app_password,email_from,email_to,email_subject,email_text):
    # #email_to = ['Mallik.Pullela@Hamiltonbeach.com', 'Mallik.Pullela@Hamiltonbeach.com']  
    # try:  
# #        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
# #        server.ehlo()
# #        server.login(uname, app_password)
        # message = """From: %s\nTo: %s\nSubject: %s\n\n%s""" % (email_from, ", ".join(email_to), email_subject, email_text)
        # server = login_email(uname, app_password)
        # server.sendmail(email_from, email_to, message)
        # server.close()
        # return ('Email sent!')
    # except:  
        # return ('Something went wrong...')

def send_email(send_to, subject, text, attachments=None, uname=None, app_pwd=None, send_from=None, html=False):
    """
    Send email. Also allows attachments. 
    Adapted from https://stackoverflow.com/questions/3362600/how-to-send-email-attachments
    Allows text formatting using HTML tags by setting HTML = True. You can either
    add the appropriate HTML tags at the top or bottom, or they will be added automatically
    if missing (the whole text will then be part of the HTML body). 
    
    
    Parameters
    ----------
    send_to: str or list 
    subject: str
    text: str
    attachments: str or list
    uname: str
    app_pwd: str
    send_from: str
    html: bool
        whether text should be sent as HTML or plain text. Default: False (plain text).
    """
    
    if uname == None:
        uname=load_config()['email']['ADDRESS']
    if app_pwd == None:
        app_pwd=load_config()['email']['APP_PWD']
    if send_from == None:
        send_from=load_config()['email']['ADDRESS']
    
    if isinstance(send_to, str):
        send_to = [send_to]
    if isinstance(attachments, str):
        attachments = [attachments]
    assert isinstance(send_to, list)
    
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = ', '.join(send_to)
    msg['Date'] = email.utils.formatdate(localtime=True)
    msg['Subject'] = subject

    # If text is HTML but does not contain the right tags at bottom and 
    # top, add them.
    if html and not text.lstrip().startswith('<html>'):
        html_top = '''
        <html>
          <head></head>
          <body>
            <p>'''

        html_bottom = '''
            </p>
          </body>
        </html>'''

        text = ''.join(
            [html_top, text, html_bottom]
        )


    # specify whether email body is plaintext or HTML
    text_format = 'html' if html == True  \
        else 'plain'
    msg.attach(
        email.mime.text.MIMEText(text, text_format)
    )

    for f in attachments or []:
        with open(f, "rb") as fil:
            part = email.mime.application.MIMEApplication(
                        fil.read(),
                        Name=os.path.basename(f)
            )
            
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
        msg.attach(part)
    
    server = login_email(uname, app_pwd)
    server.sendmail(send_from, send_to, msg.as_string())
    server.close()
    print('Sent!')
    
def format_email_addresses(addr_str):
    '''
    Takes a string of email addresses, as copoied from gmail, and converts it into format
    suitable to pass to send_email()
    '''
    # Convert to list, splitting on comma
    unformatted_list = addr_str.split(',')
    # process list elements
    return [
        a.split('<')[-1].split('>')[0]
        for a in unformatted_list                
    ]
 
 
def read_excel(filename,sheet_name,skiprows=0):
    return pd.ExcelFile(filename).parse(sheet_name=sheet_name,skiprows=skiprows)
    
def sql_connect_AL(dsn):
    return create_engine('mssql+pyodbc://'+dsn)

#Server=RICPLG-PULLELAM\PULLELAM
#Database=Shipping
# def sql_connect_py(server,database):
#     return po.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=True;')

# def sql_connect_py_pwd(server,database):
#     return po.connect(
#         f'DRIVER={{SQL Server}}; SERVER={server}; DATABASE={database}; UID={UID_L76}, PWD={PWD_L76}'
#         )

def sql_connect_py(server,database, uid=None, pwd=None):
    if not uid:  # If no uid and pwd is supplied, use Windows authentication
        cxn = po.connect(
            'DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=True;'
            )
    else:
        cxn = po.connect(
            f'DRIVER={{SQL Server}}; SERVER={server}; DATABASE={database}; UID={uid}; PWD={pwd}'
        )
    return cxn
# uid='lvsprod'

#def tableau_extract_open(Extract_Name):
#    ExtractAPI.initialize()
#    dataExtract = Extract(Extract_Name)
#    return dataExtract
#
#def tableau_extract_close():
#    ExtractAPI.close()

def del_file_list(L):
    x=[os.remove(i) for i in L]
    return x
    
class CommitMode:
    NONE = 0 #Commit immediate (*NONE)  --> QSQCLIPKGN
    CS = 1 #Read committed (*CS)        --> QSQCLIPKGS
    CHG = 2 #Read uncommitted (*CHG)    --> QSQCLIPKGC
    ALL = 3 #Repeatable read (*ALL)     --> QSQCLIPKGA
    RR = 4 #Serializable (*RR)          --> QSQCLIPKGL

class ConnectionType:
    ReadWrite = 0 #Read/Write (all SQL statements allowed)
    ReadCall = 1 #Read/Call (SELECT and CALL statements allowed)
    Readonly = 2 #Read-only (SELECT statements only)

def connstr(system, commitmode=CommitMode.CHG, connectiontype=ConnectionType.Readonly,UID='',PWD=''):
    if UID == '':
        UID=load_config()['AS400']['UID']
    if PWD == '':
        PWD=load_config()['AS400']['PWD']
    
    
    _connstr =  'DRIVER=iSeries Access ODBC Driver;'+\
                'SYSTEM='+system+';'+\
                'SIGNON=4;CCSID=1208;TRANSLATE=1;UID='+UID+';PWD='+PWD+';'
    if commitmode is not None:
        _connstr = _connstr + 'CommitMode=' + str(commitmode) + ';'
    if connectiontype is not None:
        _connstr = _connstr +'ConnectionType=' + str(connectiontype) + ';'

    return _connstr

def connstr_test(system, commitmode=CommitMode.CHG, connectiontype=ConnectionType.Readonly,UID='',PWD=''):
    if UID == '':
        UID=load_config()['AS400_QA']['UID']
    if PWD == '':
        PWD=load_config()['AS400_QA']['PWD']
    
    _connstr =  'DRIVER=iSeries Access ODBC Driver;'+\
                'SYSTEM='+system+';'+\
                'SIGNON=4;CCSID=1208;TRANSLATE=1;UID='+UID+';PWD='+PWD+';'
    if commitmode is not None:
        _connstr = _connstr + 'CommitMode=' + str(commitmode) + ';'
    if connectiontype is not None:
        _connstr = _connstr +'ConnectionType=' + str(connectiontype) + ';'

    return _connstr


def appendDFToCSV_void(df, csvFilePath, sep=","):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

def trimAllColumns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    df=trimnewline(df)
    trimStrings = lambda x: x.strip() if type(x) is str else x
    return df.applymap(trimStrings)

def trimnewline(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    df=trimrline(df)
    trimNewline = lambda x: x.replace("\n","") if type(x) is str else x
    return df.applymap(trimNewline)

def trimrline(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trimRline = lambda x: x.replace("\r","") if type(x) is str else x
    return df.applymap(trimRline)

def read_PWD_File(File_Path):
    
    with open(File_Path) as f:
      credentials = [x.strip().split(':') for x in f.readlines()]

    for Server,username,password in credentials:
        return Server,username,password
        
def search_in_files(file_path=".",search_string="Hello"):
    """
    #file_path is the path to search the Search String 
    #search_string - search string 
    
    file_path_1 = "C:/temp/scikit-learn-master/"
    ss = "def mean_"
    print(search_in_files(file_path_1,ss))

    """
    mypath=file_path
    flist = glob.glob(mypath + '/**/*.py', recursive=True)
    
    return_list = []
    try:
        for f in flist:
            with open(f, 'r') as searchfile:
                for line in searchfile:
                    if search_string in line:
                        print (f)
                        return_list.append(f)
    except:
        return_list = return_list
    return list(set(return_list))


#####  Get USD - MEXICAN Peso / CANDIAN Dollar Conversion Rate
def convert_USD_Curr(env):
    if env == 'MEX':
        queryString = 'http://apilayer.net/api/live?access_key=a2abfe7dcfa72d80c6480d31aacfa517&currencies=MXN'
        response = requests.request("GET", url=queryString)
        json_data = response.json()
        #print (json_data)
    elif env == 'CAN':
        queryString = 'http://apilayer.net/api/live?access_key=a2abfe7dcfa72d80c6480d31aacfa517&currencies=CAD'
        response = requests.request("GET", url=queryString)
        json_data = response.json()
        #print (json_data)

    if env == 'MEX':
        print ("USD MXN = ", json_data["quotes"]["USDMXN"])
        return(1/json_data["quotes"]["USDMXN"])
    elif env == 'CAN':
        print ("USD CAN = ", json_data["quotes"]["USDCAD"])
        return(1/json_data["quotes"]["USDCAD"])
    elif env == 'US':
        return 1.0


# Read Excel
def read_excel2(fn, sheet_name=0, **kwargs):
    """
    Read excel, trim whitespaces, rename columns.
    NOTE: If passing column names in kwargs (e.g., to specify dtype),
    remember to use ORIGINAL column names (before replacing white spaces
    and changing capitalization)!
    """
    return trimAllColumns(
                pd.read_excel(fn, sheet_name, **kwargs)
            ) \
            .rename(
                lambda x: str(x).lower()
                                .replace(' ', '_')
                                .replace('\n', '_'),
                axis=1
            ) \
            .replace('', np.nan)

## Read Cognos Excel 
# def read_cognos(fn, header=[5,6], skipfooter=1, index_name='index'):
    # df = read_excel2(
                # fn,
                # header=header,
                # skipfooter=skipfooter
                # )
    
    # # Set sku as index, because it doesn't fit in the multiinedxed columns
    # df = df.set_index(
            # ('unnamed:_0_level_0', 'unnamed:_0_level_1')
    # )

    # # Change index name
    # df.index.name = index_name
    
    # # Print last rows (only first 4 columns)
    # print ('Bottom rows (Make sure they do not contain a footer or Grand Total):\n\n',
           # df.iloc[-2:, :4]
           # )
    
    # return df
    
def read_cognos(fn, skipfooter=1, index_cols=None, index_name='index', pivot=False, **kwargs,):
    df = read_excel2(
                fn,
                **kwargs,
                skipfooter=skipfooter,
                )
    
    if pivot == False:
        # Rename unnamed columns as specified in index_cols
        if index_cols is not None:
            col_map = {f'unnamed:_{position}': name for (position, name) in enumerate(index_cols)}
            df = df.rename(columns=col_map)

        # Remove family from type (unless we don't have type in cols)
        try:
            df.loc[:, 'type'] = df.type.str.split(' - ').str.get(0)
        except AttributeError:
            pass

        # Forward fill missing index columns (because they are not repeated in cognos file)
        if index_cols:
            c2ffill = index_cols[:-1]  # All up to the last index col needs to be filled
            df.loc[:, c2ffill] = df.loc[:, c2ffill].fillna(method='ffill')

        # If a column called date exists, convert to dt
        try:
            df.loc[:, 'date'] = df['date'].astype('datetime64')
        except KeyError:
            pass
        
        # Rename sales and returns (This won't raise an error if not in column index)
        df = df.rename(
                    columns={
                        'gross_qty': 'sales_units',
                        'gross_quantity': 'sales_units',
                        'defective_ret_qty': 'returns_units',
                        'gross_dollars': 'sales_$',
                        'defective_ret_dollars': 'returns_$',
                        'defective_ret_$': 'returns_$'
                    }
            )
    
    # If the data comes as pivot table (w columns corresponding to months)
    # TODO: make more general purpose
    else:
        # Set sku as index, because it doesn't fit in the multiinedxed columns
        df = df.set_index(('unnamed:_0_level_0', 'unnamed:_0_level_1'))

        # Change index name
        df.index.name = index_name

    # Print top and bottom rows (only first 4 columns)
    msg = 'Top rows (Make sure they do not start before actual data by ' \
        f'checking column names):\n\n{df.iloc[:2, :4]}\n\n' \
        f'Bottom rows (Make sure they do not contain a footer or Grand ' \
        f'Total): \n\n{df.iloc[-2:, :4]}'

    # If logger is defined, log it as debug
    try:
        logger.debug(msg)
    except:
        try:
            logging.debug(msg)
        # If logging is not defined, print message
        except:
            print(msg)
    
    return df  
    

    
def split_multiindexed_cols(df):
    """
    Split a DataFrane with multi-indexed columns into separate DataFrames. 
    Returns a new df for each level of inner index. Example: Cognos
    """
    
    # Second levels of column index
    levels = (
        df.columns 
                .remove_unused_levels() 
                .get_level_values(1) # Second (bottom) level
                .unique()
        )
    
    # Dict to hold returned dfs
    dfs = {}
    # Loop over levels 
    for level in levels:
        # Slice
        df_new = df.loc[:, pd.IndexSlice[:, level]]
        # Drop first (upper) level index (it's constant now)
        df_new.columns = df_new.columns.droplevel(1)
        # Store
        dfs[level] = df_new
        
    print(f'New Dataframes created: {dfs.keys()}')    
    return dfs

    
# Pickle File
home_dir = f"{os.environ['USERPROFILE']}"           
def to_pickle(obj, fn, dir_=f'{home_dir}\pickled_files'):
    with open(
            os.path.join(dir_, f'{fn}.pickle'),
            'wb'
            ) as f:
                pickle.dump(obj, f)
            
# Unpickle
def load_pickle(fn, dir_=f'{home_dir}\pickled_files'):
    with open(
            os.path.join(dir_, f'{fn}.pickle'),
            'rb'
            ) as f:
                return pickle.load(f)


def sum_numerics(df):
    """Computes sum for all numeric columns of a dataframe"""
    sums =  df.select_dtypes(
                include=np.number
                ) \
                .sum() \
                .dropna()
    
    return(
        pd.DataFrame(sums, columns=['sum'])
            .style.format('{:,.0f}')
        )


# Make Column names unique
# From StackOverflow: https://stackoverflow.com/questions/24685012/pandas-dataframe-renaming-multiple-identically-named-columns
def uniquify_cols(df):
    new_columns = []
    for item in df.columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    return new_columns


def plot_dupl(df, date_col_name, interval='M', title=None):
    """Plot proportion of NAs in a dataframe by month (or other time interval)"""
    dupl_and_date = pd.DataFrame({
        'date': pd.to_datetime(df.loc[:, date_col_name]),
        'dupl': df.duplicated().astype(int)
    })
    
    (
    dupl_and_date.resample(
        rule=interval, 
        on='date'
        )
        .mean()
        .dropna()
        .plot()
    );
    
    plt.title(title)


# Get data from The One Report
def get_t1r():
    """Get data underlying the one report"""
    
    t1r = read_excel2(
        r"R:\_Department\Purchasing\Quality\Bennie's files\The One Report Project\The One Report - USCanfin_KFN.xlsx",
        'unpivoted data'
    )
        
    return t1r


#config = load_config() # To fill defaults of sender's email address and password


def fill_na(df, fill_str='Unknown', fill_num=0):
    """
    Fill numeric columns with 0s and fill string columns with 'Unknown'
    or custum string. Set fill_str/num to None to not fill columns of respective type.
    """
    # Find numeric and string columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    str_cols = df.select_dtypes(include=object).columns.tolist()

    # Replace
    if fill_num:
        df.loc[:, num_cols] = df.loc[:, num_cols] \
            .fillna(0)
    
    if fill_str:
        df.loc[:, str_cols] = df.loc[:, str_cols] \
            .fillna(fill_str)
    
    return df
 
    

def find_file_pattern(pattern, directory):
    '''Search directory for file pattern.'''
    result = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result



##### DATE UTILS

def date_mmddyy():
    return datetime.datetime.now().strftime("%Y-%m-%m")


def last_updated_time(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    ctime = 0
    if platform.system() == 'Windows':
        ctime = os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            ctime=stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            ctime=stat.st_mtime

    return datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d')


# Defined function that converts week code to datetime
def format_datecode(dc):
    """
    Zero-pad datecode and convert to string if necessary. If input is None/np.nan,
    return input.
    
    Parameters
    ----------
    dc: string or float
        unformated datecode
        
    Returns
    -------
    str or None or np.nan
        Formatted datecode. 
    """
    
    # If missing, return missing
    if dc in (None, np.nan):
        return dc
    
    # This implicitly checks if input is string
    try:
        dc = dc.strip()
            
    # If len() fails, the input is probably numeric. Convert to string and try again.
    except AttributeError:
        dc = str(
            # First convert to integer, in case it's a float (would be '*.0' otherwise).
            int(dc)
        )
        dc = dc.strip()
        
    # Now that we know input is string, we can check if we need to add a leading 0
    finally:  
        if len(dc) == 3:
            dc = '0' + dc
    
    # Now make sure that the code is of the right length
    assert len(dc) == 4, f'The datecode {dc} is not of length 4!'
    
    return dc
    
    
def datecode2year(dc):    
    """Convert datecode to year as int"""
    
    # If missing, return missing
    if dc in (None, np.nan):
        return dc
        
    # Convert to string and add leading 0, if necessary
    dc = format_datecode(dc)
    # get year
    year = f'20{dc[-1]}{dc[-2]}'
            
    return int(year)
    
    
def week_code2date(dc):
    raise DeprecationWarning(
        'Use datecode2date instead. Note changed output format, though.'
    )
    dc = format_datecode(dc)    
    # Add day of week (Monday), reverse year and separate from month with a white space
    date_dc = f'Day 1-Week {dc[:2]}-Year 20{dc[3]}{dc[2]}'

    # Parse
    return time.strptime(date_dc, 'Day %w-Week %V-Year %G')

# print(week_code2date('1291'))
# print(week_code2date(1291))
def datecode2date(dc, day_of_week=1, week_offset=-1):
    """
    Convert a datecode into a date.
    
    Parameters
    ----------
    dc: string or float
        unformated datecode
    day_of_week int
        day of week for return date. Default: Monday
    week_offset int
        Adjustment for what to count as first week, relative to ISO.
        For 2020, HBB week nr are one week earlier, which is set to
        default. Change default in the new year, so by default we
        still get HBB week nrs/dates!
        
    Returns
    -------
    date 
        Date of specified day of the week of datecode, formatted as %Y-%m-%d.
    """
    
    # If missing, return missing
    if dc in (None, np.nan):
        return dc
        
    # Convert to string and add leading 0, if necessary
    dc = format_datecode(dc)    
    
    # Add day of week (Monday), reverse year and separate from month with a white space
    
    week = dc[:2]
    # Adjust week
    if week_offset not in (0, None):
        week = int(week) + week_offset
        
    year = f'20{dc[3]}{dc[2]}'

    # Parse
    date = datetime.datetime.strptime(
        f'{year}-{week}-{day_of_week}',
        '%Y-%W-%w'
    )
    
    return date


def year_week2datecode(year, week):
    """Convert year and week to date code (integer)."""
    
    # If float, convert to integer (to get rid of ".0")
    try:
        year = int(year)
        week = int(week)
    except ValueError:  # "invalid literal for int() with base 10"
        pass  # If it's string, nevermind. 
    
    # If week is not zero-padded, add a leading zero
    if len(str(week)) == 1:
        week = f'0{week}'
        
    # Use only the last two digits of year
    year = str(year)[-2:]
    
    weekcode = f'{week}{year[1]}{year[0]}'
    
    return int(weekcode)
    

# # Defined function that converts week code to datetime
# def week_code2date(dc):
    # try:
        # # If the week is not zero-padded, add a leading zero
        # if (len(dc) == 3):
            # dc = '0' + dc
    # # If this fails, the input may have been numeric. Convert to a dc and try again.
    # except TypeError:
        # dc = str(dc)
        # # Try again
        # if (len(dc) == 3):
            # dc = '0' + dc
    
    # # Now make sure that the code is of the right length
    # assert (len(dc) == 4), 'The week code is not of length 4.'
    
    # # Add day of week (Monday), reverse year and separate from month with a white space
    # date_dc = f'Day 1-Week {dc[:2]}-Year 20{dc[3]}{dc[2]}'
# #    print(date_dc)

    # # Parse
    # return time.strptime(date_dc, 'Day %w-Week %V-Year %G')


def get_1st_of_this_month():
    """Get the date of the FIRST day of the current month"""
    # Beginning of month offset 
    beg_of_m_offset = pd.offsets.MonthBegin()  
    # Current date
    now = pd.to_datetime('now')  
    
    # Go to beginning of month 
    return  pd.Series(  # So we can use .dt accessor
                beg_of_m_offset.rollback(now)
                ) \
                .dt.floor('d')[0]  # Get rid of time of day and convert back to scalar
        

def get_1st_of_last_month():
    """Get the date of the FIRST day of the previous month"""
    return get_1st_of_this_month() + pd.DateOffset(months=-1)
    
def get_1st_of_last_n_month(n):
    """Get the date of the FIRST day of the last complete n months."""
    return get_1st_of_this_month() + pd.DateOffset(months=-n)

def get_last_of_last_month():
    """Get the date of the LAST day of the previous month"""
    return get_1st_of_this_month() + pd.DateOffset(days=-1)
    
    
def get_beg_of_last_12m():
    """Get the date of the first day of the last complete 12 month"""
    return get_1st_of_this_month() + pd.DateOffset(years=-1)
    
    
def get_last_n_months(df, n=12, offset=0):
    """
    Returns all columns from a data frame whose column names corresponds
    to a date within the last n months. Optionally include an offset.
    Assumes column names are of type datetime and beginning and end date 
    correspond to the first of the month.
    """
    # The last date included is by default the first of the last month.
    # If an offset is specified, go back further.    
    last_included = get_1st_of_last_month() - pd.DateOffset(months=offset)
    # Go back n-1 months from last month included to get the first date included
    first_included = last_included - pd.DateOffset(months=n-1) 
    
    # Return data frame with only columns that fall within the specified range
    return df.loc[:, (df.columns <= last_included) & (df.columns >= first_included)]


# Returns the months as a list in mmm-YYYY format
# Input Year
# Output List of strings (Month Lists)
def year_month_list(year):
    l= []
    for month in  range (1,13):
        any_day = datetime.date(year,month,1)
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
        l.append((next_month - datetime.timedelta(days=next_month.day)).strftime("%b-%Y"))
    
    return l

# Returns the all days in the year as a list in mmm-YYYY format
# Input Year
# Output List of strings (Month Lists)
def year_days_list(year):
    start_date = datetime.date(year,1,1)
    end_date = datetime.date(year,12,31)
    date_generated = [(start_date + datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (end_date-start_date).days)]
    
    return date_generated


def get_last_and_preceding_month():
    '''
    Return date of a month ago and two months ago.
    Usefule to get reporting month and month of last existing report.
    '''
    
    last_month = datetime.datetime.now() \
        - dateutil.relativedelta.relativedelta(months=1)
    preceding_month = datetime.datetime.now() \
        - dateutil.relativedelta.relativedelta(months=2)
    
    return last_month, preceding_month


def lr():
    """Print last time file was run.  """
    try:
        logger.info('Done.')
    except:
        try:
            print('This usage is deprecated. Please get logger as "logger"')
            logging.info(f'Last run: {datetime.datetime.now()}')
        except:
            print(
                'Warning: This usage is deprecated. Please import logging '
                'and instanciate logger as "logger"'
            )                       
            print(f'Last run: {datetime.datetime.now()}')
    
    
def check_err_file(py_fn, err_fn):
    '''
    Raise error and open error file if it is not empty.
    
    Args:
    py_fn str: path to python script to check
    err_fn str: path to bcp error file
    if not os.path.isfile(err_fn):
    '''
       
    with open(err_fn, 'r') as f:
        # If error occurred
        if len(f.read()) > 0:
            # Open error file
            # os.system(f'start notepad {err_fn}')
            
            # Compose email
            subject=f'Error running {py_fn}'
            text = f'''An error occurred running {py_fn}. 
                    \nPlease check the attached error file.
                    \nThis email was generated automatically.
                    '''
            attachment=err_fn
                
        # If it ran successfully
        else:
            # Compose email
            subject=f'Successfully ran {py_fn}'
            text = f'''Successfully ran {py_fn}.
                    \nThis email was generated automatically.
                    '''
            attachment=None
        
        # Send email
        send_email(
            send_to=[
                'thomas.loeber@hamiltonbeach.com',
                'mallik.pullela@hamiltonbeach.com'
             ],
            subject=subject,
            text=text,
            attachments=attachment,
        )
        
        
def check_colnames_match(dfs):
    """
    Before concatenating a list of DataFrames, this function can be used
    to check if all DataFrames have the same column names.
    
    Args:
    =====
    dfs list: List of dataframes to be concatenated
    """

    col_names = pd.DataFrame(
        [df.columns.tolist() for df in dfs]
    )

    if len(col_names.drop_duplicates()) > 1:
        print(col_names.drop_duplicates())
        raise Exception('Column names of input files do not align.')        
        

# def reload_modules():        
    # """Reload modules imported with *."""
    # import sys, importlib
    # importlib.reload(sys.modules['hbbutils'])
    # importlib.reload(sys.modules['sql_utils'])
    # importlib.reload(sys.modules['tableau_utils'])
    # importlib.reload(sys.modules['ml_utils'])
    # importlib.reload(sys.modules['ts_utils'])
    # from hbbutils import *
    # from sql_utils import *        
    
    
def create_log_path(path):
    """
    Create filename for log file. To base log file name on fn of calling python file,
    set "path=__file__". 
    (This cannot be made default argument, because this would return path to hbbutils!)
    """
    py_fn = os.path.basename(path)
    log_root = os.path.splitext(py_fn)[0] # Remove file extension
    log_fn = f'{log_root}.log'
    py_dir = os.path.dirname(path) # Dir w py files
    
    # Create log directory if it doesn't exist
    if not os.path.isdir(
        os.path.join(py_dir, 'logs')
    ):
        os.makedirs(
            os.path.join(py_dir, 'logs')
        )
    
    log_path = os.path.join(py_dir, 'logs', log_fn)
    return log_path
    

def load_log_config(conf_fn='config.yaml', conf_dir=r'R:\_Department\Analytics\logging'):
    """Reads a yaml config and configures logging."""
    conf_path = os.path.join(conf_dir, conf_fn)
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    
    
def is_part(family):
    """ Determines whether a product family falls under parts."""
    
    # Classify as part if family ends in accessories or parts
    try:
        if (family.lower().endswith('accessories') | 
            family.lower().endswith('parts')
            ):
                return 'Parts'
        
        # Otherwise classify as Retail
        else:
            return 'Products'
        
    # If we get an error, this is because of np.NaNs, so return missing value
    except AttributeError: # "float" object has no attribute "lower"
        return None
    
# combined = combined.assign(part = combined.family.map(is_part))


# In[34]:


def is_commercial(family):
    """Determines whether a product family falls under commercial"""
    try:
        if family.lower().startswith('commercial'):
            return 'Commercial'
        else:
            return 'Retail'
    
    # If we get an error, this is because of np.NaNs, so return missing value
    except AttributeError: # "float" object has no attribute "lower"
        return None    
        
        

# def test2():
    # print(x)
    # print(y)        


def obj2str(df):
    """
    Converts all columns of datatype object to str.
    
    Parameters
    ----------
    df pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    
    # Find columns to transform
    obj_cols = df.select_dtypes(include=object).columns
    
    df.loc[:, obj_cols] = df.loc[:, obj_cols].apply(
            lambda col: col.astype(str),    
            1
        )
    
    return df    
    
    
def check_cols_identical(paths2dfs, stop_on_error=True, **kwargs):
    """
    Check that all dataframes in a list have same colum names.
    
    Parameters
    ----------
    
    paths2dfs: list or dictionary 
        Paths to DataFrames to check. If dict, **keys** should contain paths.
    stop_on_error bool
        Whether to exit program if error is encountered. Setting to False can be 
        useful for debugging, because it returns dataframes.
    """ 
    
    # If paths are passed as a dict, use its keys
    try:
        paths2dfs = paths2dfs.keys()
    except AttributeError:
        pass
        

    if 'sheet_name' not in {**kwargs}.keys():
        logging.warning('No sheet name supplied. Using first sheet.')
        
    # Read all dataframes
    dfs = []
    for path in paths2dfs:
        try:
            df = (
                read_excel2(path, **kwargs)
                # Drop empty rows AFTER selecting relevant columns 
                # (if usecols is in kwargs)
                .dropna(how='all', axis=0)  
            )
            dfs.append(df)
        except: # ValueError or XLRDError:
            logging.error(f'Error reading {path}')
            
            # Check if worksheet is named differently
            s_names = xlrd.open_workbook(path, on_demand=True) \
                          .sheet_names()
            raise logging.error(
                f'Sheet names do not exist. Available sheet names: {s_names}'
            )

    # Get unique sets of column names (should only be one if identical)
    # (Note the use of sets in case order is different)
    unique_colnames = set(
        frozenset(df.columns) for df in dfs
    )

    # If column names are not idential between dfs, check which column
    # names are common to all and which are not
    if len(unique_colnames) > 1:
        # Cols common to all dfs
        common_cols = functools.reduce(
            lambda x, y: x.intersection(y),
            unique_colnames
        )

        # All unique colnames
        all_columns = set()
        for df in dfs:
            # Set union
            all_columns.update(
                df.columns
            )

        # Cols NOT common to all dfs (set difference)
        non_common_cols = all_columns - common_cols
        
        logging.exception(
            'EXCEPTION: Column names differ between dataframes.\nCommon columns:\n' \
            f'{common_cols}\n\nColumns not in all dataframes:\n{non_common_cols}'
        )
        
        # Stop execution
        if stop_on_error:
            sys.exit()
            
        # If specified, return dataframes despite error
        else:
            return dfs
        
    else:
        print('Everything looks good! All dataframes contain the same column names.')
        return dfs
    
    
def find_start_rows_and_read(files, max_rows2skip=10):
    """
    Find the first row nr that doesn't have more than 50%
    missing values, then read files.
    
    
    Parameters
    ----------
    
    files : list
        paths to files to read
    max_rows2skip : int
        Maximum rows to skip at beginning of file before giving up search.
        
    
    Returns
    -------
    list
        Dataframes, read using first valid start row  
    """
    
    dfs = []
    
    def df_with_proportion_invalid(f, rows2skip):
        """Read a dataframe, return it along w proportion of valid colnames."""

        # Read data, starting in the first row
        df = read_excel2(f, skiprows=rows2skip)

        # Check proportion of invalid columns
        inv = df.columns.str.startswith('unnamed')

        return df, inv.mean()

    for file_nr, f in enumerate(files):
        for rows2skip in range(max_rows2skip+1):
#             pdb.set_trace()
            df, inv_prop = df_with_proportion_invalid(f, rows2skip)
            if inv_prop < 0.5:
                logging.info(f'Reached inv_prop={inv_prop} for file {file_nr}, skipping {rows2skip} rows')
                break

        # Save final dataframe
        dfs.append(df)
        
        # If we didn't find a proper start row for give file, give up
        if inv_prop >= 0.5:
            logger.warning(f'Did not find a good start row for file {file_nr}.')
            
    return dfs       
    