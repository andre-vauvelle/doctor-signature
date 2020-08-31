import pandas as pd
from definitions import DATA_DIR
import os

event_data_path = 'hesinApril2019.tsv'
diag_event_data_path = 'hesin_diag10April2019.tsv'
event_data_path = os.path.join('raw', event_data_path)
diag_event_data_path = os.path.join('raw', diag_event_data_path)
patient_base_raw_path = 'patient_nov16.csv'
patient_base_raw_path = os.path.join('raw', patient_base_raw_path)

# patient_events = pd.read_csv(os.path.join(DATA_DIR, 'interim/202006012159_patient_events.csv'))
# cohort = pd.read_csv(os.path.join(DATA_DIR, 'raw/cohort_hf.csv'))

event_data = pd.read_csv(DATA_DIR + event_data_path, delimiter='\t')
diag_data = pd.read_csv(DATA_DIR + diag_event_data_path, delimiter='\t')
patient_base = pd.read_csv(os.path.join(DATA_DIR, patient_base_raw_path), low_memory=False)

# Spiros columns
patient_base.loc[:, "sex"] = patient_base.loc[:, "31-0.0"]
patient_base.loc[:, "yob"] = patient_base.loc[:, "34-0.0"]
patient_base.loc[:, "mob"] = patient_base.loc[:, "52-0.0"]
patient_base.loc[:, "dob"] = pd.to_datetime(patient_base.apply(
    lambda x: '{}/{}/01'.format(x.yob, x.mob) if x is not None else '{}/07/01'.format(x.yob), axis=1))
patient_base.loc[:, "center_ass"] = patient_base.loc[:, '54-0.0']
patient_base.loc[:, "year_ass"] = pd.to_datetime(patient_base.loc[:, '53-0.0']).dt.year
patient_base.loc[:, "age_ass"] = patient_base.loc[:, '21003-0.0']

# Get all hf events
hf_events_primary = event_data[
    (event_data.diag_icd10.str[:4].str.match('I50.|I110|I130|I132|I260')) & (event_data.admidate.notnull())]
diag_data_w_prim_record = pd.merge(event_data, diag_data, on=['eid', 'record_id'])
hf_events_secondary = diag_data_w_prim_record[
    diag_data_w_prim_record.diag_icd10_y.str[:4].str.match('I50.|I110|I130|I132|I260') & (
        diag_data_w_prim_record.admidate.notnull())]
hf_events = pd.concat([hf_events_primary, hf_events_secondary], axis=0)
# Mark first HF
hf_events.admidate = pd.to_datetime(hf_events.admidate)
first_hf_event = hf_events.groupby('eid').admidate.min().rename('d_hf_dx')
patient_base = pd.merge(patient_base, first_hf_event, on='eid', how='left', validate='one_to_one')

# needed_cols = ['eid', 'year_of_birth', 'sex', 'uk_biobank_assessment_centre', 'year_of_assessment', 'date_of_death']

# patient_base.loc[:, needed_cols].head()

# Find a columns
# column_headers[column_headers.UDI == '40000-1.0']

# Exclusion criteria for self reported HF
ex_self_reported = ['20002-0.0',
                    '20002-0.1',
                    '20002-0.2',
                    '20002-0.3',
                    '20002-0.4',
                    '20002-0.5',
                    '20002-0.6',
                    '20002-0.7',
                    '20002-0.8',
                    '20002-0.9',
                    '20002-0.10',
                    '20002-0.11',
                    '20002-0.12',
                    '20002-0.13',
                    '20002-0.14',
                    '20002-0.15',
                    '20002-0.16',
                    '20002-0.17',
                    '20002-0.18',
                    '20002-0.19',
                    '20002-0.20',
                    '20002-0.21',
                    '20002-0.22',
                    '20002-0.23',
                    '20002-0.24',
                    '20002-0.25',
                    '20002-0.26',
                    '20002-0.27',
                    '20002-0.28']

patient_base.loc[:, 'exclusions_self_reported'] = (patient_base.loc[:, ex_self_reported] == 1076).any(axis=1)
patient_base.loc[:, 'exclusions_dates'] = (2015 < patient_base.d_hf_dx.dt.year) | (patient_base.d_hf_dx.dt.year < 1997)

death_cols = ['40000-0.0', '40000-1.0', '40000-2.0']
deaths = patient_base.loc[:, death_cols]
deaths = deaths.fillna('').agg(''.join, axis=1)
deaths = pd.to_datetime(deaths.str[-10:])
patient_base.loc[:, 'dod'] = deaths
patient_base.loc[:, "dentry"] = pd.to_datetime(patient_base.loc[:, "53-0.0"])
patient_base.loc[:, "exclusions_dentry"] = patient_base.dentry > patient_base.dod

patient_base_basic = patient_base.loc[:, ['eid', 'sex', 'yob', 'mob', 'dob', 'center_ass', 'year_ass', 'age_ass',
                                          'd_hf_dx', 'exclusions_dentry', 'exclusions_self_reported',
                                          'exclusions_dates',
                                          'dod', 'dentry']]

# patient_base_basic = patient_base_basic[patient_base_basic.eid.isin(event_data.eid)]

patient_base_basic.to_csv(os.path.join(DATA_DIR, 'interim', 'patient_base_basic.csv'))

hf_cohort = patient_base_basic[
    ((patient_base_basic.loc[:, ["exclusions_dentry", "exclusions_self_reported", "exclusions_dates"]] == False).all(
        axis=1)) &
    (patient_base_basic.d_hf_dx.notnull())
    ]

hf_cohort.to_csv(os.path.join(DATA_DIR, 'interim', 'hf_cohort.csv'))


patient_base = pd.read_csv(os.path.join(DATA_DIR, 'interim', 'patient_base_basic.csv'), index_col=0, parse_dates=['d_hf_dx', 'dentry', 'dob'])
