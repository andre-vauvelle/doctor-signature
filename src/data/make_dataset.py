# -*- coding: utf-8 -*-
import os
import click
import random
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from itertools import chain

from definitions import DATA_DIR, ROOT_DIR
from src.omni.functions import save_pickle, load_pickle, save_list2csv

TEST_RATIO = 0.1
VALIDATION_RATIO = 0

random.seed(42)
MAX_CODE_LENGTH = 4


# TODO: implement click for commandline make
class BiobankDataset:
    def __init__(self,
                 patient_base_raw_path='patient_nov16.csv',
                 column_header_path='baselinedictionary_firstoccurance_992.csv',
                 cohort_path='hf_cohort.csv',
                 event_data_path='hesinApril2019.tsv',
                 diag_event_data_path='hesin_diag10April2019.tsv',
                 opcs_event_data_path='hesin_operApril2019.tsv',
                 verbose=True,
                 ):
        self.patient_base_raw_path = os.path.join('raw', patient_base_raw_path)
        self.patient_base_basic_path = os.path.join('interim', 'patient_base_basic.csv')
        self.column_header_path = os.path.join('external', column_header_path)
        self.cohort_path = os.path.join('interim', cohort_path)
        self.event_data_path = os.path.join('raw', event_data_path)
        self.diag_event_data_path = os.path.join('raw', diag_event_data_path)
        self.opcs_event_data_path = os.path.join('raw', opcs_event_data_path)
        self.verbose = verbose

    def get_patient_table(self) -> pd.DataFrame:
        """
        Extracts the raw patient baseline data. Also adds heart failure case column.
        :return patient_base: A dataframe with all patient level info
        """
        # You might need to run patient_base_and_hf_cohort.py here
        patient_base = pd.read_csv(os.path.join(DATA_DIR, self.patient_base_basic_path),
                                   parse_dates=['d_hf_dx', 'dentry', 'dob', 'dod'])

        return patient_base

    def clean_patient_table(self) -> pd.DataFrame:
        """
        :param patient_base_basic_path:
        :param basic_columns: The sub set of columns to keep (for cohort matching)
        :return patient_base_basic: just whats needed for cohort matching
        """
        patient_base = self.get_patient_table()

        # apply exclusions
        patient_base = patient_base[((patient_base.loc[:,
                                      ["exclusions_dentry", "exclusions_self_reported",
                                       "exclusions_dates"]] == False).all(axis=1)) & patient_base.dod.isnull()]
        # patients must have event data to be included!
        event_data = pd.read_csv(DATA_DIR + self.event_data_path, delimiter='\t')
        patient_base = patient_base[patient_base.eid.isin(event_data.eid)]

        patient_base.loc[:, 'is_case'] = (patient_base.d_hf_dx.notnull())

        return patient_base

    def get_case_controls_cohort(self,
                                 patient_base,
                                 match_on=(
                                         'yob', 'sex', 'center_ass', 'year_ass'),
                                 n_controls=9) -> pd.DataFrame:
        """
        Gets n controls matched on specified variables.

        Drop excluded participants, patients that died
        and any patients that did not have at least one
        consultation during their entire history.

        :param match_on: what to match patients on
        :param n_controls: number of controls per patient with a case
        :return case_controls: a dataframe with patient cases with controls
        """

        patient_base.loc[:, 'is_case'] = ((patient_base.loc[:,
                                           ["exclusions_dentry", "exclusions_self_reported",
                                            "exclusions_dates"]] == False).all(
            axis=1)) & (patient_base.d_hf_dx.notnull())
        cases = patient_base[patient_base.is_case]
        controls = patient_base[~patient_base.is_case]

        match_candidates = pd.merge(cases, controls, on=match_on, how='left')
        match_candidates.rename(columns={'eid_x': 'case_id', 'eid_y': 'control_id'}, inplace=True)

        case_ids = match_candidates.case_id.unique()
        # Loop through every case and try to extract the required number of control case_controls
        case_control_store = []
        for case_id in tqdm(case_ids):
            # Extract one case
            case_control_batch = match_candidates[match_candidates.case_id == case_id]

            # If number of controls in batch is exact keep the same
            if case_control_batch.shape[0] == n_controls:
                pass
            # Reject batches where less than required number of controls found
            elif case_control_batch.shape[0] < n_controls:
                continue
            # Random subsample where more controls than needed in batch
            elif case_control_batch.shape[0] > n_controls:
                case_control_batch = case_control_batch.sample(n_controls)
            case_control_store.append(case_control_batch)

            # Remove the controls from all other match candidates
            match_candidates = match_candidates[~match_candidates.control_id.isin(case_control_batch.control_id)]

        case_controls = pd.concat(case_control_store, axis=0)

        if case_controls.control_id.unique().shape[0] != case_controls.shape[0]:
            raise Exception(
                'Controls duplicated! N unique: {}, N Total: {}'.format(case_controls.control_id.unique().shape[0],
                                                                        case_controls.shape[0]))

        if self.verbose:
            total_cases = case_controls.case_id.unique().shape[0]
            total_cases_org = cases.shape[0]
            print(
                'Total cases with case_controls found: {} \nTotal cases lost: {} from original: {} \n'.format(
                    total_cases,
                    total_cases_org - total_cases,
                    total_cases_org) + 'Average age at assessment {}, Percentage male: {}%'.format(
                    round((case_controls.year_ass - case_controls.yob).mean(), 2),
                    100 * round(case_controls.sex.mean(), 2)
                )
            )

        return case_controls

    def get_patient_events(self,
                           case_controls,
                           exclude_post_diag=True,
                           all_events=False,
                           buffer=30 * 6) -> pd.DataFrame:
        """
        Gets the event data for the eids in your case_controls

        For example for a diagnosis on 1999-04-02 buffer=6 will return
        all diagnoses occuring before six days prior to
        1999-04-02.

        Patients with no data will be kept around...
        Patients with no data added using one row with
        code_type - diag_icd10
        code - <MISSING>
        timedelta - 0 seconds

        :param all_events: hacky if true then ignore case control ids and get all events. This is used for word2vec training
        :param exclude_post_diag: if false then only events prediagnosis and
        :param buffer: days before diagnosis to exclude!
        :return patient_event_data:
        """
        event_data = pd.read_csv(DATA_DIR + self.event_data_path, delimiter='\t')

        # Add secondary codes
        id_vars = ['eid', 'record_id']
        diag_data = pd.read_csv(DATA_DIR + self.diag_event_data_path, delimiter='\t')
        diag_data.rename(columns={'diag_icd10': 'secondary_icd10'}, inplace=True)
        opcs_data = pd.read_csv(DATA_DIR + self.opcs_event_data_path, delimiter='\t')

        # Concat all the codes with unique columns
        event_data_ = pd.concat([event_data.loc[:, id_vars + ['diag_icd10']],
                                 diag_data.loc[:, id_vars + ['secondary_icd10']],
                                 opcs_data.loc[:, id_vars + ['oper4']]], axis=0)
        # Melt so that each row is an event with a code and code type
        event_data_norm = event_data_.melt(id_vars=id_vars, value_vars=['diag_icd10', 'secondary_icd10', 'oper4'],
                                           var_name='code_type', value_name='code')

        # Merge to get back back epistart date to each event (time is the episode start date of the record id)
        event_data = pd.merge(event_data.loc[:, id_vars + ['epistart']], event_data_norm, on=id_vars)
        event_data = event_data.dropna(subset=['code', 'epistart'], how='any')

        event_data.code = event_data.code.str[:MAX_CODE_LENGTH]  # remove extra subsub chapter details

        if not all_events:
            patient_event_data = event_data[
                event_data.eid.isin(pd.concat([case_controls.case_id, case_controls.control_id]))].copy()
            # Find cases
            patient_event_data.loc[:, 'is_case'] = event_data.eid.isin(case_controls.case_id)

            if exclude_post_diag:
                case_controls.loc[:, 'd_hf_dx'] = pd.to_datetime(case_controls.d_hf_dx_x)
                patient_event_data.epistart = pd.to_datetime(patient_event_data.epistart)
                patient_data_store = []
                for case_id, control_id_series in list(case_controls.groupby('case_id').control_id):
                    ids = control_id_series.to_list()
                    ids.append(case_id)
                    diagnosis_date = case_controls[case_controls.case_id == case_id].iloc[0, :].d_hf_dx

                    patient_data_batch = patient_event_data[patient_event_data.eid.isin(ids)].copy()
                    patient_data_batch.loc[:, 'timedelta'] = (pd.Timestamp(
                        (diagnosis_date - timedelta(
                            days=buffer))) - patient_event_data.epistart)  # Calc time delta from buffer
                    patient_data_batch.timedelta = patient_data_batch.timedelta.apply(lambda x: x.total_seconds())  #
                    patient_data_batch = patient_data_batch[
                        patient_data_batch.timedelta >= 0]  # remove all data within buffer

                    # Patients with no data for primary diag re-added using one row
                    missing_ids = set(ids) - set(patient_data_batch.eid)
                    missing_patient_data = pd.DataFrame(
                        np.empty(shape=(len(missing_ids), patient_data_batch.shape[1])).fill(np.nan),
                        columns=patient_data_batch.columns)
                    missing_patient_data.loc[:, 'eid'] = list(missing_ids)
                    missing_patient_data.loc[:, 'is_case'] = (missing_patient_data.eid == case_id)
                    missing_patient_data.loc[:, 'code_type'] = 'diag_icd10'
                    missing_patient_data.loc[:, 'code'] = '<MISSING>'
                    missing_patient_data.loc[:, 'timedelta'] = 0
                    patient_data_store.append(patient_data_batch)
                    patient_data_store.append(missing_patient_data)

                patient_event_data = pd.concat(patient_data_store)
        else:
            patient_event_data = event_data
            patient_event_data.epistart = pd.to_datetime(patient_event_data.epistart)
            time_sequences_ = patient_event_data \
                .groupby(['eid'])['epistart'].apply(list) \
                .groupby(level=0).apply(lambda x: list(chain.from_iterable(x)))
            time_sequences = time_sequences_.apply(
                lambda x: [0] if pd.isnull(x[0]) else [(x[-1] - t).total_seconds() for t in x])

            patient_event_data.loc[:, 'timedelta'] = time_sequences

        # Add on the first letter of the code_type to distinguish codes among each source
        patient_event_data.code = patient_event_data.code + patient_event_data.code_type.str[:1]

        return patient_event_data

    # TODO: this is a bit of hack magic
    @staticmethod
    def get_sequences(patient_event_data, order_by='epistart', group_sequences=('eid', 'code_type')):
        """
        Turns a dataframe of events into a list of events for each patient

        :param group_sequences: A tuple containing the variables the sequences to by.. In reality should only be ('eid',) or ('eid', 'code_type')
        :param patient_event_data: a dataframe containing eid and event name
        :param order_by: events will be sorted by datetime objects in this column
        :return eid_list, sequences, timestamp_sequences, labels: list of list, outer is each patient, inner is events
        """
        if order_by:
            patient_event_data.loc[:, order_by] = pd.to_datetime(patient_event_data.loc[:, order_by])
            patient_event_data = patient_event_data.sort_values(['eid', order_by])

        # list events
        patient_sequences = patient_event_data \
            .groupby(list(group_sequences))['code'].apply(list) \
            .groupby(level=list(range(len(group_sequences)))).apply(
            lambda x: list(chain.from_iterable(x)))  # Coerce to Series of lists each row an eid
        # Get time differences in seconds
        time_sequences = patient_event_data \
            .groupby(list(group_sequences))['timedelta'].apply(list) \
            .groupby(level=list(range(len(group_sequences)))).apply(lambda x: list(chain.from_iterable(x)))

        try:
            labels = patient_event_data \
                .groupby(['eid'])['is_case'].apply(all)
            labels = labels.astype(str).values
        except KeyError:
            labels = None
            print('WARNING: no cases defined')

        # TODO: bit of a hack could do with refactor
        if len(group_sequences) == 2:
            eid_list = patient_sequences.index.get_level_values('eid').astype(int).values
            # Get a dictionary for each patient with each value being a list of codes/timestamps
            sequences = []
            for eid, data in patient_sequences.groupby('eid'):
                sequences.append(data.reset_index(level='eid', drop=True).to_dict())
            timestamp_sequences = []
            for eid, data in time_sequences.groupby('eid'):
                timestamp_sequences.append(data.reset_index(level='eid', drop=True).to_dict())
        else:
            eid_list = patient_sequences.index.astype(int).values
            sequences = patient_sequences.values
            timestamp_sequences = time_sequences.values

        # Because I'm paranoid
        if not (time_sequences.apply(len) == patient_sequences.apply(len)).all():
            raise Exception('Lengths not matching between events and times')
        if not (patient_sequences.index == time_sequences.index).all():
            raise Exception('IDs not matching between times and events')

        return eid_list, sequences, timestamp_sequences, labels

    @staticmethod
    def train_validate_test_splitter(sequences: list, timestamp_sequences: list, labels: list, eids: list,
                                     test_ratio=0.1,
                                     validation_ratio=0.1):
        """
        Splits up the data into train valid and test sets

        :param sequences: np.array
        :param timestamp_sequences:  np.array
        :param labels:  np.array
        :param test_ratio: percentage of total data in test set
        :param validation_ratio: percentage of total data in validation set
        :return:
        """
        data_size = len(sequences)
        np.random.seed(0)

        ind = np.random.permutation(data_size)
        n_test = int(test_ratio * data_size)
        n_valid = int(validation_ratio * data_size)

        test_indices = ind[:n_test]
        valid_indices = ind[n_test:n_test + n_valid]
        train_indices = ind[n_test + n_valid:]

        indices = {'test': test_indices, 'valid': valid_indices, 'train': train_indices}
        data_dict = {'x': sequences, 'y': labels, 't': timestamp_sequences, 'eid': eids}

        # Index into each seperate split
        store = {'train': {}, 'valid': {}, 'test': {}}
        for feature_n, data in data_dict.items():
            data = np.array(data)
            for n, ind in indices.items():
                if data.any() and ind.any():
                    d_set = data[ind]
                    store[n].update({feature_n: d_set})
                else:
                    store[n].update({feature_n: None})

        def len_argsort(seq):
            lengths = []
            for s in seq:
                lengths.append(sum([len(v) for v in s]))
            return np.array(lengths).argsort()

        # Sort by sequence length
        for n, feature_dict in store.items():
            x = feature_dict['x']
            sort_index = len_argsort(x) if x is not None else None
            for feature_n, feature in feature_dict.items():
                feature = [feature[i] for i in sort_index] if feature is not None else None
                store[n].update({feature_n: feature})

        return store['train'], store['valid'], store['test']

    def run(self,
            n_controls,
            code_types,
            rerun=('patient_base_load', 'case_controls_load', 'get_patient_events', 'get_sequences'),
            date_string=None,
            patient_base_path=None,
            patient_event_path=None,
            case_controls_path=None,
            sequences_path=None,
            train_valid_test_folder=None
            ):
        """
        Runs data processing scripts to turn raw data from (../ raw) into cleaned data ready to be analyzed(saved in.. /
        processed).
        """

        logger = logging.getLogger(__name__)
        logger.info('Rerunning {}'.format(' '.join(rerun)))

        # Define default formatting for folders if not overridden
        date_string = datetime.now().strftime("%Y%m%d%H%M") if date_string is None else date_string

        code_types_str = '_'.join(code_types) if code_types is not None else code_types
        patient_base_path = os.path.join('interim', '_'.join(
            [date_string, 'patient_base.csv'])) if patient_base_path is None else patient_base_path
        case_controls_path = os.path.join('interim', '_'.join(
            [date_string, 'case_controls.csv', str(n_controls)])) if case_controls_path is None else case_controls_path
        patient_event_path = os.path.join('interim', '_'.join(
            [date_string, 'patient_events.csv'])) if patient_event_path is None else patient_event_path
        sequences_path = os.path.join('interim', '_'.join(
            [date_string, 'sequences', str(n_controls), code_types_str])) if sequences_path is None else sequences_path
        train_valid_test_path = os.path.join('processed', '_'.join(['train_valid_test', str(n_controls),
                                                                    code_types_str])) if train_valid_test_folder is None else train_valid_test_folder

        # Run each step if in rerun else load
        if 'patient_base' in rerun:
            patient_base = self.clean_patient_table()
            patient_base.to_csv(os.path.join(DATA_DIR, patient_base_path), index=False)
            logger.info('Successfully reran patient base: {}'.format(patient_base_path))
        elif 'patient_base_load' in rerun:
            patient_base = pd.read_csv(os.path.join(DATA_DIR, patient_base_path))
            logger.info('Successfully loaded patient base: {}'.format(patient_base_path))
        else:
            logger.info('Skipping patient base')

        if 'case_controls' in rerun:
            case_controls = self.get_case_controls_cohort(
                patient_base,
                n_controls=n_controls)
            case_controls.to_csv(os.path.join(DATA_DIR, case_controls_path), index=False)
            logger.info('Successfully reran case controls: {}'.format(case_controls_path))
        elif 'case_controls_load' in rerun:
            case_controls = pd.read_csv(os.path.join(DATA_DIR, case_controls_path))
        else:
            logger.info('Skipping case controls')

        if 'get_patient_events' in rerun:
            patient_event_data = self.get_patient_events(case_controls=case_controls)
            patient_event_data.to_csv(os.path.join(DATA_DIR, patient_event_path), index=False)
            if self.verbose:
                n_cases = patient_event_data[patient_event_data.is_case].eid.unique().shape[0]
                n_controls = patient_event_data[~patient_event_data.is_case].eid.unique().shape[0]
                print('Total cases remaining: {}, Total controls {}, Ratio: {}, Total events: {},'.format(
                    n_cases, n_controls, n_controls / n_cases, patient_event_data.shape[0]))
            logger.info('Successfully reran patient events: {}'.format(patient_event_path))
        elif 'get_patient_events_load' in rerun:
            patient_event_data = pd.read_csv(os.path.join(DATA_DIR, patient_event_path))
        else:
            logger.info('Skipping patient events')

        if 'get_sequences' in rerun:
            # Filter out the events we want
            patient_event_data = patient_event_data[patient_event_data.code_type.isin(code_types)]
            eids, sequences, timestamp_sequences, labels = self.get_sequences(patient_event_data)
            sequences_dict = {'eids': eids,
                              'sequences': sequences,
                              'timestamp_sequences': timestamp_sequences,
                              'labels': labels}
            for name, obj in sequences_dict.items():
                save_pickle(obj=obj, filename=os.path.join(DATA_DIR, sequences_path, name + '.dill'), use_dill=True)
            # save_list2csv(obj=sequences, filename=os.path.join(sequences_path, 'sequences.csv'))
            logger.info('Successfully reran get_sequences')
        else:
            eids = load_pickle(filename=os.path.join(DATA_DIR, 'interim', sequences_path, 'eids.dill'))
            sequences = load_pickle(filename=os.path.join(DATA_DIR, 'interim', sequences_path, 'sequences.dill'))
            timestamp_sequences = load_pickle(
                filename=os.path.join(DATA_DIR, 'interim', sequences_path, 'timestamp_sequences.dill'))
            labels = load_pickle(filename=os.path.join(DATA_DIR, 'interim', sequences_path, 'labels.dill'))

        train_set, valid_set, test_set = self.train_validate_test_splitter(
            sequences, timestamp_sequences, labels, eids,
            test_ratio=TEST_RATIO, validation_ratio=VALIDATION_RATIO)

        save_pickle(obj=train_set, filename=os.path.join(DATA_DIR, train_valid_test_path, 'train.dill'), use_dill=True)
        save_pickle(obj=valid_set, filename=os.path.join(DATA_DIR, train_valid_test_path, 'valid.dill'), use_dill=True)
        save_pickle(obj=test_set, filename=os.path.join(DATA_DIR, train_valid_test_path, 'test.dill'), use_dill=True)

        logger.info(
            'Successfully finished data pipeline with train, valid, test sets: {}'.format(train_valid_test_path))

    def run_embeddings(self, code_types, rerun=(),
                       patient_event_path=None,
                       train_valid_test_path=os.path.join(DATA_DIR, 'processed/train_embeddings/')):
        """
        Creates a pickle (.dill) that will be used in the src.features.event_embedding_features.get_wordvectors

        :param code_types: which code types to include ('diag_icd10', 'secondary_icd10', 'oper4')
        :param rerun: which parts of the pipeline to rerun, can include  'get_patient_events'
        :param patient_event_path: location to save (if rerunning) or load patient events df from
        :param train_valid_test_path: location to save final pickle .dill to
        """

        logger = logging.getLogger(__name__)
        logger.info('Getting all events!')

        date_string = datetime.now().strftime("%Y%m%d%H%M")
        patient_event_path = os.path.join('interim', '_'.join(
            [date_string, '_'.join(code_types),
             'all_events.csv'])) if patient_event_path is None else patient_event_path

        if 'get_patient_events' in rerun:
            patient_event_data = self.get_patient_events(case_controls=None, all_events=True)
            patient_event_data.to_csv(os.path.join(DATA_DIR, patient_event_path), index=False)
            logger.info('Successfully reran patient events: {}'.format(patient_event_path))
        else:
            patient_event_data = pd.read_csv(os.path.join(DATA_DIR, patient_event_path))
            logger.info('Successfully loaded patient events: {}'.format(patient_event_path))

        # Restrict code_types
        patient_event_data = patient_event_data[patient_event_data.code_type.isin(code_types)]

        logger.info('All events retrieved \n {}'.format(patient_event_data.code_type.value_counts()))

        eids, sequences, timestamp_sequences, labels = self.get_sequences(patient_event_data, order_by='epistart',
                                                                          group_sequences=('eid',))

        train_set, valid_set, test_set = self.train_validate_test_splitter(sequences, timestamp_sequences, labels, eids,
                                                                           test_ratio=0, validation_ratio=0)
        logger.info('Events sequenced')

        save_path = os.path.join(train_valid_test_path, 'train_embeddings_full_{}.dill'.format('_'.join(code_types)))
        save_pickle(obj=train_set, filename=save_path, use_dill=True)

        logger.info('Saved at ' + save_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    BiobankDataset().run(n_controls=9, patient_base_path='interim/patient_base_basic.csv',
                         case_controls_path='interim/20200702001_case_controls_9.csv',
                         patient_event_path='interim/20200724001_patient_events.csv',
                         rerun=('case_controls_load', 'get_patient_events', 'get_sequences'),
                         train_valid_test_folder=os.path.join('processed', 'train_test_PRIMDX'),
                         code_types=('diag_icd10',))

    BiobankDataset().run(n_controls=9, patient_base_path='interim/patient_base_basic.csv',
                         case_controls_path='interim/20200702001_case_controls_9.csv',
                         patient_event_path='interim/20200724001_patient_events.csv',
                         rerun=('get_patient_events_load', 'get_sequences',),
                         train_valid_test_folder=os.path.join('processed', 'train_test_PRIMDX_SECDX_PROC'),
                         code_types=('diag_icd10', 'secondary_icd10', 'oper4'))

    BiobankDataset().run_embeddings(code_types=('diag_icd10',),
                                    patient_event_path=os.path.join(
                                        'interim',
                                        '202005271611_diag_icd10_secondary_icd10_oper4_all_events.csv'),
                                    rerun=())
