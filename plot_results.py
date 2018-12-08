"""
Author: Sungjae Cho
This file is a package containing functions used to plot experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from pprint import pprint


plot_fig_dir = 'plot_figures/results'
results_dir = 'user_data/results'
operators = ['add', 'subtract', 'multiply', 'divide', 'modulo']
columns = ['data_index', 'correct', 'solving_time', 'answer', 'truth',
    'operand_digits', 'operator', 'carries']
questions_per_carry_ds = {'add':10, 'subtract':10, 'multiply':5, 'divide':10, 'modulo':10}


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def filter_carries(df_result, carries):
    # Condition of carries
    cond_carries = (df_result['carries'] == carries)

    return df_result[cond_carries]


def filter_operator(df_result, operator):
    # Condition of the given operators
    cond_operator = (df_result['operator'] == operator)

    return df_result[cond_operator]


def get_unique_carries(df_result):
    carries_array = np.sort(df_result['carries'].unique())

    return carries_array


def get_all_result_files():
    result_files = list()
    list_dir = listdir(results_dir)
    for f in list_dir:
        file_path = join(results_dir, f)
        if isfile(file_path) and f != '_.txt':
            result_files.append(file_path)

    return result_files


def get_results(operator):
    '''
    Get all results of the given operator in the form of DataFrame.
    '''
    df_results = list()
    result_files = get_all_result_files()
    for f in result_files:
        if get_operator(f) == operator:
            df_result = read_result_file(f)
            df_results.append(df_result)

    return df_results


def get_total_result(operator=None):
    '''
    Get all results in one DataFrame.
    '''

    if operator == None:
        # Import results of all operators
        df_results = list()
        for op in operators:
            df_results = df_results + get_results(op)
    else:
        # Import results of the given operator
        df_results = get_results(operator)

    total_result = pd.concat(df_results, axis=0, join='outer', join_axes=None,
        ignore_index=False,keys=None, levels=None, names=None,
        verify_integrity=False, copy=True
    )

    return total_result


def get_operator(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline()
    operator_index = 6
    operator = first_line.split('\t')[6]

    return operator


def get_accuracy(groupby_operator=False, groupby_carries=False):
    total_result = get_total_result()

    if (groupby_operator == False) and (groupby_carries == False):
        accuracy = total_result['correct'].mean()

    if (groupby_operator == True) and (groupby_carries == False):
        accuracy = total_result.groupby(['operator'])['correct'].mean()

    if (groupby_operator == True) and (groupby_carries == True):
        accuracy = total_result.groupby(['operator', 'carries'])['correct'].mean()

    return accuracy

def get_mean_solving_time(groupby_operator=False, groupby_carries=False):
    total_result = get_total_result()

    if (groupby_operator == False) and (groupby_carries == False):
        mean_solving_time = total_result['solving_time'].mean()

    if (groupby_operator == True) and (groupby_carries == False):
        mean_solving_time = total_result.groupby(['operator'])['solving_time'].mean()

    if (groupby_operator == True) and (groupby_carries == True):
        mean_solving_time = total_result.groupby(['operator', 'carries'])['solving_time'].mean()

    return mean_solving_time


def read_result_file(file_path):
    df_results = pd.read_csv(file_path, sep='\t', header=None,
        names=columns)

    return df_results


def plot_solving_time(df_result, subject_index):
    # Retrieve the operator from the first row.
    operator = df_result['operator'][0]
    plt.title('{operator} subject[{sub_index}]'.format(
        operator=operatorcapitalize() ,
        sub_index=subject_index
    ))
    plt.xlabel('Experienced problems')
    plt.ylabel('Solving time (sec.)')

    x = np.arange(len(df_result))
    y = df_result['solving_time']
    plt.plot(x, y, ':o')
    plt.show()
    plt.clf()


def plot_solving_time_by_carries(df_result, subject_index, mode='save', file_format='pdf'):
    # Retrieve the operator from the first row.
    operator = df_result['operator'][0]

    carries_array = get_unique_carries(df_result)

    for carries in carries_array:
        plt.title('{operator} subject[{sub_index}] by carries'.format(
            operator=operator.capitalize() ,
            sub_index=subject_index,
            carries=carries
        ))
        plt.xlabel('Experienced problems')
        plt.ylabel('Solving time (sec.)')

        df_result_by_carries = filter_carries(df_result, carries)

        x = np.arange(len(df_result_by_carries))
        y = df_result_by_carries['solving_time']

        if carries <= 1:
            carries_label = '{} carry'.format(carries)
        else:
            carries_label = '{} carries'.format(carries)

        plt.plot(x, y, ':o', label=carries_label)
        plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(plot_fig_dir)
        plot_fig_path = '{plot_dir}/{operator}_subject-{sub_index}_by_carries.{extension}'.format(
            plot_dir=plot_fig_dir,
            operator=operator,
            sub_index=subject_index,
            carries=carries,
            extension=file_format
        )
        plt.savefig(plot_fig_path)
    plt.clf()


def plot_mean_solving_time(mode='save', file_format='pdf'):
    # for every operator
    for operator in operators:
        # len(df_results) == number_of_experiments
        df_results = get_results(operator)

        df_st_list = list()
        for df_result in df_results:
            # Select only the 'solving_time' column
            df_result = df_result[['solving_time']]
            # Reset row indexes to concatenate dataframes afterwards.
            df_result = df_result.reset_index(drop=True)
            df_st_list.append(df_result)
        # concatenate two dataframes row wise
        df_concat_st = pd.concat(df_st_list, axis=1)
        # Row wise mean.
        series_mean_solving_time = df_concat_st.mean(axis=1)

        # Data setting stage
        # x-axis and y-axis data
        x = list(series_mean_solving_time.keys())
        y = series_mean_solving_time.get_values()

        # Plot setting stage
        plt.title('Mean solving time')
        plt.xlabel('Experienced problems')
        plt.ylabel('Mean solving time (sec.)')

        # Plot stage
        plt.plot(x, y, ':o', label=operator)
        plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(plot_fig_dir)
        plot_fig_path = '{plot_dir}/mean_solving_time.{extension}'.format(
            plot_dir=plot_fig_dir,
            extension=file_format
        )
        plt.savefig(plot_fig_path)
    plt.clf()


def plot_mean_solving_time_for_each_operator(mode='save', file_format='pdf'):
    # for every operator
    for operator in operators:
        # len(df_results) == number_of_experiments
        df_results = get_results(operator)

        df_st_list = list()
        for df_result in df_results:
            # Select only the 'solving_time' column
            df_result = df_result[['solving_time']]
            # Reset row indexes to concatenate dataframes afterwards.
            df_result = df_result.reset_index(drop=True)
            df_st_list.append(df_result)
        # concatenate two dataframes row wise
        df_concat_st = pd.concat(df_st_list, axis=1)
        # Row wise mean.
        series_mean_solving_time = df_concat_st.mean(axis=1)

        # Data setting stage
        # x-axis and y-axis data
        x = list(series_mean_solving_time.keys())
        y = series_mean_solving_time.get_values()

        # Plot setting stage
        plt.title('[{operator}] Mean solving time'.format(
            operator=operator.capitalize()
        ))
        plt.xlabel('Experienced problems')
        plt.ylabel('Mean solving time (sec.)')

        # Plot stage
        plt.plot(x, y, ':o')

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(plot_fig_dir)
            plot_fig_path = '{plot_dir}/mean_solving_time_{operator}.{extension}'.format(
                plot_dir=plot_fig_dir,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path)
        plt.clf()


def plot_mean_solving_time_by_carries(mode='save', file_format='pdf'):
    # for every operator
    for operator in operators:
        # len(df_results) == number_of_experiments
        df_results = get_results(operator)
        carry_array = get_unique_carries(df_results[0])
        # for each carries
        for carries in carry_array:
            df_st_list = list()
            for df_result in df_results:
                # Filter carries
                df_result = filter_carries(df_result, carries)
                # Select only the 'solving_time' column
                df_result = df_result[['solving_time']]
                # Reset row indexes to concatenate dataframes afterwards.
                df_result = df_result.reset_index(drop=True)
                df_st_list.append(df_result)
            # concatenate two dataframes row wise
            df_concat_st = pd.concat(df_st_list, axis=1)
            # Row wise mean.
            series_mean_solving_time = df_concat_st.mean(axis=1)

            # Data setting stage
            # x-axis and y-axis data
            x = list(series_mean_solving_time.keys())
            y = series_mean_solving_time.get_values()

            # Plot setting stage
            if carries <= 1:
                carries_label = '{} carry'.format(carries)
            else:
                carries_label = '{} carries'.format(carries)

            plt.title('[{operator}] Mean solving time by carries'.format(
                operator=operator.capitalize()
            ))
            plt.xlabel('Experienced problems')
            plt.ylabel('Mean solving time (sec.)')

            # Plot stage
            plt.plot(x, y, ':o', label=carries_label)
            plt.legend()

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(plot_fig_dir)
            plot_fig_path = '{plot_dir}/mean_solving_time_by_carries_{operator}.{extension}'.format(
                plot_dir=plot_fig_dir,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path)
        plt.clf()


def plot_all(mode='save', file_format='pdf'):
    plot_mean_solving_time(mode=mode, file_format=file_format)
    plot_mean_solving_time_for_each_operator(mode=mode, file_format=file_format)
    plot_mean_solving_time_by_carries(mode=mode, file_format=file_format)
