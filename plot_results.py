"""
Author: Sungjae Cho
This file is a package containing functions used to plot experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import data_utils
from os import listdir
from os.path import isfile, join
from pprint import pprint


dir_plot_fig = 'plot_figures/results'
dir_results = 'user_data/results'
dir_st = 'user_data/results_csv/solving_time'
dir_st_correct = 'user_data/results_csv/solving_time_correct'
dir_accuracy = 'user_data/results_csv/accuracy'
operators = ['add', 'subtract', 'multiply', 'divide', 'modulo']
columns = ['data_index', 'correct', 'solving_time', 'answer', 'truth',
    'operand_digits', 'operator', 'carries']
problems_per_carry_ds = {'add':10, 'subtract':10, 'multiply':5, 'divide':10, 'modulo':10}
solving_time_normalized = False
solving_time_correctness = True
errorbar_std = 1
font_size = {'xlabel':20, 'ylabel':17, 'xtick':20, 'ytick':10}


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def filter_carries(df_result, carries):
    # Condition of carries
    cond_carries = (df_result['carries'] == carries)

    return df_result[cond_carries]


def filter_correct(df_result, correctness):
    # Condition of the given operators
    cond_operator = (df_result['correct'] == correctness)

    return df_result[cond_operator]


def filter_operator(df_result, operator):
    # Condition of the given operators
    cond_operator = (df_result['operator'] == operator)

    return df_result[cond_operator]


def filter_for_mean_solving_time(df_result):
    if solving_time_normalized:
        df_result = normalize_solving_time(df_result)
    if solving_time_correctness:
        df_result = filter_correct(df_result, True)

    return df_result


def normalize_solving_time(df_result):
    df_result[['solving_time']] = df_result[['solving_time']] * np.std(df_result[['solving_time']])

    return df_result


def get_unique_carries(df_result):
    carries_array = np.sort(df_result['carries'].unique())

    return carries_array


def get_all_result_files():
    result_files = list()
    list_dir = listdir(dir_results)
    for f in list_dir:
        file_path = join(dir_results, f)
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


def count_results(operator=None):
    ''''
    Count how many result files for a particular operator or all operators.
    '''
    result_files = get_all_result_files()
    if operator == None:
        n_results = len(result_files)
    else:
        n_results = 0
        for f in result_files:
            if get_operator(f) == operator:
                n_results = n_results + 1

    return n_results


def print_count_results():
    '''
    Print how many result files for all operators and each operator.
    '''
    print("Total results: {}".format(count_results()))
    for operator in operators:
        print("{} results: {}".format(
            operator.capitalize(),
            count_results(operator)))


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
    '''
    Return a string of an operator where a particular result file used.
    '''
    with open(file_path, 'r') as f:
        first_line = f.readline()
    operator_index = 6
    operator = first_line.split('\t')[6]

    return operator


def get_accuracy(groupby_operator=False, groupby_carries=False):
    '''
    Get accuracy of all problem instances across all participants.
    '''
    total_result = get_total_result()

    if (groupby_operator == False) and (groupby_carries == False):
        accuracy = total_result['correct'].mean()

    if (groupby_operator == True) and (groupby_carries == False):
        accuracy = total_result.groupby(['operator'])['correct'].mean()

    if (groupby_operator == True) and (groupby_carries == True):
        accuracy = total_result.groupby(['operator', 'carries'])['correct'].mean()

    return accuracy


def get_mean_solving_time(groupby_operator=False, groupby_carries=False):
    '''
    Get the mean of all solving time instances across all participants.
    '''
    total_result = get_total_result()

    if (groupby_operator == False) and (groupby_carries == False):
        mean_solving_time = total_result['solving_time'].mean()
        std_solving_time = total_result['solving_time'].std()

    if (groupby_operator == True) and (groupby_carries == False):
        mean_solving_time = total_result.groupby(['operator'])['solving_time'].mean()
        std_solving_time = total_result.groupby(['operator'])['solving_time'].std()

    if (groupby_operator == True) and (groupby_carries == True):
        mean_solving_time = total_result.groupby(['operator', 'carries'])['solving_time'].mean()
        std_solving_time = total_result.groupby(['operator', 'carries'])['solving_time'].std()

    return mean_solving_time, std_solving_time


def get_accuracy_by_operator():
    '''
    Get all instances of the mean solving time for each participant grouped by operator.
    Returns
    - df : pandas.dataframe. Each row has accuracy of a person.
    '''
    df_accuracy_operator_list = list()

    for operator in operators:
        df_results = get_results(operator)
        for i in range(len(df_results)):
            df_results[i][['correct']] = df_results[i][['correct']].astype('int')
            df_accuracy_operator = df_results[i].groupby(['operator'], as_index=False)['correct'].mean().rename(columns={'correct':'accuracy'})
            df_accuracy_operator_list.append(df_accuracy_operator)

    df_accuracy_operator = pd.concat(df_accuracy_operator_list, axis=0)
    mean_accuracy_by_operator = df_accuracy_operator.groupby(['operator'])['accuracy'].mean()
    std_accuracy_by_operator = df_accuracy_operator.groupby(['operator'])['accuracy'].std()

    return mean_accuracy_by_operator, std_accuracy_by_operator, df_accuracy_operator


def get_accuracy_by_carries(operator):
    '''
    Get all instances of the mean solving time for each participant grouped by carries if an operator is given.
    '''
    df_results = get_results(operator)

    df_accuracy_carries_list = list()

    for i in range(len(df_results)):
        df_accuracy_carries = df_results[i].groupby(['carries'], as_index=False)['correct'].mean().rename(columns={'correct':'accuracy'})
        df_accuracy_carries_list.append(df_accuracy_carries)
    df_accuracy_carries = pd.concat(df_accuracy_carries_list, axis=0)
    mean_accuract_by_carries = df_accuracy_carries.groupby(['carries'])['accuracy'].mean()
    std_accuracy_by_carries = df_accuracy_carries.groupby(['carries'])['accuracy'].std()

    return mean_accuract_by_carries, std_accuracy_by_carries, df_accuracy_carries


def get_mean_solving_time_by_operator():
    '''
    Returns
    - df : pandas.dataframe. Each row has the mean solving time of a person.
    '''
    df_mean_st_operator_list = list()

    for operator in operators:
        df_results = get_results(operator)
        for i in range(len(df_results)):
            df_results[i] = filter_for_mean_solving_time(df_results[i])
        for i in range(len(df_results)):
            df_mean_st_operator = df_results[i].groupby(['operator'], as_index=False)['solving_time'].mean().rename(columns={'solving_time':'mean_solving_time'})
            df_mean_st_operator_list.append(df_mean_st_operator)

    df_mean_st_operator = pd.concat(df_mean_st_operator_list, axis=0)
    mean_mean_solving_time_by_operator = df_mean_st_operator.groupby(['operator'])['mean_solving_time'].mean()
    std_mean_solving_time_by_operator = df_mean_st_operator.groupby(['operator'])['mean_solving_time'].std()

    return mean_mean_solving_time_by_operator, std_mean_solving_time_by_operator, df_mean_st_operator


def get_mean_solving_time_by_carries(operator):
    df_results = get_results(operator)
    for i in range(len(df_results)):
        df_results[i] = filter_for_mean_solving_time(df_results[i])

    df_mean_st_carries_list  = list()

    for i in range(len(df_results)):
        df_mean_st_carries = df_results[i].groupby(['carries'], as_index=False)['solving_time'].mean().rename(columns={'solving_time':'mean_solving_time'})
        df_mean_st_carries_list.append(df_mean_st_carries)
    df_mean_st_carries = pd.concat(df_mean_st_carries_list, axis=0)
    mean_mean_solving_time_by_carries = df_mean_st_carries.groupby(['carries'])['mean_solving_time'].mean()
    std_mean_solving_time_by_carries = df_mean_st_carries.groupby(['carries'])['mean_solving_time'].std()

    return mean_mean_solving_time_by_carries, std_mean_solving_time_by_carries, df_mean_st_carries


def read_result_file(file_path):
    df_results = pd.read_csv(file_path, sep='\t', header=None,
        names=columns)

    return df_results


def plot_accuracy_by_operator(mode='save', file_format='pdf'):

    total_accuracy = get_accuracy(groupby_operator=False, groupby_carries=False)
    accuracy_by_operator = get_accuracy(groupby_operator=True, groupby_carries=False)

    # = ('Add', 'Subtract', 'Multiply', 'Divide', 'Modulo')
    x = ['+', '−', '×', '÷', 'mod']
    y = [accuracy_by_operator['add'],
        accuracy_by_operator['subtract'],
        accuracy_by_operator['multiply'],
        accuracy_by_operator['divide'],
        accuracy_by_operator['modulo']
    ]

    plt.figure(figsize=(len(x)-1,4))

    plt.ylim(0.8, 1.0)
    plt.yticks(np.arange(0.8, 1.0, step=0.05))
    plt.grid(axis='y')
    #plt.xlabel('Operator', fontsize=font_size['xlabel'])
    plt.ylabel('Accuracy', fontsize=font_size['ylabel'])
    plt.tick_params(axis='x', labelsize=font_size['xtick'])
    plt.tick_params(axis='y', labelsize=font_size['ytick'])
    #plt.title('Accuracy by operator')

    plt.plot(x, y, 'r:o', label='Each operator')
    #plt.bar(x, y, align='center')
    plt.hlines(total_accuracy, xmin=-0.5, xmax=len(x)-0.5, colors='g', label='All operators')
    plt.legend(loc='lower left')

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(dir_plot_fig)
        plot_fig_path = '{plot_dir}/accuracy_by_operator.{extension}'.format(
            plot_dir=dir_plot_fig,
            extension=file_format
        )
        plt.savefig(plot_fig_path, bbox_inches='tight')
    plt.clf()


def plot_accuracy_by_carries(mode='save', file_format='pdf'):

    accuracy_by_operator = get_accuracy(groupby_operator=True, groupby_carries=False)
    accuracy_by_carries = get_accuracy(groupby_operator=True, groupby_carries=True)

    for operator in operators:

        carries_list = list(accuracy_by_carries[operator].keys())
        x = [str(carries) for carries in carries_list]
        y = list(accuracy_by_carries[operator].get_values())

        plt.figure(figsize=(len(x)-1,4))
        plt.ylim(0.8, 1.0)
        plt.yticks(np.arange(0.8, 1.05, step=0.05))
        plt.grid(axis='y')
        plt.ylabel('Accuracy')
        plt.xlabel('Carries', fontsize=font_size['xlabel'])
        plt.ylabel('Accuracy', fontsize=font_size['ylabel'])
        plt.tick_params(axis='x', labelsize=font_size['xtick'])
        plt.tick_params(axis='y', labelsize=font_size['ytick'])
        #plt.title('[{operator}] Accuracy by carries'.format(operator=operator.capitalize()))

        #plt.bar(x, y, align='center')
        plt.plot(x, y, 'r:o', label='Carry datasets')
        plt.hlines(accuracy_by_operator[operator],
            xmin=-0.5, xmax=len(x)-0.5, colors='g',
            label='Operation dataset'.format(operator=operator.capitalize()))
        plt.legend(loc='lower left')

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(dir_plot_fig)
            plot_fig_path = '{plot_dir}/accuracy_by_carries_{operator}.{extension}'.format(
                plot_dir=dir_plot_fig,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path, bbox_inches='tight')
        plt.clf()

def plot_mean_accuracy_by_operator(mode='save', file_format='pdf'):
    accuracy_by_operator, std_accuracy_by_operator, _ = get_accuracy_by_operator()

    #x = ['Add', 'Subtract', 'Multiply', 'Divide', 'Modulo']
    x = ['+', '−', '×', '÷', 'mod']
    y = [accuracy_by_operator['add'],
        accuracy_by_operator['subtract'],
        accuracy_by_operator['multiply'],
        accuracy_by_operator['divide'],
        accuracy_by_operator['modulo']
    ]
    e = [errorbar_std * std_accuracy_by_operator['add'],
        errorbar_std * std_accuracy_by_operator['subtract'],
        errorbar_std * std_accuracy_by_operator['multiply'],
        errorbar_std * std_accuracy_by_operator['divide'],
        errorbar_std * std_accuracy_by_operator['modulo']
    ]

    plt.figure(figsize=(len(x)-1,4))
    #plt.xlabel('Operator')
    plt.ylabel('Accuracy', fontsize=font_size['ylabel'])
    plt.tick_params(axis='x', labelsize=font_size['xtick'])
    plt.tick_params(axis='y', labelsize=font_size['ytick'])

    #plt.ylim(0.0, 60.0)
    #plt.ylim(0.0, 100.0)
    #plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(axis='y')
    #plt.title('Mean solving time by operator')

    #plt.plot(x, y, ':o', label='Each operator')
    plt.errorbar(x, y, e, fmt='r:o', ecolor='orange', capsize=3)

    #plt.bar(x, y, align='center')
    '''plt.hlines(total_mean_solving_time, xmin=-0.5, xmax=len(x)-0.5, colors='g', label='All operators')
    plt.legend()'''

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(dir_plot_fig)
        plot_fig_path = '{plot_dir}/mean_accuracy_by_operator.{extension}'.format(
            plot_dir=dir_plot_fig,
            extension=file_format
        )
        plt.savefig(plot_fig_path, bbox_inches='tight')
    plt.clf()


def plot_mean_accuracy_by_carries(mode='save', file_format='pdf'):
    for operator in operators:
        mean_accuracy_by_carries, std_accuracy_by_carries, _ = get_accuracy_by_carries(operator)

        carries_list = list(mean_accuracy_by_carries.keys())
        x = [str(carries) for carries in carries_list]
        y = list(mean_accuracy_by_carries.get_values())
        e = list(std_accuracy_by_carries.get_values())
        for i in range(len(e)):
            e[i] = errorbar_std * e[i]

        plt.figure(figsize=(len(x)-1,4))
        plt.xlabel('Carries', fontsize=font_size['xlabel'])
        plt.ylabel('Accuracy', fontsize=font_size['ylabel'])
        plt.tick_params(axis='x', labelsize=font_size['xtick'])
        plt.tick_params(axis='y', labelsize=font_size['ytick'])

        #plt.ylim(0.0, 60.0)
        #plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis='y')
        #plt.title('[{operator}] Mean solving time by carries'.format(operator=operator.capitalize()))

        #plt.bar(x, y, align='center')
        #plt.plot(x, y, ':o', label='Carry datasets')
        plt.errorbar(x, y, e, fmt='r:o', ecolor='orange', capsize=3)
        '''plt.hlines(mean_solving_time_by_operator[operator],
            xmin=-0.5, xmax=len(x)-0.5, colors='g',
            label='[{operator}] Operator dataset'.format(operator=operator.capitalize()))
        plt.legend()'''

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(dir_plot_fig)
            plot_fig_path = '{plot_dir}/mean_accuracy_by_carries_{operator}.{extension}'.format(
                plot_dir=dir_plot_fig,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path, bbox_inches='tight')
        plt.clf()

def plot_mean_solving_time_by_operator(mode='save', file_format='pdf'):

    total_mean_solving_time, total_std_solving_time = get_mean_solving_time(groupby_operator=False, groupby_carries=False)
    mean_solving_time_by_operator, std_solving_time_by_operator = get_mean_solving_time(groupby_operator=True, groupby_carries=False)
    #mean_solving_time_by_operator, std_solving_time_by_operator, _ = get_mean_solving_time_by_operator()

    #x = ['Add', 'Subtract', 'Multiply', 'Divide', 'Modulo']
    x = ['+', '−', '×', '÷', 'mod']
    y = [mean_solving_time_by_operator['add'],
        mean_solving_time_by_operator['subtract'],
        mean_solving_time_by_operator['multiply'],
        mean_solving_time_by_operator['divide'],
        mean_solving_time_by_operator['modulo']
    ]
    e = [errorbar_std * std_solving_time_by_operator['add'],
        errorbar_std * std_solving_time_by_operator['subtract'],
        errorbar_std * std_solving_time_by_operator['multiply'],
        errorbar_std * std_solving_time_by_operator['divide'],
        errorbar_std * std_solving_time_by_operator['modulo']
    ]

    plt.figure(figsize=(len(x)-1,4))
    #plt.xlabel('Operator')
    plt.ylabel('Response time (sec.)', fontsize=font_size['ylabel'])
    plt.tick_params(axis='x', labelsize=font_size['xtick'])
    plt.tick_params(axis='y', labelsize=font_size['ytick'])

    #plt.ylim(0.0, 60.0)
    #plt.ylim(0.0, 100.0)
    #plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(axis='y')
    #plt.title('Mean solving time by operator')

    #plt.plot(x, y, ':o', label='Each operator')
    plt.errorbar(x, y, e, fmt='r:o', ecolor='orange', capsize=3)

    #plt.bar(x, y, align='center')
    '''plt.hlines(total_mean_solving_time, xmin=-0.5, xmax=len(x)-0.5, colors='g', label='All operators')
    plt.legend()'''

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(dir_plot_fig)
        plot_fig_path = '{plot_dir}/mean_solving_time_by_operator.{extension}'.format(
            plot_dir=dir_plot_fig,
            extension=file_format
        )
        plt.savefig(plot_fig_path, bbox_inches='tight')
    plt.clf()


def plot_mean_solving_time_by_carries(mode='save', file_format='pdf'):

    for operator in operators:
        mean_solving_time_by_operator, std_solving_time_by_operator = get_mean_solving_time(groupby_operator=True, groupby_carries=True)

        carries_list = list(mean_solving_time_by_operator[operator].keys())
        x = [str(carries) for carries in carries_list]
        y = list(mean_solving_time_by_operator[operator].get_values())
        e = list(std_solving_time_by_operator[operator].get_values())
        for i in range(len(e)):
            e[i] = errorbar_std * e[i]

        plt.figure(figsize=(len(x)-1,4))
        plt.xlabel('Carries', fontsize=font_size['xlabel'])
        plt.ylabel('Response time (sec.)', fontsize=font_size['ylabel'])
        #plt.tick_params(axis='x', labelsize=font_size['xtick'])
        plt.tick_params(axis='y', labelsize=font_size['ytick'])

        #plt.ylim(0.0, 60.0)
        #plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis='y')
        #plt.title('[{operator}] Mean solving time by carries'.format(operator=operator.capitalize()))

        #plt.bar(x, y, align='center')
        #plt.plot(x, y, ':o', label='Carry datasets')
        plt.errorbar(x, y, e, fmt='r:o', ecolor='orange', capsize=3)
        '''plt.hlines(mean_solving_time_by_operator[operator],
            xmin=-0.5, xmax=len(x)-0.5, colors='g',
            label='[{operator}] Operator dataset'.format(operator=operator.capitalize()))
        plt.legend()'''

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(dir_plot_fig)
            plot_fig_path = '{plot_dir}/mean_solving_time_by_carries_{operator}.{extension}'.format(
                plot_dir=dir_plot_fig,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path, bbox_inches='tight')
        plt.clf()


def plot_mean_mean_solving_time_by_operator(mode='save', file_format='pdf'):

    #total_mean_solving_time, total_std_solving_time = get_mean_solving_time(groupby_operator=False, groupby_carries=False)
    #mean_solving_time_by_operator, std_solving_time_by_operator = get_mean_solving_time(groupby_operator=True, groupby_carries=False)
    mean_solving_time_by_operator, std_solving_time_by_operator, _ = get_mean_solving_time_by_operator()

    #x = ['Add', 'Subtract', 'Multiply', 'Divide', 'Modulo']
    x = ['+', '−', '×', '÷', 'mod']
    y = [mean_solving_time_by_operator['add'],
        mean_solving_time_by_operator['subtract'],
        mean_solving_time_by_operator['multiply'],
        mean_solving_time_by_operator['divide'],
        mean_solving_time_by_operator['modulo']
    ]
    e = [errorbar_std * std_solving_time_by_operator['add'],
        errorbar_std * std_solving_time_by_operator['subtract'],
        errorbar_std * std_solving_time_by_operator['multiply'],
        errorbar_std * std_solving_time_by_operator['divide'],
        errorbar_std * std_solving_time_by_operator['modulo']
    ]

    plt.figure(figsize=(len(x)-1,4))
    #plt.xlabel('Operator')
    plt.ylabel('Response time (sec.)', fontsize=font_size['ylabel'])
    plt.tick_params(axis='x', labelsize=font_size['xtick'])
    plt.tick_params(axis='y', labelsize=font_size['ytick'])

    #plt.ylim(0.0, 60.0)
    #plt.ylim(0.0, 100.0)
    #plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(axis='y')
    #plt.title('Mean solving time by operator')

    #plt.plot(x, y, ':o', label='Each operator')
    plt.errorbar(x, y, e, fmt='r:o', ecolor='orange', capsize=3)

    #plt.bar(x, y, align='center')
    '''plt.hlines(total_mean_solving_time, xmin=-0.5, xmax=len(x)-0.5, colors='g', label='All operators')
    plt.legend()'''

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(dir_plot_fig)
        plot_fig_path = '{plot_dir}/mean_mean_solving_time_by_operator.{extension}'.format(
            plot_dir=dir_plot_fig,
            extension=file_format
        )
        plt.savefig(plot_fig_path, bbox_inches='tight')
    plt.clf()


def plot_mean_mean_solving_time_by_carries(mode='save', file_format='pdf'):

    for operator in operators:
        mean_mean_solving_time_by_carries, std_mean_solving_time_by_carries, _ = get_mean_solving_time_by_carries(operator)

        carries_list = list(mean_mean_solving_time_by_carries.keys())
        x = [str(carries) for carries in carries_list]
        y = list(mean_mean_solving_time_by_carries.get_values())
        e = list(std_mean_solving_time_by_carries.get_values())
        for i in range(len(e)):
            e[i] = errorbar_std * e[i]

        plt.figure(figsize=(len(x)-1,4))
        plt.xlabel('Carries', fontsize=font_size['xlabel'])
        plt.ylabel('Response time (sec.)', fontsize=font_size['ylabel'])
        plt.tick_params(axis='x', labelsize=font_size['xtick'])
        plt.tick_params(axis='y', labelsize=font_size['ytick'])

        #plt.ylim(0.0, 60.0)
        #plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis='y')
        #plt.title('[{operator}] Mean solving time by carries'.format(operator=operator.capitalize()))

        #plt.bar(x, y, align='center')
        #plt.plot(x, y, ':o', label='Carry datasets')
        plt.errorbar(x, y, e, fmt='r:o', ecolor='orange', capsize=3)
        '''plt.hlines(mean_solving_time_by_operator[operator],
            xmin=-0.5, xmax=len(x)-0.5, colors='g',
            label='[{operator}] Operator dataset'.format(operator=operator.capitalize()))
        plt.legend()'''

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(dir_plot_fig)
            plot_fig_path = '{plot_dir}/mean_mean_solving_time_by_carries_{operator}.{extension}'.format(
                plot_dir=dir_plot_fig,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path, bbox_inches='tight')
        plt.clf()


def boxplot_mean_solving_time_by_operator(mode='save', file_format='pdf'):

    mean_mean_solving_time_by_operator, std_mean_solving_time_by_operator, df_mean_st_operator = get_mean_solving_time_by_operator()

    total_result = df_mean_st_operator['mean_solving_time'].get_values()
    total_result_add = filter_operator(df_mean_st_operator, 'add')['mean_solving_time'].get_values()
    total_result_subtract = filter_operator(df_mean_st_operator, 'subtract')['mean_solving_time'].get_values()
    total_result_multiply = filter_operator(df_mean_st_operator, 'multiply')['mean_solving_time'].get_values()
    total_result_divide = filter_operator(df_mean_st_operator, 'divide')['mean_solving_time'].get_values()
    total_result_modulo = filter_operator(df_mean_st_operator, 'modulo')['mean_solving_time'].get_values()

    max_solving_time = total_result.max()

    x_labels = ['All', 'Add', 'Subtract', 'Multiply', 'Divide', 'Modulo']
    x = np.arange(0,len(x_labels))

    y_mean = (total_result.mean(),
        mean_mean_solving_time_by_operator['add'],
        mean_mean_solving_time_by_operator['subtract'],
        mean_mean_solving_time_by_operator['multiply'],
        mean_mean_solving_time_by_operator['divide'],
        mean_mean_solving_time_by_operator['modulo']
    )

    plt.figure(figsize=(8,8))

    #max_ylim = int(np.ceil(max_solving_time / 10) * 10)
    #plt.ylim(0.0, max_ylim)
    #plt.yticks([0, 10, 20, 30, 40, 50] + list(range(60,max_ylim,30)))

    plt.grid(axis='y')
    plt.title('Response by operator')
    plt.ylabel('Response time (sec.)')
    plt.xlabel('Operator')

    plt.boxplot([total_result,
                    total_result_add,
                    total_result_subtract,
                    total_result_multiply,
                    total_result_divide,
                    total_result_modulo],
                labels=x_labels,
                positions=x
    )
    plt.plot(x, y_mean, 'r:o', label='Mean mean solving time')

    plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(dir_plot_fig)
        plot_fig_path = '{plot_dir}/solving_time_by_operator.{extension}'.format(
            plot_dir=dir_plot_fig,
            extension=file_format
        )
        plt.savefig(plot_fig_path, bbox_inches='tight')
    plt.clf()


def boxplot_mean_solving_time_by_carries(mode='save', file_format='pdf'):

    for operator in operators:

        plt.figure(figsize=(8,8))
        plt.grid(axis='y')
        plt.title('[{operator}] Solving time by carries'.format(operator=operator.capitalize()))
        plt.ylabel('Response time (sec.)')
        plt.xlabel('Carries')

        mean_mean_solving_time_by_carries, std_mean_solving_time_by_carries, df_mean_st_carries = get_mean_solving_time_by_carries(operator)

        carries_list = list(mean_mean_solving_time_by_carries.keys())
        x_labels = ['All']
        y_boxplot_list = [df_mean_st_carries['mean_solving_time'].get_values()]
        y_mean = [df_mean_st_carries['mean_solving_time'].mean()]

        for carries in carries_list:
            total_result_of_carries = filter_carries(df_mean_st_carries, carries)['mean_solving_time'].get_values()
            x_labels.append(str(carries))
            y_boxplot_list.append(total_result_of_carries)
            y_mean.append(mean_mean_solving_time_by_carries[carries])

        x = np.arange(0,len(x_labels))

        plt.boxplot(y_boxplot_list,
                    labels=x_labels,
                    positions=x
        )
        plt.plot(x_labels, y_mean, 'r:o', label='Mean mean solving time')

        plt.legend()

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(dir_plot_fig)
            plot_fig_path = '{plot_dir}/solving_time_by_carries_{operator}.{extension}'.format(
                plot_dir=dir_plot_fig,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path, bbox_inches='tight')
        plt.clf()


def plot_mean_solving_time_by_problems(mode='save', file_format='pdf'):
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
        plt.title('Mean solving time by experienced problems')
        plt.xlabel('Experienced problems')
        plt.ylabel('Response time (sec.)')

        # Plot stage
        plt.plot(x, y, ':o', label=operator)
        plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(dir_plot_fig)
        plot_fig_path = '{plot_dir}/mean_solving_time_by_experienced_problems.{extension}'.format(
            plot_dir=dir_plot_fig,
            extension=file_format
        )
        plt.savefig(plot_fig_path, bbox_inches='tight')
    plt.clf()


def plot_solving_time(df_result, subject_index):
    # Retrieve the operator from the first row.
    operator = df_result['operator'][0]
    plt.title('{operator} subject[{sub_index}]'.format(
        operator=operatorcapitalize() ,
        sub_index=subject_index
    ))
    plt.xlabel('Experienced problems')
    plt.ylabel('Response time (sec.)')

    x = np.arange(len(df_result))
    y = df_result['solving_time']
    plt.plot(x, y, ':o')
    plt.show()
    plt.clf()


def plot_solving_time_for_carries(df_result, subject_index, mode='save', file_format='pdf'):
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
        plt.ylabel('Response time (sec.)')

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
        create_dir(dir_plot_fig)
        plot_fig_path = '{plot_dir}/{operator}_subject-{sub_index}_by_carries.{extension}'.format(
            plot_dir=dir_plot_fig,
            operator=operator,
            sub_index=subject_index,
            carries=carries,
            extension=file_format
        )
        plt.savefig(plot_fig_path, bbox_inches='tight')
    plt.clf()


def plot_mean_solving_time_by_problems_for_operators(mode='save', file_format='pdf'):
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
        plt.title('[{operator}] Mean solving time by experienced problems'.format(
            operator=operator.capitalize()
        ))
        plt.xlabel('Experienced problems')
        plt.ylabel('Response time (sec.)')

        # Plot stage
        plt.plot(x, y, ':o')

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(dir_plot_fig)
            plot_fig_path = '{plot_dir}/mean_solving_time_by_experienced_problems_{operator}.{extension}'.format(
                plot_dir=dir_plot_fig,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path, bbox_inches='tight')
        plt.clf()


def plot_mean_solving_time_by_problems_for_carries(mode='save', file_format='pdf'):
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

            plt.title('[{operator}] Mean solving time by experienced problems for carries'.format(
                operator=operator.capitalize()
            ))
            plt.xlabel('Experienced problems')
            plt.ylabel('Response time (sec.)')

            # Plot stage
            plt.plot(x, y, ':o', label=carries_label)
            plt.legend()

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(dir_plot_fig)
            plot_fig_path = '{plot_dir}/mean_solving_time_by_experienced_problems_for_carries_{operator}.{extension}'.format(
                plot_dir=dir_plot_fig,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path, bbox_inches='tight')
        plt.clf()


def plot_all_mst_by_problems(mode='save', file_format='pdf'):
    plot_mean_solving_time_by_problems(mode=mode, file_format=file_format)
    plot_mean_solving_time_by_problems_for_operators(mode=mode, file_format=file_format)
    plot_mean_solving_time_by_problems_for_carries(mode=mode, file_format=file_format)


def save_csv_files(experiment_name):
    '''
    Create CSV files for ANOVA.
    '''
    if experiment_name == 'cogsci2019':
        dir_save = join(dir_accuracy, experiment_name)
        create_dir(dir_save)

        _, _, df_accuracy_carries = get_accuracy_by_operator()
        df_accuracy_carries.to_csv(join(dir_save, 'operators.csv'), index=False)

        for operator in data_utils.operators_list:
            _, _, df_accuracy_carries = get_accuracy_by_carries(operator)
            df_accuracy_carries.to_csv(join(dir_save, 'carries_{}.csv'.format(operator)), index=False)

        dir_save = join(dir_st_correct, experiment_name)
        create_dir(dir_save)
        _, _, df_mean_st_operator = get_mean_solving_time_by_operator()
        df_mean_st_operator.to_csv(join(dir_save, 'operators.csv'), index=False)

        for operator in data_utils.operators_list:
            _, _, df_mean_st_carries = get_mean_solving_time_by_operator(operator)
            df_mean_st_carries.to_csv(join(dir_save, 'carries_{}.csv'.format(operator)), index=False)


    if experiment_name == 'iccm2019':
        operators_list = ['add', 'subtract']

        dir_save = join(dir_accuracy, experiment_name)
        create_dir(dir_save)

        for operator in operators_list:
            _, _, df_accuracy_carries = get_accuracy_by_carries(operator)
            df_accuracy_carries.to_csv(join(dir_save, 'carries_{}.csv'.format(operator)), index=False)

        dir_save = join(dir_st_correct, experiment_name)
        create_dir(dir_save)

        for operator in operators_list:
            _, _, df_mean_st_carries = get_mean_solving_time_by_operator(operator)
            df_mean_st_carries.to_csv(join(dir_save, 'carries_{}.csv'.format(operator)), index=False)
