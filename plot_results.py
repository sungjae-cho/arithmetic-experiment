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


def count_results(operator=None):
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


def plot_accuracy_by_operator(mode='save', file_format='pdf'):

    total_accuracy = get_accuracy(groupby_operator=False, groupby_carries=False)
    accuracy_by_operator = get_accuracy(groupby_operator=True, groupby_carries=False)

    x = ('Add', 'Subtract', 'Multiply', 'Divide', 'Modulo')
    y = (accuracy_by_operator['add'],
        accuracy_by_operator['subtract'],
        accuracy_by_operator['multiply'],
        accuracy_by_operator['divide'],
        accuracy_by_operator['modulo']
    )

    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(axis='y')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by operator')

    plt.plot(x, y, ':o', label='Each operator')
    #plt.bar(x, y, align='center')
    plt.hlines(total_accuracy, xmin=-0.5, xmax=len(x)-0.5, colors='r', label='All operators')
    plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(plot_fig_dir)
        plot_fig_path = '{plot_dir}/accuracy_by_operator.{extension}'.format(
            plot_dir=plot_fig_dir,
            extension=file_format
        )
        plt.savefig(plot_fig_path)
    plt.clf()


def plot_accuracy_by_carries(mode='save', file_format='pdf'):

    accuracy_by_operator = get_accuracy(groupby_operator=True, groupby_carries=False)
    accuracy_by_carries = get_accuracy(groupby_operator=True, groupby_carries=True)

    for operator in operators:

        carries_list = list(accuracy_by_carries[operator].keys())
        x = [str(carries) for carries in carries_list]
        y = list(accuracy_by_carries[operator].get_values())

        plt.ylim(0.0, 1.0)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis='y')
        plt.ylabel('Accuracy')
        plt.title('[{operator}] Accuracy by carries'.format(operator=operator.capitalize()))

        #plt.bar(x, y, align='center')
        plt.plot(x, y, ':o', label='Carry datasets')
        plt.hlines(accuracy_by_operator[operator],
            xmin=-0.5, xmax=len(x)-0.5, colors='r',
            label='[{operator}] Operator dataset'.format(operator=operator.capitalize()))
        plt.legend()

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(plot_fig_dir)
            plot_fig_path = '{plot_dir}/accuracy_by_carries_{operator}.{extension}'.format(
                plot_dir=plot_fig_dir,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path)
        plt.clf()


# TODO: Implement!
def plot_mean_solving_time_by_operator(mode='save', file_format='pdf'):

    total_mean_solving_time = get_mean_solving_time(groupby_operator=False, groupby_carries=False)
    mean_solving_time_by_operator = get_mean_solving_time(groupby_operator=True, groupby_carries=False)

    x = ('Add', 'Subtract', 'Multiply', 'Divide', 'Modulo')
    y = (mean_solving_time_by_operator['add'],
        mean_solving_time_by_operator['subtract'],
        mean_solving_time_by_operator['multiply'],
        mean_solving_time_by_operator['divide'],
        mean_solving_time_by_operator['modulo']
    )

    plt.ylim(0.0, 60.0)
    #plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(axis='y')
    plt.ylabel('Mean solving time (sec.)')
    plt.title('Mean solving time by operator')

    plt.plot(x, y, ':o', label='Each operator')
    #plt.bar(x, y, align='center')
    plt.hlines(total_mean_solving_time, xmin=-0.5, xmax=len(x)-0.5, colors='r', label='All operators')
    plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(plot_fig_dir)
        plot_fig_path = '{plot_dir}/mean_solving_time_by_operator.{extension}'.format(
            plot_dir=plot_fig_dir,
            extension=file_format
        )
        plt.savefig(plot_fig_path)
    plt.clf()


def plot_mean_solving_time_by_carries(mode='save', file_format='pdf'):

    mean_solving_time_by_operator = get_mean_solving_time(groupby_operator=True, groupby_carries=False)
    mean_solving_time_by_carries = get_mean_solving_time(groupby_operator=True, groupby_carries=True)

    for operator in operators:

        carries_list = list(mean_solving_time_by_carries[operator].keys())
        x = [str(carries) for carries in carries_list]
        y = list(mean_solving_time_by_carries[operator].get_values())

        plt.ylim(0.0, 60.0)
        #plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis='y')
        plt.ylabel('Mean solving time (sec.)')
        plt.title('[{operator}] Mean solving time by carries'.format(operator=operator.capitalize()))

        #plt.bar(x, y, align='center')
        plt.plot(x, y, ':o', label='Carry datasets')
        plt.hlines(mean_solving_time_by_operator[operator],
            xmin=-0.5, xmax=len(x)-0.5, colors='r',
            label='[{operator}] Operator dataset'.format(operator=operator.capitalize()))
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


def plot_solving_time_by_operator(mode='save', file_format='pdf'):

    total_mean_solving_time = get_mean_solving_time(groupby_operator=False, groupby_carries=False)
    mean_solving_time_by_operator = get_mean_solving_time(groupby_operator=True, groupby_carries=False)

    total_result = get_total_result()['solving_time'].get_values()
    total_result_add = get_total_result('add')['solving_time'].get_values()
    total_result_subtract = get_total_result('subtract')['solving_time'].get_values()
    total_result_multiply = get_total_result('multiply')['solving_time'].get_values()
    total_result_divide = get_total_result('divide')['solving_time'].get_values()
    total_result_modulo = get_total_result('modulo')['solving_time'].get_values()

    max_solving_time = total_result.max()

    x_labels = ['All', 'Add', 'Subtract', 'Multiply', 'Divide', 'Modulo']
    x = np.arange(0,len(x_labels))

    y_mean = (total_mean_solving_time,
        mean_solving_time_by_operator['add'],
        mean_solving_time_by_operator['subtract'],
        mean_solving_time_by_operator['multiply'],
        mean_solving_time_by_operator['divide'],
        mean_solving_time_by_operator['modulo']
    )

    plt.figure(figsize=(8,8))

    max_ylim = int(np.ceil(max_solving_time / 10) * 10)
    plt.ylim(0.0, max_ylim)
    plt.yticks([0, 10, 20, 30, 40, 50] + list(range(60,max_ylim,30)))

    plt.grid(axis='y')
    plt.title('Solving time by operator')
    plt.ylabel('Solving time (sec.)')
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
    plt.plot(x, y_mean, 'r:o', label='Mean solving time')

    plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(plot_fig_dir)
        plot_fig_path = '{plot_dir}/solving_time_by_operator.{extension}'.format(
            plot_dir=plot_fig_dir,
            extension=file_format
        )
        plt.savefig(plot_fig_path)
    plt.clf()


# TODO: Implement!
def plot_solving_time_by_carries(mode='save', file_format='pdf'):

    mean_solving_time_by_operator = get_mean_solving_time(groupby_operator=True, groupby_carries=False)

    for operator in operators:

        plt.figure(figsize=(8,8))
        plt.grid(axis='y')
        plt.title('[{operator}] Solving time by carries'.format(operator=operator.capitalize()))
        plt.ylabel('Solving time (sec.)')
        plt.xlabel('Carries')

        total_result = get_total_result(operator)
        mean_solving_time_by_carries = get_mean_solving_time(groupby_operator=True, groupby_carries=True)[operator]
        carries_list = list(mean_solving_time_by_carries.keys())

        x_labels = ['All']
        y_boxplot_list = [total_result['solving_time'].get_values()]
        y_mean = [mean_solving_time_by_operator[operator]]

        for carries in carries_list:
            total_result_of_carries = filter_carries(total_result, carries)
            total_result_of_carries = total_result_of_carries['solving_time'].get_values()
            x_labels.append(str(carries))
            y_boxplot_list.append(total_result_of_carries)
            y_mean.append(mean_solving_time_by_carries[carries])

        x = np.arange(0,len(x_labels))

        plt.boxplot(y_boxplot_list,
                    labels=x_labels,
                    positions=x
        )
        plt.plot(x_labels, y_mean, 'r:o', label='Mean solving time')

        plt.legend()

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(plot_fig_dir)
            plot_fig_path = '{plot_dir}/solving_time_by_carries_{operator}.{extension}'.format(
                plot_dir=plot_fig_dir,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path)
        plt.clf()

    pass


def plot_mean_solving_time_by_questions(mode='save', file_format='pdf'):
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
        plt.title('Mean solving time by experienced questions')
        plt.xlabel('Experienced questions')
        plt.ylabel('Mean solving time (sec.)')

        # Plot stage
        plt.plot(x, y, ':o', label=operator)
        plt.legend()

    if mode == 'show':
        plt.show()
    if mode == 'save':
        create_dir(plot_fig_dir)
        plot_fig_path = '{plot_dir}/mean_solving_time_by_experienced_questions.{extension}'.format(
            plot_dir=plot_fig_dir,
            extension=file_format
        )
        plt.savefig(plot_fig_path)
    plt.clf()


def plot_mean_solving_time_by_questions_for_operators(mode='save', file_format='pdf'):
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
        plt.title('[{operator}] Mean solving time by experienced questions'.format(
            operator=operator.capitalize()
        ))
        plt.xlabel('Experienced questions')
        plt.ylabel('Mean solving time (sec.)')

        # Plot stage
        plt.plot(x, y, ':o')

        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(plot_fig_dir)
            plot_fig_path = '{plot_dir}/mean_solving_time_by_experienced_questions_{operator}.{extension}'.format(
                plot_dir=plot_fig_dir,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path)
        plt.clf()


def plot_mean_solving_time_by_questions_for_carries(mode='save', file_format='pdf'):
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

            plt.title('[{operator}] Mean solving time by experienced questions for carries'.format(
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
            plot_fig_path = '{plot_dir}/mean_solving_time_by_experienced_questions_for_carries_{operator}.{extension}'.format(
                plot_dir=plot_fig_dir,
                operator=operator,
                extension=file_format
            )
            plt.savefig(plot_fig_path)
        plt.clf()


def plot_all(mode='save', file_format='pdf'):
    plot_mean_solving_time_by_questions(mode=mode, file_format=file_format)
    plot_mean_solving_time_by_questions_for_operators(mode=mode, file_format=file_format)
    plot_mean_solving_time_by_questions_for_carries(mode=mode, file_format=file_format)
