import numpy as np
import pickle
import os
import csv # write_carry_dataset_statistics
import pandas as pd # plot_carry_dataset_statistics
import matplotlib.pyplot as plt # plot_carry_dataset_statistics
import random # import_random_sampled_carry_datasets


data_dir = 'data'
plot_fig_dir = 'plot_figures'
carry_dataset_statistics_name = 'carry_dataset_statistics.csv'
operand_digits_list = [4, 6, 8]
operators_list = ['add', 'subtract', 'multiply', 'divide', 'modulo']
np_type = np.int


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_result_digits(operand_digits, operator):
    if operator == 'add':
        result_digits = operand_digits + 1
    if operator == 'subtract':
        result_digits = operand_digits
    if operator == 'multiply':
        result_digits = operand_digits * 2
    if operator == 'divide':
        result_digits = operand_digits
    if operator == 'modulo':
        result_digits = operand_digits
    return result_digits


def get_str_bin(int_dec):
    '''
    Parameters
    ----------
    int_dec: int. a decimal number.

    Returns
    -------
    str_bin: str. the string of int_dec
    - If int_dec >=0, then no sign character in str_bin.
    - If int_dec < 0, then '-' becomes the first character of str_bin.
    '''
    if int_dec >= 0:
        str_bin = bin(int_dec)[2:]
    else:
        str_bin =  bin(int_dec)[0] + bin(int_dec)[3:]
    return str_bin


def get_int_dec(str_bin):
    '''
    Parameters
    ----------
    str_bin : str. the string of a binary number

    Returns
    -------
    int_dec : int. decimal interger.
    '''
    int_dec = int(str_bin, 2)
    return int_dec


def get_np_bin(str_bin, np_bin_digits):
    '''
    Parameters
    ----------
    str_bin

    Return
    ------
    np_bin: numpy.ndarry. binary number. The smaller index, the higher digit.
    '''
    assert str_bin[0] != '-'

    np_bin = np.zeros((np_bin_digits), dtype=np_type) # Should be initialized as 0.

    for i in range(1, len(str_bin)+1):
        np_bin[-i] = int(str_bin[-i])

    return np_bin


def get_leading_zeros(operand):
    '''
    Parameters
    ----------
    operand : np.ndarray. 1-dimension. shape==(operand_digits).

    Returns
    -------
    n_leading_zeros : int. The number of leading zeros.
    - If operand is [0,0,1,1,0,1], the number of leading zeros is 2.
    '''
    operand_digits = operand.shape[0]
    n_leading_zeros = 0
    for i in range(operand_digits):
        if operand[i] == 0:
            n_leading_zeros = n_leading_zeros + 1
        else:
            break
    return n_leading_zeros


def get_carry_ds_stat_path():
    carry_ds_stat_path = '{}/{}'.format(data_dir, carry_dataset_statistics_name)
    return carry_ds_stat_path


def less_than(operand1, operand2):
    '''
    Parameters
    ----------
    operand1 : np.ndarray. 1-dimension. shape==(operand_digits).
    operand2 : np.ndarray. 1-dimension. shape==(operand_digits).

    Returns
    -------
    is_less_than : bool. operand1 < operand2.
    '''
    operand_digits = operand1.shape[0]
    for i in range(operand_digits):
        if operand1[i] > operand2[i]:
            return False
        if operand1[i] < operand2[i]:
            return True
    # All same digits
    return False


def str_binary_operation(str_operand1, str_operator, str_operand2):
    int_dec_operand1 = get_int_dec(str_operand1)
    int_dec_operand2 = get_int_dec(str_operand2)
    if str_operator in ['add', '+']:
        int_dec_result = int_dec_operand1 + int_dec_operand2
    if str_operator in ['subtract', '-']:
        int_dec_result = int_dec_operand1 - int_dec_operand2
    if str_operator in ['multiply', '*']:
        int_dec_result = int_dec_operand1 * int_dec_operand2
    if str_operator in ['divide', '/', '//']:
        int_dec_result = int_dec_operand1 // int_dec_operand2
    if str_operator in ['modulo', '%']:
        int_dec_result = int_dec_operand1 % int_dec_operand2
    str_bin_result = get_str_bin(int_dec_result)
    return str_bin_result


def add_two_digits(digit1, digit2, carry):
    '''
    Parameters
    ----------
    digit1 : int. digit1 in [0, 1].
    digit2 : int. digit2 in [0, 1].
    carry : the carry from the lower addtion.

    Returns
    -------
    carry : the carry for the next digit addition.
    result : the current digit result of addition.
    '''
    digit_sum = digit1 + digit2 + carry

    if digit_sum == 3:
        (carry, result) = (1, 1)
    if digit_sum == 2:
        (carry, result) = (1, 0)
    if digit_sum == 1:
        (carry, result) = (0, 1)
    if digit_sum == 0:
        (carry, result) = (0, 0)

    return (carry, result)


def add_two_numbers(operand1, operand2):
    '''
    Parameters
    ----------
    operand1 : np.dnarray. 1-dimension.
    operand2 : np.dnarray. 1-dimension. This should have the same dimension as operand2.

    Returns
    -------
    result : np.dnarray. 1-dimension. The result of addtion.
    n_carries : int. The number of carries occurred while addition.
    '''
    operand_digits = operand1.shape[0]
    result_digits = get_result_digits(operand_digits, 'add')
    result = np.empty((result_digits), dtype=np_type)
    carry = 0
    n_carries = 0
    for i in range(1, operand_digits + 1):
        (carry, digit_result) = add_two_digits(operand1[-i], operand2[-i], carry)
        n_carries = n_carries + carry
        result[-i] = digit_result
        if i == (operand_digits): # Last digit
            result[-(i+1)] = carry
    return (result, n_carries)


def subtract_two_numbers(operand1, operand2):
    '''
    Parameters
    ----------
    operand1 : np.ndarray. 1-dimension. shape==(operand_digits).
    operand2 : np.ndarray. 1-dimension. shape==(operand_digits).
    - Always operand1 >= operand2.

    Returns
    -------
    result : np.ndarray. result = operand1 - operand2. 1-D. shape==(operand_digits).
    - Beacuse operand1 >= operand2, result >= 0.
    n_carries : int. The number of carries that occurred while subtraction.
    '''
    operand_digits = operand1.shape[0]
    result_digits = get_result_digits(operand_digits, 'subtract')
    cp_operand1 = np.copy(operand1)
    cp_operand2 = np.copy(operand2)
    result = np.empty((result_digits), dtype=np_type)
    n_carries = 0
    for i in range(1, operand_digits + 1):
        if cp_operand1[-i] >= cp_operand2[-i]:
            result[-i] = cp_operand1[-i] - cp_operand2[-i]
        else:
            for j in range(i + 1, operand_digits + 1):
                n_carries = n_carries + 1
                if cp_operand1[-j] == 1:
                    cp_operand1[-j] = 0
                    for k in range(i + 1, j):
                        cp_operand1[-k] = 1
                    break
            result[-i] = 1

    return (result, n_carries)


def multiply_two_numbers(operand1, operand2):
    '''
    Parameters
    ----------
    operand1 : np.ndarray. 1-dimension. shape==(operand_digits).
    operand2 : np.ndarray. 1-dimension. shape==(operand_digits).

    Returns
    -------
    result : np.ndarray. result = operand1 - operand2. 1-D. shape==(operand_digits).
    n_carries : int. The number of carries that occurred while multiplication.
    '''
    operand_digits = operand1.shape[0]
    result_digits = get_result_digits(operand_digits, 'multiply')
    result = np.empty((result_digits), dtype=np_type) # To return
    carry_buffer = np.zeros((result_digits), dtype=np_type) # To save carries while addition

    # The multiplying phase
    multiply_result_to_sum = np.zeros((operand_digits, result_digits), dtype=np_type)
    for i in range(operand_digits):
        if operand2[-(i+1)] == 1:
            start_index = (result_digits - operand_digits - i)
            end_index = (result_digits - i)
            multiply_result_to_sum[i, start_index:end_index] = operand1

    # The summation and carrying phase
    n_carries = 0 # total carries in one multiplication operation.
    for i in range(1, result_digits+1):
        digit_wise_sum = np.sum(multiply_result_to_sum[:,-i]) + carry_buffer[-i]
        carry, remainder = divmod(digit_wise_sum, 2)
        n_carries = n_carries + carry
        if i < result_digits: # except the last digit
            carry_buffer[-(i+1)] = carry
        result[-i] = remainder

    return (result, n_carries)


def divide_two_numbers(operand1, operand2):
    '''
    Parameters
    ----------
    operand1 : np.ndarray. 1-dimension. shape==(operand_digits).
    operand2 : np.ndarray. 1-dimension. shape==(operand_digits).
    - operand2 must not be zero.

    Returns
    -------
    result : np.ndarray. result = operand1 // operand2. 1-D. shape==(operand_digits)
    n_carries : int. The number of carries that occurred while multiplication.
    remainder : np.ndarray. shape==(operand_digits).
    '''
    operand_digits = operand1.shape[0]
    result_digits = get_result_digits(operand_digits, 'divide')
    result = np.zeros((result_digits), dtype=np_type)

    leading_zeros = get_leading_zeros(operand2)
    valid_operand2_digits = operand_digits - leading_zeros

    division_steps = operand_digits - valid_operand2_digits + 1

    n_total_carries = 0
    for i in range(division_steps):
        division_index = valid_operand2_digits + i - 1
        division_range = division_index + 1

        # Assignment: local_divide_operand1
        local_divide_operand1 = np.zeros((division_range), dtype=np_type)
        if i == 0:
            local_divide_operand1 = operand1[:division_range]
        else:
            local_divide_operand1[:division_index] = local_subtract_result
            local_divide_operand1[division_index] = operand1[division_index]

        # Assignment: local_divide_operand2
        local_divide_operand2 = np.zeros((division_range), dtype=np_type)
        local_divide_operand2[-division_range:] = operand2[-division_range:]

        # Division: If condition. less_than
        # Subtraction: Get a remainder
        if less_than(local_divide_operand1, local_divide_operand2):
            result[division_index] = 0 # Division result
            local_subtract_result = np.copy(local_divide_operand1[:division_range]) # Get the remainder
            n_carries = 0
        else:
            result[division_index] = 1 # Division result
            local_subtract_result, n_carries = subtract_two_numbers(local_divide_operand1, local_divide_operand2) # Get the remainder

        n_total_carries = n_total_carries + n_carries

    remainder = local_subtract_result

    return (result, n_carries, remainder)


def modulo_two_numbers(operand1, operand2):
    '''
    Parameters
    ----------
    operand1 : np.ndarray. 1-dimension. shape==(operand_digits).
    operand2 : np.ndarray. 1-dimension. shape==(operand_digits).
    - operand2 must not be zero.

    Returns
    -------
    result : np.ndarray. result = operand1 % operand2. 1-D. shape==(operand_digits).
    n_carries : int. The number of carries that occurred while multiplication.
    remainder : np.ndarray. shape==(operand_digits).
    '''

    _, n_carries, result = divide_two_numbers(operand1, operand2)

    return (result, n_carries)


def operate_two_numbers(operand1, operand2, operator):
    '''
    Parameters
    ----------
    operand1 : np.ndarray. 1-dimension. shape==(operand_digits).
    operand2 : np.ndarray. 1-dimension. shape==(operand_digits).
    operator : str. ['add', 'substract', 'multiply', 'divide', 'modulo']

    Returns
    -------
    return_vector : The reult of an operation.
    - For division, the size of it will be 3 but the size of the others will be 2.
    '''
    if operator == 'add':
        return_vector = add_two_numbers(operand1, operand2)
    if operator == 'subtract':
        return_vector = subtract_two_numbers(operand1, operand2)
    if operator == 'multiply':
        return_vector = multiply_two_numbers(operand1, operand2)
    if operator == 'divide':
        return_vector = divide_two_numbers(operand1, operand2)
    if operator == 'modulo':
        return_vector = modulo_two_numbers(operand1, operand2)

    return return_vector


def generate_datasets(operand_digits, operator):
    if operator == 'add':
        carry_datasets = generate_add_datasets(operand_digits)
    if operator == 'subtract':
        carry_datasets = generate_subtract_datasets(operand_digits)
    if operator == 'multiply':
        carry_datasets = generate_multiply_datasets(operand_digits)
    if operator == 'divide':
        carry_datasets = generate_divide_datasets(operand_digits)
    if operator == 'modulo':
        carry_datasets = generate_modulo_datasets(operand_digits)

    return carry_datasets


def generate_add_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries addition.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries addition.
    -- result_digits == operand_digits + 1

    '''
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(2**operand_digits):
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of an adding operation
            result, n_carries = add_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of addition.
            carry_datasets[n_carries]['input'].append(np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1))

            # Append the output of addition.
            carry_datasets[n_carries]['output'].append(result.reshape(1,-1))

    # List to one numpy.ndarray
    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    return carry_datasets


def generate_subtract_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries subtraction.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries subtraction.
    -- result_digits == operand_digits
    '''
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(2**operand_digits):
            if dec_op1 >= dec_op2:
                # Get numpy.ndarray binary operands.
                np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
                np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

                # The phase of a subtracting operation
                result, n_carries = subtract_two_numbers(np_bin_op1, np_bin_op2)

                # Create a list to store operations
                if n_carries not in carry_datasets:
                    carry_datasets[n_carries] = dict()
                    carry_datasets[n_carries]['input'] = list()
                    carry_datasets[n_carries]['output'] = list()

                # Append the input of subtraction.
                carry_datasets[n_carries]['input'].append(np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1))

                # Append the output of subtraction.
                carry_datasets[n_carries]['output'].append(result.reshape(1,-1))

    # List to one numpy.ndarray
    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    return carry_datasets


def generate_multiply_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries multiplication.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries multiplication.
    -- result_digits == operand_digits * 2
    '''
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(2**operand_digits):
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of a multiplying operation
            result, n_carries = multiply_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of multiplication.
            carry_datasets[n_carries]['input'].append(np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1))

            # Append the output of multiplication.
            carry_datasets[n_carries]['output'].append(result.reshape(1,-1))

    # List to one numpy.ndarray
    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    return carry_datasets


def generate_divide_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries division.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries division.
    -- result_digits == operand_digits
    '''
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(1, 2**operand_digits): # Exclude `dec_op2 = 0`
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of a dividing operation
            result, n_carries, _ = divide_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of division.
            carry_datasets[n_carries]['input'].append(np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1))

            # Append the output of division.
            carry_datasets[n_carries]['output'].append(result.reshape(1,-1))

    # List to one numpy.ndarray
    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    return carry_datasets


def generate_modulo_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries modulo(division).
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries modulo(division).
    -- result_digits == operand_digits
    '''

    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(1, 2**operand_digits): # Exclude `dec_op2 = 0`
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of a dividing operation
            result, n_carries = modulo_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of division.
            carry_datasets[n_carries]['input'].append(np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1))

            # Append the output of division.
            carry_datasets[n_carries]['output'].append(result.reshape(1,-1))

    # List to one numpy.ndarray
    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    return carry_datasets


def generate_and_save_all_datasets():
    for operator in operators_list:
        for operand_digits in operand_digits_list:
            carry_datasets = generate_datasets(operand_digits, operator)
            save_carry_datasets(carry_datasets, operand_digits, operator)


def print_carry_datasets_info(carry_datasets):
    data_len_list = list()
    for key in carry_datasets.keys():
        data_len_list.append(carry_datasets[key]['input'].shape[0])
    total_operations = sum(data_len_list)

    for key in carry_datasets.keys():
        print('{}-carry dataset'.format(key))

        print('- #input dimension: {}'.format(carry_datasets[key]['input'].shape[1]))
        print('- #output dimension: {}'.format(carry_datasets[key]['output'].shape[1]))
        print('- #operations: {}'.format(carry_datasets[key]['input'].shape[0]))
        print('- Perceptage of {}-carry operations: {} %'.format(
            key, (carry_datasets[key]['input'].shape[0] / total_operations * 100)))


def get_carry_dataset_info_list(carry_datasets, operator):
    data_len_list = list()
    for key in carry_datasets.keys():
        data_len_list.append(carry_datasets[key]['input'].shape[0])
    total_operations = sum(data_len_list)

    carry_dataset_info_list = list()


    for n_carries in carry_datasets.keys():
        carry_dataset_info = dict()

        carry_dataset_info['operator'] = operator
        carry_dataset_info['carries'] = n_carries
        carry_dataset_info['operand digits'] = carry_datasets[n_carries]['input'].shape[1] // 2
        carry_dataset_info['input dimension'] = carry_datasets[n_carries]['input'].shape[1]
        carry_dataset_info['output dimension'] = carry_datasets[n_carries]['output'].shape[1]
        carry_dataset_info['carry operations'] = carry_datasets[n_carries]['input'].shape[0]
        carry_dataset_info['total operations'] = total_operations
        carry_dataset_info['carry percentage'] = (carry_datasets[n_carries]['input'].shape[0] / total_operations * 100)

        carry_dataset_info_list.append(carry_dataset_info)

    return carry_dataset_info_list


def write_carry_dataset_statistics():
    carry_dataset_info_list = list()
    csv_file_path = get_carry_ds_stat_path()
    create_dir(data_dir)

    for operator in operators_list:
        for operand_digits in operand_digits_list:
            carry_datasets = generate_datasets(operand_digits, operator)
            carry_dataset_info_list = carry_dataset_info_list + get_carry_dataset_info_list(carry_datasets, operator)

    with open(csv_file_path, mode='w') as csv_file:
        fieldnames = ['operator', 'operand digits',
                      'input dimension', 'output dimension', 'total operations',
                     'carries', 'carry operations', 'carry percentage']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for carry_dataset_info in carry_dataset_info_list:
            writer.writerow(carry_dataset_info)

    print('{} saved!'.format(csv_file_path))


def plot_carry_dataset_statistics(mode='save', file_format='svg'):
    df_carry_ds_stat = pd.read_csv(get_carry_ds_stat_path())
    df_carry_ds_stat = df_carry_ds_stat[['operator', 'operand digits', 'carries', 'carry percentage']]
    for operand_digits in operand_digits_list:
        plt.title('Percentage of operations by required carries ({}-bit operand)'.format(operand_digits))
        plt.xlabel('Carries')
        plt.ylabel('Percentage (%)')
        #plt.yticks(np.arange(0, 101, step=20))
        plt.ylim(0, 101)
        for operator in operators_list:
            if operator == 'modulo':
                break
            if operator == 'divide':
                operator_label = 'divide/modulo'
            else:
                operator_label = operator

            df = df_carry_ds_stat.loc[(df_carry_ds_stat['operator'] == operator) & (df_carry_ds_stat['operand digits'] == operand_digits)]
            df = df[['carries', 'carry percentage']]

            plt.plot(df['carries'], df['carry percentage'], ':o', label=operator_label)
            #plt.bar(df['carries'], df['carry percentage'], label=operator)
            plt.legend()
        if mode == 'show':
            plt.show()
        if mode == 'save':
            create_dir(plot_fig_dir)
            plot_fig_path = '{}/carry_dataset_statistics_{}-bit_operand.{}'.format(plot_fig_dir, operand_digits, file_format)
            plt.savefig(plot_fig_path)
            print('{} saved!'.format(plot_fig_path))
        plt.clf()


def save_carry_datasets(carry_datasets, operand_digits, operator):
    save_dir = 'data/{}-bit/{}'.format(operand_digits, operator)
    create_dir(save_dir)
    save_path = '{}/carry_datasets.pickle'.format(save_dir)

    with open(save_path, 'wb') as f:
        pickle.dump(carry_datasets, f)

    print("Saved in '{}'.".format(save_path))


def import_carry_datasets(operand_digits, operator):
    '''
    Parameters
    ----------
    operand_digits: int. The number of digits of an operand.
    operantor: str. one of ['add', 'substract', 'multiply', 'divide', 'modulo']

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: shape == (n_operations, input_dim).
    - carry_datasets[n_carries]['output']: shape == (n_operations, output_dim).
    '''
    import_path = 'data/{}-bit/{}/carry_datasets.pickle'.format(operand_digits, operator)

    with open(import_path, 'rb') as f:
        carry_datasets = pickle.load(f)

    print("Imported from '{}'.".format(import_path))

    return carry_datasets


def import_random_sampled_carry_datasets(operand_digits, operator, n_samples):
    '''
    "Import carry datasets that `n_samples` operations are sampled from each carry dataset."

    Parameters
    ----------
    operand_digits: int. The number of digits of an operand.
    operantor: str. one of ['add', 'substract', 'multiply', 'divide', 'modulo'].
    n_samples : int. The number of operations to sample from each carry.

    Returns
    -------
    carry_datasets : dict. Carry datasets that `n_samples` operations are sampled from each carry dataset.
    - carry_datasets[n_carries]['input']: shape == (n_samples, input_dim) or (n_operations, input_dim).
    - carry_datasets[n_carries]['output']: shape == (n_samples, output_dim) or (n_operations, output_dim).
    - If `n_samples` > n_operations in a carry dataset, then import all operations in it.
    '''
    carry_datasets = import_carry_datasets(operand_digits, operator)
    for n_carries in carry_datasets.keys():
        n_operations = carry_datasets[n_carries]['input'].shape[0]

        if n_samples > n_operations:
            sampled_indexes = random.sample(range(n_operations), n_operations)
        else:
            sampled_indexes = random.sample(range(n_operations), n_samples)

        carry_datasets[n_carries]['input'] = carry_datasets[n_carries]['input'][sampled_indexes,:]
        carry_datasets[n_carries]['output'] = carry_datasets[n_carries]['output'][sampled_indexes,:]

    return carry_datasets


def test_func_add_two_numbers():
    is_all_correct = True
    for operand_digits in operand_digits_list:
        # varying part
        result_digits = get_result_digits(operand_digits, 'add')
        for dec_op1 in range(2**operand_digits):
            for dec_op2 in range(2**operand_digits):
                # varying part
                bin_result = get_str_bin(dec_op1 + dec_op2)
                np_bin_result = get_np_bin(bin_result, result_digits)

                np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
                np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)
                np_bin_result_algo, _ = add_two_numbers(np_bin_op1, np_bin_op2)

                is_equal = np.array_equal(np_bin_result, np_bin_result_algo)
                is_all_correct = is_all_correct and is_equal
    return is_all_correct


def test_func_subtract_two_numbers():
    is_all_correct = True
    for operand_digits in operand_digits_list:
        # varying part
        result_digits = get_result_digits(operand_digits, 'subtract')
        for int_dec_operand1 in range(2**operand_digits):
            for int_dec_operand2 in range(2**operand_digits):
                if int_dec_operand1 >= int_dec_operand2: # Only these cases are dealth with.
                    # varying part
                    bin_result = get_str_bin(int_dec_operand1 - int_dec_operand2)
                    np_result = get_np_bin(bin_result, result_digits)

                    np_operand1 = get_np_bin(get_str_bin(int_dec_operand1), operand_digits)
                    np_operand2 = get_np_bin(get_str_bin(int_dec_operand2), operand_digits)
                    np_bin_result_algo, _ = subtract_two_numbers(np_operand1, np_operand2)

                    is_equal = np.array_equal(np_result, np_bin_result_algo)
                    is_all_correct = is_all_correct and is_equal
    return is_all_correct


def test_func_multiply_two_numbers():
    is_all_correct = True
    for operand_digits in operand_digits_list:
        # varying part
        result_digits = get_result_digits(operand_digits, 'multiply')
        for int_dec_operand1 in range(2**operand_digits):
            for int_dec_operand2 in range(2**operand_digits):
                # varying part
                bin_result = get_str_bin(int_dec_operand1 * int_dec_operand2)
                np_result = get_np_bin(bin_result, result_digits)

                np_operand1 = get_np_bin(get_str_bin(int_dec_operand1), operand_digits)
                np_operand2 = get_np_bin(get_str_bin(int_dec_operand2), operand_digits)
                np_bin_result_algo, _ = multiply_two_numbers(np_operand1, np_operand2)

                is_equal = np.array_equal(np_result, np_bin_result_algo)
                is_all_correct = is_all_correct and is_equal
    return is_all_correct


def test_func_divide_two_numbers():
    is_all_correct = True
    for operand_digits in operand_digits_list:
        # varying part
        result_digits = get_result_digits(operand_digits, 'divide')
        for int_dec_operand1 in range(2**operand_digits):
            for int_dec_operand2 in range(1, 2**operand_digits): # Exclude `int_dec_operand2 = 0`
                # varying part
                bin_result = get_str_bin(int_dec_operand1 // int_dec_operand2)
                np_result = get_np_bin(bin_result, result_digits)

                np_operand1 = get_np_bin(get_str_bin(int_dec_operand1), operand_digits)
                np_operand2 = get_np_bin(get_str_bin(int_dec_operand2), operand_digits)
                np_bin_result_algo, _, _ = divide_two_numbers(np_operand1, np_operand2)

                is_equal = np.array_equal(np_result, np_bin_result_algo)
                is_all_correct = is_all_correct and is_equal
    return is_all_correct


def test_func_modulo_two_numbers():
    is_all_correct = True
    for operand_digits in operand_digits_list:
        # varying part
        result_digits = get_result_digits(operand_digits, 'modulo')
        for int_dec_operand1 in range(2**operand_digits):
            for int_dec_operand2 in range(1, 2**operand_digits): # Exclude `int_dec_operand2 = 0`
                # varying part
                bin_result = get_str_bin(int_dec_operand1 % int_dec_operand2)
                np_result = get_np_bin(bin_result, result_digits)

                np_operand1 = get_np_bin(get_str_bin(int_dec_operand1), operand_digits)
                np_operand2 = get_np_bin(get_str_bin(int_dec_operand2), operand_digits)
                np_bin_result_algo, _ = modulo_two_numbers(np_operand1, np_operand2)

                is_equal = np.array_equal(np_result, np_bin_result_algo)
                is_all_correct = is_all_correct and is_equal
    return is_all_correct


def test_multiply_symmetric_carries():
    '''
    Purpose : To test whether the number of carries while multipication is same for a * b and b * a.
    Result  : The number of carries is always same for a * b and b * a.
    '''
    is_all_symmetric = True
    for operand_digits in operand_digits_list:
        for int_dec_operand1 in range(2**operand_digits):
            for int_dec_operand2 in range(2**operand_digits):
                operand1 = get_np_bin(get_str_bin(int_dec_operand1), operand_digits)
                operand2 = get_np_bin(get_str_bin(int_dec_operand2), operand_digits)
                result1, _ = multiply_two_numbers(operand1, operand2)
                result2, _ = multiply_two_numbers(operand2, operand1)

                is_equal = np.array_equal(result1, result2)
                is_all_symmetric = is_all_symmetric and is_equal
    return is_all_symmetric


def test_import_random_sampled_carry_datasets(n_samples=10):
    '''
    "To test the function `import_random_sampled_carry_datasets`"
    '''
    is_all_correct = True

    for operand_digits in operand_digits_list:
        for operator in operators_list:
            carry_datasets = import_random_sampled_carry_datasets(operand_digits, operator, n_samples)
            for n_carries in carry_datasets.keys():
                n_operations = carry_datasets[n_carries]['input'].shape[0]
                for i_operation in range(n_operations):
                    operand1 = carry_datasets[n_carries]['input'][i_operation, :operand_digits]
                    operand2 = carry_datasets[n_carries]['input'][i_operation, operand_digits:]
                    result = carry_datasets[n_carries]['output'][i_operation, :]

                    result_by_computing = operate_two_numbers(operand1, operand2, operator)[0] # Get the first element

                    is_equal = np.array_equal(result, result_by_computing)
                    if not is_equal:
                        print(operand1)
                        print(operand2)
                        print(result)
                        print(result_by_computing)
                        print('================')

                    is_all_correct = is_all_correct and is_equal

    return is_all_correct
