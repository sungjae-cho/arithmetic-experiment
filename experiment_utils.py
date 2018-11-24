from data_utils import *
import numpy
from itertools import cycle
from random import shuffle, choice
QUESTION_TYPE = ["add", "subtract", "multiply", "divide", "modulo"]
DIGIT_OPERANDS_CARRIES = {4: [1, 2, 3],
                          6: [1, 2, 3, 4, 5],
                          8: [1, 2, 3, 4, 5]}


def load_questions(add=0, subtract=0, multiply=0, divide=0, modulo=0):
    question_nums = [add, subtract, multiply, divide, modulo]
    questions = {}
    for num_questions, question_type in zip(question_nums, QUESTION_TYPE):
        if not num_questions:
            continue

        question_set = {}
        carry_digit_combos = _get_carry_operand_combos(DIGIT_OPERANDS_CARRIES)
        shuffle(carry_digit_combos)
        question_iterator = cycle(carry_digit_combos)
        for _ in range(num_questions):
            carrier_num, digit_operants = next(question_iterator)
            if digit_operants not in question_set.keys():
                question_set[digit_operants] = generate_datasets(digit_operants, question_type)

            # TODO: Figure out how to delete rows from numpy arrays
            question_index = choice(range(len(question_set[digit_operants][carrier_num]["input"])))
            question = question_set[digit_operants][carrier_num]["input"][question_index]
            answer = question_set[digit_operants][carrier_num]["output"][question_index]
            if question_type in questions:
                questions[question_type].append((question, answer, carrier_num))
            else:
                questions[question_type] = [(question, answer, carrier_num)]
    return questions


def _get_carry_operand_combos(carry_operand_dict):

    combo_list = []
    for operand_digits, carries in carry_operand_dict.items():
        for carry in carries:
            combo_list.append((carry, operand_digits))
    return combo_list


def test_load_questions():

    for _ in range(10):
        questions = load_questions(add=5, subtract=0, multiply=5, divide=0, modulo=0)
        for question_type ,question_set in questions.items():
            for question in question_set:
                operands, answer, carrier_num = question
                len_operand = len(operands) // 2
                operand1, operand2 = operands[:len_operand], operands[len_operand:]
                if question_type == "add":
                    numpy.testing.assert_equal(answer, add_two_numbers(operand1, operand2)[0])
                elif question_type == "subtract":
                    numpy.testing.assert_equal(answer, subtract_two_numbers(operand1, operand2)[0])
                elif question_type == "multiply":
                    numpy.testing.assert_equal(answer, multiply_two_numbers(operand1, operand2)[0])
                elif question_type == "divide":
                    numpy.testing.assert_equal(answer, divide_two_numbers(operand1, operand2)[0])
                elif question_type == "modulo":
                    numpy.testing.assert_equal(answer, modulo_two_numbers(operand1, operand2)[0])
                else:
                    raise Exception

test_load_questions()

