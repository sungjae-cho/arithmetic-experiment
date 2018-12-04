from data_utils import *
import numpy
from random import shuffle, choice
QUESTION_TYPE = ["add", "subtract", "multiply", "divide", "modulo"]
TEST_DIGIT_OPERANDS = 4
NUM_TEST_QUESTIONS_PER_CARRY = 10


def evenly_load_questions(digit_operands, add=0, subtract=0, multiply=0, divide=0, modulo=0):
    """
    This method will use a sample by replacement technique to provide the given number of questions
    for each num_carry sub dataset for that operation with the given number of digit_operands
    """
    question_nums = [add, subtract, multiply, divide, modulo]
    questions = {}
    for num_questions, question_type in zip(question_nums, QUESTION_TYPE):
        if not num_questions:
            continue

        questions[question_type] = []
        question_set = generate_datasets(digit_operands, question_type)
        all_carries = question_set.keys()

        for num_carries in all_carries:
            num_q = num_questions
            chosen_indices = []
            possible_indices = list(range(len(question_set[num_carries]["input"])))
            while len(possible_indices) < num_q:
                chosen_indices += possible_indices
                num_q -= len(possible_indices)
            chosen_indices = chosen_indices + \
                     [possible_indices.pop(choice(range(len(possible_indices)))) for _ in range(num_q)]
            shuffle(chosen_indices)
            question_nums = [question_set[num_carries]["input"][index] for index in chosen_indices]
            answer_nums = [question_set[num_carries]["output"][index] for index in chosen_indices]
            for q, a, i in zip(question_nums, answer_nums, chosen_indices):
                questions[question_type].append((q[:digit_operands], q[digit_operands:], a, num_carries, i))
        shuffle(questions[question_type])
    return questions


def test_evenly_load_questions_loads_valid_questions():

    for _ in range(10):
        questions = evenly_load_questions(TEST_DIGIT_OPERANDS, add=10, subtract=10, multiply=10, divide=10, modulo=10)
        for question_type, question_set in questions.items():

            for question in question_set:
                operand1, operand2, answer, _, _ = question
                int1, int2, int_answer = binary2decimal(operand1), binary2decimal(operand2), binary2decimal(answer)
                if question_type == "add":
                    numpy.testing.assert_equal(answer, add_two_numbers(operand1, operand2)[0])
                    assert int1 + int2 == int_answer
                elif question_type == "subtract":
                    numpy.testing.assert_equal(answer, subtract_two_numbers(operand1, operand2)[0])
                    assert int1 - int2 == int_answer
                elif question_type == "multiply":
                    numpy.testing.assert_equal(answer, multiply_two_numbers(operand1, operand2)[0])
                    assert int1 * int2 == int_answer
                elif question_type == "divide":
                    numpy.testing.assert_equal(answer, divide_two_numbers(operand1, operand2)[0])
                    assert int1 // int2 == int_answer
                elif question_type == "modulo":
                    numpy.testing.assert_equal(answer, modulo_two_numbers(operand1, operand2)[0])
                    assert int1 % int2 == int_answer
                else:
                    raise Exception


def binary2decimal(binary_array):
    return int("".join([str(int(i)) for i in binary_array]), 2)


def test_evenly_load_questions_loads_correct_number_of_questions():

    for question_type in QUESTION_TYPE:
        kwargs = {"add": 0, "subtract": 0, "multiply": 0, "divide": 0, "modulo": 0}
        kwargs[question_type] = NUM_TEST_QUESTIONS_PER_CARRY
        questions = evenly_load_questions(TEST_DIGIT_OPERANDS, **kwargs)
        actual_question_set = generate_datasets(TEST_DIGIT_OPERANDS, question_type)
        assert len(actual_question_set.keys()) * NUM_TEST_QUESTIONS_PER_CARRY == len(questions[question_type])


def test_question_indices_map_back_to_correct_questions():

    for question_type in QUESTION_TYPE:
        kwargs = {"add": 0, "subtract": 0, "multiply": 0, "divide": 0, "modulo": 0}
        kwargs[question_type] = NUM_TEST_QUESTIONS_PER_CARRY
        questions = evenly_load_questions(TEST_DIGIT_OPERANDS, **kwargs)
        actual_question_set = generate_datasets(TEST_DIGIT_OPERANDS, question_type)
        for operand1, operand2, answer, num_carries, index in questions[question_type]:
            numpy.testing.assert_equal(operand1, actual_question_set[num_carries]["input"][index][:TEST_DIGIT_OPERANDS])
            numpy.testing.assert_equal(operand2, actual_question_set[num_carries]["input"][index][TEST_DIGIT_OPERANDS:])
            numpy.testing.assert_equal(answer, actual_question_set[num_carries]["output"][index])


def test_sample_with_replacement():
    for _ in range(10):
        questions = evenly_load_questions(TEST_DIGIT_OPERANDS, multiply=NUM_TEST_QUESTIONS_PER_CARRY)
        actual_question_set = generate_datasets(TEST_DIGIT_OPERANDS, "multiply")
        total_indices_chosen_per_carry = {key: [] for key in actual_question_set.keys()}
        for _, _, _, num_carries, index in questions["multiply"]:
            total_indices_chosen_per_carry[num_carries].append(index)
        for num_carries in actual_question_set.keys():
            if NUM_TEST_QUESTIONS_PER_CARRY <= len(actual_question_set[num_carries]["output"]):
                assert len(total_indices_chosen_per_carry[num_carries]) == len(set(total_indices_chosen_per_carry[num_carries]))
            else:
                assert len(total_indices_chosen_per_carry[num_carries]) != len(set(total_indices_chosen_per_carry[num_carries]))
                assert len(total_indices_chosen_per_carry[num_carries]) == NUM_TEST_QUESTIONS_PER_CARRY


def test_files_give_correct_output():
    from experiment import PRACTICE_RESULTS_DIR
    datasets = {}
    for result_file in os.listdir(PRACTICE_RESULTS_DIR):
        if not result_file or "_.txt" in result_file:
            continue
        with open(os.path.join(PRACTICE_RESULTS_DIR, result_file)) as fh:
            for line in fh:
                line_info = [i for i in line.split() if i]
                index, correct, time, user_answer, correct_answer, operand_digits, question_type, num_carries = line_info
                correct = True if correct == "True" else False
                index, operand_digits, num_carries = int(index), int(operand_digits), int(num_carries)
                user_answer = [int(i) for i in list(user_answer.strip())]
                correct_answer = [int(j) for j in list(correct_answer.strip())]
                if question_type not in datasets.keys():
                    datasets[question_type] = {}
                if operand_digits not in datasets[question_type]:
                    datasets[question_type][operand_digits] = generate_datasets(operand_digits, question_type)
                # Make sure if we mark answer correct they actually are correct
                if correct:
                    assert binary2decimal(user_answer) == binary2decimal(correct_answer)
                else:
                    assert binary2decimal(user_answer) != binary2decimal(correct_answer)

                # Now make sure the correct answer matches with our stored correct answer
                stored_answer = datasets[question_type][operand_digits][num_carries]["output"][index]
                assert binary2decimal(correct_answer) == binary2decimal(stored_answer)


if __name__ == "__main__":
    test_files_give_correct_output()
    test_evenly_load_questions_loads_valid_questions()
    test_evenly_load_questions_loads_correct_number_of_questions()
    test_question_indices_map_back_to_correct_questions()
    test_sample_with_replacement()


