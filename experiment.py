from experiment_utils import evenly_load_questions
import random, time, os, numpy, datetime, pytz
from gui_ex import Quiz
from copy import deepcopy
import sys # to get arguments from the command line execution.
SEOUL = pytz.timezone('Asia/Seoul')
PROJ_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
USER_DATA_PATH = os.path.join(PROJ_PATH, "user_data")
ACTUAL_RESULTS_DIR = os.path.join(USER_DATA_PATH, "results")
PRACTICE_RESULTS_DIR = os.path.join(USER_DATA_PATH, ".practice_results")
MALE, FEMALE = range(1, 3)
BAD, BAVERAGE, AAVERAGE, GOOD = range(1, 5)
DIGIT_OPERANDS = 4
N_PRACTICES_PER_CARRIES = 1
N_CARRIES_DICT = {'add':5, 'subtract':4, 'multiply':9, 'divide':4, 'modulo':4}


OPERATION_DICT = {"ADD": "+", "SUBTRACT": "—", "MULTIPLY": "x", "DIVIDE": "÷", "MODULO": "%"}


def get_results_dir():
    if sys.argv[1].lower() == 'practice':
        PRACTICE = True
    if sys.argv[1].lower() == 'experiment':
        PRACTICE = False
    RESULTS_DIR = PRACTICE_RESULTS_DIR if PRACTICE else ACTUAL_RESULTS_DIR

    return RESULTS_DIR


def get_question_set():
    question_set = sys.argv[2].upper()
    return question_set


def get_problems_per_carries():
    problems_per_carries = int(sys.argv[3])
    question_set = get_question_set()

    operators = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MODULO"]
    problems_list = [0] * len(operators)

    i_op = operators.index(question_set)
    problems_list[i_op] = problems_per_carries

    return problems_list


class ImBored(Exception):

    def __init__(self, results):
        super(ImBored, self).__init__("Thank you for our time ^^")
        self.results = results


def _get_input():
    response = ""
    try:
        response = input().strip()
    except UnicodeDecodeError:
        print("Make sure that you set your keyboard to English")

    return response


def welcome():
    """
    Nice welcome message

    :return: None
    """
    welcome_msg = "Welcome to the Arithmetic Cognitive Science Experiment ^^"
    print(welcome_msg)
    print("-" * len(welcome_msg))


def get_user_data():
    """
    Function for reading in user data from console

    :return: name, age, gender, ability
    """
    info_correct = False
    while not info_correct:

        year = ""
        # Make sure age is valid assuming no one is over 100
        while not _valid_birth_year(year):
            print("Please enter your birth year :")
            year = _get_input()
        print("Great !!\n")

        gender = None
        while not gender:
            print("Are you male or female (M/F) :")
            g = _get_input()
            if not g or g[0].upper() not in ["M", "F"]:
                continue

            gender = MALE if g[0].upper() == "M" else FEMALE
        print("Excellent\n")

        major = None
        while not major:
            print("What is your major :")
            m = _get_input()
            if not m or len(m) < 2:
                continue

            major = m.upper()
        print("Wonderful\n")

        ability = None
        print("How would you rate your math ability on a scale of 1 -> 4")
        print("4: Very Good")
        print("3: Above Average")
        print("2: Below Average")
        print("1: Very Bad")
        while not ability:
            print("Please enter ability level here :")
            ab = _get_input()
            if not ab.isdigit() or int(ab) not in [BAD, BAVERAGE, AAVERAGE, GOOD]:
                continue

            ability = int(ab)
        print("Fantastic\n")

        print("Have you entered all your input correct (Y/N):\t")
        c = _get_input()
        info_correct = True if c and c[0].upper() == "Y" else False

    # Make sure we have everything we need
    assert all([year, gender, ability, major])
    print("Okay !! Let's get started !!!\n\n")

    return year, gender, ability, major


def _valid_birth_year(year):
    return year.isdigit() and (1920 < int(year) < 2005)


def wait_for_user():
    ready = False
    print("You are now going to answer some binary math questions.")
    print("These questions will involve {0} operations.".format(get_question_set()))
    print("Please answer these questions as quickly and accurately as possible.")
    print("Wait for your instructor to go over the experiment in detail before beginning the experiment.")
    #print("You can leave the experiment anytime by typing 'QUIT' in response to the "
    #      "'Are you ready for next question (Y/N) ??' question")
    while not ready:
        print("Are you ready (Y/N) ??")
        r = _get_input()
        ready = True if r and r[0].upper() == "Y" else False


def record_results(year, gender, ability, major, results):
    """
    Record user id and results

    return: None
    """
    id = str(SEOUL.localize(datetime.datetime.now()))[:-7].replace(" ", "_").replace(":", "_")
    file_name = "{id}_{year}_{gender}_{ability}_{major}.result".format(id=id, year=year, gender=gender, ability=ability, major=major)
    file_name = os.path.join(get_results_dir(), file_name)
    print("Writing results to file {}".format(file_name))
    with open(file_name, "w+") as fh:
        for result in results:
            result_string = "{qid}\t{correct}\t{duration}\t{user_answer}\t{correct_answer}\t{operand_digits}\t" \
                            "{question_type}\t{num_carries}\t{button_order}\n".format(**result)
            fh.write(result_string)


def evaluate_readiness(input, results):
    if input.upper() == "QUIT":
        raise ImBored(results)
    return True if input and input[0].upper() == "Y" else False


def run_experiment():
    [ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULO] = get_problems_per_carries()
    question_set = evenly_load_questions(DIGIT_OPERANDS, add=ADD, subtract=SUBTRACT, multiply=MULTIPLY, divide=DIVIDE, modulo=MODULO)
    results = []
    total_nun_questions = sum([len(qs) for qs in question_set.values()])
    try:
        while question_set:
            completed_questions = total_nun_questions - sum([len(qs) for qs in question_set.values()])
            ready = False
            while not ready:
                print("Completed {completed}/{total} questions".format(completed=completed_questions, total=total_nun_questions))
                print("Completed {percentage}% of quiz".format(percentage=round(completed_questions/float(total_nun_questions) * 100, 2)))
                print("Are you ready for next question (Y/N) ??")
                r = _get_input()
                ready = evaluate_readiness(r, results=results)
            question_type = random.choice(list(question_set.keys()))
            os.system("clear")
            operand1, operand2, answer, num_carries, qid = _get_question(question_set, question_type)
            result = ask_question(question_type, operand1, operand2, answer, num_carries, qid)
            results.append(result)
    except ImBored as e:
        print(e)
        results = e.results
    print("\n Great Job: You got {correct}/{total}".format(correct=len([1 for result in results if result["correct"]]),
                                                           total=len(results)))
    print("Thank You !!!")
    return results

def run_ui_experiment():
    [ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULO] = get_problems_per_carries()
    question_set = evenly_load_questions(DIGIT_OPERANDS, add=ADD, subtract=SUBTRACT, multiply=MULTIPLY, divide=DIVIDE, modulo=MODULO, n_practices=N_PRACTICES_PER_CARRIES)
    extracted_question_bank = [(question[0], question[1], question[2], question[3]) for question in question_set[get_question_set().lower()]]
    question_bank_copy = deepcopy(extracted_question_bank)
    quiz = Quiz(extracted_question_bank, OPERATION_DICT[get_question_set().upper()])
    quiz.open_question()
    print("\n Great Job: You got {correct}/{total}".format(correct=len([1 for result in quiz.responses if result[2]]),
                                                           total=len(quiz.responses)))
    print("Thank You !!!")
    results = []
    for i in range(len(quiz.responses)):
        results.append(dict(qid=question_bank_copy[i][0].tolist() + question_bank_copy[i][1].tolist(), correct=quiz.responses[i][2], duration=round(quiz.responses[i][1], 3),
                            user_answer=quiz.responses[i][0], correct_answer=question_bank_copy[i][2], operand_digits=DIGIT_OPERANDS,
                            question_type=get_question_set(), num_carries=question_bank_copy[i][3], button_order=quiz.button_order[i]))

    assert len(results) == len(question_bank_copy)
    n_practice_questions = N_PRACTICES_PER_CARRIES * N_CARRIES_DICT[get_question_set().lower()]
    return results[n_practice_questions:]


def ask_question(question_type, operand1, operand2, answer, num_carries, qid):
    if question_type in ["modulo", "divide"]:
        print()
        print("┌{0}".format("─" * (2 * len(operand2) - 1)).rjust(30))
        print("{operand2} ┃{operand1}".format(operand1=" ".join(str(d) for d in operand1), operand2=" ".join(str(d) for d in operand2)).rjust(30))
        print("-" * 30)
    else:
        print()
        print("{operand1}".format(operand1=" ".join(str(d) for d in operand1)).rjust(30))
        print("{operation} {operand2}".format(operation=OPERATION_DICT[question_type.upper()], operand2=" ".join(str(d) for d in operand2)).rjust(30))
        print("-" * 30)
    start_time = time.time()
    valid_answer = False
    while not valid_answer:
        print("Your answer: ")
        user_answer = _get_input().replace(" ", "")
        valid_answer = _valid_answer(user_answer)
    duration = round(time.time() - start_time, 3)
    correct = validate_answer(answer, user_answer)
    string_answer = "".join([str(int(i)) for i in answer])
    if correct:
        print(" Correct !! Good Job".rjust(30))
    else:
        print("Hard luck :(".rjust(30))
        print("The correct answer was {answer}".format(answer=string_answer).rjust(30))
    return {"correct": correct, "duration": duration, "user_answer":user_answer, "correct_answer": string_answer,
            "question_type": question_type, "num_carries": num_carries, "operand_digits": len(operand1), "qid": qid}


def validate_answer(answer, user_answer):

    user_answer_array = [int(i) for i in user_answer]
    if len(answer) > len(user_answer_array):
        user_answer_array = [0] * (len(answer) - len(user_answer_array)) + user_answer_array
    correct = (numpy.array(user_answer_array) == answer)
    if not isinstance(correct, bool):
        correct = correct.all()
    return correct


def _valid_answer(answer, first=False):
    valid = True
    if len(answer) < 1 or not all([i in ["0", "1"] for i in answer]):
        if not first:
            print("Invalid answer !!!")
        valid = False
    return valid


def _get_question(question_set, question_type):
    """

    :param question_set:
    :param question_type:
    :return:
    """
    num_questions = len(question_set[question_type])
    question_index = random.choice(range(num_questions))
    question = question_set[question_type].pop(question_index)
    if not question_set[question_type]:
        del question_set[question_type]
    return question


def main():

    welcome()
    year, gender, ability, major = get_user_data()
    wait_for_user()
    experiment_results = run_ui_experiment()
    record_results(year, gender, ability, major, experiment_results)


if __name__ == "__main__":
    main()
