from experiment_utils import load_questions
import random, time, os, numpy
PROJ_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
USER_DATA_PATH = os.path.join(PROJ_PATH, "user_data")
USER_INFO_FILE = os.path.join(USER_DATA_PATH, "user_info", "user_id.txt")
RESULTS_DIR = os.path.join(USER_DATA_PATH, "results")
MALE, FEMALE = range(1, 3)
BAD, OKAY, GOOD = range(1, 4)
ADD = 5
SUBTRACT = 5
MULTIPLY = 5
DIVIDE = 5
MODULO = 5
OPERATION_DICT = {"ADD": "+", "SUBTRACT": "—", "MULTIPLY": "x", "DIVIDE": "÷", "MODULO": "%"}


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
        name = ""
        while not name:
            name = input("Please enter your name :\t")
        print("Thank You!!\n")

        age = ""
        # Make sure age is valid assuming no one is over 100
        while age.isdigit() is False or len(age) > 2:
            age = input("Please enter your age :\t").strip()
        print("Great !!\n")

        gender = None
        while not gender:
            g = input("Are you male or female (M/F) :\t").strip()
            if not g or g[0].upper() not in ["M", "F"]:
                continue

            gender = MALE if g[0].upper() == "M" else FEMALE
        print("Excellent\n")

        ability = None
        print("How would you rate your math ability on a scale of 1 -> 3")
        print("3: Above Average")
        print("2: Average")
        print("1: Below Average")
        while not ability:
            ab = input("Please enter ability level here :\t").strip()
            if not ab.isdigit() or int(ab) not in [GOOD, OKAY, BAD]:
                continue

            ability = int(ab)
        print("Fantastic\n")

        c = input("Have you entered all your input correct (Y/N):\t").strip()
        info_correct = True if c and c[0].upper() == "Y" else False

    # Make sure we have everything we need
    assert all([name, age, gender, ability])
    print("Okay !! Let's get started !!!\n\n")

    return name, age, gender, ability


def wait_for_user():
    ready = False
    print("You are now going to answer some binary math questions.")
    print("These questions will involve +, -, x, / and % (modulo) operations.")
    print("Please answer these questions as quickly and accurately as possible.")
    print("Wait for your instructor to go over the experiment in detail before beginning the experiment.")
    while not ready:
        r = input("Are you ready (Y/N) ??\t")
        ready = True if r and r[0].upper() == "Y" else False


def record_results(name, age, gender, ability, results):
    """
    Record user id and results

    return: None
    """
    id = get_id()
    with open(USER_INFO_FILE, "a+") as fh:
        fh.write("{id}={name}\n".format(id=id, name=name))

    file_name = "{id}_{age}_{gender}_{ability}.result".format(id=id, age=age, gender=gender, ability=ability)
    file_name = os.path.join(RESULTS_DIR, file_name)
    with open(file_name, "w+") as fh:
        for result in results:
            result_string = "{qid}\t{correct}\t{duration}\t{user_answer}\t{correct_answer}\t{operand_digits}\t" \
                            "{question_type}\t{num_carries}\n".format(**result)
            fh.write(result_string)


def get_id():
    return len(os.listdir(RESULTS_DIR)) + 1


def run_experiment():
    question_set = load_questions(add=ADD, subtract=SUBTRACT, multiply=MULTIPLY, divide=DIVIDE, modulo=MODULO)
    results = []
    while question_set:
        ready = False
        while not ready:
            r = input("Are you ready for next question (Y/N) ??")
            ready = True if r and r[0].upper() == "Y" else False
        question_type = random.choice(list(question_set.keys()))
        operand1, operand2, answer, num_carries, qid = _get_question(question_set, question_type)
        result = ask_question(question_type, operand1, operand2, answer, num_carries, qid)
        results.append(result)
    print("\n Great Job: You got {correct}/{total}".format(correct=len([1 for result in results if result["correct"]]),
                                                           total=len(results)))
    print("Thank You !!!")
    return results


def ask_question(question_type, operand1, operand2, answer, num_carries, qid):
    print("{operand1}".format(operand1=operand1).rjust(30))
    print("{operation} {operand2}".format(operation=OPERATION_DICT[question_type.upper()], operand2=operand2).rjust(30))
    print("-" * 30)
    start_time = time.time()
    valid_answer = False
    while not valid_answer:
        user_answer = input("Your answer: ").replace(" ", "")
        valid_answer = _valid_answer(user_answer)
    duration = time.time() - start_time
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
    name, age, gender, ability = get_user_data()
    wait_for_user()
    experiment_results = run_experiment()
    record_results(name, age, gender, ability, experiment_results)


if __name__ == "__main__":
    main()
