import os
from experiment_utils import load_questions
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
MODULOS = 0

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
            if g[0].upper() not in ["M", "F"]:
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
        info_correct = True if c[0].upper() == "Y" else False

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
        r = input("Are you ready (Y/N) ??\n\n\n")
        ready = True if r[0].upper() == "Y" else False


def record_results(name, age, gender, ability, results):
    """
    Record user id and results

    return: None
    """
    id = get_id()
    with open(USER_INFO_FILE, "a+") as fh:
        fh.write("{id}={name}".format(id=id, name=name))

    file_name = "{id}_{age}_{gender}_{ability}.result".format(id=id, age=age, gender=gender, ability=ability)
    file_name = os.path.join(RESULTS_DIR, file_name)
    with open(file_name, "wb") as fh:
        for result in results:
            pass


def get_id():
    return len(os.listdir(RESULTS_DIR)) + 1


def run_experiment():
    question_set = load_questions(add=ADD, subtract=SUBTRACT, multiply=MULTIPLY, divide=DIVIDE)


def main():

    welcome()
    name, age, gender, ability = get_user_data()
    wait_for_user()
    experiment_results = run_experiment()
    record_results(name, age, gender, ability, experiment_results)


if __name__ == "__main__":
    main()
