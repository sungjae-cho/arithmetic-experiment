MALE, FEMALE = "MALE", "FEMALE"
BAD, OKAY, GOOD = range(1, 4)

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

    return name, age, gender, ability


def main():

    welcome()
    name, age, gender, ability = get_user_data()
    print(name, age, gender, ability)


if __name__ == "__main__":
    main()

