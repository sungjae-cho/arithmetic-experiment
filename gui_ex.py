from tkinter import *
import tkinter.font
import time


class Quiz(object):

    def __init__(self, question_bank, operator):
        self.master = self.setup()
        self.responses = []
        self.operator = operator
        self.question_bank = question_bank
        self.extra_panels = []
        self.start_time = None
        self.font_default = tkinter.font.Font(size=36, weight='bold')
        self.font_true_answer = tkinter.font.Font(size=34, weight='bold')


    def setup(self):
        master = Tk()

        window_h = 8
        window_w = 5
        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(window_h+1, weight=1)
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(window_w+1, weight=1)
        return master


    def open_question(self):
        if not self.question_bank:
            self.master.destroy()
            return

        # Full screen
        width_value = self.master.winfo_screenwidth()
        height_value = self.master.winfo_screenheight()
        self.master.geometry("{}x{}+0+0".format(width_value, height_value))

        operand1, operand2, true_answer, _ = self.question_bank.pop(0)
        for _ in range(len(self.extra_panels)):
            self.extra_panels.pop().destroy()



        digits = [0, 1]
        n_result_digits = 5 if self.operator == "+" else 4
        n_operand_digits = 4
        n_operands = 2
        operands = [operand1, operand2]

        for i_row in range(n_operands):
            for i_col in range(n_operand_digits):
                op_digit = Label(self.master, text=str(operands[i_row][i_col]), font=self.font_default)
                op_digit.grid(row=i_row+1, column=i_col+2)

        sign = Label(self.master, text=self.operator, font=self.font_default)
        sign.grid(row=n_operands, column=1, ipadx=1)

        v_list = list()

        for i in range(n_result_digits):

            v = IntVar()
            v.set(-1) # initialize
            v_list.append(v)

            for digit in digits:
                b = Radiobutton(self.master, text=str(digit), variable=v, value=digit,
                    indicatoron=0, height=1, width=2, font=self.font_default)
                b.grid(row=digit+3, column=i+1 if self.operator == "+" else i+2)
                #b.pack(side=LEFT)

        self.start_time = time.time()
        cal = lambda : self.callback(true_answer, n_operand_digits, v_list)
        empty_space = Label(self.master, text="", font=self.font_true_answer)
        empty_space.grid(row=5, column=1, columnspan=n_operand_digits+1)
        button_submit = Button(self.master, text="Submit", font=self.font_default, command=cal)
        button_submit.grid(row=6,column=1 if self.operator == "+" else 2, columnspan=n_result_digits)
        self.master.mainloop()


    def callback(self, true_answer, n_operand_digits, v_list):
        if not self.start_time:
            return

        answer, rt = [v.get() if v.get() >=0 else 0 for v in v_list], time.time() - self.start_time
        true_answer = true_answer.tolist()
        if len(true_answer) > len(answer):
            for i in range(len(true_answer) - len(answer)):
                answer = [0] + answer
        elif len(answer) > len(true_answer):
            for i in range(len(answer) - len(true_answer)):
                true_answer = [0] + true_answer

        self.responses.append((answer, rt, true_answer == answer))
        self.start_time = None
        str_message = "{}\nTrue answer:\n{}".format("Wrong" if answer != true_answer else "Correct", "".join(str(i) for i in true_answer))
        Submit_message = Label(self.master, text=str_message, font=self.font_true_answer)
        Submit_message.grid(row=7, column=1, columnspan=n_operand_digits+1)
        button_next = Button(self.master, text="Next", font=self.font_default, command=self.open_question)
        button_next.grid(row=8, column=1, columnspan=n_operand_digits+1)
        self.extra_panels = [Submit_message, button_next]
