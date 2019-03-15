from tkinter import *
import tkinter.font

def next_problem():
    pass


def callback():
    str_message = "Wrong!\nTrue answer:\n{}".format("10010")
    Submit_message = Label(master, text=str_message, font=font_obj)
    Submit_message.grid(row=6,column=1, columnspan=n_result_digits)

    button_next = Button(master, text="Next", font=font_obj, command=next_problem)
    button_next.grid(row=7,column=1, columnspan=n_result_digits)

master = Tk()

window_h = 7
window_w = 5
master.grid_rowconfigure(0, weight=1)
master.grid_rowconfigure(window_h+1, weight=1)
master.grid_columnconfigure(0, weight=1)
master.grid_columnconfigure(window_w+1, weight=1)

font_obj = tkinter.font.Font(size=36, weight='bold')
digits = [0, 1]
n_result_digits = 5
n_operand_digits = 4
n_operands = 2
operator_sign = '+'
operand1 = [0, 1, 0, 1]
operand2 = [1, 1, 0, 1]
operands = [operand1, operand2]

for i_row in range(n_operands):
    for i_col in range(n_operand_digits):
        op_digit = Label(master, text=str(operands[i_row][i_col]), font=font_obj)
        op_digit.grid(row=i_row+1, column=i_col+2)

sign = Label(master, text=operator_sign, font=font_obj)
sign.grid(row=n_operands, column=1)

v_list = list()

for i in range(n_result_digits):

    v = IntVar()
    v.set(-1) # initialize
    v_list.append(v)

    for digit in digits:
        b = Radiobutton(master, text=str(digit), variable=v, value=digit,
            indicatoron=0, height=1, width=2, font=font_obj)
        b.grid(row=digit+3, column=i+1)
        #b.pack(anchor=W)

button_submit = Button(master, text="Submit", font=font_obj, command=callback)
button_submit.grid(row=5,column=1, columnspan=n_result_digits)

mainloop()
