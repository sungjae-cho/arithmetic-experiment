# Arithmetic experiment
To make a program that can do the psychology experiment of arithmetic cognition on human subjects

# Subject information
1. ID : `int`
1. Name : `str`
1. Age : `int`
1. Gender = `{'male', 'female'}`
1. Math ability = `{'good', 'okay', 'bad'}`

# Experiment record

# Data descriptions

# Import data structure
```Python
from data_utils import *
operand_digits = 4 # operand_digits in [4, 6, 8]
operator = 'add' # operator in ['add', 'substract', 'multiply', 'divide', 'modulo']
carry_dataset = import_carry_datasets(operand_digits, operator)
```

# How to access the data of `carry_dataset`
How see what kinds of carries exist in the dataset.
```Python
carry_dataset.keys()
```

How to access the 2-carry dataset
```Python
carry_dataset[2]
```

How to access the input the 2-carry dataset
```Python
carry_dataset[2]['input']
```
`carry_dataset[2]['input'].shape == (n_operations, 2 * operand_digits)`

How to access the input the 2-carry dataset
```Python
carry_dataset[2]['output']
```
`carry_dataset[2]['output'].shape == (n_operations, operand_digits)`

How to access the i-th operation in the 2-carry dataset
```Python
carry_dataset[2]['input'][i,:]
carry_dataset[2]['output'][i,:]
```

# Program dependency
1. Python 3.xx
1. Numpy
