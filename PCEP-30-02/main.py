# This is a sample Python script.

def main():
    print_hi('Krzysztof')

def print_hi(name):
    print(f'Hi, {name}!')

if __name__ == '__main__':
    main()

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

# Python keywords

# 'False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

print("\\")

print("Hello", "World")

print("H", "E", "L", "L", "O", sep="-")

print("Monty", "Python.", sep="*", end="*\n")

print("I like \"Monty Python\"")

print("+" + 10 * "-" + "+")
print(("|" + " " * 10 + "|\n") * 5, end="")
print("+" + 10 * "-" + "+")

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

# Lists

# Create a list of countries
countries = ['Greece', 'India', 'USA', 'Canada']

# Print the list
print(countries)

# Print the first element of the list
print(countries[0])

# Print the last element of the list
print(countries[-1])

# Change the value of an element in the list
countries[1] = 'Italy'

# Get the length of the list
print(len(countries))

# Delete an item from the list
del countries[2]
print(countries)

print(len(countries))

the_list = ['1', 1, 1, 1]
print(the_list.index('1') == 0)
print(the_list.count(1) == 3)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

# 0b1010 is a binary number with a (decimal) value equal to 10.
print(0b1010)

# 0o123 is an octal number with a (decimal) value equal to 83.
print(0o123)

# 0x123 is a hexadecimal number with a (decimal) value equal to 291
print(0x123)

print(3_0000_0000 == 3E8)
print(3_0000_0000 == 3 * 10**8)

print(0.00_0000000000_0000000001 == 1E-22)

# twenty point twelve times ten raised to the power of eight

2.012E8

print(0.1 + 0.2 == 0.3)

list = [False, True, "2", 3, 4 ]
b = 0 in list
print(b)

x = [0, 1, 2]
x.insert(0,1)
del x[1]
print(sum(x))

mums = []
vals = mums
vals.append(1)

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

def fun(x):
    return 1 /(x + 1 / (x + 1/(x + 1/x)))

x = 100
print("y =", fun(x))

def check_value(x):
    return "Positive" if x > 0 else "Negative" if x < 0 else "Zero"

print(check_value(-10))
print(check_value(0))
print(check_value(10))

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

print(int("4") + float("5.6"))

nums = [1, 2, 3]
vals = nums
del vals[1:2]

print(6 // 3)
print(6. // 4)

print(14 % 4)
print(14 // 4)
print(3 * 4)
print(14 - 12)

x = 11  # User enters 11
y = 4  # User enters 4
x = x % y         # 11 % 4 = 3 → x becomes 3
x = x % y         # 3 % 4 = 3 → x stays 3
y = y % x         # 4 % 3 = 1 → y becomes 1
print(y)          # Output is 1

print(3 ** 2 // 2)

x = 1 / 2 + 3 // 3 + 4 ** 2
# 4 ** 2 = exponentiation = 16
# 3 // 3 = integer division = 1
# 1 / 2 = floating-point division = 0.5
x = 0.5 + 1 + 16  # which equals 17.5

1//3 * 3 ** 0
4//3

4/2 - 2 ** 1
4/2 - 2 ** 0

print(2 ** 2 ** 3)
print("2 ** 3 = 8", "2 ** 8 = 256", sep=";")

print(9 % 6 % 2)
print("9 % 6 = 3", "3 % 2 = 1", sep=";")

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

try:
    print(x)
except:
    print("An exception occurred")

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

try:
    print("Hello")
except:
    print("Something went wrong")
else:
    print("Nothing went wrong")

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

try:
    print(x)
except:
    print("Something went wrong")
finally:
    print("The 'try except' is finished")

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

try:
    n = 0
    res = 100 / n

except ZeroDivisionError:
    print("You can't divide by zero!")

except ValueError:
    print("Enter a valid number!")

else:
    print("Result is", res)

finally:
    print("Execution complete.")

# ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

# You can use try and except to handle errors from application

try:
    user_input = int(input("Enter a number: "))
    result = 10 / user_input
    print(result)
except ZeroDivisionError:
    print("You can't divide by zero")
except ValueError:
    print("You must enter a number")
# Python doesn't have a general exception handler and needs
# to catch each exception which can be in the app separately
# except:
#    print("Something went wrong")

# Hierarchical Exceptions
# You can use the hierarchy of exceptions to catch multiple exceptions
try:
    user_input = int(input("Enter a number: "))
    result = 10 / user_input
    print(result)
except ArithmeticError:
    # You can use the parent exception to catch all the child exceptions,
    # like ZeroDivisionError and others
    print("You can't divide by zero")
except ValueError:
    print("You must enter a number")

# Order matters in exception handling branches so if you want to catch
# some exceptions in the same level you can do by using ():
try:
    user_input = int(input("Enter a number: "))
    result = 10 / user_input
    print(result)
except (ZeroDivisionError, ValueError):
    print("Same level exceptions")
except ValueError:
    print("You must enter a number")

# You can use assert to validate your data and raise an exception
# if something wrong
try:
    user_input = int(input("Enter a number: "))
    assert user_input != 0
    result = 10 / user_input
    print(result)
except AssertionError:
    print("Error in validation of data")