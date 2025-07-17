# This is a sample Python script.

def main():
    print_hi('Krzysztof')

def print_hi(name):
    print(f'Hi, {name}!')

if __name__ == '__main__':
    main()

#~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.

print("\\")

print("Hello", "World")

print("My", "name", "is", "Monty", "Python.", sep="-")

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

print(3 ** 2 // 2)

1//3 * 3 ** 0
4//3

4/2 - 2 ** 1
4/2 - 2 ** 0

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