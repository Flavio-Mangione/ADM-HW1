"""           Algorithmic Methods of Data Mining
                        HOMEWORK 1
Problem 1: This section contains exercises from HackeRank site
"""
# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Introduction """

#Exercise 1: Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")

#Exercise 2: Python If-Else

if __name__ == '__main__':
    n = int(input().strip())
if n%2==0 and n<=5 and n>=2:
    print("Not Weird")
elif n%2==0 and n<=20 and n>=6 or n%2!=0:
    print("Weird")

#Exercise 3: Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#Exercise 4: Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#Exercise 5: Loops

if __name__ == '__main__':
    n = int(input())
    print(*[number**2 for number in range(n)],sep="\n")

#Exercise 6: Write a function

def is_leap(year):
    leap = False
    if year % 4 == 0 and year % 100 != 0:
        leap = True
    elif year % 400 == 0 and year % 100 == 0:
        leap = True
    return leap

#Exercise 7: Print Function

if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i,end="")

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Data types """

#Exercise 8: List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

list=[[i,j,k] for i in range(x+1) for j in range (y+1) for k in range(z+1) if sum([i,j,k])!=n]
print(list)

#Exercise 9: Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
l=list(arr)
max_score=max([number for number in l if number!=max(l)])
print(max_score)

#Exercise 10: Nested Lists

if __name__ == '__main__':
    student=[]
    score_results=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        score_results.append(score)
        student.append([name,score])
    while score_results.count(min(score_results))>1:
        score_results.remove(min(score_results))
    score_results.sort()
    student=[student_list[0] for student_list in student if student_list[1]==min(score_results[1:])]
    print(*sorted(student),sep="\n")

#Exercise 11: Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    print(format(sum(student_marks[query_name])/len(student_marks[query_name]),".2f"))

#Exercise 12: Lists

if __name__ == '__main__':
    N = int(input())
initial_list=[]
for i in range(N):
    command = input().split()
    if command[0] == "insert":
        i = int(command[1])
        e = int(command[2])
        initial_list.insert(i, e)
    elif command[0] == "print":
        print(initial_list)
    elif command[0] == "remove":
        e = int(command[1])
        initial_list.remove(e)
    elif command[0] == "append":
        e = int(command[1])
        initial_list.append(e)
    elif command[0] == "sort":
        initial_list.sort()
    elif command[0] == "pop" and len(initial_list)>=0:
        initial_list.pop()
    elif command[0] == "reverse":
        initial_list.reverse()

#Exercise 13: Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
hash_tuple=tuple(integer_list)
print(hash(hash_tuple))

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Strings """

#Exercise 14: sWAP cASE

def swap_case(s):
    new_string=""
    for index in range (len(s)):
        if s[index]==s[index].lower():
            new_string+=s[index].upper()
        else:
            new_string+=s[index].lower()
    return new_string

#Exercise 15: String Split and Join
def split_and_join(line):
    my_string=line.split(" ")
    my_string="-".join(my_string)
    return my_string

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#Exercise 16: What's Your Name?

def print_full_name(first, last):
    print("Hello "+first+" "+last+"! "+"You just delved into python.")

#Exercise 17: Mutations

def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    return "".join(l)

#Exercise 18: Find a string

def count_substring(string, sub_string):
    sub_counter=0
    for index in range(len(string)):
        if string[index]==sub_string[0] and string[index:index+len(sub_string)]==sub_string:
            sub_counter+=1
    return sub_counter

#Exercise 19: String Validators

if __name__ == '__main__':
    s = input()
def verify_alpha(s):
    for character in s:
        if character.isalnum():
            return True
    return False
def verify_alphabet(s):
    for character in s:
        if character.isalpha():
            return True
    return False
def verify_digit(s):
    for character in s:
        if character.isdigit():
            return True
    return False
def verify_lower(s):
    for character in s:
        if character.islower():
            return True
    return False
def verify_upper(s):
    for character in s:
        if character.isupper():
            return True
    return False
print(verify_alpha(s))
print(verify_alphabet(s))
print(verify_digit(s))
print(verify_lower(s))
print(verify_upper(s))

#Exercise 20: Text Alignment

thickness = int(input())
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Exercise 21: Text Wrap

def wrap(string, max_width):
    for i in range(0,len(string)):
        if max_width*i<len(string):
            print(string[max_width*i:max_width*(i+1)])
    return ""

#Exercise 22: String Formatting
def print_formatted(number):
    n=len(bin(number)[2:])
    for i in range(1,number+1):
        print(str(i).rjust(n),oct(i)[2:].rjust(n),hex(i)[2:].upper().rjust(n),bin(i)[2:].rjust(n))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

#Exercise 23: Capitalize!

import os
def solve(s):
    s_new = ""
    capitalize_verify = True
    for element in s:
        if element == " ":
            s_new += element
            capitalize_verify = True
        elif capitalize_verify:
            s_new += element.upper()
            capitalize_verify = False
        else:
            s_new += element
    return s_new

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

#Exercise 24: Merge the Tools!
def merge_the_tools(string, k):
    n = len(string)
    for i in range(0, n, k):
        sub_s = []
        for j in range(i, i+k):
            if string[j] not in sub_s:
                sub_s.append(string[j])
        print(''.join(sub_s))

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Sets """

#Exercise 25: Introduction to Sets

def average(array):
    return sum(set(array))/len(set(array))

#Exercise 26: Set .add()

n=int(input())
void_list=[]
for i in range(n):
    void_list.append(input())
print(len(set(void_list)))

#Exercise 27: Set.union() Operation

if 0<int(input())<1000:
    student1=set(map(int, input().split()))
if 0<int(input())<1000:
    student2=set(map(int, input().split()))
total_student=student1.union(student2)
print(len(total_student))

#Exercise 28: Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
for _ in range(int(input())):
    i=input().split()
    if "pop" in i:
        s.pop()
    if "remove" in i:
        s.remove(int(i[1]))
    if "discard" in i:
        s.discard(int(i[1]))
print(sum(s))

#Exercise 29: Symmetric Difference

M=input()
a=set(map(int,input().split()))
N=input()
b=set(map(int,input().split()))
c=a.difference(b).union(b.difference(a))
print(*sorted(c),sep="\n")


#Exercise 30: Set .difference() Operation

if 0<int(input())<1000:
    student1=set(map(int, input().split()))
if 0<int(input())<1000:
    student2=set(map(int, input().split()))
total_student=student1.difference(student2)
print(len(total_student))

#Exercise 31: Set .symmetric_difference() Operation

if 0<int(input())<1000:
    student1=set(map(int, input().split()))
if 0<int(input())<1000:
    student2=set(map(int, input().split()))
total_student=student1.symmetric_difference(student2)
print(len(total_student))

#Exercise 32: Set Mutations

n=int(input())
s=set(map(int,input().split()))
for _ in range(int(input())):
    command=input().split()
    if "intersection_update" in command:
        s.intersection_update(set(map(int,input().split())))
    if "update" in command:
        s.update(set(map(int,input().split())))
    if "symmetric_difference_update" in command:
        s.symmetric_difference_update(set(map(int,input().split())))
    if "difference_update" in command:
        s.difference_update(set(map(int,input().split())))
print(sum(s))

#Exercise 33: The Captain's Room

from collections import Counter
k = int(input())
l = list(map(int, input().split()))
count = Counter(l)
for number in l:
    if count[number] == 1:
        print(number)
        break

#Exercise 34: Check Subset

number_of_test=int(input())
if 0<number_of_test<21:
    for i in range(number_of_test):
        element_of_a=input()
        set_a=set(map(int,input().split()))
        element_of_b=input()
        set_b=set(map(int,input().split()))
        result=all([element in set_b for element in set_a])
        print(result)

#Exercise 35: Check Strict Superset

A=set(map(int,input().split()))
n,l=int(input()),[]
for _ in range(n):
    s=set(map(int,input().split()))
    inter=A.intersection(s)
    l.append(inter==s)
print(all(l))

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Collections """

#Exercise 36: collections.Counter()

from collections import Counter
l_items=[]
number_of_shoes=int(input())
shoe_size= list(map(int,input().split()))
count=Counter(shoe_size)
customers=int(input())
d=dict(count.items())
for _ in range(customers):
    l_items.append(tuple(map(int,input().split())))
tot=0
for item in l_items:
    size, price = item
    if size in d and d[size]>0:
        tot+=price
        d[size] -= 1
print(tot)

#Exercise 37: DefaultDict Tutorial

from collections import defaultdict
A=defaultdict(list)
B=[]
len_A,len_B=input().split()
for i in range(1,int(len_A)+1):
    A[input()].append(i)
for  _ in range(int(len_B)):
    B.append(input())
for element in B:
    if element in A.keys():
        print(*A[element])
    else:
        print(-1)

#Exercise 38: Collections.namedtuple()

from collections import namedtuple
tot,number_of_students,Student=0,int(input()),namedtuple("Student",input().split())
for _ in range(number_of_students):
    stud=Student(*input().split())
    tot+=int(stud.MARKS)
print(tot/number_of_students)

#Exercise 39: Collections.OrderedDict()

from collections import OrderedDict
d,n= OrderedDict(), int(input())
for _ in range(n):
    key,value=input().rsplit(maxsplit=1)
    if key in d:
        d[key]+=int(value)
    else:
        d[key]=int(value)
for key in d:
    print(key,d[key])

#Exercise 40: Word Order

from collections import OrderedDict
number_of_words,d=int(input()),OrderedDict()
for _ in range(number_of_words):
    word=input()
    if word in d:
        d[word]+=1
    else:
        d[word]=1
print(len(d.keys()))
print(*d.values())

#Exercise 41: Collections.deque()

from collections import deque
number_of_operations,d=int(input()),deque()
for _ in range(number_of_operations):
    operation=input().split()
    if operation[0]=="append":
        d.append(operation[1])
    elif operation[0]=="appendleft":
        d.appendleft(operation[1])
    elif operation[0]=="pop":
        d.pop()
    elif operation[0]=="popleft":
        d.popleft()
print(*d)

#Exercise 42: Company Logo

from collections import OrderedDict
if __name__ == '__main__':
    s = input()
d=OrderedDict()
for element in s:
    if element in d:
        d[element]+=1
    else:
        d[element]=1
d=sorted(d.items(),key=lambda item: (-item[1],item[0]))
if len(d)>=3:
    for item in d[:3]:
        print(*item)

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Date and Time """

#Exercise 43: Calendar Module

import calendar
month, day, year = map(int, input().split())
day_of_week = calendar.weekday(year, month, day)
names_of_day=["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
print(names_of_day[day_of_week])

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Errors and Exceptions """

#Exercise 44: Exceptions

n_test=int(input())
for _ in range(n_test):
    a,b=input().split()
    try:
        a=int(a)
    except ValueError:
        print(f"Error Code: invalid literal for int() with base 10: '{a}'")
        continue
    try:
        b=int(b)
    except ValueError:
        print(f"Error Code: invalid literal for int() with base 10: '{b}'")
        continue
    try:
        print(a//b)
    except ZeroDivisionError:
        print("Error Code: integer division or modulo by zero")

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Built-ins """

#Exercise 45: Zipped!

N,X=input().split()
l=[]
for student in range(int(X)):
    l.append(list(map(float,input().split())))
result = [sum(i)/len(i) for i in list(zip(*l))]
print(*result,sep="\n")

#Exercise 46: Athlete Sort

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
arr=sorted(arr,key=lambda x: x[k])
for l in arr:
    print(*l)

#Exercise 47: ginortS!
def order_string_rule(string):
    if string.islower():
        return (0,string)
    elif string.isupper():
        return (1,string)
    elif string.isdigit():
        digit = int(string)
        if digit % 2 == 1:
            return (2,string)
        else:
            return (3,string)

my_string=input()
new_string=sorted(my_string,key=order_string_rule)
print("".join(new_string))

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Python Functionals """

#Exercise 48: Map and Lambda Function

cube = lambda x: x ** 3
def recursive_fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)
def fibonacci(n):
    l = []
    for i in range(n):
        l.append(recursive_fibonacci(i))
    return l
if __name__ == '__main__':
    n = int(input())

    print(list(map(cube, fibonacci(n))))

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Regex and Parsing challenges """

#Exercise 49: Detect Floating Point Number

n=int(input())
for _ in range(n):
    try:
        i=input()
        float(i)
        if "." in i:
            print(True)
        else:
            print(False)
    except ValueError:
        print(False)

#Exercise 50: Re.split()

regex_pattern = r'[,.]'
import re
print("\n".join(re.split(regex_pattern, input())))

#Exercise 51: Group(), Groups() & Groupdict()

import re
s=input()
pattern=re.search(r'([a-zA-Z0-9])\1',s)
if pattern:
    print(pattern.group(1))
else:
    print(-1)

#Exercise 52: Re.start() & Re.end()

import re
s,k=input(),input()
pattern = f"(?=({k}))"
matches = re.finditer(pattern, s)
found=False
for match in matches:
    found=True
    print((match.start(), match.start() + len(k) - 1))
if not found:
    print((-1,-1))

#Exercise 53: Validating Roman Numerals

regex_pattern = r"^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
import re
print(str(bool(re.match(regex_pattern, input()))))

#Exercise 54: Validating phone numbers

import re
n=int(input())
pattern=r"^[789]\d{9}$"
for _ in range(n):
    if re.match(pattern,input()):
        print("YES")
    else:
        print("NO")

#Exercise 55: Validating and Parsing Email Addresses

import email.utils
import re
n=int(input())
for _ in range(n):
    name,m=email.utils.parseaddr(input())
    pattern=r"^[a-zA-Z]+[a-zA-Z0-9._%+-]+@[a-zA-Z]+\.[a-zA-Z]{0,3}$"
    if re.match(pattern,m):
        print(f"{name} <{m}>")

#Exercise 56: Matrix Script

import re
first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
s=""
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
for i in range(m):
    for j in range(n):
        s+=matrix[j][i]
new_s=re.sub(r'(?<=[a-zA-Z0-9])[^a-zA-Z0-9]+(?=[a-zA-Z0-9])', ' ', s)
transformed_string = re.sub(r'\s+', ' ', new_s)
print(new_s)

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: XML """

#Exercise 56: XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree
def get_attr_number(node):
    count = len(node.attrib)
    for child in node:
        count += get_attr_number(child)
    return count

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Closures and Decorations """

#Exercise 57: Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        l= ["+91 " + num[-10:-5] + " " + num[-5:] for num in l]
        return f(l)
    return fun
@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

#Exercise 58: Name Directory

import operator
def person_lister(f):
    def inner(people):
        people = sorted(people, key=lambda x: int(x[2]))
        return [f(person) for person in people]
    return inner
@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

# ----------------------------------------------------------------------------------------------------------------------

""" SUBDOMAINS: Numpy """

#Exercise 59: Arrays

import numpy
def arrays(arr):
    arr=list(map(float,arr))
    arr.reverse()
    return numpy.array(arr)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#Exercise 60: Shape and Reshape

import numpy
arr=list(map(int,input().split()))
array=numpy.array(arr)
print(array.reshape(3,3))

#Exercise 61: Transpose and Flatten

import numpy
n,m=input().split()
l=[]
for _ in range(int(n)):
    l.append(list(map(int,input().split())))
array=numpy.array(l)
print(numpy.transpose(array))
print(array.flatten())

#Exercise 62: Concatenate

import numpy
n,m,p=input().split()
l=[]
for _ in range(int(n)+int(m)):
    l.append(list(map(int,input().split())))
ar=numpy.array([l[0]])
for i in range(1,len(l)):
    ar=numpy.concatenate((ar,numpy.array([l[i]])),axis=0)
print(ar)

#Exercise 63: Zeros and Ones

import numpy
i=tuple(map(int,input().split()))
print(numpy.zeros(i,dtype = int))
print(numpy.ones(i,dtype = int))

#Exercise 64: Eye and Identity

import numpy
row,col=input().split()
row,col=int(col),int(row)
numpy.set_printoptions(legacy="1.13")
print(numpy.eye(col,row))

#Exercise 65: Array Mathematics

import numpy
n,m=input().split()
n=int(n)
A=numpy.array([list(map(int,input().split()))])
for _ in range(1,n):
    A_2=numpy.array([list(map(int,input().split()))])
    A=numpy.concatenate((A,A_2),axis=0)
B=numpy.array([list(map(int,input().split()))])
for _ in range(1,n):
    B_2=numpy.array([list(map(int,input().split()))])
    B=numpy.concatenate((B,B_2),axis=0)
print(numpy.add(A,B))
print(numpy.subtract(A,B))
print(numpy.multiply(A,B))
print(numpy.floor_divide(A,B))
print(numpy.mod(A,B))
print(numpy.power(A,B))

#Exercise 66: Floor, Ceil and Rint

import numpy
ar=numpy.array(list(map(float,input().split())))
numpy.set_printoptions(legacy="1.13")
print(numpy.floor(ar))
print(numpy.ceil(ar))
print(numpy.rint(ar))

#Exercise 67: Sum and Prod

import numpy
n,m=input().split()
l=[]
for _ in range(int(n)):
    l.append(list(map(int,input().split())))
my_array=numpy.array(l)
array_sum=numpy.sum(my_array,axis=0)
print(numpy.prod(array_sum))

#Exercise 68: Min and Max

import numpy
n,m=input().split()
l=[]
for _ in range(int(n)):
    l.append(list(map(int,input().split())))
my_array=numpy.array(l)
min_array=numpy.min(my_array,axis=1)
print(max(min_array))

#Exercise 69: Mean, Var, and Std

import numpy
n,m=input().split()
n,l=int(n),[]
for _ in range(n):
    l.append(list(map(int,input().split())))
my_array=numpy.array(l)
print(numpy.mean(my_array,axis=1))
print(numpy.var(my_array,axis=0))
print(round(numpy.std(my_array),11))

#Exercise 70: Dot and Cross

import numpy
n=int(input())
A=numpy.array([list(map(int,input().split()))])
for _ in range(1,n):
    A_2=numpy.array([list(map(int,input().split()))])
    A=numpy.concatenate((A,A_2),axis=0)
B=numpy.array([list(map(int,input().split()))])
for _ in range(1,n):
    B_2=numpy.array([list(map(int,input().split()))])
    B=numpy.concatenate((B,B_2),axis=0)
print(numpy.dot(A,B))

#Exercise 71: Inner and Outer

import numpy
A=numpy.array(list(map(int,input().split())))
B=numpy.array(list(map(int,input().split())))
print(numpy.inner(A,B))
print(numpy.outer(A,B))

#Exercise 72: Inner and Outer

import numpy
my_array=numpy.array(list(map(float,input().split())))
print(numpy.polyval(my_array,int(input())))

#Exercise 73: Linear Algebra

import numpy
n=int(input())
l=[]
for _ in range(n):
    l.append(list(map(float,input().split())))
det=numpy.linalg.det(l)
print(round(det,2))

# ----------------------------------------------------------------------------------------------------------------------

"""           Algorithmic Methods of Data Mining
                        HOMEWORK 1
Problem 2: The implementation of some algorithms in Python
"""

#Exercise 1: Birthday Cake Candles
def birthdayCakeCandles(candles):
    maximum_height = max(candles)
    occurence_maximum_height = candles.count(maximum_height)
    return occurence_maximum_height

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Exercise 2: Number Line Jumps

def kangaroo(x1, v1, x2, v2):
    if (x1<x2 and v1<v2) or (x2<x1 and v2<v1):
        return "NO"
    elif (v1-v2)!=0:
        if (x2 - x1) % (v1 - v2) == 0:
            return "YES"
    return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Exercise 3: Viral Advertising
def viralAdvertising(n):
    people_reached = 5
    tot_likes=0
    for day in range(n):
        likes=people_reached//2
        tot_likes+=likes
        people_reached=likes*3
    return tot_likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Exercise 4: Recursive Digit Sum
def superDigit(n, k):
    n=str(sum(int(digit) for digit in n)*k)
    if len(n)==1:
        return n
    else:
        n =sum(int(digit) for digit in n)
        return superDigit(str(n),1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Exercise 5: Insertion Sort - Part 1

def insertionSort1(n, arr):
    max_value=arr[n-1]
    j=1
    for i in reversed(range(len(arr)-1)):
        if arr[i]>max_value:
            arr[n-j]=arr[i]
            j+=1
            print(*arr)
        else:
            arr[i+1]=max_value
            print(*arr)
            break
    if n-j == 0:
        arr[0] = max_value
        print(*arr)
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Exercise 6: Insertion Sort - Part 2

def insertionSort2(n, arr):
    lower_value = arr[0]
    index = 0
    for i in range(1, n):
        swap = False
        for j in reversed(range(i)):
            if arr[i] < arr[j]:
                lower_value = arr[i]
                index = j
                swap = True
        if swap:
            arr.remove(lower_value)
            arr.insert(index, lower_value)
        print(*arr)


if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
