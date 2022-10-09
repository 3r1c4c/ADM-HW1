
#INTRODUCTION

#Python if-else no
import random
if __name__ == '__main__':
    n=int(input())
    if n %2 ==1 :
        print("Weird")
    elif n%2 ==0 & n>=5 & n<=2:
        print("Not Weird")
    elif n%2 ==0 & n>=6 & n<=4:
        print("Weird")
    elif n%2 ==0 & n>20:
        print("Not Weird")

#say "Hello world! with python
print("Hello, World!")

#Arithmetic operation
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#Python:Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#Loops
if __name__ == '__main__':
    n = int(input())
    i=0
    while i<n:
        print(i**2)
        i=i+1

#Write a function
def is_leap(year):
    leap = False
    if year % 400 == 0:
        leap = True
    elif (year % 4 == 0) & (year % 100 != 0):
        leap = True

    return leap


year = int(input())
print(is_leap(year))

#Print function
if __name__ == '__main__':
    n = int(input())
    print(*range(1,n+1), sep='')

#BASIC DATA TYPES

#list comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
l=[]

for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if i+j+k !=n:
                l.append([i,j,k])
print(l)

#Find th runner-up score!
if __name__ == '__main__':
    n = int(input())
    A=map(int,input().split())
    l=[]
    for i in A:
        if i not in l:
          l.append(i)
        l.sort()
    print(l[-2])

#nasted list
if __name__ == '__main__':
    l=[]
    s=[]
    n=int(input())
    for i in range(n):
        name = input()
        score = float(input())
        l.append([name,score])
        s.append(score)
    s.sort()
    l.sort()
    for i in range(len(s)):
        if s[i]>s[0]:
            smin=s[i]
            break
    for i in range(len(l)):
        if l[i][1]==smin:
            print(l[i][0])

#finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    marks=0
    for i in student_marks[query_name]:
        marks=marks+i
    avg=marks/3
    print("%.2f"%avg)

#Lists
if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(N):
        A=list(input().split())
        if A[0] == 'insert':
            l.insert(int(A[1]), int(A[2]))
        if A[0] == 'print':
            print(l)
        if A[0] == 'remove':
            l.remove(int(A[1]))
        if A[0] == 'append':
            l.append(int(A[1]))
        if A[0] == 'pop':
            l.pop()
        if A[0] == 'reverse':
            l.reverse()
        if A[0] == 'sort':
            l.sort()

#tuples
import builtins
n = int(input())
t = tuple(map(int, input().split()))
print(builtins.hash(t))

#STRINGS

#Mutations
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

#Swap case
def swap_case(s):
    g=s.swapcase()
    return(g)

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

#string validators
if __name__ == '__main__':
    s = input()
    print(any(map(str.isalnum, s)))
    print(any(map(str.isalpha, s)))
    print(any(map(str.isdigit, s)))
    print(any(map(str.islower, s)))
    print(any(map(str.isupper, s)))

#strings split and joint
def split_and_join(line):
    line=line.split()
    line="-".join(line)
    return line

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#what's your name?
def print_full_name(a, b):
    print("Hello " + a, b + "! You just delved into python.")


if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

#find a string
def count_substring(string, sub_string):
    c = 0
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i + len(sub_string)] == sub_string:
            c = c + 1
    return c

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)

#Text wrap
import textwrap

def wrap(string, max_width):
    s_wrap_list = textwrap.wrap(string, max_width)
    r='\n'.join(s_wrap_list)
    return r

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

#Capitalize
def solve(s):
    s=s.title()
    return s

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


#SETS

#introduction to sets
def average(array):
    s=set(array)
    c=0
    for i in s:
        c=c+i
    avarage=c/len(s)
    return avarage

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

#no idea!
n,m=list(map(int, input().split()))
N=list(map(int, input().split()))
A=set(map(int, input().split()))
B=set(map(int, input().split()))

happiness = 0
for i in N:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1

print (happiness)

#symmetric difference
m=int(input())
one=set(map(int, input().split()))
n=int(input())
two=set(map(int, input().split()))

diff1=one.difference(two)
diff2=two.difference(one)

ins=diff1.union(diff2)
sor=sorted(list(ins))

for i in sor:
    print(i)

#set.add()
N = int(input())

s = set()
for i in range(N):
    s.add(input())

print(len(s))

#set.union() operation
n=int(input())
n1=set(map(int, input().split()))
b=int(input())
b1=set(map(int, input().split()))

print(len(n1.union(b1)))

#set.intersection() operation
n=int(input())
s1=set(map(int, input().split()))
b=int(input())
b1=set(map(int, input().split()))

print(len(s1.intersection(b1)))

#set.difference() operation
n=int(input())
n1=set(map(int, input().split()))
b=int(input())
b1=set(map(int, input().split()))

print(len(n1.difference(b1)))

#set.symmetric_difference() operatio
n=int(input())
n1=set(map(int, input().split()))
b=int(input())
b1=set(map(int, input().split()))

print(len(n1.symmetric_difference(b1)))

#set mutation
n = int(input())
n1 = set(map(int, input().split()))
b = int(input())

for i in range(b):
    C, N = input().split()
    set1 = set(map(int, input().split()))
    getattr(n1, C)(set1)
print(sum(n1))

#COLLECTIONS

#collections.Counter()
from collections import Counter

X = int(input())
x = list(input().split())
costumers = int(input())
c = Counter(x)
somma = 0
for i in range(costumers):
    size, price = input().split()
    if c[size] > 0:
        c[size] -= 1
        somma += int(price)
print(somma)

#default dict tutorial
from collections import defaultdict
d = defaultdict(list)

n, m = map(int,input().split())
for i in range(1, n+1):
    d[input()].append(str(i))


for i in range(m):
    b = input()
    if b in d: print(' '.join(d[b]))
    else: print(-1)

#word-order
from collections import OrderedDict
N=int(input())

d=OrderedDict()

for i in range(N):
    word=input()
    if word  in d:
        d[word]+=1
    else:
        d[word]=1

print(len(d))

for k,v in d.items():
    print(v,end = " ");

#collections.deque()
from collections import deque
from six.moves import input as raw_input
d = deque()
N=int(raw_input())
for i in range(N):
    A=list(raw_input().split())
    if A[0]=='append':
        d.append(int(A[1]))
    elif A[0]=='appendleft':
        d.appendleft(int(A[1]))
    elif A[0]=='pop':
        d.pop()
    elif A[0]=='popleft':
        d.popleft()
for i in d:
    print(i, end=' ')

#DATE AND TIME

#calendar module
import calendar
MM, DD, YYYY = map(int, input().split())
print (calendar.day_name[calendar.weekday(YYYY,MM,DD)].upper())

#EXCEPTIONS
n=int(input())
for i in range(n):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

#BUILT-INS
#zipped
N, M = map(int, input().split())
c = []
for i in range(int(M)):
    s = list(map(float, input().split()))
    c.append(s)

for i in zip(*c):
    print(sum(i) / M)

#athlete sort


#ginortS
def sort_s(s):
    lo = []
    up = []
    o_d = []
    e_d = []
    for i in s:
        if i.islower():
            lo.append(i)
        elif i.isupper():
            up.append(i)
        elif i.isdigit():
            if int(i) % 2 == 0:
                e_d.append(i)
            else:
                o_d.append(i)
    return (''.join(sorted(lo) + sorted(up) + sorted(o_d) + sorted(e_d)))


if __name__ == '__main__':
    S = input()
    print(sort_s(S))

#PYTHON FUNCTIONALS

#map and lambda function
cube = lambda x: pow(x,3)

def fibonacci(n):
    l = [0,1]
    for i in range(2,n):
        l.append(l[i-2] + l[i-1])
    return(l[0:n])



if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))




#NUMPHY

#array
import numpy

def arrays(arr):
    l=[]
    for i in arr:
        if i not in l:
            l.append(i)
    l.reverse()
    return numpy.array(l,float)
arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#shape and reshape
import numpy

n=list(map(int, input().split()))
my_array=numpy.array(n)
print(numpy.reshape(my_array, (3,3)))

#traspose and flatten
import numpy
n=list(map(int, input().split()))[0]
l=[]
for i in range(n):
    i=list(map(int, input().split()))
    l.append(i)
my_array=numpy.array(l)
print(numpy.transpose(my_array))
print(my_array.flatten())

#concatenete
import numpy

l = list(map(int, input().split()))
a = []
for i in range(l[0]):
    c = list(map(int, input().split()))
    a.append(c)

b = []
for j in range(l[1]):
    d = list(map(int, input().split()))
    b.append(d)
array1 = numpy.array(a)
array2 = numpy.array(b)
print(numpy.concatenate((array1, array2), axis=0))

#zeros and ones
import numpy
n=tuple(list(map(int, input().split())))


print(numpy.zeros(n, dtype=numpy.int))

print(numpy.ones(n, dtype=numpy.int))

#Eye and Identity
import numpy as np
N,M=list(map(int, input().split()))
np.set_printoptions(legacy='1.13')
print(np.eye(N,M))

#Array Mathematics
import numpy as np
N,M=map(int, input().split())

A = np.array([input().split() for i in range(N)], int)
B = np.array([input().split() for i in range(N)], int)

functions=[np.add, np.subtract, np.multiply, np.floor_divide, np.mod, np.power]
for f in functions:
    print(f(A,B))

#Floor, ceil, rint
import numpy as np

np.set_printoptions(legacy='1.13')
A=np.array(input().split(),float)
print(np.floor(A))
print(np.ceil(A))
print(np.rint(A))

#sum and prod
import numpy as np
N,M=map(int, input().split())
A=[]
for i in range(N):
    A.append(list(map(int, input().split())))
arr=np.array(A)

print(np.prod(np.sum(arr, axis=0)))

#min and max
import numpy as np
N,M=map(int, input().split())
A=[]
for i in range(N):
    A.append(list(map(int, input().split())))
arr=np.array(A)
print(np.max(np.min(arr, axis=1)))

#mean var, std
import numpy as np
np.set_printoptions(legacy='1.13')
N,M=map(int, input().split())
A=[]
for i in range(N):
    A.append(list(map(int, input().split())))
arr=np.array(A)
print(np.mean(arr,axis=1))
print( np.var(arr,axis=0))
print(round(np.std(arr),11) )

#dot and cross
import numpy as np
n=int(input())
a=[]
b=[]
for i in range(n):
    a.append(list(map(int, input().split())))
for i in range(n):
    b.append(list(map(int, input().split())))
arr1=np.array(a)
arr2=np.array(b)

print(np.dot(arr1,arr2))

#inner and outer
import numpy as np
A=np.array(list(map(int, input().split())))
B=np.array(list(map(int, input().split())))

print(np.inner(A,B))
print(np.outer(A,B))

#Polynomials
import numpy as np
C=np.array(input().split(),float)
x=int(input())

print(np.polyval(C,int(x)))

#linear algebra
import numpy as np

n=int(input())
A=[]
for i in range(n):
    A.append(list(map(float, input().split())))
arr=np.array(A)

print(round(np.linalg.det(arr),2))

#Birthday cake candles
def birthdayCakeCandles(candles):
    a = []

    m = max(candles)
    n = 0
    for j in candles:

        if m == j:
            n += 1
    return (n)

#Number line Jumps
def kangaroo(x1, v1, x2, v2):
    result = 'NO'
    if (x1 < x2) == (v1 > v2):
        s = abs(x1 - x2)
        v = abs(v1 - v2)
        if (s % v) == 0:
            result = 'YES'

    return result


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

#viral adverising

import math
import os
import random
import re
import sys


def viralAdvertising(n):
    shared=5
    cumulative=0
    for i in range(1,n+1):
        liked=shared//2
        cumulative=cumulative+liked
        shared=liked*3
    return cumulative

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

