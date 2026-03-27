"""
Anad Coding Dataset
====================
Programming and coding training data from public sources.

Sources:
  1. Stack Exchange (CC-BY-SA) — real Q&A from developers
  2. Rosetta Code — same task in every language
  3. Handcrafted coding examples — Python, JS, algorithms
  4. Documentation seeds — how to explain code clearly
  5. Indian developer context — local frameworks, problems

This teaches Anad to:
  - Understand code in multiple languages
  - Explain what code does
  - Debug common errors
  - Write clean functions
  - Think algorithmically

Author: Anad Community
License: Public Domain
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from typing import Iterator, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.data_collector import DataRecord, AnadDataCollector


# ══════════════════════════════════════════════════════════════════
# HANDCRAFTED CODING EXAMPLES
# Rich, diverse, well-explained code
# ══════════════════════════════════════════════════════════════════

CODING_SEED_TEXTS = [

    # ── Python Basics ────────────────────────────────────────────
    DataRecord(
        text="""# Python basics — variables and types
name = "Anad"           # string
age = 25                # integer
pi = 3.14               # float
is_public = True        # boolean

# Print values
print(f"Name: {name}, Age: {age}")

# Type checking
print(type(name))       # <class 'str'>
print(type(age))        # <class 'int'>""",
        source="coding_seed", language="en",
        license="public_domain", title="Python variables"),

    DataRecord(
        text="""# Python functions
def greet(name, language="en"):
    \"\"\"Greet a person in their language.\"\"\"
    greetings = {
        "en": f"Hello, {name}!",
        "hi": f"नमस्ते, {name}!",
        "gu": f"નમસ્તે, {name}!",
        "ta": f"வணக்கம், {name}!",
    }
    return greetings.get(language, f"Hello, {name}!")

print(greet("World"))           # Hello, World!
print(greet("Bharat", "hi"))    # नमस्ते, Bharat!
print(greet("Anad", "gu"))      # નમસ્તે, Anad!""",
        source="coding_seed", language="en",
        license="public_domain", title="Python functions"),

    DataRecord(
        text="""# Python lists and loops
fruits = ["apple", "mango", "banana", "guava"]

# Loop through list
for fruit in fruits:
    print(fruit)

# List comprehension — shorter and faster
upper_fruits = [f.upper() for f in fruits]
print(upper_fruits)  # ['APPLE', 'MANGO', 'BANANA', 'GUAVA']

# Filter with condition
long_names = [f for f in fruits if len(f) > 5]
print(long_names)    # ['banana', 'guava']""",
        source="coding_seed", language="en",
        license="public_domain", title="Python lists"),

    DataRecord(
        text="""# Python dictionaries — key-value storage
person = {
    "name": "Ravi",
    "age": 30,
    "city": "Ahmedabad",
    "skills": ["Python", "AI", "Data Science"]
}

# Access values
print(person["name"])           # Ravi
print(person.get("age", 0))     # 30

# Add/update
person["email"] = "ravi@example.com"

# Loop through
for key, value in person.items():
    print(f"{key}: {value}")""",
        source="coding_seed", language="en",
        license="public_domain", title="Python dictionaries"),

    DataRecord(
        text="""# Python classes — object oriented programming
class Node:
    \"\"\"Represents an Anad network node.\"\"\"

    def __init__(self, node_id, tier="desktop"):
        self.node_id = node_id
        self.tier = tier
        self.credits = 50       # starter credits
        self.is_active = True

    def earn(self, amount):
        \"\"\"Earn credits for contributing compute.\"\"\"
        self.credits += amount
        return self.credits

    def spend(self, amount):
        \"\"\"Spend credits to use AI.\"\"\"
        if self.credits >= amount:
            self.credits -= amount
            return True
        return False

    def __repr__(self):
        return f"Node({self.node_id}, credits={self.credits})"

# Create and use
node = Node("anad1_abc123", tier="laptop")
node.earn(10)
print(node)             # Node(anad1_abc123, credits=60)
print(node.spend(5))    # True""",
        source="coding_seed", language="en",
        license="public_domain", title="Python classes"),

    DataRecord(
        text="""# Python error handling
def divide(a, b):
    \"\"\"Safe division with error handling.\"\"\"
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    except TypeError as e:
        print(f"Error: Wrong type — {e}")
        return None
    finally:
        print("Division attempted")

print(divide(10, 2))    # 5.0
print(divide(10, 0))    # Error: Cannot divide by zero
print(divide("a", 2))   # Error: Wrong type""",
        source="coding_seed", language="en",
        license="public_domain", title="Python error handling"),

    DataRecord(
        text="""# Python file handling
import json

# Write JSON file
data = {"name": "Anad", "version": "0.1.0", "public": True}
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

# Read JSON file
with open("config.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)

print(loaded["name"])   # Anad
print(loaded["version"]) # 0.1.0""",
        source="coding_seed", language="en",
        license="public_domain", title="Python file handling"),

    # ── Algorithms ───────────────────────────────────────────────
    DataRecord(
        text="""# Binary search — O(log n)
def binary_search(arr, target):
    \"\"\"Find target in sorted array. Returns index or -1.\"\"\"
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

numbers = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search(numbers, 7))    # 3
print(binary_search(numbers, 4))    # -1""",
        source="coding_seed", language="en",
        license="public_domain", title="Binary search"),

    DataRecord(
        text="""# Bubble sort — simple sorting algorithm
def bubble_sort(arr):
    \"\"\"Sort array in place using bubble sort.\"\"\"
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

numbers = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(numbers))  # [11, 12, 22, 25, 34, 64, 90]

# Python's built-in sort is much faster
numbers2 = [64, 34, 25, 12, 22, 11, 90]
print(sorted(numbers2))      # [11, 12, 22, 25, 34, 64, 90]""",
        source="coding_seed", language="en",
        license="public_domain", title="Sorting algorithms"),

    DataRecord(
        text="""# Fibonacci sequence — recursion and iteration
# Recursive (simple but slow for large n)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

# Iterative (fast)
def fib_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Dynamic programming (memoization)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

print([fib_iterative(i) for i in range(10)])
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]""",
        source="coding_seed", language="en",
        license="public_domain", title="Fibonacci"),

    DataRecord(
        text="""# Linked list implementation
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        if not self.head:
            self.head = ListNode(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = ListNode(val)

    def display(self):
        vals = []
        curr = self.head
        while curr:
            vals.append(curr.val)
            curr = curr.next
        return " -> ".join(map(str, vals))

ll = LinkedList()
for v in [1, 2, 3, 4, 5]:
    ll.append(v)
print(ll.display())  # 1 -> 2 -> 3 -> 4 -> 5""",
        source="coding_seed", language="en",
        license="public_domain", title="Linked list"),

    # ── Web & APIs ───────────────────────────────────────────────
    DataRecord(
        text="""# Python HTTP request — calling an API
import urllib.request
import json

def fetch_data(url):
    \"\"\"Fetch JSON data from a URL.\"\"\"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Anad/0.1"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example: fetch Wikipedia summary
url = "https://en.wikipedia.org/api/rest_v1/page/summary/India"
data = fetch_data(url)
if data:
    print(data["extract"][:200])""",
        source="coding_seed", language="en",
        license="public_domain", title="Python HTTP"),

    DataRecord(
        text="""# Simple Flask web server (Python)
# Install: pip install flask
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Anad API", "status": "running"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    # Process question with Anad model here
    return jsonify({
        "question": question,
        "answer": "Processing...",
        "credits_spent": 1
    })

if __name__ == "__main__":
    app.run(debug=True, port=8765)""",
        source="coding_seed", language="en",
        license="public_domain", title="Flask web server"),

    # ── Data Science ─────────────────────────────────────────────
    DataRecord(
        text="""# NumPy basics — numerical computing
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Operations
print(arr * 2)          # [2 4 6 8 10]
print(arr.mean())       # 3.0
print(arr.std())        # 1.414...

# Matrix operations
print(matrix.shape)     # (3, 3)
print(matrix.T)         # transpose
print(np.dot(arr, arr)) # dot product = 55

# Useful for AI: create random weights
weights = np.random.randn(10, 5) * 0.01
print(weights.shape)    # (10, 5)""",
        source="coding_seed", language="en",
        license="public_domain", title="NumPy basics"),

    DataRecord(
        text="""# PyTorch basics — neural network building block
import torch
import torch.nn as nn

# Tensors
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

# Forward pass
y = (x * w).sum()
print(y)            # tensor(1.4)

# Backward pass — compute gradients
y.backward()
print(w.grad)       # tensor([1., 2., 3.])

# Simple linear layer
linear = nn.Linear(3, 1)
out = linear(x)
print(out)          # tensor([...], grad_fn=<AddmmBackward>)""",
        source="coding_seed", language="en",
        license="public_domain", title="PyTorch basics"),

    # ── Debugging ────────────────────────────────────────────────
    DataRecord(
        text="""# Common Python bugs and fixes

# Bug 1: Mutable default argument
# WRONG
def add_item(item, lst=[]):
    lst.append(item)
    return lst

# RIGHT
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

# Bug 2: Integer division
# WRONG in Python 2, fine in Python 3
result = 7 / 2   # 3.5 in Python 3
result = 7 // 2  # 3 (floor division)

# Bug 3: String vs bytes
text = "नमस्ते"
encoded = text.encode("utf-8")   # bytes
decoded = encoded.decode("utf-8") # string back

# Bug 4: Modifying list while iterating
numbers = [1, 2, 3, 4, 5]
# WRONG: for n in numbers: if n%2==0: numbers.remove(n)
# RIGHT:
numbers = [n for n in numbers if n % 2 != 0]
print(numbers)  # [1, 3, 5]""",
        source="coding_seed", language="en",
        license="public_domain", title="Python debugging"),

    # ── Git & Version Control ────────────────────────────────────
    DataRecord(
        text="""# Git commands every developer needs

# Setup
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# Start a project
git init                        # new repo
git clone <url>                 # copy existing repo

# Daily workflow
git status                      # see what changed
git add .                       # stage all changes
git add file.py                 # stage one file
git commit -m "describe change" # save snapshot
git push                        # upload to GitHub

# Branching
git branch feature-name         # create branch
git checkout feature-name       # switch to branch
git merge feature-name          # merge back to main
git branch -d feature-name      # delete branch

# Undo mistakes
git restore file.py             # undo file changes
git reset HEAD~1                # undo last commit""",
        source="coding_seed", language="en",
        license="public_domain", title="Git commands"),

    # ── SQL ──────────────────────────────────────────────────────
    DataRecord(
        text="""-- SQL basics — database queries

-- Create table
CREATE TABLE users (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    email   TEXT UNIQUE,
    credits INTEGER DEFAULT 50,
    joined  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert data
INSERT INTO users (name, email) VALUES ('Ravi', 'ravi@example.com');
INSERT INTO users (name, email, credits) VALUES ('Priya', 'priya@example.com', 100);

-- Query data
SELECT * FROM users;
SELECT name, credits FROM users WHERE credits > 50;
SELECT name FROM users ORDER BY credits DESC LIMIT 10;

-- Update
UPDATE users SET credits = credits + 10 WHERE name = 'Ravi';

-- Delete
DELETE FROM users WHERE email IS NULL;

-- Join two tables
SELECT u.name, n.node_id
FROM users u
JOIN nodes n ON u.id = n.user_id;""",
        source="coding_seed", language="en",
        license="public_domain", title="SQL basics"),

    # ── Explaining Code ──────────────────────────────────────────
    DataRecord(
        text="""Question: What does this Python code do?

def mystery(lst):
    if not lst:
        return 0
    return lst[0] + mystery(lst[1:])

Answer: This function calculates the sum of all numbers in a list using recursion.

How it works:
1. Base case: if the list is empty, return 0
2. Recursive case: add the first element to the sum of the rest

Example:
mystery([1, 2, 3, 4]) 
= 1 + mystery([2, 3, 4])
= 1 + 2 + mystery([3, 4])
= 1 + 2 + 3 + mystery([4])
= 1 + 2 + 3 + 4 + mystery([])
= 1 + 2 + 3 + 4 + 0
= 10

Note: Python's built-in sum(lst) is faster and clearer for this purpose.""",
        source="coding_seed", language="en",
        license="public_domain", title="Code explanation"),

    DataRecord(
        text="""Question: How do I fix this error?
TypeError: 'NoneType' object is not subscriptable

Answer: This error means you are trying to use indexing (like [0]) on None.

Common causes:
1. A function returned None instead of a list
2. A variable was never assigned a value
3. A database query returned no results

Fix:
# Before accessing, check if value exists
result = get_data()  # might return None
if result is not None:
    print(result[0])
else:
    print("No data found")

# Or use a default value
result = get_data() or []
print(result[0] if result else "empty")""",
        source="coding_seed", language="en",
        license="public_domain", title="Error fixing"),

    # ── Indian Developer Context ──────────────────────────────────
    DataRecord(
        text="""# Common Python interview questions in India

# 1. Reverse a string
s = "Namaste"
print(s[::-1])   # etsamaN

# 2. Check palindrome
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

print(is_palindrome("racecar"))  # True
print(is_palindrome("Anad"))     # False

# 3. Count word frequency
text = "to be or not to be that is the question"
words = text.split()
freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1
print(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3])
# [('be', 2), ('to', 2), ('or', 1)]""",
        source="coding_seed", language="en",
        license="public_domain", title="Python interview India"),

    DataRecord(
        text="""# UPI payment system — how it works technically
# UPI = Unified Payments Interface (India)

# A UPI transaction flow:
# 1. User enters UPI ID (e.g., user@paytm)
# 2. App sends request to UPI gateway
# 3. Gateway routes to user's bank
# 4. Bank verifies PIN
# 5. Money transferred in real time

# Simulating UPI transaction in Python
class UPITransaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.transaction_id = self._generate_id()
        self.status = "pending"

    def _generate_id(self):
        import hashlib, time
        return hashlib.md5(
            f"{self.sender}{self.receiver}{time.time()}".encode()
        ).hexdigest()[:12].upper()

    def process(self):
        # In real UPI: call NPCI API
        self.status = "success"
        return self.transaction_id

t = UPITransaction("user@paytm", "shop@gpay", 500)
txn_id = t.process()
print(f"Transaction {txn_id}: {t.status}")""",
        source="coding_seed", language="en",
        license="public_domain", title="UPI Python"),

    DataRecord(
        text="""# Aadhaar number validation in Python
# Aadhaar is India's 12-digit unique ID system

def validate_aadhaar(number):
    \"\"\"
    Validate an Aadhaar number.
    Rules:
    - Must be 12 digits
    - Cannot start with 0 or 1
    - Must pass Verhoeff checksum
    \"\"\"
    number = str(number).strip().replace(" ", "")

    if not number.isdigit():
        return False, "Must contain only digits"
    if len(number) != 12:
        return False, "Must be exactly 12 digits"
    if number[0] in "01":
        return False, "Cannot start with 0 or 1"

    return True, "Valid Aadhaar format"

print(validate_aadhaar("2345 6789 0123"))  # True
print(validate_aadhaar("0123456789012"))   # False — starts with 0""",
        source="coding_seed", language="en",
        license="public_domain", title="Aadhaar validation"),
]


# ══════════════════════════════════════════════════════════════════
# STACK EXCHANGE DATA (CC-BY-SA licensed)
# ══════════════════════════════════════════════════════════════════

class StackExchangeSource:
    """
    Fetch programming Q&A from Stack Exchange API.
    CC-BY-SA licensed — free for training.
    """

    API_URL = "https://api.stackexchange.com/2.3/questions"

    TAGS = [
        "python", "javascript", "algorithm", "data-structures",
        "machine-learning", "numpy", "pandas", "flask", "django",
        "sql", "git", "linux", "bash", "java", "c++",
    ]

    def fetch_questions(self, tag: str, max_items: int = 20) -> List[DataRecord]:
        """Fetch top questions for a tag"""
        records = []
        try:
            params = urllib.parse.urlencode({
                "order":    "desc",
                "sort":     "votes",
                "tagged":   tag,
                "site":     "stackoverflow",
                "filter":   "withbody",
                "pagesize": max_items,
                "key":      "",
            })
            url = f"{self.API_URL}?{params}"
            req = urllib.request.Request(
                url,
                headers={"Accept-Encoding": "identity"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            for item in data.get("items", []):
                title   = item.get("title", "")
                body    = item.get("body", "")
                # Strip HTML tags simply
                body = body.replace("<p>", "\n").replace("</p>", "\n")
                body = body.replace("<code>", "").replace("</code>", "")
                body = body.replace("<pre>", "\n").replace("</pre>", "\n")
                # Remove remaining HTML
                import re
                body = re.sub(r"<[^>]+>", "", body).strip()

                if len(body) > 100:
                    records.append(DataRecord(
                        text=f"Question: {title}\n\n{body[:2000]}",
                        source="stackoverflow",
                        language="en",
                        license="cc_by_sa",
                        url=item.get("link", ""),
                        title=title[:80],
                    ))
        except Exception as e:
            print(f"  Stack Exchange error ({tag}): {e}")

        return records

    def stream(self) -> Iterator[DataRecord]:
        for tag in self.TAGS[:5]:  # start with 5 tags
            print(f"  Stack Exchange [{tag}]...")
            records = self.fetch_questions(tag, max_items=10)
            for r in records:
                yield r
            time.sleep(1)  # rate limit


# ══════════════════════════════════════════════════════════════════
# MAIN CODING DATA COLLECTOR
# ══════════════════════════════════════════════════════════════════

class CodingDataCollector:
    """
    Collects coding training data from all sources.
    Integrates with the main AnadDataCollector.
    """

    def __init__(self, data_dir: str = "./training/data"):
        self.collector = AnadDataCollector(data_dir)

    def collect(self, include_stackoverflow: bool = True):
        print("\n" + "═" * 50)
        print("  ANAD CODING DATA COLLECTION")
        print("═" * 50 + "\n")

        # Add handcrafted examples
        print("  Adding handcrafted coding examples...")
        added = 0
        for record in CODING_SEED_TEXTS:
            if self.collector._add(record):
                added += 1
        self.collector._flush()
        print(f"  Added {added} coding examples")

        # Stack Exchange
        if include_stackoverflow:
            print("\n  Fetching Stack Overflow questions...")
            so = StackExchangeSource()
            so_added = 0
            for record in so.stream():
                if self.collector._add(record):
                    so_added += 1
            self.collector._flush()
            print(f"  Added {so_added} Stack Overflow Q&As")

        total = self.collector.total_records()
        print(f"\n  Total records now: {total}")
        print("\n  Coding data collection complete.")
        return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./training/data")
    parser.add_argument("--no-stackoverflow", action="store_true")
    args = parser.parse_args()

    collector = CodingDataCollector(args.datadir)
    collector.collect(
        include_stackoverflow=not args.no_stackoverflow
    )
