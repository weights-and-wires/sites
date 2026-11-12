"""One triplet (p, i, o) checked three ways (Section 3.2)."""

from azr_lib import check_answer, run_program

p = "def f(x):\n    return sum(i for i in range(x) if i % 2 == 0)"
i = 5
o = run_program(p, i)
print("program: sum of even numbers below x")
print("f(5) =", o, "| f(6) =", run_program(p, 6), "| f(7) =", run_program(p, 7))
print("deduction, answer 6 :", check_answer("deduction", p, i, o, 6))
print("deduction, answer 5 :", check_answer("deduction", p, i, o, 5))
print("abduction, answer 5 :", check_answer("abduction", p, i, o, 5))
print("abduction, answer 6 :", check_answer("abduction", p, i, o, 6), "(a different input with the same output)")
print("abduction, answer 3 :", check_answer("abduction", p, i, o, 3))
pairs = [(x, run_program(p, x)) for x in [1, 2, 3, 4, 8, 9]]
print("held-out pairs      :", pairs)
good = "def f(x):\n    return sum(range(0, x, 2))"
bad = "def f(x):\n    return x // 2"
print("induction, equal program:", check_answer("induction", p, pairs, None, good))
print("induction, x // 2       :", check_answer("induction", p, pairs, None, bad))
