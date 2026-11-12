# A tiny rule-based normalizer: written forms -> spoken words.
# This is our own toy. It is NOT Woof's normalizer, which the post says
# handles far more cases (and which we have not read).
import re

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
MONTHS = "January February March April May June July August September October November December".split()
ORD = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth", 20: "twentieth", 30: "thirtieth"}


def below_1000(n):
    out = []
    if n >= 100:
        out.append(ONES[n // 100] + " hundred")
        n %= 100
        if n:
            out.append("and")
    if n >= 20:
        out.append(TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10]))
    elif n or not out:
        out.append(ONES[n])
    return " ".join(out)


def number(n):
    if n < 1000:
        return below_1000(n)
    th, rest = divmod(n, 1000)
    s = below_1000(th) + " thousand"
    if rest:
        s += (" and " if rest < 100 else " ") + below_1000(rest)
    return s


def ordinal(n):
    if n in ORD:
        return ORD[n]
    if n < 20:
        return ONES[n] + "th"
    t, o = divmod(n, 10)
    return TENS[t] + (" " + ordinal(o) if o else "")


def digits(s):
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


RULES = []


def rule(pattern):
    def deco(fn):
        RULES.append((re.compile(pattern), fn))
        return fn
    return deco


@rule(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?")
def money(m):
    d = int(m.group(1).replace(",", ""))
    s = number(d) + (" dollar" if d == 1 else " dollars")
    if m.group(2) and int(m.group(2)):
        s += " " + number(int(m.group(2))) + " cents"
    return s


@rule(r"(\d{1,2})/(\d{1,2})")
def date(m):
    d, mo = int(m.group(1)), int(m.group(2))
    return f"the {ordinal(d)} of {MONTHS[mo - 1]}"


@rule(r"(\d{1,2}):(\d{2})")
def clock(m):
    h, mi = int(m.group(1)), int(m.group(2))
    return number(h) + ("" if mi == 0 else " " + ("oh " + ONES[mi] if mi < 10 else number(mi)))


@rule(r"\b([A-Z]{2,})-(\d+)\b")
def code(m):
    return " ".join(m.group(1)) + " " + digits(m.group(2))


@rule(r"(\d+)\.(\d+) ?mg")
def dose(m):
    return f"{number(int(m.group(1)))} point {digits(m.group(2))} milligrams"


@rule(r"\b\d{3} \d{4} \d{4}\b")
def phone(m):
    return ", ".join(digits(g) for g in m.group(0).split())


def normalize(text):
    for pat, fn in RULES:
        text = pat.sub(lambda m, fn=fn: fn(m), text)
    return text


tests = [
    "Call 020 7946 0018 before 6:15.",
    "Your order ships on 14/03.",
    "The deposit is $25.",
    "Total: $1,249.50",
    "Part XK-4471 ships today.",
    "Take 2.5 mg daily.",
]
rows = []
for t in tests:
    n = normalize(t)
    print(t, "->", n)
    rows.append([t, n])
result = rows
