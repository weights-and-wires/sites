"""Equation 4 on five completions of "47 + 38" (gold answer 85)."""

from r1_lib import accuracy_reward, format_reward, rule_reward

GOLD = 85
COMPLETIONS = {
    "A tags, right answer": "<think>40 + 30 = 70 and 7 + 8 = 15, so 85.</think><answer>85</answer>",
    "B tags, wrong answer": "<think>47 + 38 = 75.</think><answer>75</answer>",
    "C no tags, right answer": "The answer is 85.",
    "D answer tag missing": "<think>47 + 38 = 85.</think> 85",
    "E tags in wrong order": "<answer>85</answer><think>47 + 38 = 85.</think>",
}

print(f"{'completion':26s} acc  fmt  total")
for name, text in COMPLETIONS.items():
    print(f"{name:26s} {accuracy_reward(text, GOLD):.0f}    {format_reward(text):.0f}    {rule_reward(text, GOLD):.0f}")
