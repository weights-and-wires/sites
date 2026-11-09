"""Equation 7 on a toy chain of thought, with this page's word rule."""

from r1_lib import language_reward

cots = {
    "English only": "First add the ones digits 7 and 8 to get 15 then carry the 1",
    "mixed": "First add the ones digits 我们先算 7 and 8 to get 15 then 进位 the 1",
    "no letters": "7 + 8 = 15",
}
for name, cot in cots.items():
    words = [w for w in cot.split() if any(c.isalpha() for c in w)]
    print(f"{name:13s} words counted: {len(words):2d}  reward: {language_reward(cot):.4f}")
