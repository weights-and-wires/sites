---
id: "alphago"
title: "Mastering the game of Go with deep neural networks and tree search"
authors: "Silver et al."
year: 2016
venue: "Nature 529"
arxiv: null
url: "https://doi.org/10.1038/nature16961"
track: rl
tier: extended
est_hours: 5
status: stub
summary: "Combines policy and value networks with MCTS; sets up AlphaZero."
related_course: "rl: planning and search"
prereqs:
  - id: dqn
    reason: "Q-network trained on TD targets"
background: [monte-carlo-tree-search, supervised-policy-learning]
course_links:
  - title: "Monte Carlo tree search"
    url: "https://rl.ifkash.dev/lessons/022-monte-carlo-tree-search"
  - title: "Many agents and self-play"
    url: "https://rl.ifkash.dev/lessons/048-multi-agent-and-self-play"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Policy network from human games, refined by self-play
- Value network for position evaluation
- MCTS guided by learned priors
