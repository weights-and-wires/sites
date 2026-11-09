---
id: "alphazero"
title: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
authors: "Silver et al."
year: 2017
venue: "arXiv preprint (Science 2018)"
arxiv: "1712.01815"
url: "https://arxiv.org/abs/1712.01815"
track: rl
tier: extended
est_hours: 6
status: stub
summary: "Pure self-play with MCTS and no human data; the template for search plus learning."
related_course: "rl: planning and search"
prereqs:
  - id: alphago
    reason: "Policy network from human games, refined by self-play"
background: [self-play, puct]
course_links:
  - title: "Monte Carlo tree search"
    url: "https://rl.ifkash.dev/lessons/022-monte-carlo-tree-search"
  - title: "Many agents and self-play"
    url: "https://rl.ifkash.dev/lessons/048-multi-agent-and-self-play"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Single network with policy and value heads
- MCTS visit counts as policy targets
- Self-play generates training data
