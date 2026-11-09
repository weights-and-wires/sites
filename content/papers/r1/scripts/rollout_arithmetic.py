"""The batch arithmetic of Section 2.1 and Section 3.2.1."""

questions_per_step = 32
group_size = 16
outputs_per_step = questions_per_step * group_size
print("outputs per training step:", outputs_per_step)

rollout_outputs = 8192
minibatches = 16
print("outputs per minibatch:", rollout_outputs // minibatches)
print("questions per rollout:", rollout_outputs // group_size)
print("steps per rollout (one inner epoch, one step per minibatch):", minibatches)

steps_r1_zero = 10_400
print("rollouts in 10,400 steps:", steps_r1_zero // minibatches)
print("completions sampled in 10,400 steps:", steps_r1_zero * outputs_per_step)
