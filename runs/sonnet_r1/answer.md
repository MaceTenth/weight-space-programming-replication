**Design**

The policy is a reactive paddle tracker. It tries to put the paddle where the ball will be on its next step.

- **Target position:** the ball moves exactly one column per step. The network reads that direction from the trail cell and sets target = ball_x + (ball_x − trail_x) = 2·ball_x − trail_x.
- **Offset:** d = target − paddle_x. This is a linear function of the one-hot inputs: the ball cell (channel 3), the trail cell (channel 4) and the paddle cell (channel 0, row 9).
- **Why lead by one step:** the paddle moves at the same speed as the ball's sideways motion, and a catch needs the paddle on the ball's current or next column. Aiming at the next column keeps the paddle inside that window, even after a sticky-action slip.
- **Behaviour at the edges:** at a wall the target can land off the grid, and the paddle just clamps there.
- **Behaviour at the start:** the trail equals the ball, so the target is the ball's own column.

**Hidden units**

- **Layer 1, unit 0 (A):** tanh(4·(d − 0.5)). It is about +1 if d ≥ 1 and about −1 if d ≤ 0.
- **Layer 1, unit 1 (B):** tanh(4·(d + 0.5)). It is about +1 if d ≥ 0 and about −1 if d ≤ −1.
- **Layer 2, unit 0:** tanh(2A), the "move right" indicator.
- **Layer 2, unit 1:** tanh(2B), which is −1 exactly when the paddle should move left.
- **Layer 2, unit 2:** tanh(2(B − A)). It is about 1 only when d = 0 and about 0 otherwise.
- **Layer 2, other units:** unused.

**Outputs**

- **Right:** 8 · unit 0.
- **Left:** −8 · unit 1.
- **No-op:** 8 · unit 2.
- **Up, down and fire:** a bias of −10, so they are never chosen.

The margin between the correct action and the others is about 7.7 logits. Sampling noise is therefore negligible, and only the game's own 10% sticky actions cause mistakes.

```edits
b1[0] = -2
b1[1] = 2
W1[3:60:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[63:120:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[123:180:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[183:240:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[243:300:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[303:360:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[363:420:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[423:480:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[483:540:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[543:600:6, 0] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[3:60:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[63:120:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[123:180:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[183:240:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[243:300:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[303:360:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[363:420:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[423:480:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[483:540:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[543:600:6, 1] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
W1[4:60:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[64:120:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[124:180:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[184:240:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[244:300:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[304:360:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[364:420:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[424:480:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[484:540:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[544:600:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[4:60:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[64:120:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[124:180:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[184:240:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[244:300:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[304:360:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[364:420:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[424:480:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[484:540:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[544:600:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[540:600:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W1[540:600:6, 1] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
W2[0, 0] = 2
W2[1, 1] = 2
W2[0, 2] = -2
W2[1, 2] = 2
W3[0, 3] = 8
W3[1, 1] = -8
W3[2, 0] = 8
b3[2] = -10
b3[4] = -10
b3[5] = -10
```

One thing to check in the block above: the last two `W1[540:600:6, ...]` lines write the paddle weights for row 9, channel 0. They are in the right place. The trail statement for row 9 (`W1[544:600:6, ...]`) is separate, because trail is channel 4.