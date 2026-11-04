**Observations**

*Grid structure from W1 (which inputs carry large weights at all).* Every input for the border (rows 0 and 9, columns 0 and 9) is at initialisation scale (|w| ≲ 0.2), e.g. inputs 0–49 and 450–499, and the (row 1, col 0)/(row 1, col 9) groups. Inside the playable area the large rows of W1 are not uniform: 16 interior cells have *all five* channels at init scale — (2,2),(2,3),(2,6),(2,7),(3,2),(3,7),(4,4),(4,5),(5,4),(5,5),(6,2),(6,7),(7,2),(7,3),(7,6),(7,7) (e.g. inputs 110–119, 130–139, 160–164, 185–189, 220–229, 270–279, 310–314, 335–339, 360–369, 380–389 are all |w| < 0.3). Nothing ever varies there, so they are permanent obstacles. Laid out:

```
      c1 c2 c3 c4 c5 c6 c7 c8
r1    .  .  .  .  .  .  .  .
r2    .  X  X  .  .  X  X  .
r3    .  X  .  .  .  .  X  .
r4    .  .  .  X  X  .  .  .
r5    .  .  .  X  X  .  .  .
r6    .  X  .  .  .  .  X  .
r7    .  X  X  .  .  X  X  .
r8    .  .  .  .  .  .  .  .
```
A 4-fold-symmetric maze: an outer corridor ring, four L-shaped corner blocks, a 2×2 centre block, 48 open corridor cells.

*Channel 0 (agent):* huge weights at every open cell and only there — input 55 (1,1): `1.706 -3.242 … 4.081`; input 65 (1,3): `4.742 … -3.778 … 3.821`; input 210 (4,2): `… 7.277 …` (unit 17); input 175 (3,5): `-5.208 3.306 3.336 …`; input 285 (5,7): `-4.928 5.834 …`; input 405 (8,1): `2.446 … -5.493`. So the agent can stand on any corridor cell and its exact position matters a lot.

*Channel 1 (moving objects):* equally large at the same 48 open cells — input 126 (2,5): `… -5.209 …`; input 236 (4,7): `… 4.226 … -5.925`; input 376 (7,5): `… 5.421 … -4.042`; input 331 (6,6): `… -4.838 …`; input 241 (4,8): `… -4.641 … 4.077`. Moving objects roam the whole maze. Hidden unit 9 sums channel 1 *negatively* in the upper rows ((1,5) −2.968, (1,6) −3.386, (1,7) −3.354, (2,8) −3.343, (3,8) −3.074, (4,8) −4.641) and ≈0 in the bottom rows ((8,1) −0.380, (8,4) −0.075); its output row W3[9] = `0.740 0.334 -3.070 1.267 0.856`, so "movers in the top half" ⇒ large *down* logit. That is avoidance, not pursuit: the moving objects are hazards.

*Channel 2 (reward targets):* clearly learned but much weaker than ch 0/1, and present at open cells only: input 122 (2,4): `… -4.876` (unit 6); input 62 (1,2): `… -2.361`; input 67 (1,3): `… -1.905`; input 57 (1,1): `… -1.490`; input 322 (6,4): `-0.862 … 1.596 -1.426 … 1.171 … -1.319`; the rest are ±0.2–0.5 (e.g. inputs 257, 407, 432). Reward items are spread over the corridors but their exact map is weakly informative — what you expect when they are initially almost everywhere.

*Channel 3 (obstacles):* at init scale at every single cell (inputs 58, 63, 123, 173, 323, 428 … all |w| < 0.2). Obstacles never change during play: the maze is static and nothing else (no growing body, no bricks) ever enters this channel.

*Channel 4 (previous frame):* moderate, spatially smooth weights at open cells: unit 6 receives ≈+0.6…+1.0 from nearly every open cell (input 69: 1.043, 169: 0.747, 234: 0.857, 394: 0.941, 444: 0.941), unit 9 receives negative values concentrated in the top rows (input 94: −2.849, 144: −2.307, 84: −2.070). Combined with the current-frame weights this gives the direction of motion of the movers.

*Output layer:* b3 = `-0.873 0.703 0.712 0.595 -0.092` — no-op is strongly disfavoured; the agent is meant to keep moving. W3 is dominated by big single-action vetoes/pushes: row 9 `-3.070` down, row 7 `-2.498` down, row 19 `-2.927` down, row 25 `-2.491` left, row 16 `+2.490` left, row 24 `+2.502` down, row 28 `+2.500` right, row 30 `+2.223` right — hidden units encode "don't go that way / go this way" decisions, typical of corridor navigation with threats.

**Reasoning**

- Static, symmetric interior maze with corridors, and an obstacle channel that never varies ⇒ rules out Snake (body would appear in ch 3), Breakout (bricks vanish), Bomberman (lattice pillars + destructible blocks), Sokoban-like box pushing (boxes would be obstacles).
- Agent weights on *every* corridor cell ⇒ rules out paddle/ship games (Pong, Breakout, Space Invaders) and lane games where the agent lives on one row.
- Moving objects on every corridor cell, with hidden units that map "movers above me" → "go down" ⇒ free-roaming enemies to avoid, not lane-bound cars (Frogger/Freeway) and not falling items to catch.
- Reward channel present across the corridors but used weakly ⇒ ubiquitous collectibles (pellets) rather than a single goal cell.
- No-op strongly disfavoured ⇒ continual movement. 

Everything matches a small Pac-Man: static maze, ghosts roaming, pellets everywhere, keep moving, avoid ghosts.

**Game:** Pac-Man (a mini 8×8 maze version — "Pac-Man / ghost-maze" gridworld).

**Description:** The agent moves through the fixed corridors of a symmetric maze collecting the reward dots that fill the corridors, while several ghosts (channel 1) wander the same corridors; touching a ghost ends the episode, so the policy mostly tracks its own position and the ghosts' positions/velocities to pick a safe corridor direction.

**Confidence:** 75%