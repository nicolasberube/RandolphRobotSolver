# RandolphRobotSolver
Randolph's robot game homemade solver

This is the first version of an algorithm to solve Randolph's robot game. It is not optimised in any way, and was just to test my own first try to solve this problem without any external help.

More info could be found in the literature, which I deliberetaly have not researched or read before coding this
- On the Complexity of Randolph’s Robot Game,
  Birgit Engels Tom Kamphans, 2005
- The Parameterized Complexity of Ricochet Robots,
  Adam Hesterberg Justin Kopinsky, 2017
(Could Dijkstra's algorithm have helped?)

# First version

This code just passed the integration test. However, many key features are missing:
- The Gray robot and the BlackHole goal is not coded yet.
- Bumpers are not implemented.
- All tiles data are not entered in tiles.json. Only one group of tiles was done so far.

The way the algorithm works is that is finds all accessible position of a single robot (without moving the others) and keeps track of them (on a "tracking map" object, or TMap, which tracks the position and relevant information for quick calculation, as well as the shortest path necessary to reach the position). Then, from all those positions, it tries to moves, and sees if it collides with another's robot possibly accessible positions. This will generate a new reachable position, and from that one, it will keep exploring new positions. Each position keeps track of where the collision happened, as a "conditional position" of the other robot. These conditional positions are used as a proxy to explore the map of reachable positions without having to compare the whole of the paths. This step is then repeated a set number of times, and then we check the destination's shortest path on this TMap.

This particular algorithm does not handle re-collisions. In other words, for efficiency's sake, the algorithm can't process that a robot can collide twice on the same robot. It could also probably benefit from a pruning algorithm once a solution has been found.