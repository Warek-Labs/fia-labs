# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import time
import heapq

from pacman import GameState
from util import manhattanDistance, INFINITY
from game import Directions
import random, util, sys
from itertools import count

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()  # Pacman position after moving
        newFood = successorGameState.getFood()  # Remaining food
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        listFood = newFood.asList()  # All remaining food as list
        ghostPos = successorGameState.getGhostPositions()  # Get the ghost position
        # Initialize with list 
        mFoodDist = []
        mGhostDist = []

        # Find the distance of all the foods to the pacman 
        for food in listFood:
            mFoodDist.append(manhattanDistance(food, newPos))

        # Find the distance of all the ghost to the pacman
        for ghost in ghostPos:
            mGhostDist.append(manhattanDistance(ghost, newPos))

        if currentGameState.getPacmanPosition() == newPos:
            return (-(float("inf")))

        for ghostDistance in mGhostDist:
            if ghostDistance < 2:
                return (-(float("inf")))

        if len(mFoodDist) == 0:
            return float("inf")
        else:
            minFoodDist = min(mFoodDist)
            maxFoodDist = max(mFoodDist)

        return 1000 / sum(mFoodDist) + 10000 / len(mFoodDist)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

    """
        Your improved evaluation function here
    """

    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def minimax(self, agent: int, depth: int, game_state: GameState) -> float:
        if game_state.isLose() or game_state.isWin() or depth == self.depth:
            return self.evaluationFunction(game_state)

        if agent == 0:
            return self.max_value(agent, depth, game_state)
        else:
            return self.min_value(agent, depth, game_state)

    def max_value(self, agent: int, depth: int, game_state: GameState) -> float:
        legal_actions: list[str] = game_state.getLegalActions(agent)
        scores: list[float] = []

        for action in legal_actions:
            successor_state = game_state.generateSuccessor(agent, action)
            score = self.minimax(1, depth, successor_state)
            scores.append(score)

        return max(scores)

    def min_value(self, agent: int, depth: int, game_state: GameState) -> float:
        next_agent_index = agent + 1
        if next_agent_index == game_state.getNumAgents():
            next_agent_index = 0
            depth += 1

        legal_actions: list[str] = game_state.getLegalActions(agent)
        scores: list[float] = []

        for action in legal_actions:
            successor_state: GameState = game_state.generateSuccessor(agent, action)
            score: float = self.minimax(next_agent_index, depth, successor_state)
            scores.append(score)

        return min(scores)

    def getAction(self, gameState: GameState) -> int:
        possible_actions: list[str] = gameState.getLegalActions(0)
        action_scores: list[float] = []

        for action in possible_actions:
            successor_state: GameState = gameState.generateSuccessor(0, action)
            score: float = self.minimax(0, 0, successor_state)
            action_scores.append(score)

        max_action: float = max(action_scores)
        max_indices: list[int] = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosen_index: int = random.choice(max_indices)

        return possible_actions[chosen_index]


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState) -> Directions:
        def mini_max(game_state: GameState, depth: int, alpha: float, beta: float, agent_index: int) -> float:
            if depth == self.depth or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state)

            if agent_index == 0:
                return get_max_value(game_state, depth, alpha, beta)
            else:
                return get_min_value(game_state, depth, agent_index, alpha, beta)

        def get_max_value(game_state: GameState, depth: int, alpha: float, beta: float) -> float:
            max_value = -float('inf')
            for action in game_state.getLegalActions(0):
                successor_state = game_state.generateSuccessor(0, action)
                max_value = max(max_value, mini_max(successor_state, depth, alpha, beta, 1))
                if max_value >= beta:
                    return max_value  # Prune
                alpha = max(alpha, max_value)
            return max_value

        def get_min_value(game_state: GameState, depth: int, agent_index: int, alpha: float, beta: float) -> float:
            min_value = INFINITY
            num_agents = game_state.getNumAgents()
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                if agent_index == num_agents - 1:
                    min_value = min(min_value, mini_max(successor_state, depth + 1, alpha, beta, 0))
                else:
                    min_value = min(min_value, mini_max(successor_state, depth, alpha, beta, agent_index + 1))
                if min_value <= alpha:
                    return min_value  #
                beta = min(beta, min_value)
            return min_value

        return max(gameState.getLegalActions(0),
                   key=lambda action: mini_max(gameState.generateSuccessor(0, action), 0, -INFINITY, INFINITY,
                                               1))


class ProgressiveDeepeningAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState) -> Directions:
        num_agents: int = gameState.getNumAgents()
        start_time: float = time.time()
        best_action: Directions = Directions.STOP
        time_limit: float = 0.5

        def mini_max(game_state: GameState, depth: int, alpha: float, beta: float) -> Directions:
            best_action: Directions = Directions.STOP
            max_score = -INFINITY

            for pacman_action in game_state.getLegalActions(0):
                successor_state = game_state.generateSuccessor(0, pacman_action)
                score = get_min_value(successor_state, 1, depth, alpha, beta)

                if score > max_score:
                    max_score = score
                    best_action = pacman_action

            return best_action

        def get_max_value(game_state: GameState, ghost_index: int, depth: int, alpha: float, beta: float) -> int:
            if game_state.isWin() or game_state.isLose() or depth == self.depth:
                return self.evaluationFunction(game_state)

            # do not go deeper if time limit is exceeded
            if time.time() - start_time > time_limit:
                return self.evaluationFunction(game_state)

            ghost_index %= (num_agents - 1)
            max_score = -INFINITY

            for ghost_action in game_state.getLegalActions(ghost_index):
                successor_state = game_state.generateSuccessor(ghost_index, ghost_action)
                score = get_min_value(successor_state, ghost_index + 1, depth, alpha, beta)
                max_score = max(max_score, score)

                if max_score >= beta:
                    return max_score
                alpha = max(alpha, max_score)

            return max_score

        def get_min_value(game_state: GameState, agent_index: int, depth: int, alpha: float, beta: float) -> int:
            if game_state.isWin() or game_state.isLose() or depth == self.depth:
                return self.evaluationFunction(game_state)

            # do not go deeper if time limit is exceeded
            if time.time() - start_time > time_limit:
                return self.evaluationFunction(game_state)

            min_score = INFINITY
            legal_actions = game_state.getLegalActions(agent_index)

            if agent_index + 1 == num_agents:
                for ghost_action in legal_actions:
                    successor_state = game_state.generateSuccessor(agent_index, ghost_action)
                    score = get_max_value(successor_state, 0, depth + 1, alpha, beta)
                    min_score = min(min_score, score)

                    if min_score <= alpha:
                        return min_score
                    beta = min(beta, min_score)
            else:
                for ghost_action in legal_actions:
                    successor_state = game_state.generateSuccessor(agent_index, ghost_action)
                    score = get_min_value(successor_state, agent_index + 1, depth, alpha, beta)
                    min_score = min(min_score, score)

                    if min_score <= alpha:
                        return min_score
                    beta = min(beta, min_score)

            return min_score

        for depth in range(1, self.depth + 1):
            # do not go deeper if time limit is exceeded
            if time.time() - start_time > time_limit:
                break

            best_action = mini_max(gameState, 0, -INFINITY, INFINITY)

        return best_action


class AStarMinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState) -> Directions:
        num_agents = gameState.getNumAgents()

        def a_star(game_state: GameState, depth: int) -> Directions:
            pq = []
            tiebreaker = count()
            start_actions = game_state.getLegalActions(0)
            best_action = Directions.STOP
            best_score = float('-inf')

            for action in start_actions:
                successor_state = game_state.generateSuccessor(0, action)
                f_score = self.evaluationFunction(successor_state) + heuristic(successor_state)
                heapq.heappush(pq, (f_score, next(tiebreaker), depth, 1, action, successor_state, float('-inf'), float('inf')))

            while pq:
                priority, tie, current_depth, agent_index, first_action, current_state, alpha, beta = heapq.heappop(pq)

                if current_state.isWin() or current_state.isLose() or current_depth == self.depth:
                    score = self.evaluationFunction(current_state)
                    if score > best_score:
                        best_score = score
                        best_action = first_action
                    continue

                if agent_index == 0:  # Pacman's turn (maximizing)
                    for next_action in current_state.getLegalActions(agent_index):
                        next_state = current_state.generateSuccessor(agent_index, next_action)
                        g_score = self.evaluationFunction(next_state)
                        f_score = g_score + heuristic(next_state)

                        if f_score > beta:  # Beta pruning
                            continue

                        heapq.heappush(pq, (f_score, next(tiebreaker), current_depth, agent_index + 1, first_action, next_state, alpha, beta))

                        if f_score > alpha:  # Update alpha
                            alpha = f_score

                else:  # Ghost's turn (minimizing)
                    legal_actions = current_state.getLegalActions(agent_index)
                    next_agent = (agent_index + 1) % num_agents
                    next_depth = current_depth + 1 if next_agent == 0 else current_depth

                    for next_action in legal_actions:
                        next_state = current_state.generateSuccessor(agent_index, next_action)
                        g_score = self.evaluationFunction(next_state)
                        f_score = g_score + heuristic(next_state)

                        if f_score < alpha:  # Alpha pruning
                            continue

                        heapq.heappush(pq, (f_score, next(tiebreaker), next_depth, next_agent, first_action, next_state, alpha, beta))

                        if f_score < beta:  # Update beta
                            beta = f_score

            return best_action

        def heuristic(game_state: GameState) -> int:
            """Estimate remaining cost."""
            pacman_position = game_state.getPacmanPosition()
            food_list = game_state.getFood().asList()
            ghost_positions = [game_state.getGhostPosition(i) for i in range(1, num_agents)]

            food_distance = min([manhattanDistance(pacman_position, food) for food in food_list]) if food_list else 0
            ghost_distance = min([manhattanDistance(pacman_position, ghost) for ghost in ghost_positions]) if ghost_positions else float('inf')

            return food_distance - ghost_distance

        return a_star(gameState, 0)

class AStarAlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState) -> Directions:
        num_agents = gameState.getNumAgents()

        def a_star_alpha_beta(game_state: GameState, depth: int) -> Directions:
            pq = []
            tiebreaker = count()
            start_actions = game_state.getLegalActions(0)
            best_action = Directions.STOP
            best_score = -INFINITY
            alpha = -INFINITY
            beta = INFINITY

            for action in start_actions:
                successor_state = game_state.generateSuccessor(0, action)
                f_score = self.evaluationFunction(successor_state) + heuristic(successor_state)
                heapq.heappush(pq, (f_score, next(tiebreaker), depth, 1, action, successor_state))

            while pq:
                priority, tie, current_depth, agent_index, first_action, current_state = heapq.heappop(pq)

                if current_state.isWin() or current_state.isLose() or current_depth == self.depth:
                    score = self.evaluationFunction(current_state)
                    if score > best_score:
                        best_score = score
                        best_action = first_action
                    continue

                if agent_index == 0:
                    best_value = -INFINITY
                    for next_action in current_state.getLegalActions(agent_index):
                        next_state = current_state.generateSuccessor(agent_index, next_action)
                        g_score = self.evaluationFunction(next_state)
                        f_score = g_score + heuristic(next_state)

                        best_value = max(best_value, g_score)
                        if best_value > beta:
                            break
                        alpha = max(alpha, best_value)

                        heapq.heappush(pq, (
                            f_score, next(tiebreaker), current_depth, agent_index + 1, first_action, next_state))

                else:
                    best_value = INFINITY
                    legal_actions = current_state.getLegalActions(agent_index)
                    next_agent = (agent_index + 1) % num_agents
                    next_depth = current_depth + 1 if next_agent == 0 else current_depth

                    for next_action in legal_actions:
                        next_state = current_state.generateSuccessor(agent_index, next_action)
                        g_score = self.evaluationFunction(next_state)
                        f_score = g_score + heuristic(next_state)

                        best_value = min(best_value, g_score)
                        if best_value < alpha:
                            break
                        beta = min(beta, best_value)

                        heapq.heappush(pq,
                                       (f_score, next(tiebreaker), next_depth, next_agent, first_action, next_state))

            return best_action

        def heuristic(game_state: GameState) -> int:
            pacman_position = game_state.getPacmanPosition()
            food_list = game_state.getFood().asList()
            ghost_positions = [game_state.getGhostPosition(i) for i in range(1, num_agents)]

            food_distance = min([manhattanDistance(pacman_position, food) for food in food_list]) if food_list else 0
            ghost_distance = min([manhattanDistance(pacman_position, ghost) for ghost in
                                  ghost_positions]) if ghost_positions else INFINITY

            return food_distance - ghost_distance

        return a_star_alpha_beta(gameState, 0)


def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates()
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    if currentGameState.isWin():
        return INFINITY
    if currentGameState.isLose():
        return -INFINITY
    foodDistList = []
    for each in foods.asList():
        foodDistList = foodDistList + [util.manhattanDistance(each, pacmanPos)]
    minFoodDist = min(foodDistList)
    ghostDistList = []
    scaredGhostDistList = []
    for each in ghostList:
        if each.scaredTimer == 0:
            ghostDistList = ghostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
        elif each.scaredTimer > 0:
            scaredGhostDistList = scaredGhostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
    minGhostDist = -1
    if len(ghostDistList) > 0:
        minGhostDist = min(ghostDistList)
    minScaredGhostDist = -1
    if len(scaredGhostDistList) > 0:
        minScaredGhostDist = min(scaredGhostDistList)
    # Evaluate score
    """
        Your improved evaluation here
    """

    # 1000 / sum(minFoodDist) + 10000 / len(minGhostDist)
    score = scoreEvaluationFunction(currentGameState)

    # encourage chasing scared ghost
    if minScaredGhostDist > 0:
        # if the scared ghost is farther, the score should be lower
        # since feasible depth is 3, pacman will not look farther than 3 steps so hardcode 3
        score += 10000 / max(0, minScaredGhostDist)

    # penalty for standing still
    if currentGameState.getPacmanState().getDirection() == Directions.STOP:
        score -= 10 / 10000

    return score


# Abbreviation
better = betterEvaluationFunction
