# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#
import math

from util import manhattanDistance
from game import Directions
import random, util

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
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()  # food available from current state
        newFood = successorGameState.getFood()  # food available from successor state (excludes food@successor)
        # currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
        # newCapsules = successorGameState.getCapsules()  # capsules available from successor (excludes
        # capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        stopPunish = -50 if action == 'Stop' else 0
        foodReward = 500 if (len(newFood.asList()) < len(currentFood.asList())) else 0
        scaredReward = 50 * max(newScaredTimes)
        distance2food = [manhattanDistance(newPos, food) for food in newFood.asList()]
        distance2ghost = [manhattanDistance(newPos, state.getPosition()) for state in newGhostStates]
        min2food = min(distance2food) if distance2food else 0
        min2ghost = min(distance2ghost) if distance2ghost else 0
        ghostPunish = -999999 if min2ghost <= 2 else 0
        return foodReward + 100 / (min2food + 1) - 10 / (min2ghost + 1) + stopPunish + scaredReward + ghostPunish


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
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
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def maxValue(gameState, currentDepth):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState), None
            else:
                vMax = - float('inf')
                aMax = None
                for a in gameState.getLegalActions(agentIndex=0):
                    s = gameState.generateSuccessor(agentIndex=0, action=a)
                    v = minValue(gameState=s, agentIndex=1, currentDepth=currentDepth)
                    if v > vMax:
                        vMax, aMax = v, a
                return vMax, aMax

        def minValue(gameState, agentIndex, currentDepth):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)
            else:
                vMin = float('inf')
                for a in gameState.getLegalActions(agentIndex):
                    s = gameState.generateSuccessor(agentIndex, action=a)
                    if agentIndex >= gameState.getNumAgents() - 1:
                        v, a = maxValue(gameState=s, currentDepth=currentDepth + 1)
                    else:
                        v = minValue(gameState=s, agentIndex=agentIndex + 1, currentDepth=currentDepth)
                    vMin = min(vMin, v)
                return vMin

        v, a = maxValue(gameState, 0)
        return a


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxValue(gameState, currentDepth, alpha, beta):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState), None
            else:
                vMax = - float('inf')
                aMax = None
                for a in gameState.getLegalActions(agentIndex=0):
                    s = gameState.generateSuccessor(agentIndex=0, action=a)
                    v = minValue(gameState=s, agentIndex=1, currentDepth=currentDepth, alpha=alpha, beta=beta)
                    if v > vMax:
                        vMax, aMax = v, a
                    if vMax > beta:
                        return vMax, None
                    alpha = max(alpha, vMax)
                return vMax, aMax

        def minValue(gameState, agentIndex, currentDepth, alpha, beta):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)
            else:
                vMin = float('inf')
                for a in gameState.getLegalActions(agentIndex):
                    s = gameState.generateSuccessor(agentIndex, action=a)
                    if agentIndex >= gameState.getNumAgents() - 1:
                        v, a = maxValue(gameState=s, currentDepth=currentDepth + 1, alpha=alpha, beta=beta)
                    else:
                        v = minValue(gameState=s, agentIndex=agentIndex + 1, currentDepth=currentDepth, alpha=alpha,
                                     beta=beta)
                    vMin = min(vMin, v)
                    if vMin < alpha:
                        return vMin
                    beta = min(beta, vMin)
                return vMin

        v, a = maxValue(gameState, 0, -float('inf'), float('inf'))
        return a


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def maxValue(gameState, currentDepth):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState), None
            else:
                vMax = - float('inf')
                aMax = None
                for a in gameState.getLegalActions(agentIndex=0):
                    # print(a)
                    s = gameState.generateSuccessor(agentIndex=0, action=a)
                    v = expValue(gameState=s, agentIndex=1, currentDepth=currentDepth)
                    if v > vMax:
                        vMax, aMax = v, a
                return vMax, aMax

        def expValue(gameState, agentIndex, currentDepth):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)
            else:
                values = []
                for a in gameState.getLegalActions(agentIndex):
                    s = gameState.generateSuccessor(agentIndex, action=a)
                    if agentIndex >= gameState.getNumAgents() - 1:
                        v, a = maxValue(gameState=s, currentDepth=currentDepth + 1)
                    else:
                        v = expValue(gameState=s, agentIndex=agentIndex + 1, currentDepth=currentDepth)
                    values.append(v)
                return sum(values) / len(values)

        v, a = maxValue(gameState, 0)
        return a


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: using bfs to find the distance to the nearest food, capsule and non-scared ghost.
      get punishment for distance to the nearest food, capsule and numbers of food and capsules left,
      as well as too close ghost, get reward for distance to the nearest ghost
    """
    def closest_bfs(pos, locations, walls):
        q = util.Queue()
        q.push(pos)
        distances = {pos: 0}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while not q.isEmpty():
            x, y = q.pop()
            dis = distances[(x, y)] + 1
            if (x, y) in locations:
                break
            for direction in directions:
                x1 = x + direction[0]
                y1 = y + direction[1]
                if not walls[x1][y1] and (x1, y1) not in distances:
                    distances[(x1, y1)] = dis
                    q.push((x1, y1))
        return dis

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()  # food available from current state
    walls = currentGameState.getWalls()
    capsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    ghostStates = currentGameState.getGhostStates()
    # find distance to the closest food
    min2food = closest_bfs(pos, food.asList(), walls)
    # find distance to the closest capsule
    min2capsule = closest_bfs(pos, capsules, walls)
    # find distance to the closest ghost
    ghosts = [state.getPosition() for state in ghostStates]
    min2ghost = closest_bfs(pos, ghosts, walls)
    ghostPunish = -999999 if min2ghost <= 2 else 0
    # find distance to the closest active ghost
    activeGhosts = [state.getPosition() for state in ghostStates if state.scaredTimer == 0]
    min2activeGhost = closest_bfs(pos, activeGhosts, walls)

    return - 20 * min2food + 1 * min2activeGhost - 10 * min2capsule - 5000 * len(capsules) \
           - 1000 * len([f for f in food.asList()]) + ghostPunish


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = betterEvaluationFunction
        self.depth = int(depth)

    def getAction(self, gameState):
        def maxValue(gameState, currentDepth, alpha, beta):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState), None
            else:
                vMax = - float('inf')
                aMax = None
                for a in gameState.getLegalActions(agentIndex=0):
                    s = gameState.generateSuccessor(agentIndex=0, action=a)
                    v = minValue(gameState=s, agentIndex=1, currentDepth=currentDepth, alpha=alpha, beta=beta)
                    if v > vMax:
                        vMax, aMax = v, a
                    if vMax > beta:
                        return vMax, None
                    alpha = max(alpha, vMax)
                return vMax, aMax

        def minValue(gameState, agentIndex, currentDepth, alpha, beta):
            if not gameState.getLegalActions(agentIndex=0) or currentDepth >= self.depth:
                return self.evaluationFunction(gameState)
            else:
                vMin = float('inf')
                for a in gameState.getLegalActions(agentIndex):
                    s = gameState.generateSuccessor(agentIndex, action=a)
                    if agentIndex >= gameState.getNumAgents() - 1:
                        v, a = maxValue(gameState=s, currentDepth=currentDepth + 1, alpha=alpha, beta=beta)
                    else:
                        v = minValue(gameState=s, agentIndex=agentIndex + 1, currentDepth=currentDepth, alpha=alpha,
                                     beta=beta)
                    vMin = min(vMin, v)
                    if vMin < alpha:
                        return vMin
                    beta = min(beta, vMin)
                return vMin

        v, a = maxValue(gameState, 0, -float('inf'), float('inf'))
        return a

