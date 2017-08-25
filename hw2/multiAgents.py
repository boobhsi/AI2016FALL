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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        /8"""
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print newPos
        #print newFood
        #print newGhostStates
        #print newScaredTimes
        #return successorGameState.getScore()
        walls = currentGameState.getWalls()
        food_list_old = currentGameState.getFood().asList()
        food_list = newFood.asList()
        cap_list_old = currentGameState.getCapsules()
        cap_list = successorGameState.getCapsules()
        dist = lambda x: util.manhattanDistance(x, newPos)
        pq_for_food = util.PriorityQueueWithFunction(dist)
        pq_for_ghost = util.PriorityQueueWithFunction(dist)
        for i in newGhostStates:
            pq_for_ghost.push(i.getPosition())
        for i in food_list:
            pq_for_food.push(i)
        nearest_ghost_position = pq_for_ghost.pop()
        if len(food_list) != 0: nearest_food_position = pq_for_food.pop()
        if len(cap_list_old) != len(cap_list):
            #print "Eating casulate"
            return 1e70
        for i in range(len(newGhostStates)):
            if newGhostStates[i].getPosition() == nearest_ghost_position:
                nearest_ghost_time = newScaredTimes[i]
                nearest_ghost_index = i
        if successorGameState.isLose():
            #print "Escaping from losing"
            return -1000
        if dist(nearest_ghost_position) <= 3 and nearest_ghost_time == 0:
            #print "Escaping from ghost"
            return -1
        if dist(nearest_ghost_position) <= nearest_ghost_time and nearest_ghost_time != 0:
            #print "Chasing nearest ghost"
            return 1.0 / dist(nearest_ghost_position)
        else:
            if successorGameState.isWin():
                #print "Attempting to win!!!"
                return 1e100
            if len(food_list_old) != len(food_list):
                #print "Attempting to eat"
                return 1e50
            #print "Attempting to find food"
            return 1.0 / dist(nearest_food_position)

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def max(self, a, b):
        if a >= b: return a
        return b

    def min(self, a, b):
        if a <= b: return a
        return b

    def minimax(self, gameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin(): return self.evaluationFunction(gameState)
        if agentIndex == 0:
            pq = util.PriorityQueue()
            for i in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, i)
                result = self.minimax(nextState, 1, depth)
                pq.push((result, i), -result)
            max_eva = pq.pop()
            if depth == 0:
                #print max_eva[0]
                return max_eva[1]
            return max_eva[0]
        else:
            pq = util.PriorityQueue()
            for i in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, i)
                result = 0
                if agentIndex == gameState.getNumAgents() - 1:
                    if depth == self.depth - 1:
                        result = self.evaluationFunction(nextState)
                    else: result = self.minimax(nextState, 0, depth + 1)
                else:
                    result = self.minimax(nextState, agentIndex + 1, depth)
                pq.push((result, i), result)
            min_eva = pq.pop()
            return min_eva[0]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin(): return self.evaluationFunction(gameState)
        if agentIndex == 0:
            v = -1e10
            max_action = "Null"
            for i in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, i)
                result = self.alphabeta(nextState, 1, depth, alpha, beta)
                v = self.max(v, result)
                if v == result: max_action = i
                if v > beta:
                    if depth == 0: return max_action
                    return v
                alpha = self.max(alpha, v)
            if depth == 0:
                #print v
                return max_action
            return v
        else:
            v = 1e10
            for i in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, i)
                if agentIndex == gameState.getNumAgents() - 1:
                    if depth == self.depth - 1:
                        result = self.evaluationFunction(nextState)
                    else: result = self.alphabeta(nextState, 0, depth + 1, alpha, beta)
                else:
                    result = self.alphabeta(nextState, agentIndex + 1, depth, alpha, beta)
                v = self.min(v, result)
                if alpha > v: return v
                beta = self.min(beta, v)
            return v

    def expectMnm(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)
        if agentIndex == 0:
            pq = util.PriorityQueue()
            for i in gameState.getLegalActions(0):
                nextState = gameState.generateSuccessor(0, i)
                result = self.expectMnm(nextState, 1, depth)
                pq.push((result, i), -result)
            max_eva = pq.pop()
            if depth == 0:
                #print max_eva[0]
                return max_eva[1]
            return max_eva[0]
        else:
            result = 0.0
            for i in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, i)
                if agentIndex == gameState.getNumAgents() - 1:
                    if depth == self.depth - 1:
                        result += self.evaluationFunction(nextState)
                    else: result += self.expectMnm(nextState, 0, depth + 1)
                else:
                    result += self.expectMnm(nextState, agentIndex + 1, depth)
            #result = float(result) / float(len(gameState.getLegalActions(agentIndex)))
            return result

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
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState, 0, 0, -1e10, 1e10)

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
        "*** YOUR CODE HERE ***"
        return self.expectMnm(gameState, 0, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      There are 6 features to evaluate the current state: feature_for_food, feature_for_food_count, feature_for_escaping, feature_for_cap(capsule), feature_for_chasing_ghost, and eventually, the score of current state.

      Feature_for_food is obtained from reciprocal of the distance from the pacman to the nearest food. To avoid the possibility of stuck in symmetic food mapping, the pacman will tend to move upward and rihgtward since the feature extracting formula is 1.1 / distance(food), while the opposite is 1.0 / distance(food).

      Feature_for_food_count is obtained from the formulation 1.0 / (1.3 ** len(list_of_food), which encourage pacman to eat food rather than stop as result of the change of the nearest food.

      Feature_for_escaping is obtained from the reciporal of the distance from the pacman to the nearest ghost, which is in the range of 5 manhattan distace from the pacman and not scared.                                                                     
      Feature_for_cap is obtained from the sum of total scared timers multiplied by the reciprocal of the distance between pacman and the nearest food and between foods which are not scared. It will get maximum when all ghosts are near the pacman and have largest scared time. This situation is the best for pacman to eat ghost.

      Feature_for_chasing_ghost is obtained from reciprocal of the distance from pacman to the nearest ghost if the nearest one is eatable.

      All features are multiplied by weights which are tuned by experiment. Finally we sum 5 * current score to get as higher score as possible.

    """
    "*** YOUR CODE HERE ***"
    foodNum = currentGameState.getNumFood()
    foodList = currentGameState.getFood().asList()
    capList = currentGameState.getCapsules()
    nowPos = currentGameState.getPacmanPosition()
    dist = lambda x: util.manhattanDistance(x, nowPos)
    ghostStates = currentGameState.getGhostStates()
    ghostPos = currentGameState.getGhostPositions()

    if currentGameState.isLose(): return 1e-9
    if currentGameState.isWin(): return 1e9

    pq_for_food = util.PriorityQueue()
    for i in range(len(foodList)):
        pq_for_food.push((i, foodList[i]), dist(foodList[i]))

    pq_for_ghost = util.PriorityQueue()
    pq_for_ghost2 = util.PriorityQueue()
    for i in range(len(ghostStates)):
        pq_for_ghost.push((i, ghostPos[i]), dist(ghostPos[i]))
        pq_for_ghost2.push((i, ghostPos[i]), dist(ghostPos[i]))

    feature_for_food = 0.0
    if pq_for_food.isEmpty(): feature_for_food = 0.0
    else:
        nearest_food = pq_for_food.pop()
        if nearest_food[1][0] >= nowPos[0] or nearest_food[1][1] >= nowPos[1]:
            feature_for_food = 1.1 / dist(nearest_food[1])
        else: feature_for_food = 1.0 / dist(nearest_food[1])

    feature_for_chasing_ghost = 0.0
    feature_for_escaping = 5.0
    nearest_ghost = pq_for_ghost.pop()

    if ghostStates[nearest_ghost[0]].scaredTimer >= dist(nearest_ghost[1]):
        feature_for_chasing_ghost += 1.0 / dist(nearest_ghost[1])
    elif dist(nearest_ghost[1]) < 5:
        feature_for_escaping = dist(nearest_ghost[1])

    feature_for_cap = 0.0


    if len(capList) != 0:
        pq_for_cap = util.PriorityQueue()
        for i in range(len(capList)):
            pq_for_cap.push((i, capList[i]), dist(capList[i]))
        totalTimer = 0
        for i in ghostStates:
            totalTimer += i.scaredTimer
        nearest_cap = pq_for_cap.pop()
        temp = 1.0
        tn = pq_for_ghost2.pop()
        if ghostStates[tn[0]].scaredTimer != 0: temp /= dist(tn[1])
        pv = tn
        while(not pq_for_ghost2.isEmpty()):
            tn = pq_for_ghost2.pop()
            if ghostStates[tn[0]].scaredTimer == 0:
                if tn[1] != pv[1]: temp /= util.manhattanDistance(tn[1], pv[1])
                pv = tn
        feature_for_cap = totalTimer * temp


    feature_for_food_count = 1.0 / (1.3 ** len(foodList))

    final_score = 10 * feature_for_food + 100 * feature_for_food_count + 100 * 0.2 * feature_for_escaping + 100 * feature_for_cap +  5 * currentGameState.getScore() + 20 * feature_for_chasing_ghost

    return final_score
# Abbreviation
better = betterEvaluationFunction

