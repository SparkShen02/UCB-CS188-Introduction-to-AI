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

    The code below is provided as a guide. You are welcome to change
    it in any way you s ee fit, so long as you don't touch our method
    headers.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        curFood = currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # find the closet food
        minDisFood = float('inf')
        for food in newFood:
            minDisFood = min(minDisFood, manhattanDistance(newPos, food))

        # find the closet ghost
        minDisGhost = float('inf')
        for ghost in newGhostStates:
            minDisGhost = min(minDisGhost, manhattanDistance(newPos, ghost.getPosition()))

        if minDisGhost <= 1:
            return -float('inf')

        if len(curFood) > len(newFood):
            return float('inf')
        
        # the closer the food, the better
        return successorGameState.getScore() - minDisFood

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
    multi-agent searchers. Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents. Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended. Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        maxScore, maxAction = -float('inf'), None
        agentIndex = 0 # pacman
        for action in gameState.getLegalActions(agentIndex): 
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            curScore = self.value(successorGameState, agentIndex+1, 0) # agentIndex = 1, curDepth = 0
            if curScore > maxScore:
                maxScore = curScore
                maxAction = action
        return maxAction

    def value(self, gameState, agentIndex, curDepth):
        if agentIndex == gameState.getNumAgents(): # next depth
            return self.value(gameState, 0, curDepth+1)
        if agentIndex == 0: # pacman
            return self.maxValue(gameState, agentIndex, curDepth)
        else: # ghosts
            return self.minValue(gameState, agentIndex, curDepth)
    
    def minValue(self, gameState, agentIndex, curDepth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)        
        v = float('inf')            
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            # last ghost in max depth
            if agentIndex == gameState.getNumAgents()-1 and curDepth == self.depth-1:
                v = min(v, self.evaluationFunction(successorGameState))
            else:
                v = min(v, self.value(successorGameState, agentIndex+1, curDepth))
        return v
    
    def maxValue(self, gameState, agentIndex, curDepth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successorGameState, agentIndex+1, curDepth))
        return v
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        maxScore, maxAction = -float('inf'), None
        alpha, beta = -float('inf'), float('inf')
        agentIndex = 0 # pacman
        for action in gameState.getLegalActions(agentIndex): 
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            # agentIndex = 1, curDepth = 0
            curScore = self.value(successorGameState, agentIndex+1, 0, alpha, beta)
            if curScore > maxScore:
                maxScore = curScore
                maxAction = action
            # alpha-beta pruning
            alpha = max(alpha, maxScore)
        return maxAction
    
    def value(self, gameState, agentIndex, curDepth, alpha, beta):
        if agentIndex == gameState.getNumAgents(): # next depth
            return self.value(gameState, 0, curDepth+1, alpha, beta)
        if agentIndex == 0: # pacman
            return self.maxValue(gameState, agentIndex, curDepth, alpha, beta)
        else: # ghosts
            return self.minValue(gameState, agentIndex, curDepth, alpha, beta)
    
    def minValue(self, gameState, agentIndex, curDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)        
        v = float('inf')            
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            # last ghost in max depth
            if agentIndex == gameState.getNumAgents()-1 and curDepth == self.depth-1:
                v = min(v, self.evaluationFunction(successorGameState))
            else:
                v = min(v, self.value(successorGameState, agentIndex+1, curDepth, alpha, beta))
            # alpha-beta pruning
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def maxValue(self, gameState, agentIndex, curDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successorGameState, agentIndex+1, curDepth, alpha, beta))
            # alpha-beta pruning
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

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
        maxScore, maxAction = -float('inf'), None
        agentIndex = 0 # pacman
        for action in gameState.getLegalActions(agentIndex): 
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            curScore = self.value(successorGameState, agentIndex+1, 0) # agentIndex = 1, curDepth = 0
            if curScore > maxScore:
                maxScore = curScore
                maxAction = action
        return maxAction

    def value(self, gameState, agentIndex, curDepth):
        if agentIndex == gameState.getNumAgents(): # next depth
            return self.value(gameState, 0, curDepth+1)
        if agentIndex == 0: # pacman
            return self.maxValue(gameState, agentIndex, curDepth)
        else: # ghosts
            return self.expValue(gameState, agentIndex, curDepth)

    def expValue(self, gameState, agentIndex, curDepth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)
        v = 0
        prob = 1.0 / float(len(legalActions))
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            # last ghost in max depth
            if agentIndex == gameState.getNumAgents()-1 and curDepth == self.depth-1:
                cur_v = float(self.evaluationFunction(successorGameState))
            else:
                cur_v = float(self.value(successorGameState, agentIndex+1, curDepth))
            v += cur_v * prob
        return v
    
    def maxValue(self, gameState, agentIndex, curDepth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successorGameState, agentIndex+1, curDepth))
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    # Useful information you can extract from a GameState (pacman.py)
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood().asList()
    curGhostStates = currentGameState.getGhostStates()

    # distance to the ghost
    disGhost = manhattanDistance(curPos, curGhostStates[0].getPosition())
    if disGhost <= 0:
        return -float('inf')
        
    # find the closet food
    minDisFood = float('inf')
    for food in curFood:
        minDisFood = min(minDisFood, manhattanDistance(curPos, food))

    if len(curFood) == 0:
        return currentGameState.getScore()
    
    return currentGameState.getScore()-minDisFood

# Abbreviation
better = betterEvaluationFunction
