# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.defending = False
    self.sideLimit = gameState.getAgentPosition(self.getOpponents(gameState)[0])[0]//2 if self.red else self.start[0]//2 + 1
    self.eatingPower = True
  
  def findPath(self, gameState, isGoal, ghostScaring):
    s, discovered = util.Queue(), set()
    s.push((gameState.getAgentPosition(self.index), [], gameState))
    while s.list:
      v, path, curGameState = s.pop()
      if isGoal(v): return path
      if v not in discovered:
        discovered.add(v)
        if not ghostScaring and v == gameState.getAgentPosition(self.index):
          actions = self.avoidDangActions(gameState)
        else:
          actions = curGameState.getLegalActions(self.index)
        for action in actions:
          if action == 'Stop': continue
          successor = curGameState.generateSuccessor(self.index, action)
          w = successor.getAgentPosition(self.index)
          cur_path = path[:]
          cur_path.append(action)
          s.push((w, cur_path, successor))

  def goCloest(self, gameState):
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
    return random.choice(bestActions)
  
  def inMySide(self, gameState):
    return (self.red and gameState.getAgentPosition(self.index)[0] <= self.sideLimit) or (not self.red and gameState.getAgentPosition(self.index)[0] >= self.sideLimit)

  def avoidDangActions(self, gameState):
    legalActions = gameState.getLegalActions(self.index)
    actions = legalActions[:]
    for action in actions:
      successor = gameState.generateSuccessor(self.index, action)
      succPos =  successor.getAgentPosition(self.index)
      if self.getMazeDistance(succPos, gameState.getAgentPosition(self.index)) > 1:
        legalActions.pop(legalActions.index(action))
        continue
      if self.inMySide(successor):  continue
      if 1 in [self.getMazeDistance(succPos, gameState.getAgentPosition(opponent)) for opponent in self.getOpponents(gameState)]: legalActions.pop(legalActions.index(action))
    if legalActions:  return legalActions
    else:  return ['Stop']
    
  def chooseAction(self, gameState):
    if self.defending:
      print('Defending...')
      return self.goCloest(gameState)
    else:
      if self.eatingPower:
        print('Moving to a power capsule...')
        def isGoal(v):  return v in self.getCapsules(gameState)
        toPowerPath = self.findPath(gameState, isGoal, False)
        if toPowerPath == None: return random.choice(self.avoidDangActions(gameState))
        if len(toPowerPath) == 1: 
          self.stepsAfterPower = 0
          self.eatingPower = False
        return toPowerPath[0]       

      if self.stepsAfterPower <= 20:
        print('Moving 20 steps after eating a power capsule...')
        self.stepsAfterPower += 1
        return self.goCloest(gameState)

      elif not self.inMySide(gameState):
        print('Returning to the red/blue side...')
        def isGoal(v):  return v[0] == self.sideLimit
        toSidePath = self.findPath(gameState, isGoal, True)
        if toSidePath == None: return random.choice(self.avoidDangActions(gameState))
        return toSidePath[0]

      elif self.getCapsules(gameState):
        print('Has returned to the red/blue side and there is still power capsule left.')
        def isGoal(v):  return v in self.getCapsules(gameState)
        toPowerPath = self.findPath(gameState, isGoal, False)
        self.eatingPower = True
        if toPowerPath == None: return random.choice(self.avoidDangActions(gameState))
        return toPowerPath[0]

      else:
        print('Has returned to the red/blue side and there is no power capsule left.')
        self.defending = True
        return 'Stop'

  def getFeatures(self, gameState, action):
    if self.defending:
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)

      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)

      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1

      return features
    else:
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      foodList = self.getFood(successor).asList()    
      features['successorScore'] = -len(foodList)#self.getScore(successor)

      # Compute distance to the nearest food

      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistance
      return features

  def getWeights(self, gameState, action):
    if self.defending:
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
    else:
      return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
