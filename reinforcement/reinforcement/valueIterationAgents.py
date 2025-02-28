# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        eachState = mdp.getStates()
        counter = -1
        sub_dic = util.Counter()
        #print "initialize"
        for i in eachState:
            #actions = self.mdp.getPossibleActions(i)
            #temp_dic = util.Counter()
            #for j in actions:
            #    temp_dic[j] = 0
            sub_dic[i] = 0.0
        #print sub_dic
        #print "iteration start"
        while counter < self.iterations:
            #for i in sub_dic:
            #    self.values[i] = sub_dic[i].copy()
            self.values = sub_dic.copy()
            #print "copy completed"
            for i in eachState:
                if not self.mdp.isTerminal(i):
                    actions_for_now = self.mdp.getPossibleActions(i)
                    act_dic = util.Counter()
                    for j in actions_for_now:
                        #temp_value = 0.0
                        #temp = self.mdp.getTransitionStatesAndProbs(i, j)
                        act_dic[j] = self.getQValue(i, j)
                        #for k in temp:
                        #    if not self.mdp.isTerminal(k[0]):
                        #        temp_value += k[1] * self.values[k[0]][self.values[k[0]].argMax()]
                        #sub_dic[i][j] = self.mdp.getReward(i, None, None) + self.discount * temp_value
                    sub_dic[i] = act_dic[act_dic.argMax()]
            counter += 1
            #print counter
        #print "iteration over"
        #print self.values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        dic = util.Counter()
        for i in nextStates:
            dic[i[0]] = i[1] * (self.mdp.getReward(state, action, i[0]) + self.discount * self.getValue(i[0]))
        return dic.totalCount()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state): return None
        actions = self.mdp.getPossibleActions(state)
        dic = util.Counter()
        for i in actions:
            nextStates = self.mdp.getTransitionStatesAndProbs(state, i)
            temp_dic = util.Counter()
            for j in nextStates:
                temp_dic[j[0]] = j[1] * self.getValue(j[0])
            dic[i] = temp_dic.totalCount()
        return dic.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
