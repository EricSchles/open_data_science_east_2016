#Understanding Game AI

##Reinforcement Learning - Basics

The central idea of reinforcement learning, is to learn a strategy over time, based on interacting with an environment.  There are strong correlaries between this notion of environment interaction and graph traversal.  For this talk, we'll make use of Q learning to understand the basics of reinforcement learning.  There are other techniques one could employ, namely SARSA and Temporal difference learning to name a few.  

The basic idea of Q learning is we start with a set of possible choices, and if we've seen all those choices before, we choose the choice that is the best (based on past experience).  The way the program evaluates which new possible choice is the best, is via looking at the rewards associated with each choice.  Let's look at a method to evaluate a set of choices, in the abstract:

```
def choose_action(state,actions,representation):
    q = [evaluate(representation,(state,action)) for action in actions]
    q = remove_all(q,None)
    if q == []: return random.choice(actions)
    else:
        max_q = max(q)
        num_equally_valued_actions = q.count(max_q)
        if num_equally_valued_actions > 1:
            possible_choices = [i for i in range(len(actions)) if q[i] == max_q]
            action_choice = random.choice(possible_choices)
        else:
            action_choice = q.index(max_q)
        return actions[action_choice]
```

Here state, represents the current state of the "player", the actions represent the possible actions, and the representation is the model rewards associated with each state,action pair.  Using the representation, the program chooses which action to execute next.  Here the representation is a dictionary, with keys as state,action pairs, and values as the associated rewards.  

The code should be fairly straight forward: 

The full set of possible actions is observed and their associated rewards are recoded in a list: 

`q = [evaluate(representation,(state,action)) for action in actions]`

This gives us a list "q" which gives us all the relevant rewards, indexed by possible actions.  We then choose the action which yields the greatest reward:

`max_q = max(q)`

If there are multiple rewards that yield the best possible scenario, we simply randomly choose one such reward.  If we have no information about our action set, we simply choose an action at random: 

`if q == []: return random.choice(actions)`

This is more or else the crux of basic Q-Learning.  We only deal with rewards and learn representations of what actions to take in each state.

[Demo Code](https://github.com/EricSchles/open_data_science_east_2016/blob/master/code/basic_rl.py)

As you can see from the above demo we are able to solve the simple "game" of finding positive reward, in a 2-d matrix.  Of course, when dealing with real games, as we will do in later demos there will be some feature transformation we will carry out.  But this gets us pretty close to a realistic representation of how we might carry out simple game AI.

##Adding in Polcies

