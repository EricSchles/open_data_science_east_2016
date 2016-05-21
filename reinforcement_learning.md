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

This is more or else the crux of basic Q-Learning.  

Now let's dig in a little bit to how our representation get's updated:

```
def update_representation(representation,state,action,reward,alpha=0.3):
    try:
        current_value = representation[(state,action)]
    except:
        current_value = None
    if current_value:
        representation[(state,action)] = current_value + alpha * reward
    else:
        representation[(state,action)] = reward
    return representation
```

Here we do the most naive thing possible, simply discounting the present state against any past state.  In this way, our first impression dominates and any future iterations simply contribute to a lesser extent.  Over many repeated plays, this tends to cancel out and the long term trend will dominate eventually.  

##Playing with update_representation

The long term representation that we use, after many iterations is very important to the learning mechanism we employ.  Intuitively, we can think of this long term representation as a `policy`.  By finding the right way to generate long term views, we can have significant gains in how our AI performs.  

Let's look at some alternative ways to update our representation:

```
def update_representation(representation,counts,state,action,reward,alpha=0.3):
    try:
        current_value = representation[(state,action)]
    except:
        current_value = None
    if current_value:
        representation[(state,action)] = current_value + (1/counts[(state,action)]*( reward - current_value ))
        counts[(state,action)] += 1
    else:
        counts[(state,action)] = 1
        representation[(state,action)] = reward
    return representation,counts
```

In the above code instead of simply allowing our first value to dominate (until the long term), we use the average value of the representation.  Here the current value is the average up until this point and the somewhat confusing looking thing:

`(1/counts[(state,action)]*( reward - current_value ))`

Is the updated additive contribution of the reward: `reward - current_value`, discounted by the total number of terms seen `1/counts[(state,action)]`.  The reason that we need the count for each state,action combination is because we want the average for each state,action pair, rather than the overall state.  However, I assure you, this is the average.  

A more natural, albeit less space efficient version of this code is below:

```
def update_representation(representation,counts,state,action,reward,alpha=0.3):
    try:
        current_value = representation[(state,action)]
    except:
        current_value = None
    if current_value:
	    counts[(state,action)] += [reward]
        representation[(state,action)] = sum(counts[(state,action)])/float(len(counts[(state,action)]))
    else:
        counts[(state,action)] = [reward]
        representation[(state,action)] = reward
    return representation,counts
```

##How do these two policy update mechanisms differ?

In order to fully motivate how updating may affect things, let's construct a simple "game".  

The goal of this game is to get to at least 100 points, in the fewest number of moves.  The player sits on a given square and can move up, down, left, or right.  When the player moves to the new tile, they are given either some treasure or forced to fight a bad guy.  This is represented by positive or negative reward values (internal to the system).  For this first simple game, the player lives forever and keeps going until he has amassed a 100 points of treasure, no negative affects are contributed to the overall score.  

We train the AI to play the game via the following two methods:

```
def train(board,iterations):
    representation = {}
    counts = {}
    state = (0,0)
    score = 0
    actions = ("up","down","left","right")
    for _ in range(iterations):
        while score < 100:
            states = generate_new_board_states(board)
            action = choose_action(state,actions,representation)
            state = update_state(action,state,len(states),len(states[0]))
            representation,counts = update_representation(representation,counts,state,action,states[state[0]][state[1]],alpha=0.3)
            score = sum([elem for elem in representation.values() if elem > 0])
    return representation
```

AND

```
def play(board,representation):
    state = (0,0)
    count = 0
    score = 0
    actions = ("up","down","left","right")
    traversal = []
    path = []
    while score < 100:
        states = generate_new_board_states(board)
        action = choose_action(state,actions,representation)
        state = update_state(action,state,len(board),len(board[0]))
        traversal.append(states[state[0]][state[1]])
        path.append(state)
        score = sum([elem for elem in traversal if elem > 0])
        count += 1
    return count,traversal,path
```

Now that we know how our game is played, let's watch it playing out.

[Demo Code](https://github.com/EricSchles/open_data_science_east_2016/blob/master/code/basic_rl.py)

As you can see from the above demo we are able to solve the simple "game" of finding positive reward, in a 2-d matrix.  Of course, when dealing with real games, as we will do in later demos there will be some feature transformation we will carry out.  But this gets us pretty close to a realistic representation of how we might carry out simple game AI.

Now that we have an explicit game construction, let's investigate how our two representations match up.

[Demo Code](https://github.com/EricSchles/open_data_science_east_2016/blob/master/code/naive_vs_informed.py)

(Best of two)

```
Average basic path length 19.159
Average policy path length 22.128

Best basic path:
(99, 0) -> (98, 0) -> (99, 0) -> (98, 0) -> (97, 0) -> (97, 1) -> (97, 0) -> (96, 0) -> (97, 0) -> (97, 1) -> (96, 1)

Best policy path:
(0, 1) -> (1, 1) -> (2, 1) -> (2, 2) -> (2, 1) -> (1, 1) -> (2, 1) -> (2, 0) -> (3, 0)
```
The above is a typical run of the naive implementation versus one looking at central tendency.  

Now let's look at one other measure of central tendency, the median:

```
def update_representation_median(representation,counts,state,action,reward,alpha=0.3):
    try:
        current_value = representation[(state,action)]
    except:
        current_value = None
    if current_value:
        counts[(state,action)] += [reward]
        representation[(state,action)] = statistics.median(counts[(state,action)])
    else:
        counts[(state,action)] = [reward]
        representation[(state,action)] = reward
    return representation,counts
```

Running all three algorithms gives us:

```
started training
learned representations
Average basic path length 22.3
Average policy path length 19.2
Average median path length 22.3

Best basic path:
(99, 0) -> (0, 0) -> (0, 1) -> (1, 1) -> (0, 1) -> (0, 0) -> (0, 1) -> (0, 2) -> (1, 2) -> (2, 2) -> (3, 2) -> (4, 2) -> (3, 2) -> (3, 1) -> (4, 1)

Best policy path:
(1, 0) -> (1, 1) -> (2, 1) -> (3, 1) -> (4, 1) -> (5, 1) -> (6, 1) -> (7, 1) -> (8, 1) -> (9, 1) -> (10, 1) -> (11, 1) -> (12, 1)

Best median path:
(1, 0) -> (1, 99) -> (1, 98) -> (1, 97) -> (2, 97) -> (1, 97) -> (1, 96) -> (1, 95) -> (0, 95) -> (0, 94) -> (0, 93) -> (0, 92) -> (99, 92) -> (0, 92)
```

[Demo Code](https://github.com/EricSchles/open_data_science_east_2016/blob/master/code/naive_vs_informed.py)

(Best of three)

