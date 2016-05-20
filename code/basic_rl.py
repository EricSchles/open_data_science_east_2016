import random

def remove_all(listing,val):
    while val in listing:
        listing.remove(val)
    return listing

def evaluate(representation,state_action):
    try:
        return representation[state_action]
    except:
        return None
    
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

#column,row
def update_state(action,state,board_height,board_width):
    if action == "up":
        if state[0]+1 == board_height:
            state = (0,state[1])
        else:
            state = (state[0]+1,state[1])
    elif action == "down":
        if state[0] == 0:
            state = (board_height - 1,state[1])
        else:
            state = (state[0]-1,state[1])
    elif action == "left":
        if state[1] == 0:
            state = (state[0],board_width - 1)
        else:
            state = (state[0],state[1]-1)
    elif action == "right":
        if state[1]+1 == board_width:
            state = (state[0],0)
        else:
            state = (state[0],state[1]+1)
    return state        

def train(board_height,board_width):
    representation = {}
    states = []
    for cur_height in range(board_height):
        states.append([random.randint(-5,5) for _ in range(board_width)])
    state = (0,0)
    score = 0
    actions = ("up","down","left","right")
    while score < 100:
        action = choose_action(state,actions,representation)
        state = update_state(action,state,board_height,board_width)
        representation = update_representation(representation,state,action,states[state[0]][state[1]],alpha=0.3)
        score = sum([elem for elem in representation.values() if elem > 0])
    return states,representation

def play(board,representation):
    state = (0,0)
    count = 0
    score = 0
    actions = ("up","down","left","right")
    traversal = []
    path = []
    while score < 100:
        action = choose_action(state,actions,representation)
        state = update_state(action,state,len(board),len(board[0]))
        representation = update_representation(representation,state,action,board[state[0]][state[1]],alpha=0.3)
        traversal.append(board[state[0]][state[1]])
        path.append(state)
        score = sum([elem for elem in traversal if elem > 0])
        count += 1
    return count,traversal,path

print("started training")
board,representation = train(100,100)
print("learned representation")
count,traversal,path = play(board,representation)
print("It took",count)
print(" -> ".join([str(elem) for elem in path]))
