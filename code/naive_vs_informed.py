import policy_rl
import basic_rl

def best_of_three():
    board_height,board_width = 100,100
    print("started training")
    board = basic_rl.create_board(board_height,board_width)
    proportionality_constant = 0.2
    num_iterations = int(board_height * proportionality_constant)
    basic_representation = basic_rl.train(board,num_iterations)
    policy_representation = policy_rl.train(board,num_iterations)
    median_policy_representation = policy_rl.train_with_median(board,num_iterations)
    print("learned representations")
    basic_counts = 0
    policy_counts = 0
    median_counts = 0
    lowest_basic_count = 10000
    lowest_policy_count = 10000
    lowest_median_count = 10000
    best_basic_path = None
    best_policy_path = None
    best_median_path = None
    num_runs = 10
    for i in range(num_runs):
        #if i%10==0:print(str(i)+"th count") 
        basic_count,basic_traversal,basic_path = basic_rl.play(board,basic_representation)
        policy_count,policy_traversal,policy_path = policy_rl.play(board,policy_representation)
        median_count,median_traversal,median_path = policy_rl.play(board,median_policy_representation)
        basic_counts += basic_count
        policy_counts += policy_count
        median_counts += median_count
        if lowest_basic_count > basic_count:
            best_basic_path = basic_path
            lowest_basic_count = basic_count
        if lowest_policy_count > policy_count:
            best_policy_path = policy_path
            lowest_policy_count = policy_count
        if lowest_median_count > median_count:
            best_median_path = median_path
            lowest_median_count = median_count
    print("Average basic path length",basic_counts/num_runs)
    print("Average policy path length",policy_counts/num_runs)
    print("Average median path length",median_counts/num_runs)
    print("Best basic path:")
    print(" -> ".join([str(elem) for elem in best_basic_path]))
    print("Best policy path:")
    print(" -> ".join([str(elem) for elem in best_policy_path]))
    print("Best median path:")
    print(" -> ".join([str(elem) for elem in best_median_path]))

def best_of_two():
    board_height,board_width = 100,100
    print("started training")
    board = basic_rl.create_board(board_height,board_width)
    proportionality_constant = 0.2
    num_iterations = int(board_height * proportionality_constant)
    basic_representation = basic_rl.train(board,num_iterations)
    policy_representation = policy_rl.train(board,num_iterations)
    print("learned representations")
    basic_counts = 0
    policy_counts = 0
    lowest_basic_count = 10000
    lowest_policy_count = 10000
    best_basic_path = None
    best_policy_path = None
    num_runs = 10
    for i in range(num_runs):
        #if i%10==0:print(str(i)+"th count") 
        basic_count,basic_traversal,basic_path = basic_rl.play(board,basic_representation)
        policy_count,policy_traversal,policy_path = policy_rl.play(board,policy_representation)
        basic_counts += basic_count
        policy_counts += policy_count
        if lowest_basic_count > basic_count:
            best_basic_path = basic_path
            lowest_basic_count = basic_count
        if lowest_policy_count > policy_count:
            best_policy_path = policy_path
            lowest_policy_count = policy_count
    print("Average basic path length",basic_counts/num_runs)
    print("Average policy path length",policy_counts/num_runs)
    print("Best basic path:")
    print(" -> ".join([str(elem) for elem in best_basic_path]))
    print("Best policy path:")
    print(" -> ".join([str(elem) for elem in best_policy_path]))

best_of_two()
