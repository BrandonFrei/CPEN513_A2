from copy import deepcopy
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np

def parse_netlist(rel_path):
    """Parses the netlist

    Args:
        rel_path (string): relative path to location of the input file

    Returns:
        list: list version of netlist
    """
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

    abs_file_path = os.path.join(script_dir, rel_path)
    # reading in the file
    data = []
    with open(abs_file_path) as rfile:
        data_raw = rfile.readlines()
        for line in data_raw:
            data.append(line.strip())
    data = list(filter(('').__ne__, data))
    split_data = []
    for i in range(int(len(data))):
        temp = (data[i].split())
        temp = list(map(int, temp))
        split_data.append(temp)
    return split_data

def get_values(netlist):
    """gets descriptive values of netlist

    Args:
        netlist (list): list version of netlist

    Returns:
        num_blocks (int): number of blocks in netlist
        num_connections (int): number of connections between cells
        num_rows (int): number of grid rows for circuit to be placed
        num_columns (int): number of grid columns for circuit to be placed
        new_netlist (dict): the key is the net number, the 1st list associated is the respective net, the 2nd list
                            will contain the cost of the given net
    """
    num_blocks, num_connections, num_rows, num_columns = netlist[0]
    netlist = netlist[1:]
    new_netlist = {}
    for i in range(num_connections):
        new_netlist[int(i)] = []
        new_netlist[int(i)].append(netlist[i][1:])
        new_netlist[int(i)].append([])
    return num_blocks, num_connections, num_rows, num_columns, new_netlist


def get_netlist_cost(this_netlist, net_number, block_locations):
    """Gets the cost of a given net in the netlist

    Args:
        this_netlist (dict): [description]
        net_number (int): net we're trying to get the cost for
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]

    Returns:
        int: cost for the net_number net
    """
    net_cost = 0
    # We calculate the distance between each block (e.g. 3 blocks have 2 distances: a->b, b->c)
    # print(this_netlist[net_number][0][0] - 1)
    max_x = -9999999
    min_x = 999999
    max_y = -9999999
    min_y = 999999
    for i in range(int(len(this_netlist[net_number][0]))):
        x_loc = (block_locations[this_netlist[net_number][0][i]][0][0])
        y_loc = (block_locations[this_netlist[net_number][0][i]][0][1])
        if (x_loc > max_x):
            max_x = x_loc
        if (x_loc < min_x):
            min_x = x_loc
        if (y_loc > max_y):
            max_y = y_loc
        if (y_loc < min_y):
            min_y = y_loc
    distance_x = abs(max_x - min_x)
    distance_y = abs(max_y - min_y)
    net_cost = distance_x + distance_y

    # for i in range(int(len(this_netlist[net_number][0])) - 1):
    #     net_cost += calc_distance(this_netlist[net_number][0][i], this_netlist[net_number][0][i + 1], block_locations)
    return net_cost

def calc_distance(block_1, block_2, block_locations):
    """Returns the distance between 2 blocks

    Args:
        block_1 (int): block number
        block_2 (int): block number
        block_locations (list): list containing the location of each block

    Returns:
        int: distance (dx + dy) between the 2 blocks
    """
    location_x_1, location_y_1 = block_locations[block_1][0]
    location_x_2, location_y_2 = block_locations[block_2][0]
    distance = abs(location_x_1 - location_x_2) + abs(location_y_1 - location_y_2)
    return distance


def init_cell_placements(num_blocks, num_rows, num_connections, num_columns, netlist):
    """Places cells into the nxm grid as specified by the input file (random locations)

    Args:
        num_blocks (int): [description]
        num_rows (int): number of rows in cell grid
        num_connections (int): number of nets in cell grid
        num_columns (int): number of columns in cell grid
        netlist (dict): the key is the net number, the 1st list associated is the respective net, the 2nd list
                        will contain the cost of the given net (left for init_cell_placement)

    Returns:
        dict: contains the locations of each block in the format of:
              block: [[current_cell_x, current_cell_y], [associated netlist nets]]
    """
    block_locations = {}
    avail_locations = []
    for i in range(num_rows):
        for j in range(num_columns):
            temp_loc = [i, j]
            avail_locations.append(temp_loc)
    random.shuffle(avail_locations)

    # after this, block locations looks like: block: [[current_cell_x, current_cell_y], [associated netlist nets]]
    for i in range(num_blocks):
        block_locations[int(i)] = []
        block_locations[int(i)].append(avail_locations[i])
        associated_nets = []
        for j in range(num_connections):
            if int(i) in netlist[j][0][1:]:
                associated_nets.append(j)
        block_locations[int(i)].append(associated_nets)
    grid = update_grid(block_locations, num_rows, num_columns, num_blocks)
    avail_locations = []
    # check and see which locations are still blank
    for i in range(num_rows):
        for j in range(num_columns):
            if (grid[i][j] < 0):
                avail_locations.append([i, j])
    # fill in blank block locations
    for i in range(len(avail_locations)):
        relative_index = num_blocks + i
        block_locations[relative_index] = []
        block_locations[relative_index].append(avail_locations[i])
        block_locations[relative_index].append([])
    return block_locations

def update_grid(block_locations, num_rows, num_columns, num_blocks):
    """Returns a grid with the current location of the blocks

    Args:
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]
        num_rows (int): number of rows in cell grid
        num_columns (int): number of columns in cell grid

    Returns:
        grid (np array): numpy array containing location of blocks
    """
    grid = np.zeros((num_rows, num_columns))
    grid[:,:] = -1
    for i in range(num_blocks):
        grid[block_locations[i][0][0]][block_locations[i][0][1]] = i

    return grid

def update_grid_block_swap(grid, block_1, block_2, block_location_1, block_location_2):
    """Updates the grid 

    Args:
        grid (np array): grid containing cell locations
        block_1 (int): 1st block to be swapped
        block_2 (int): 2nd block to be swapped
        block_location_1 (list): location of block 1
        block_location_2 (list): location of block 2

    Returns:
        grid (np array): numpy array containing location of blocks
    """

    grid[block_location_1[0]][block_location_1[1]] = block_2
    grid[block_location_2[0]][block_location_2[1]] = block_1
    return grid

def get_initial_cost(new_netlist, block_locations, num_connections):
    """Generates the initial temperature of the grid

    Args:
        new_netlist (dict): the key is the net number, the 1st list associated is the respective net, the 2nd list
                            contains the cost of the given net
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]
    Returns:
        int: temperature of grid
    """
    total_cost = 0
    for i in range(num_connections):
        curr_cost = get_netlist_cost(new_netlist, i, block_locations)
        new_netlist[int(i)][1] = curr_cost
        total_cost += curr_cost
    return new_netlist, total_cost

def get_block_cost(new_netlist, block_locations, block):
    """Generates the cost of a given block on the grid by summing up all of the nets
       it is a part of.

    Args:
        new_netlist (dict): the key is the net number, the 1st list associated is the respective net, the 2nd list
                            contains the cost of the given net
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]
        block (int): the block we want to get the cost of
    Returns:
        int: temperature of grid
    """
    total_cost = 0
    for i in range(len(block_locations[block][1])):
        curr_cost = new_netlist[block_locations[block][1][i]][1]
        total_cost += curr_cost
    return total_cost

def update_netlist_values(block_to_swap_1, block_to_swap_2, netlist, block_locations, num_connections):
    """Updates the cost for the netlist for each of the nets that were effected by a block switch

    Args:
        block_to_swap_1 (int): block 1 that was swapped
        block_to_swap_2 (int): block 2 that was swapped
        netlist (list): the key is the net number, the 1st list associated is the respective net, the 2nd list
                        contains the cost of the given net 
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]
    Returns:
        netlist (list): netlist with updated cost for updated block locations 
    """
        # don't need to update block locations, just need to update netlist with new values for each net:
    for i in range(len(block_locations[block_to_swap_1][1])):
        net = block_locations[block_to_swap_1][1][i]
        netlist[net][1] = get_netlist_cost(netlist, net, block_locations)
    for i in range(len(block_locations[block_to_swap_2][1])):
        net_2 = block_locations[block_to_swap_2][1][i]
        netlist[net_2][1] = get_netlist_cost(netlist, net_2, block_locations)
    return netlist, block_locations

def swap_back(block_locations, block_to_swap_1, block_to_swap_2):
    temp = deepcopy(block_locations[block_to_swap_1][0])
    block_locations[block_to_swap_1][0] = deepcopy(block_locations[block_to_swap_2][0])
    block_locations[block_to_swap_2][0] = temp
    return block_locations

def swap_block_locations(block_locations, netlist, block_to_swap_1, block_to_swap_2):
    """Swaps the location of 2 blocks in the block_locations list

    Args:
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]
        netlist (list): netlist with updated cost for updated block locations 
        block_to_swap_1 (int): The first block number to be swapped
        block_to_swap_2 (int): The second block number to be swapped

    Returns:
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]
                                updated with swapped blocks
        block1_cost: Sum of the associated netlist values
        block2_cost: Sum of the associated netlist values
        new_block_1_cost: Sum of the associated netlist values post swap
        new_block_2_cost: Sum of the assocciated netlist values post swap
    """
    # print("Block 1: " + str(block_to_swap_1) + ", Block 2: " + str(block_to_swap_2))

    temp_block_1 = deepcopy(block_locations[block_to_swap_1][0])
    temp_block_2 = deepcopy(block_locations[block_to_swap_2][0])

    block2_cost = get_block_cost(netlist, block_locations, block_to_swap_2)
    block1_cost = get_block_cost(netlist, block_locations, block_to_swap_1)

    block_locations[block_to_swap_1][0] = temp_block_2
    block_locations[block_to_swap_2][0] = temp_block_1

    new_block_1_cost = 0
    new_block_2_cost = 0
    for i in range(len(block_locations[block_to_swap_1][1])):
        new_block_1_cost += get_netlist_cost(netlist, block_locations[block_to_swap_1][1][i], block_locations)
    for i in range(len(block_locations[block_to_swap_2][1])):
        new_block_2_cost += get_netlist_cost(netlist, block_locations[block_to_swap_2][1][i], block_locations)


    return block_locations, block1_cost, block2_cost, new_block_1_cost, new_block_2_cost

def sum_cost(netlist):
    """_summary_

    Args:
        netlist (list): netlist (list): netlist with updated cost for updated block locations 

    Returns:
        sum (int): total cost
    """
    sum = 0
    for i in range(len(netlist)):
        sum += netlist[i][1]
    return sum

def range_window_select(block_1_location, window_size, grid, block_locations):
    max_height = len(grid) - 1
    max_width = len(grid[0]) - 1
    x_relative = random.randint(-window_size, window_size)
    y_relative = random.randint(-window_size, window_size)
    while (x_relative == 0 and y_relative == 0):
        x_relative = random.randint(-window_size, window_size)
        y_relative = random.randint(-window_size, window_size)

    new_row = block_1_location[0] + x_relative
    new_column = block_1_location[1] + y_relative

    # ensure our new block is within the x and y bounds of the circuit
    x_location = max( min(new_column, max_width), 0)
    y_location = max( min(new_row, max_height), 0)

    new_block = grid[y_location][x_location]
    if (new_block == -1):
        for i in range(len(block_locations)):
            if(block_locations[i][0] == [y_location, x_location]):
                new_block = i
                break

    return new_block

def main():

    # Parsing the initial input file and getting the initial cost set up
    initial_cost = 999999999
    block_locations = []
    netlist = []
    # Try a few placements to see which is the best to start in
    for i in range(50):
        # Change the next line to change the input file
        new_netlist = parse_netlist("ass2_files/apex1.txt")
        num_blocks, num_connections, num_rows, num_columns, new_netlist = get_values(new_netlist)
        # print(new_netlist)
        new_block_locations = init_cell_placements(num_blocks, num_rows, num_connections, num_columns, new_netlist)
        new_netlist, new_cost = get_initial_cost(new_netlist, new_block_locations, num_connections)
        print(new_cost)
        if (new_cost < initial_cost):
            netlist = deepcopy(new_netlist)
            block_locations = deepcopy(new_block_locations)
            initial_cost = deepcopy(new_cost)
    print("netlist: ")
    print(netlist)
    print("block locations: ")
    print(block_locations)

    # Can print the grid to get an idea of what the cells look like
    grid = update_grid(block_locations, num_rows, num_columns, num_blocks)
    current_cost = deepcopy(initial_cost)
    temperature = 22
    iterations_per_temp = num_connections
    print("num connections: " + str(num_connections))
    N_iterations = math.ceil(20 * math.pow(iterations_per_temp, 1.33))
    print("number of iterations per temp: " + str(N_iterations))
    current_cost_array = []
    current_temperature_array = []

    # plotting
    x = []
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("Cost vs. Number of Steps")
    ax1.set_xlabel("Number of Steps")
    ax1.set_ylabel("Cost (Total Wirelength)", c="red")
    ax2.set_ylabel("Temperature", c="blue")
    accepted_move_percentages = []
    plt.ion()
    i = 0
    temp_threshold = .7

    # 3x3 window (1 on either side)
    window_size = max(num_columns, num_rows)
    
    while (temperature > temp_threshold or temperature == 0):
        number_moves_accepted = 0

        for _ in range(N_iterations):
            # ================ Next 4 lines are for standard annealing (no range windows) ==============
            # # Ensures we're swapping an occupied cell
            # block_to_swap_1 = random.randint(0, num_connections - 1)
            # # Can potentially swap occupied cell with an unoccupied cell
            # block_to_swap_2 = random.randint(0, num_columns * num_rows - 1)
            # ================ end of standard annealing ===============

            # ================ next 4 lines are for range windows==================

            block_to_swap_1 = random.randint(0, num_connections - 1)
            block_1_location = block_locations[block_to_swap_1][0]
            block_to_swap_2 = range_window_select(block_1_location, window_size, grid, block_locations)
            block_2_location = block_locations[block_to_swap_1][0]
            
            #================= end range windows =================================

            # Make sure their values aren't the same
            while block_to_swap_1 == block_to_swap_2:
                block_to_swap_2 = random.randint(0, num_columns * num_rows - 1)

            # swap the blocks
            block_locations, block1_cost, block2_cost, new_block_1_cost, new_block_2_cost = swap_block_locations(block_locations, netlist,
                block_to_swap_1, block_to_swap_2)

            new_cost = new_block_1_cost + new_block_2_cost
            old_cost = block1_cost + block2_cost
            # This might need to be the other way around
            change_in_cost = new_cost - old_cost
            
            rand_val = random.random()
            
            # accepting all good moves, and accepting some of the bad ones
            if ((temperature == 0 and change_in_cost > 0) or ((temperature != 0) and (rand_val < math.exp((-1) * (change_in_cost / temperature))))):
                grid = update_grid_block_swap(grid, block_to_swap_1, block_to_swap_2, block_1_location, block_2_location)
                number_moves_accepted += 1
                current_cost = sum_cost(netlist)
                # don't need to update block locations, just need to update netlist with new values for each net:
                netlist, block_locations = update_netlist_values(block_to_swap_1, block_to_swap_2, netlist, block_locations, num_connections)
            else:
                # need to switch the blocks back (netlist was unmodified) back
                block_locations = swap_back(block_locations, block_to_swap_1, 
                                            block_to_swap_2)

        accepted_move_percentages.append(number_moves_accepted/N_iterations)
        current_cost_array.append(current_cost)
        current_temperature_array.append(temperature)

        # ====================== just for range windows ===========================
        if(number_moves_accepted/N_iterations < .40):
            window_size -= int(math.pow((max(num_columns, num_rows)), 1/3))
        if(number_moves_accepted/N_iterations > .50):
            window_size += int(math.pow((max(num_columns, num_rows)), 1/3))
        if(window_size < 1):
            window_size = 1
        if(window_size > max(num_rows, num_columns)):
            window_size = max(num_rows, num_columns)
        # ====================== range window ends =========================

        print(accepted_move_percentages[-1])
        x.append(i)
        ax1.scatter(i, current_cost, c="Red")
        ax1.plot(x, current_cost_array, c="red", linestyle='-')
        ax2.plot(x, current_temperature_array, c="Blue")
        plt.pause(0.1)
        print("temperature: " + str(temperature))
        temperature *= .9
        if(temperature < temp_threshold):
            break
        if((temperature < temp_threshold) or (accepted_move_percentages[i] < .001)):
            temperature = 0
            N_iterations = math.ceil(100 * math.pow(iterations_per_temp, 1.33))
        

        i += 1
        # print(temperature)
    print("accepted move percentages: " + str(accepted_move_percentages))
    plt.pause(10)
    print(update_grid(block_locations, num_rows, num_columns, num_blocks))
    print("Final Cost: " + str(current_cost_array[-1]))
    
main()

