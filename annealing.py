from copy import deepcopy
import os
import random
import numpy as np
import math
import time
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

    # CHANGE FOLLOWING LINE FOR CHANGING THE INFILE
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
        new_netlist[int(i)].append(netlist[i])
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
    for i in range(int(this_netlist[net_number][0][0]) - 1):
        net_cost += calc_distance(this_netlist[net_number][0][i + 1], this_netlist[net_number][0][i + 2], block_locations)
    #print("Cost for net " + str(net_number) + ": " + str(net_cost))
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
    """Prints a grid with the current location of the blocks

    Args:
        block_locations (dict): contains the locations of each block in the format of:
                                block: [[current_cell_x, current_cell_y], [associated netlist nets]]
        num_rows (int): number of rows in cell grid
        num_columns (int): number of columns in cell grid

    Returns:
        (np array): numpy array containing location of blocks
    """
    grid = np.zeros((num_rows, num_columns))
    grid[:,:] = -1
    for i in range(num_blocks):
        grid[block_locations[i][0][0]][block_locations[i][0][1]] = i

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
        #print(new_netlist[block_locations[block][1][i]])
        #print("current_cost: " + str(curr_cost) + ", block_locations: " + str(block_locations[block][1][i]) + ", block: " + str(block))
    return total_cost

def update_netlist_values(block_to_swap_1, block_to_swap_2, netlist, block_locations, num_connections):
    """Updates the cost for the netlist for each of the nets that were effected by a block switch

    Args:
        block_to_swap_1 (int): block 1 that was swapped
        block_to_swap_2 (int): block 2 that was swapped
        netlist (list): the key is the net number, the 1st list associated is the respective net, the 2nd list
                        contains the cost of the given net 
        block_locations (list): [description]
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
    print("Block 1: " + str(block_to_swap_1) + ", Block 2: " + str(block_to_swap_2))

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
    sum = 0
    for i in range(len(netlist)):
        sum += netlist[i][1]
    return sum

def main():

    random.seed(4)
    # Parsing the initial input file and getting the initial cost set up
    netlist = parse_netlist("ass2_files/cm138a.txt")
    num_blocks, num_connections, num_rows, num_columns, netlist = get_values(netlist)
    block_locations = init_cell_placements(num_blocks, num_rows, num_connections, num_columns, netlist)
    netlist, initial_cost = get_initial_cost(netlist, block_locations, num_connections)

    # Can print the grid to get an idea of what the cells look like
    grid = update_grid(block_locations, num_rows, num_columns, num_blocks)

    current_cost = deepcopy(initial_cost)
    temperature = 20
    iterations_per_temp = num_connections
    N_iterations = math.ceil(10 * math.pow(iterations_per_temp, 1.33))
    current_cost_array = np.zeros((21, 1))
    # setting up the environment for blit plotting
    x = np.arange(0, 21, 1)
    
    i = 0
    while (temperature > 2):
        for _ in range(N_iterations):
            # print("CURRENT ITERATION: " + str(i))
            # Ensures we're swapping an occupied cell
            block_to_swap_1 = random.randint(0, num_connections - 1)
            # Can potentially swap occupied cell with an unoccupied cell
            block_to_swap_2 = random.randint(0, num_columns * num_rows - 1)

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
            if ((change_in_cost < 0) or (rand_val < math.exp((-1) * (change_in_cost) / temperature))):
                current_cost = sum_cost(netlist)
                # don't need to update block locations, just need to update netlist with new values for each net:
                netlist, block_locations = update_netlist_values(block_to_swap_1, block_to_swap_2, netlist, block_locations, num_connections)
            else:
                # need to switch the blocks back (netlist was unmodified) back
                block_locations = swap_back(block_locations, block_to_swap_1, 
                                            block_to_swap_2)
            print(current_cost)
        temperature *= .8
        print(temperature)
        time.sleep(3)
    current_cost_array[i] = current_cost
    
    
    # print(netlist)
    # print(block_locations)
    # print(update_grid(block_locations, num_rows, num_columns, num_blocks))
    # print("num rows: " + str(num_rows))
    # print("num cols: " + str(num_columns))

    
main()

