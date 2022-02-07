import os
import random

def parse_netlist(rel_path):
    """Parses the netlist

    Args:
        rel_path (string): relative path to location of the input file

    Returns:
        [list]: list version of netlist
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
        num_cells (int): number of cells in netlist
        num_connections (int): number of connections between cells
        num_rows (int): number of grid rows for circuit to be placed
        num_columns (int): number of grid columns for circuit to be placed
    """
    num_cells, num_connections, num_rows, num_columns = netlist[0]
    return num_cells, num_connections, num_rows, num_columns



## quick thoughts:
## would be easy to have a grid that stored the block number
## on top of that, could have a list that stores the current location of each net (2d, containing x + y locations)
## might be easiest
def calc_distance(block_1, block_2, block_locations):
    distance_x_1, distance_y_1 = block_locations[block_1]
    distance_x_2, distance_y_2 = block_locations[block_2]
    distance = abs(distance_x_1 - distance_x_2) + abs(distance_y_1 - distance_y_2)
    return distance

def init_cell_placements(num_cells, num_rows, num_connections, num_columns, netlist):
    block_locations = {}
    avail_locations = []
    for i in range(num_rows):
        for j in range(num_columns):
            temp_loc = [i, j]
            avail_locations.append(temp_loc)
    random.shuffle(avail_locations)
    for i in range(num_cells):
        block_locations[int(i)] = avail_locations[i]
    
    return block_locations


netlist = parse_netlist("ass2_files/cm138a.txt")
num_cells, num_connections, num_rows, num_columns = get_values(netlist)
block_locations = init_cell_placements(num_cells, num_rows, num_connections, num_columns, netlist)
print(block_locations)


