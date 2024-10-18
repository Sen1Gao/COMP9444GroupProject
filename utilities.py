"""
Group: Team1024
File name: utilities.py
Author: Sen Gao
"""

# RGB format
index_color_mapping={0:(0,0,0),         # void
                     1:(108,64,20),     # dirt
                     2:(255,229,204),   # sand
                     3:(0,102,0),       # grass
                     4:(0,255,0),       # tree
                     5:(0,153,153),     # ploe
                     6:(0,128,255),     # water
                     7:(0,0,255),       # sky
                     8:(255,255,0),     # vehicle
                     9:(255,0,127),     # container/generic-object
                     10:(64,64,64),     # asphalt
                     11:(255,128,0),    # gravel
                     12:(255,0,0),      # building
                     13:(153,76,0),     # mulch
                     14:(102,102,0),    # rock-bed
                     15:(102,0,0),      # log
                     16:(0,255,128),    # bicycle
                     17:(204,153,255),  # person
                     18:(102,0,204),    # fence
                     19:(255,153,204),  # bush
                     20:(0,102,102),    # sign
                     21:(153,204,255),  # rock
                     22:(102,255,255),  # bridge
                     23:(101,101,11),   # concrete
                     24:(114,85,47)     # picnic-table
                     }

# RGB format
color_index_mapping={(0,0,0):0,         # void
                     (108,64,20):1,     # dirt
                     (255,229,204):2,   # sand
                     (0,102,0):3,       # grass
                     (0,255,0):4,       # tree
                     (0,153,153):5,     # ploe
                     (0,128,255):6,     # water
                     (0,0,255):7,       # sky
                     (255,255,0):8,     # vehicle
                     (255,0,127):9,     # container/generic-object
                     (64,64,64):10,     # asphalt
                     (255,128,0):11,    # gravel
                     (255,0,0):12,      # building
                     (153,76,0):13,     # mulch
                     (102,102,0):14,    # rock-bed
                     (102,0,0):15,      # log
                     (0,255,128):16,    # bicycle
                     (204,153,255):17,  # person
                     (102,0,204):18,    # fence
                     (255,153,204):19,  # bush
                     (0,102,102):20,    # sign
                     (153,204,255):21,  # rock
                     (102,255,255):22,  # bridge
                     (101,101,11):23,   # concrete
                     (114,85,47):24     # picnic-table
                     }

def index_lookup(color)->int:
    """
    Get index of color from color_index_mapping where the format of color is RGB.\n
    Therefore, you must convert color format to RGB before you pass variable 'color'\n
    The variable 'color' is a tuple.
    """
    return color_index_mapping[color]
        
    