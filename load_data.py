#This file returns the block of data for particular run
import numpy as np
import re
import itertools

def data_load(filename, Run_num_start, Run_num_end):
        data = []
        i=0
        with open(filename,'r') as infile:
                copy = False
                for line in infile:
                        if line.strip() == "Run #"+str(Run_num_start):
                                copy = True
                        elif line.strip() == "Run #"+str(Run_num_end+1):
                                copy = False
                        elif copy:
                                line = line.split()
                                line = line[1:]
                                data.append(np.array(line))
                                i=i+1
        data = np.array(data)    
        return data


def main():
        data = data_load('13A-no_0',0,1)
        print data.shape,i


main()      

