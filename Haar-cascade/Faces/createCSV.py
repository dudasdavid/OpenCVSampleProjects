#!/usr/bin/env python

import sys
import os.path
import os

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":

    if len(sys.argv) != 2:
        # print "usage: create_csv <base_path>"
        # sys.exit(1)
        BASE_PATH = os.getcwd()
    else:
        BASE_PATH=sys.argv[1]
        

    file = open("faces.csv", "w")
    # BASE_PATH=sys.argv[1]
    SEPARATOR=";"

    # print BASE_PATH
    
    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        # print dirnames
        for subdirname in sorted(dirnames):
            # print subdirname
            subject_path = os.path.join(dirname, subdirname)
            if "src" in str(subject_path):continue
            for filename in sorted(os.listdir(subject_path)):
                abs_path = "%s/%s" % (subject_path, filename)
                print "%s%s%d" % (abs_path, SEPARATOR, label)
                file.write("%s%s%d\n" % (abs_path, SEPARATOR, label))
            label = label + 1
            
            
            
            