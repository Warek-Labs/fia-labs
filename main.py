from production import *
from rules_example_zookeeper import ZOOKEEPER_RULES, ZOOKEEPER_DATA
import re

from utils import islist

if __name__ == '__main__':
    rules = ZOOKEEPER_RULES
    data = ZOOKEEPER_DATA

    while True:
        print('Would you like forward of backward chaining?')
        res = input().lower()

        if res == 'f' or res == 'forward' or res == 'forward chaining':
            forward_chain(rules, data)
        elif res == 'b' or res == 'backward' or res == 'backward chaining':
            pass

        print('====================================================')
        input('Press enter to restart')
