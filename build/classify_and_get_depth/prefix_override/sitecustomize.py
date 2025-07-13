import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/csy/My_baby_blue/Road_fixer/install/classify_and_get_depth'
