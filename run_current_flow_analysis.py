import argparse
import allopath
import numpy as np

parser = argparse.ArgumentParser(epilog='Current flow analysis of residue-residue networks. Annie Westerlund.')

parser = allopath.set_CF_parser(parser)
args, kwargs = allopath.set_CF_args(parser)

CF = allopath.CurrentFlow(*args, **kwargs)
CF.run()
