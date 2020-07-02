import allopath
import argparse

parser = argparse.ArgumentParser(epilog='Estimating mutual information between residues. Annie Westerlund.')

parser = allopath.set_traj_init_parser(parser)
parser = allopath.set_MI_parser(parser)

args, kwargs = allopath.set_MI_input_args(parser)

MI = allopath.MutualInformation(*args, **kwargs)
MI.run()
