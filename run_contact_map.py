import allopath
import argparse

parser = argparse.ArgumentParser(epilog='Residue-residue semi-binary contact maps averaged over frames. Annie Westerlund.')

parser = allopath.set_traj_init_parser(parser)
parser = allopath.set_cmap_parser(parser)

args, kwargs = allopath.set_cmap_input_args(parser)

CM = allopath.ContactMap(*args, **kwargs)
CM.run()
