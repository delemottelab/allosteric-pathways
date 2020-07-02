import allopath
import argparse

parser = argparse.ArgumentParser(epilog='Computing protein residue-cofactor node interactions ' +
	'and cofactor nodefluctuations for expanded network analysis. Annie Westerlund.')

parser = allopath.set_traj_init_parser(parser)
parser = allopath.set_CI_parser(parser)

args, kwargs = allopath.set_CI_args(parser)
CI = allopath.CofactorInteractors(*args, **kwargs)
CI.run()
