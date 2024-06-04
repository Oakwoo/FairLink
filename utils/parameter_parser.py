import argparse

def parameter_parser():
	"""

	"""

	parser = argparse.ArgumentParser(description = "Run FairLink.")

	parser.add_argument("--folder-path",
						nargs = "?",
						default = "./data/graphs/",
					help = "path to input graph directory")

	parser.add_argument("--file-name",
						nargs = "?",
						default = "pokec_zilinsky_kraj-bytca.pk",
					help = "data file name")
	
	parser.add_argument("--algorithm",
					nargs = "?",
					default = "jac",
				help = "name of link prediciton baseline algorithm to use: jac for Jacard, adar for adamic_adar and  prf for preferential_attachment. Default is jac")
	
	parser.add_argument("--test-size",
						type = float,
						default = 0.8,
					help = "link prediction test size. Default is 0.8")
	
	parser.add_argument("--add-radio",
					type = float,
					default = "0.1",
				help = "precent of test to be predicted .Default is 0.1")
				
	parser.add_argument("--batch-step",
					type = int,
					default = "10",
				help = "batch step size. Default is 10")
				
	parser.add_argument("--acc-bound",
					type = float,
					default = "0.5",
				help = "The bottom line of tolerance for accuracy during GridSearch. Default is 0.5")
	
	parser.add_argument("--gamma",
					type = float,
					default = "1.0",
				help = "importance weight between accuracy and benefit during Reward Update. Default is 1.0")
				
	parser.add_argument("--slot-number",
					type = int,
					default = "20",
				help = "number of generated slots. Default is 20")
				
	parser.add_argument("--epsilon",
					type = float,
					default = "0.3",
				help = "Epsilon-Greedy algorithm hyper-parameter. Default is 0.3")
	
	parser.add_argument("--decay",
					type = float,
					default = "0.8",
				help = "decay parameter during benefit update. Default is 0.8")
	
	parser.add_argument("--cross-validate",
					type = int,
					default = "3",
				help = "cross validate hyper-parameter in grid search. Default is 3")
	
	parser.add_argument('--file', type=open, action=LoadFromFile)
	
	return parser.parse_args()

class LoadFromFile (argparse.Action):
	def __call__ (self, parser, namespace, values, option_string = None):
		with values as f:
			parser.parse_args(f.read().split(), namespace)
