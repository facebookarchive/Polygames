import random

params = {
    "history": [0, 2, 8],
    "nnsize": [4, 8, 16],
    "nb_nets": [8, 16],
    "nb_layers_per_net": [2, 6],
    "global_pooling": [0, 0.1, 0.25],
    "random_features": [0, 4],
#    "layer_repeats": [0, 1],
    "activation_function": ["relu", "gelu", "celu"],
#    "num_rollouts": ["20", "100", "200", "400", "600", "1000"],
#    "tournament_mode": ["true", "false"],
#    "batchnorm_momentum": ["0.01", "0.1"],
#    "layer_dropout": ["0", "0.5"],
    "rewind": [0, 8],
    "randomized_rollouts": ["true", "false"],
    "num_rollouts": [20, 100, 200],
    "tournament_mode": ["true", "false"],
    "lr": ["1e-3", "1e-4", "1e-5"],
    "sample_before_step_idx": [0, 8, 32, 1000],
    "sampling_mcts" : ["true", "false"],
#    "move_select_use_mcts_value": ["true", "false"],
#    "persistent_tree": ["false"],
    "predict_end_state": ["true", "false"],
    "predict_n_states": [0, 1, 4, 16],
}

k = list(params.keys())

all = []

def stringify(p):
  s = "test.sh sweep21"
  for key, v in p.items():
    s += "/" + key + "_" + str(v)
  s += " "
  for key, v in p.items():
    if v == "true":
      s += "--" + key + " true "
    elif v == "false":
      s += ""
    else:
      s += "--" + key + " " + str(v) + " "
  return s

def visit(n, p):
  if n >= len(k):
    #s = "sbatch -p uninterrupted,learnfair -c 80 --gres=gpu:8 --mem 420G --time 420 -J csw test.sh sweep07_rbx1"
    #s = "sbatch -p dev,uninterrupted,learnfair -c 80 --gres=gpu:8 --mem 420G --time 420 -J csw test.sh sweep13"
    s = stringify(p)
    all.append(s)
    return
  for x in params[k[n]]:
    p[k[n]] = x
    visit(n + 1, p)
#visit(0, {})
def generate():
  p = {}
  for k, v in params.items():
    p[k] = v[random.randrange(len(v))]
  s = stringify(p)
  if not s in all:
    all.append(s)

while len(all) < 256:
  generate()

while len(all) > 256:
  all.pop(random.randrange(len(all)))

for v in all:
  print(v)
