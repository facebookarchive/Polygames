import subprocess
import sys
import random

cmd = "CUDA_CACHE_DISABLE=1 TORCH_USE_RTLD_GLOBAL=1 python -um pypolygames human --init_checkpoint '$$' --num_rollouts 20 --seed " + str(random.randint(1, 65536))

#if len(sys.argv) > 3:
#  cmd += " --device_eval " + sys.argv[3]
if len(sys.argv) > 4:
  cmd += " --num_rollouts " + sys.argv[4]

a = subprocess.Popen(cmd.replace("$$", sys.argv[1]), stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, text=True)
b = subprocess.Popen(cmd.replace("$$", sys.argv[2]), stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, text=True)

moves = ""

def wait(x, parseaction = False):
  #print("wait()")
  global moves
  str = ""
  while str.find("Input action:") == -1:
    if str.find("result for the human human_player:") != -1:
      result = -1 * float(x.stdout.readline().strip())
      print(moves)
      print(result)
      sys.exit(0)
    if parseaction and str.find("MCTS value:") != -1:
      v = x.stdout.readline().strip()
      #print("MCTS value: " + v)
      str = ""
    if parseaction and str.find("Performing action") != -1:
      #print("!action!")
      r = x.stdout.readline().strip()
      moves += r + " "
      return r
    buf = x.stdout.read(2)
    if buf == '':
      raise RuntimeError("EOF")
    str += buf
    #print("str ", str)
  #print("final str ", str)

wait(a)

a.stdin.write("m\n")
a.stdin.flush()
wait(a)
a.stdin.write("reset\n")
a.stdin.flush()
wait(a)
a.stdin.write("printmoves\n")
a.stdin.flush()
wait(a)
a.stdin.write("c\n")
a.stdin.flush()

wait(b)
b.stdin.write("m\n")
b.stdin.flush()
wait(b)
b.stdin.write("reset\n")
b.stdin.flush()
wait(b)
b.stdin.write("swap\n")
b.stdin.flush()
wait(b)
b.stdin.write("printmoves\n")
b.stdin.flush()
wait(b)
b.stdin.write("c\n")
b.stdin.flush()
wait(b)

m = b
n = a

while True:
  action = wait(n, True)
  if action is None:
    #print("no action")
    m, n = n, m
  else:
    m.stdin.write(action + "\n")
    m.stdin.flush()
    #print("Passing action " + action)
