# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import requests
from time import sleep 
from bs4 import BeautifulSoup
import argparse
import os


def lg_connect(thelogin, thepassword):
    ''' Send login request and get session/login cookies. '''
    myresponse = requests.post("http://www.littlegolem.net/jsp/login/", 
            data = {'login': thelogin, 'password': thepassword})
    if myresponse.status_code != requests.codes.ok:
        raise ConnectionError("failed to request littlegolem (lg_connect)")
    if not 'login2' in myresponse.cookies:
        raise ConnectionError("failed to connect as '{}' (lg_connect)".format(thelogin))
    return myresponse.cookies

def lg_clean_str(thestr):
    ''' Remove \t, \n, \r and # in a string. '''
    return thestr.translate({ord(c): None for c in '\t\n\r#'})

def lg_get_onmove_games(thecookies):
    ''' Get the list of games "on move". Return a list [(game id, game name)]. '''
    myresponse = requests.get("http://www.littlegolem.net/jsp/game/", 
            cookies = thecookies)
    if myresponse.status_code != requests.codes.ok:
        raise ConnectionError("failed to request littlegolem (lg_get_onmove_games)")
    myhtml = BeautifulSoup(myresponse.text, "html.parser")
    mydivs = myhtml.select("div.portlet.box.blue-madison")
    if not mydivs:
        return []
    mytrs = mydivs[0].select("tbody")[0].select("tr")
    return [ (lg_clean_str(mytds[0].a.text), mytds[4].text) 
             for mytds in (mytr.select("td") for mytr in mytrs) ]

def lg_get_hsgf(thegid):
    ''' Get the description of a game, in the hsgf format. '''
    myurl = "http://www.littlegolem.net/servlet/sgf/{}/game{}.hsgf".format(thegid, thegid)
    myresponse = requests.get(myurl)
    if myresponse.status_code != requests.codes.ok:
        raise ConnectionError("failed to request littlegolem (lg_get_hsgf)")
    #import pdb;pdb.set_trace()
    return myresponse.text

def lg_play(thecookies, thegid, themove):
    myurl = "http://www.littlegolem.net/jsp/game/game.jsp?sendgame={}&sendmove={}".format(thegid, themove)
    myresponse = requests.post(myurl, cookies = thecookies)
    if myresponse.status_code != requests.codes.ok:
        raise ConnectionError("failed to request littlegolem (lg_play)")

def einstein_convert_txt_to_polygames(myhsgf, gid):
    #requests.get("http://www.littlegolem.net/jsp/game/game.jsp?gid=2127403").text
    myurl = "http://www.littlegolem.net/jsp/game/game.jsp?gid={}".format(gid) 
    myresponse = requests.get(myurl)
    if myresponse.status_code != requests.codes.ok:
        raise ConnectionError("failed to request littlegolem (einstein html)")
    myhtml = BeautifulSoup(myresponse.text, "html.parser")
    imgs = myhtml.select("img")
    assert(len(imgs) == 32)
    # 2 is dice img
    # 3-27 is number img

    num_myhsgf = len(myhsgf.split("/"))

    turn = num_myhsgf % 2
    dice = int(imgs[2]['src'][28])
    assert(dice >= 1 and dice <= 6)
    state_str = ""
    for i in range(25):
      current = None
      num_img = imgs[i + 3]['src']
      color = None
      num = None
      if len(num_img) < 27:
        color = num_img[18]
      else:
        color = num_img[27]
        num = num_img[29]
      if color == 'b':
        current = chr(ord("A") + int(num) - 1)
      elif color == 'r':
        current = chr(ord("a") + int(num) - 1)
      elif color == '0':
        current = "0"
      else:
        print("parse image error, unexpected color")
        assert(False)
      state_str += current
    s = ""
    s += str(dice) + "\n" #input dice value
    s += "m\n"  # we switch to manual mode
    s += "singlemovemode\n" # make one move, print it, exit
    s += "set_" + state_str + str(turn) + "\n" # set state string
    if turn == 0:  # By default we assume that we play first.
        s += "swap\n"  # Please note that this has nothing to do with the pie rule.
    s += "c\n"  # Resume; this is the genmove.
    s += str(dice) + "\n" #input dice value (unused)
    #s += "exit\n" # Safety exit
    return s, state_str, dice, turn


#[Event "Tournament null"] 
#[Site "www.littlegolem.net"] 
#[White "luffy_bot"] 
#[Black "gzero_bot"] 
#[Result "0-1"] 
#1. h2-g3 e7-e6 2. a2-b3 g7-f6 3. b3-c4 f6-e5 4. f2-e3 b7-c6 5. g3-f4 f7-f6 6. g2-f3 h7-g6 7. e1-f2 a7-b6 8. h1-g2 b6-c5 9. d2-c3 h8-g7 10. c2-d3 e8-f7 11. b2-b3 c7-d6 12. b3-b4 a8-b7 13. d3-d4 c5xd4 14. c3xd4 g6-g5 15. f2-g3 g7-g6 16. b4-a5 c6-c5 17. d4xc5 d6xc5 18. d1-d2 d7-c6 19. f3-e4 g6-f5 20. e2-f3 f7-g6 21. a1-b2 d8-c7 22. b2-c3 g6-h5 23. e4xf5 e6xf5 24. d2-d3 h5-h4 25. g3xh4 e5xf4 26. resign 0-1

def breakthrough_convert_txt_to_polygames(txt):
                    turn = 0
                    last_action = None
                    s = "m\n"  # we switch to manual mode
                    s += "singlemovemode\n" # make one move, print it, exit
                    elements = txt.split(".")
                    swapped = False
                    print(txt)
                    for e in elements:
                      if e[0] != " ":
                        continue
                      if e[2:] == "resign])":
                        s += "exit\n"  # We stop everything.
                        continue
                      if len(e) < 6 or (e[3] != "-" and e[3] != "x"):
                        continue
                      es = e.split()
                      e0 = es[0]
                      e1 = es[1]
                      #print(e0)
                      #print(e1)
                      y = ord(e0[0]) - ord('a')
                      z = 8 - int(e0[1])
                      x = ord(e0[3]) - ord('a') - y + 1
                      last_action = str(x) + str(y) + str(z)
                      s += last_action + "\n"
                      turn = 1 - turn
                      if e1 == "*":
                        continue
                      else:
                        y = ord(e1[0]) - ord('a')
                        z = 8 - int(e1[1])
                        x = ord(e1[3]) - ord('a') - y + 1
                        last_action = str(x) + str(y) + str(z)
                        s += last_action + "\n" 
                        turn = 1 - turn
                    if turn == 0:  # By default we assume that we play first.
                        s += "swap\n"  # Please note that this has nothing to do with the pie rule.
                    s += "c\n"  # Resume; this is the genmove.
                    s += "exit\n" # Safety exit
                    #print("turn is" + str(turn))
                    return s, swapped, turn
    
def hex_convert_hsgf_to_polygames(hsgf):
                    turn = 0
                    last_action = None
                    s = "m\n"  # we switch to manual mode
                    s += "singlemovemode\n" # make one move, print it, exit
                    elements = hsgf.split(";")
                    swapped = False
                    print(hsgf)
                    for e in elements:
                      if e[2:] == "resign])":
                        s += "exit\n"  # We stop everything.
                        continue
                      if len(e) < 5:
                        continue
                      if e[2:6] == "swap":
                        # swap is implemented differently in littlegolem and polygames.
                        # we convert by flipping all remaining moves along the long diagonal.
                        swapped = True
                        s += last_action + "\n"
                        turn = 1 - turn
                        continue
                      if (e[0] == "W" or e[0] == "B") and e[1] == "[" and e[4] == "]":
                        x = ord(e[2]) - ord('a')
                        y = ord(e[3]) - ord('a')
                        if swapped:
                          x, y = y, x
                        last_action = chr(ord('a') + x) + str(1 + y)
                        s += last_action + "\n"  # Swap is implemented as replaying the last action in Hex.
                        turn = 1 - turn
                    if turn == 0:  # By default we assume that we play first.
                        s += "swap\n"  # Please note that this has nothing to do with the pie rule.
                    s += "c\n"  # Resume; this is the genmove.
                    s += "exit\n" # Safety exit
                    return s, swapped, last_action
                #(;FF[4]EV[null]PB[leela_bot]PW[gzero_bot]SZ[13]RE[B]GC[ game #2103276]
                #SO[http://www.littlegolem.com];W[ma];B[swap];W[jd];B[ej];W[ji];B[if];
                #W[ck];B[ed];W[di];B[he];W[hf];B[ie];W[dd];B[ec];W[cc];B[cj];W[ef];B[dl];
                #W[ke];B[jg];W[cl];B[dj];W[le];B[kd];W[je];B[lf];W[ig];B[jf];W[hg];B[db];
                #W[ff];B[hb];W[cb];B[resign])

def havannah_convert_hsgf_to_polygames(hsgf, boardsize):
                    turn = 0
                    last_action = None
                    s = "m\n"  # we switch to manual mode
                    s += "singlemovemode\n" # make one move, print it, exit
                    elements = hsgf.split(";")
                    swapped = False
                    print(hsgf)
                    for e in elements:
                      if e[2:] == "resign])":
                        s += "exit\n"  # We stop everything.
                        continue
                      if len(e) < 5:
                        continue
                      if e[2:6] == "swap":
                        # swap is implemented differently in littlegolem and polygames.
                        # we convert by flipping all remaining moves along the long diagonal.
                        swapped = True
                        s += last_action + "\n"
                        turn = 1 - turn
                        continue
                      if (e[0] == "W" or e[0] == "B") and e[1] == "[" and (e[4] == "]" or e[5] == "]"):
                        # in littlegolem x, y = y, x in polygames
                        x = int(e[3]) 
                        if (e[4] != "]"):
                            x = int(e[3:5])
                        y = ord(e[2]) - ord('A') 
                        v = 1
                        if (y >= boardsize):
                            v = y - boardsize + 2
                        last_action = str((x * -1)+(boardsize*2-v)) + "," + str(y)
                        s += last_action + "\n"  # Swap is implemented as replaying the last action in Hex.
                        turn = 1 - turn
                    if turn == 0:  # By default we assume that we play first.
                        s += "swap\n"  # Please note that this has nothing to do with the pie rule.
                    s += "c\n"  # Resume; this is the genmove.
                    s += "exit\n" # Safety exit
                    return s, swapped, last_action

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Play polygames on littlegolem.')
  parser.add_argument('--username', type=str, help='Username for login')
  parser.add_argument('--password', type=str, help='Password for login')
  parser.add_argument('--hex11_model', type=str, help='Model to use for playing hex11pie')
  parser.add_argument('--hex13_model', type=str, help='Model to use for playing hex13pie')
  parser.add_argument('--havannah8_model', type=str, help='Model to use for playing havannah8pie')
  parser.add_argument('--breakthrough_model', type=str, help='Model to use for playing breakthrough')
  parser.add_argument('--havannah10_model', type=str, help='Model to use for playing havannah10pie')
  parser.add_argument('--einstein_model', type=str, help='Model to use for playing einstein')

  args = parser.parse_args()

  try:
    mylogin = args.username
    mypassword = args.password
    mycookies = lg_connect(mylogin, mypassword)
    mygames = lg_get_onmove_games(mycookies)
  except ConnectionError as e:
    print("error:", e)
    exit(1)

  if not mygames:
    print("no turn to play")
  else:
    played = []
    not_played = []
    for mygame in mygames:
      try:
        (mygid, mygname) = mygame
        print("playing game #{} ({})...".format(mygid, mygname))

        myhsgf = lg_get_hsgf(mygid)
        #if "river" not in myhsgf:    # uncomment this if you want to play only against someone with "river" in the name
        #                             # (e.g. rookDriver, a.k.a the other Teytaud)
        #  print("I do not play ", myhsgf)
        #  not_played.append(mygname)
        #  continue
        print("I play ", myhsgf)

        resign_score = -0.99
        model_path = None
        swapped = False
        last_action = None
        turn = None
        state_str = None
        dice = None
        if mygname == "Hex Size 11" and args.hex11_model:
          polygames_commands, swapped, last_action = hex_convert_hsgf_to_polygames(myhsgf)
          model_path = args.hex11_model
        elif mygname == "Hex Size 13" and args.hex13_model:
          polygames_commands, swapped, last_action = hex_convert_hsgf_to_polygames(myhsgf)
          model_path = args.hex13_model
        elif mygname == "Havannah Size 8" and args.havannah8_model:
          polygames_commands, swapped, last_action = havannah_convert_hsgf_to_polygames(myhsgf, 8)
          model_path = args.havannah8_model
        elif mygname == "Breakthrough Size 8" and args.breakthrough_model:
          polygames_commands, swapped, turn = breakthrough_convert_txt_to_polygames(myhsgf)
          model_path = args.breakthrough_model
        elif mygname == "Havannah Size 10" and args.havannah10_model:
          polygames_commands, swapped, last_action = havannah_convert_hsgf_to_polygames(myhsgf, 10)
          model_path = args.havannah10_model
        elif mygname[:7] == "havannah"[:7] and "ize 8" in mygname and args.havannah8_model:
          polygames_commands, swapped, last_action = havannah_convert_hsgf_to_polygames(myhsgf, 8)
          model_path = args.havannah8_model
        elif mygname[:7] == "havannah"[:7] and "ize 10" in mygname and args.havannah10_model:
          polygames_commands, swapped, last_action = havannah_convert_hsgf_to_polygames(myhsgf, 10)
          model_path = args.havannah10_model
        elif mygname[:8] == "EinStein würfelt nicht! 3-points match"[:8] and args.einstein_model:
          #pass in gid to handle specially for Einstein. e.g. parse board and dice value from html
          polygames_commands, state_str, dice, turn = einstein_convert_txt_to_polygames(myhsgf, mygid)
          model_path = args.einstein_model
        else:
          not_played.append(mygname)
          continue
        played.append(mygname)

        # 60 seconds per move. Human first. 8 threads.
        # Singularity command line below might be old fashioned ?
        # command = "singularity exec --nv --overlay overlay.img /checkpoint/polygames/polygames_190927.simg python -m pypolygames human --init_checkpoint " + model_path
        command = "python -m pypolygames human --init_checkpoint " + model_path
        command += " --total_time 60000 --time_ratio 0.01 --human_first --num_actor 8"
        import subprocess
        command = "echo -e \"" + polygames_commands.translate({ord(c): '\\n' for c in '\n'}) + "\" | " + command
        print(command)

        mcts_value = None
        move = None
        if mygname[:8] == "EinStein würfelt nicht! 3-points match"[:8]:
          # Unfortunately EinStein game needs special handling
          # Somehow if I put -e here it gets passed to the program, need to investigate
          command = "echo" + command[7:]
          #print(command)
          result = subprocess.check_output(command, shell=True)
          mcts_value = result.splitlines()[-2].decode()
          move = result.splitlines()[-4].decode()
          print(move)
          move_tokens = move.split()
          origin = move_tokens[-3]
          origin_num = int(origin[1])
          target = move_tokens[-1]
          origin_idx = -1
          for i in range(5):
            for j in range(5):
              idx = i * 5 + j
              if ord(state_str[idx]) - ord('a') == origin_num - 1 and origin[0] == 'x':
                 origin_idx = idx
                 break
              if ord(state_str[idx]) - ord('A') == origin_num - 1 and origin[0] == 'o':
                 origin_idx = idx
                 break
          #print(origin_idx)
          char0 = chr(4 - origin_idx % 5 + ord('a'))
          char1 = chr(4 - origin_idx // 5 + ord('a'))
          char2 = chr(ord('E') - ord(target[0]) + ord('a'))
          char3 = chr(5 - int(target[1]) + ord('a'))
          move = char0 + char1 + char2 + char3
        
        else:
          result = subprocess.check_output(command, shell=True)

                                                                # Maybe "cwd = '..' " ? not if we assume
                                                                # run from the root of polygames
          #print(result)
          (mcts_value, move) = [i.decode() for i in result.splitlines()[-2:]]

        mcts_value = mcts_value.split(":")[-1]
        print("MCTS value: " + mcts_value)
        print("Making move: " + move)
        print("in game " + mygname)

        mymove = None

        if mygname == "Hex Size 11" or mygname == "Hex Size 13":
          print("playing hex")
          x = ord(move[0]) - ord('a')
          y = int(move[1:]) - 1
          if swapped:
            x, y = y, x # in littlegolem, swap is implemented by mirror move and not switching colors
          mymove = chr(ord('a') + int(x)) + chr(ord('a') + int(y))
          if last_action != None and last_action.lower() == move.lower():
            mymove = "swap"
        elif (mygname[:10] == "Havannah Size 8"[:10] or mygname[:4] == "havannah.in"[:4]) and "Size 8" in mygname:
          print("playing havannah 8")
          boardsize = 8  # ONLY FOR SIZE 8
          listmove = move.split(',')
          x = int(listmove[0])
          y = int(listmove[1])
          mymove = chr(ord('a') + y + boardsize - 1) + chr(ord('a') + x + boardsize - 1) # ,11
          if last_action != None and last_action.lower() == move.lower():
            mymove = "swap"
        elif (mygname[:4] == "havannah.in"[:4] or mygname[:10] == "Havannah Size 10"[:10]) and "Size 10" in mygname:
          print("playing havannah 10")
          boardsize = 10  # ONLY FOR SIZE 10
          listmove = move.split(',')
          x = int(listmove[0])
          y = int(listmove[1])
          mymove = chr(ord('a') + y + boardsize - 5) + chr(ord('a') + x + boardsize - 5) # ,11
          if last_action != None and last_action.lower() == move.lower():
            mymove = "swap"
        elif mygname == "Breakthrough Size 8":
          boardsize = 8
          listmove = move.split(',')
          x = int(listmove[0])
          y = int(listmove[1])
          z = boardsize - int(listmove[2]) - 1
          z1 = z + 1 if turn == 0 else z - 1
          mymove = str(y) + str(z) + str(y + x - 1) + str(z1)
        elif mygname[:8] == "EinStein würfelt nicht! 3-points match"[:8]:
          mymove = move
        else:
          print("implement mymove for " + mygname)
          exit(1)

        mymove = mymove.lower()

        # No resign in Einstein, due to the complications on LittleGolem.
        # Anyway, in a multigame (i.e. the best of k games), resigning is complicated and can not save up much time.
        if float(mcts_value) < resign_score and "nStein w" not in mygname:
          print("Resigning!")
          mymove = "resign"

        print("Sending move " + mymove)

        lg_play(mycookies, mygid, mymove)

      except ConnectionError as e:
        print("error:", e)
        exit(1)
    if len(played):
      print("Made a move in ", played)
    if len(not_played):
      print("Did not make a move in ", not_played)

# requests: https://2.python-requests.org/en/master/
# beautifulsoup4: https://www.crummy.com/software/BeautifulSoup/bs4/doc/

