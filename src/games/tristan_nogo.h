/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

extern const int White;
extern const int Black;
extern const int Empty;

const int Dx = 9;
const int Dy = 9;

const int MaxLegalMoves = Dx * Dy;
const int MaxPlayoutLength = 1000;

const int SizeTable = 1048575;  // une puissance de 2 moins 1

extern unsigned long long HashArray[3000];

extern bool useOrderMoves;

extern bool MonteCarloMoveOrdering;

extern bool printGame;

extern unsigned long long nbPlay;

extern int level;

extern timeval stop, start;
extern unsigned long long previousTime;

extern bool useNotLosing;

class NogoBoard;

extern void initHash();

/*
class Player {
 public:
  int player;

  bool operator==(Player p) {
    return (p.player == player);
  }
};
*/

const int MaxMoveNumber = 256 * 2 * (Dx * Dy + 1) + 1;

extern bool useCode;

class NogoMove {
 public:
  int inter, color;
  int code;

  int number() {
    int c = 0;
    if (useCode)
      c = code;
    if (color == White)
      return c * 2 * (Dx * Dy + 1) + inter;
    else
      return c * 2 * (Dx * Dy + 1) + Dx * Dy + 1 + inter;
  }
};

extern double history[MaxMoveNumber];

const int MaxSize = (Dx + 2) * (Dy + 2);
const int MaxIntersections = Dx * Dy;

const int Exterieur = 3;

const int Haut = 0;
const int Bas = 1;
const int Gauche = 2;
const int Droite = 3;

extern int interMove[MaxSize], moveInter[MaxSize];

extern bool ajoute(int* stack, int elt);

class NogoBoard {
 public:
  int start, end, size, dxNogoBoard, dyNogoBoard;
  char board[MaxSize];
  unsigned long long hash;
  int turn;
  int orderMove[MaxLegalMoves];

  NogoMove rollout[MaxPlayoutLength];
  int length;

  int nbVides, vides[MaxSize], indiceVide[MaxSize];
  int nbChaines, chaines[MaxSize], indiceChaine[MaxSize];

  int nbPierres[MaxSize];
  int premierePierre[MaxSize];
  int pierreSuivante[MaxSize];

  int nbPseudoLibertes[MaxSize];
  int premierePseudoLiberte[MaxSize];
  int pseudoLiberteSuivante[4 * MaxSize];
  int pseudoLibertePrecedente[4 * MaxSize];

  NogoBoard() {
    init();
  }

  void init() {
    dxNogoBoard = Dx + 2;
    dyNogoBoard = Dy + 2;
    start = dxNogoBoard + 1;
    end = dxNogoBoard * dyNogoBoard - dxNogoBoard - 1;
    size = Dx * Dy;
    nbVides = 0;
    nbChaines = 0;
    hash = 0;
    for (int i = 0; i < dxNogoBoard * dyNogoBoard; i++) {
      if ((i < start) || (i % dxNogoBoard == 0) ||
          ((i + 1) % dxNogoBoard == 0) || (i >= end))
        board[i] = Exterieur;
      else {
        board[i] = Empty;
        vides[nbVides] = i;
        indiceVide[i] = nbVides;
        interMove[nbVides] = i;
        moveInter[i] = nbVides;
        nbVides++;
      }
    }
    turn = White;
    length = 0;
  }

  void winningMove(int depth, NogoMove m, NogoMove) {
    unsigned long long l = 4ULL << (depth - 70);
    // fprintf (stderr, "depth = %d, history [%d] += %llu, ", depth - 70,
    // m.number (), l);
    history[m.number()] += l;
  }

  int order(NogoMove m) {
    return 4;

    // return rand () % 1000;
    return history[m.number()];
  }

  int legalNogoMoves(int joueur, NogoMove moves[MaxLegalMoves]) {
    NogoMove coup;
    int nb = 0;
    coup.color = joueur;
    for (coup.inter = 0; coup.inter < size; coup.inter++)
      if (board[interMove[coup.inter]] == Empty)
        if (legal(interMove[coup.inter], joueur)) {
          // - ajouter joueur ?
          coup.code = board[interMove[coup.inter] - dxNogoBoard] +
                      4 * board[interMove[coup.inter] - 1] +
                      16 * board[interMove[coup.inter] + 1] +
                      64 * board[interMove[coup.inter] + dxNogoBoard];
          moves[nb] = coup;
          nb++;
        }
    if (useOrderMoves) {
      for (int i = 0; i < nb; i++)
        orderMove[i] = order(moves[i]);
      for (int i = 0; i < nb; i++) {
        int imin = i;
        int o = orderMove[i];
        for (int j = i + 1; j < nb; j++) {
          int o1 = orderMove[j];
          if (o1 < o) {
            imin = j;
            o = o1;
          }
        }
        NogoMove m = moves[i];
        moves[i] = moves[imin];
        moves[imin] = m;
        o = orderMove[i];
        orderMove[i] = orderMove[imin];
        orderMove[imin] = o;
      }
    }
    return nb;
  }

  bool losingMove(NogoMove) {
    return false;
  }

  bool legalMove(NogoMove m) {
    return legal(interMove[m.inter], m.color);
  }

  bool legal(int inter, char color) {
    if (board[inter] != Empty)
      return false;

    char autre = opponent(color);

    // check if capture
    if (board[inter - 1] == autre)
      if (atari(premierePierre[inter - 1]))
        return false;
    if (board[inter + 1] == autre)
      if (atari(premierePierre[inter + 1]))
        return false;
    if (board[inter - dxNogoBoard] == autre)
      if (atari(premierePierre[inter - dxNogoBoard]))
        return false;
    if (board[inter + dxNogoBoard] == autre)
      if (atari(premierePierre[inter + dxNogoBoard]))
        return false;

    // check if suicide
    if (board[inter - 1] == Empty)
      return true;
    if (board[inter + 1] == Empty)
      return true;
    if (board[inter - dxNogoBoard] == Empty)
      return true;
    if (board[inter + dxNogoBoard] == Empty)
      return true;

    int nb = 0;

    if (board[inter - 1] == color) {
      nb += nbPseudoLibertes[premierePierre[inter - 1]] - 1;
      if (board[inter + 1] == color) {
        if (premierePierre[inter - 1] != premierePierre[inter + 1])
          nb += nbPseudoLibertes[premierePierre[inter + 1]] - 1;
        else
          nb--;
        if (board[inter - dxNogoBoard] == color) {
          if ((premierePierre[inter - 1] !=
               premierePierre[inter - dxNogoBoard]) &&
              (premierePierre[inter + 1] !=
               premierePierre[inter - dxNogoBoard]))
            nb += nbPseudoLibertes[premierePierre[inter - dxNogoBoard]] - 1;
          else
            nb--;
          if (board[inter + dxNogoBoard] == color) {
            if ((premierePierre[inter - 1] !=
                 premierePierre[inter + dxNogoBoard]) &&
                (premierePierre[inter + 1] !=
                 premierePierre[inter + dxNogoBoard]) &&
                (premierePierre[inter - dxNogoBoard] !=
                 premierePierre[inter + dxNogoBoard]))
              nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
            else
              nb--;
          }
        } else if (board[inter + dxNogoBoard] == color) {
          if ((premierePierre[inter - 1] !=
               premierePierre[inter + dxNogoBoard]) &&
              (premierePierre[inter + 1] !=
               premierePierre[inter + dxNogoBoard]))
            nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
          else
            nb--;
        }
      } else {
        if (board[inter - dxNogoBoard] == color) {
          if ((premierePierre[inter - 1] !=
               premierePierre[inter - dxNogoBoard]))
            nb += nbPseudoLibertes[premierePierre[inter - dxNogoBoard]] - 1;
          else
            nb--;
          if (board[inter + dxNogoBoard] == color) {
            if ((premierePierre[inter - 1] !=
                 premierePierre[inter + dxNogoBoard]) &&
                (premierePierre[inter - dxNogoBoard] !=
                 premierePierre[inter + dxNogoBoard]))
              nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
            else
              nb--;
          }
        } else if (board[inter + dxNogoBoard] == color) {
          if ((premierePierre[inter - 1] !=
               premierePierre[inter + dxNogoBoard]))
            nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
          else
            nb--;
        }
      }
    } else {
      if (board[inter + 1] == color) {
        nb += nbPseudoLibertes[premierePierre[inter + 1]] - 1;
        if (board[inter - dxNogoBoard] == color) {
          if ((premierePierre[inter + 1] !=
               premierePierre[inter - dxNogoBoard]))
            nb += nbPseudoLibertes[premierePierre[inter - dxNogoBoard]] - 1;
          else
            nb--;
          if (board[inter + dxNogoBoard] == color) {
            if ((premierePierre[inter + 1] !=
                 premierePierre[inter + dxNogoBoard]) &&
                (premierePierre[inter - dxNogoBoard] !=
                 premierePierre[inter + dxNogoBoard]))
              nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
            else
              nb--;
          }
        } else if (board[inter + dxNogoBoard] == color) {
          if ((premierePierre[inter + 1] !=
               premierePierre[inter + dxNogoBoard]))
            nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
          else
            nb--;
        }
      } else {
        if (board[inter - dxNogoBoard] == color) {
          nb += nbPseudoLibertes[premierePierre[inter - dxNogoBoard]] - 1;
          if (board[inter + dxNogoBoard] == color) {
            if ((premierePierre[inter - dxNogoBoard] !=
                 premierePierre[inter + dxNogoBoard]))
              nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
            else
              nb--;
          }
        } else if (board[inter + dxNogoBoard] == color) {
          nb += nbPseudoLibertes[premierePierre[inter + dxNogoBoard]] - 1;
        }
      }
    }

    if (nb > 0)
      return true;

    return false;
  }

  bool atari(int p) {
    if (nbPseudoLibertes[premierePierre[p]] > 4)
      return false;
    int premiere = premierePseudoLiberte[premierePierre[p]];
    int lib = premiere >> 2;
    int suivante = pseudoLiberteSuivante[premiere];

    while (suivante != premiere) {
      if ((suivante >> 2) != lib)
        return false;
      suivante = pseudoLiberteSuivante[suivante];
    }

    return true;
  }

  void ajouteChaine(int chaine1, int chaine2) {
    int pierre = chaine1;

    do {
      premierePierre[pierre] = chaine2;
      pierre = pierreSuivante[pierre];
    } while (pierre != chaine1);

    int suivante = pierreSuivante[chaine1];
    pierreSuivante[chaine1] = pierreSuivante[chaine2];
    pierreSuivante[chaine2] = suivante;

    nbPierres[chaine2] += nbPierres[chaine1];

    chaines[indiceChaine[chaine1]] = chaines[nbChaines - 1];
    indiceChaine[chaines[nbChaines - 1]] = indiceChaine[chaine1];
    nbChaines--;

    if (nbPseudoLibertes[chaine1] > 0) {
      if (nbPseudoLibertes[chaine2] == 0) {
        premierePseudoLiberte[chaine2] = premierePseudoLiberte[chaine1];
      } else {
        int premiereChaine2 = premierePseudoLiberte[chaine2];
        int suivanteChaine2 = pseudoLiberteSuivante[premiereChaine2];
        int premiereChaine1 = premierePseudoLiberte[chaine1];
        int derniereChaine1 = pseudoLibertePrecedente[premiereChaine1];
        pseudoLiberteSuivante[premiereChaine2] = premiereChaine1;
        pseudoLibertePrecedente[premiereChaine1] = premiereChaine2;
        pseudoLiberteSuivante[derniereChaine1] = suivanteChaine2;
        pseudoLibertePrecedente[suivanteChaine2] = derniereChaine1;
      }
    }
    nbPseudoLibertes[chaine2] += nbPseudoLibertes[chaine1];
  }

  void otePseudoLiberte(int chaine, int lib) {
    nbPseudoLibertes[chaine]--;
    if (nbPseudoLibertes[chaine] > 0) {
      int precedente = pseudoLibertePrecedente[lib];
      int suivante = pseudoLiberteSuivante[lib];
      pseudoLiberteSuivante[precedente] = suivante;
      pseudoLibertePrecedente[suivante] = precedente;
      if (premierePseudoLiberte[chaine] == lib)
        premierePseudoLiberte[chaine] = suivante;
    }
  }

  void ajoutePseudoLiberte(int chaine, int lib) {
    if (nbPseudoLibertes[chaine] == 0) {
      premierePseudoLiberte[chaine] = lib;
      pseudoLiberteSuivante[lib] = lib;
      pseudoLibertePrecedente[lib] = lib;
    } else {
      int premiere = premierePseudoLiberte[chaine];
      int suivante = pseudoLiberteSuivante[premiere];
      pseudoLiberteSuivante[premiere] = lib;
      pseudoLibertePrecedente[lib] = premiere;
      pseudoLiberteSuivante[lib] = suivante;
      pseudoLibertePrecedente[suivante] = lib;
    }
    nbPseudoLibertes[chaine]++;
  }

  void play(NogoMove c) {
    // std::cerr << " before:" << std::endl;
    // print(stderr);
    // std::cerr << c.inter % Dx << "," << c.inter / Dx << std::endl;
    joue(interMove[c.inter], c.color);
    // std::cerr << " after:" << std::endl;
    // print(stderr);
    turn = opponent(turn);
    if (length < MaxPlayoutLength) {
      rollout[length] = c;
      length++;
    } else
      fprintf(stderr, "Pb play,");
  }

  void joue(int inter, char color) {
    nbPlay++;
    board[inter] = color;
    if (color == Black)
      hash ^= HashArray[moveInter[inter]];
    else
      hash ^= HashArray[MaxIntersections + moveInter[inter]];

    nbVides--;
    indiceVide[vides[nbVides]] = indiceVide[inter];
    vides[indiceVide[inter]] = vides[nbVides];

    nbPierres[inter] = 1;
    premierePierre[inter] = inter;
    pierreSuivante[inter] = inter;
    nbPseudoLibertes[inter] = 0;

    indiceChaine[inter] = nbChaines;
    chaines[nbChaines] = inter;
    nbChaines++;

    if (board[inter - 1] == color) {
      if (premierePierre[inter] != premierePierre[inter - 1])
        ajouteChaine(premierePierre[inter], premierePierre[inter - 1]);
      otePseudoLiberte(premierePierre[inter], (inter << 2) | Gauche);
    } else if (board[inter - 1] == Empty) {
      ajoutePseudoLiberte(premierePierre[inter], ((inter - 1) << 2) | Droite);
    } else if (board[inter - 1] != Exterieur) {
      otePseudoLiberte(premierePierre[inter - 1], (inter << 2) | Gauche);
    }

    if (board[inter + 1] == color) {
      if (premierePierre[inter] != premierePierre[inter + 1])
        ajouteChaine(premierePierre[inter], premierePierre[inter + 1]);
      otePseudoLiberte(premierePierre[inter], (inter << 2) | Droite);
    } else if (board[inter + 1] == Empty) {
      ajoutePseudoLiberte(premierePierre[inter], ((inter + 1) << 2) | Gauche);
    } else if (board[inter + 1] != Exterieur) {
      otePseudoLiberte(premierePierre[inter + 1], (inter << 2) | Droite);
    }

    if (board[inter - dxNogoBoard] == color) {
      if (premierePierre[inter] != premierePierre[inter - dxNogoBoard])
        ajouteChaine(
            premierePierre[inter], premierePierre[inter - dxNogoBoard]);
      otePseudoLiberte(premierePierre[inter], (inter << 2) | Haut);
    } else if (board[inter - dxNogoBoard] == Empty) {
      ajoutePseudoLiberte(
          premierePierre[inter], ((inter - dxNogoBoard) << 2) | Bas);
    } else if (board[inter - dxNogoBoard] != Exterieur) {
      otePseudoLiberte(
          premierePierre[inter - dxNogoBoard], (inter << 2) | Haut);
    }

    if (board[inter + dxNogoBoard] == color) {
      if (premierePierre[inter] != premierePierre[inter + dxNogoBoard])
        ajouteChaine(
            premierePierre[inter], premierePierre[inter + dxNogoBoard]);
      otePseudoLiberte(premierePierre[inter], (inter << 2) | Bas);
    } else if (board[inter + dxNogoBoard] == Empty) {
      ajoutePseudoLiberte(
          premierePierre[inter], ((inter + dxNogoBoard) << 2) | Haut);
    } else if (board[inter + dxNogoBoard] != Exterieur) {
      otePseudoLiberte(premierePierre[inter + dxNogoBoard], (inter << 2) | Bas);
    }
  }

  int choisitUnCoup(char color) {
    int debut = rand() % nbVides;

    for (int i = debut; i < nbVides; i++)
      if (legal(vides[i], color))
        return vides[i];

    for (int i = 0; i < debut; i++)
      if (legal(vides[i], color))
        return vides[i];

    return -1;
  }

  /**/
  int fastPlayout(int color) {
    for (;;) {
      int pos = choisitUnCoup(color);
      if (pos == -1)
        break;
      joue(pos, color);
      color = opponent(color);
    }
    if (color == Black)
      return 1;
    return 0;
  }
  /**/

  void print(FILE* fp) {
    int i;
    fprintf(fp, "       ");
    for (i = 0; i < dxNogoBoard - 2; i++)
      fprintf(fp, "%-3c", 'A' + i + (i > 7));
    fprintf(fp, "\n");
    fprintf(fp, "    /  ");
    for (i = 0; i < dxNogoBoard - 2; i++)
      fprintf(fp, "-  ");
    fprintf(fp, "\\ \n");
    for (i = start - 1; i <= end; i++) {
      if (((i) % dxNogoBoard == 0))
        fprintf(fp, "%3d |  ", dyNogoBoard - 1 - (i / dxNogoBoard));
      else if (((i + 1) % (dxNogoBoard) == 0))
        fprintf(fp, "| %2d\n", dyNogoBoard - 1 - (i / dxNogoBoard));
      else if (board[i] == Empty)
        fprintf(fp, "+  ");
      else if (board[i] == Black)
        fprintf(fp, "@  ");
      else if (board[i] == White)
        fprintf(fp, "O  ");
      else
        fprintf(fp, "%d  ", board[i]);
    }
    fprintf(fp, "    \\  ");
    for (i = 0; i < dxNogoBoard - 2; i++)
      fprintf(fp, "-  ");
    fprintf(fp, "/ \n");
    fprintf(fp, "       ");
    for (i = 0; i < dxNogoBoard - 2; i++)
      fprintf(fp, "%-3c", 'A' + i + (i > 7));
    fprintf(fp, "\n");
    fprintf(fp, "hash = %llu\n", hash);
  }

  bool won(int color) {
    NogoMove moves[MaxLegalMoves];
    int nb = legalNogoMoves(opponent(color), moves);
    return nb == 0;
  }

  float evaluation(int color) {
    NogoMove moves[MaxLegalMoves];
    int nb = legalNogoMoves(turn, moves);
    if (nb == 0) {
      if (color == turn)
        return -1000000.0;
      else
        return 1000000.0;
    }
    int nbOpponent = legalNogoMoves(opponent(turn), moves);
    if (color == turn)
      return (float)(nb - nbOpponent);
    return (float)(nbOpponent - nb);
  }

  bool terminal() {
    NogoMove moves[MaxLegalMoves];
    int nb = legalNogoMoves(turn, moves);
    return nb == 0;
  }

  int score() {
    if (turn == Black)
      return 1;
    return 0;
  }

  int opponent(int joueur) {
    if (joueur == White)
      return Black;
    return White;
  }

  int playout(int joueur) {
    return fastPlayout(joueur);
    NogoMove listeCoups[MaxLegalMoves];
    while (true) {
      int nb = legalNogoMoves(joueur, listeCoups);
      if (nb == 0) {
        if (joueur == Black)
          return 1;
        else
          return 0;
      }
      int n = rand() % nb;
      play(listeCoups[n]);
      if (length >= MaxPlayoutLength - 20) {
        return 0;
      }
      joueur = opponent(joueur);
    }
  }

  float discountedPlayout(int joueur, int maxLength = MaxPlayoutLength - 20) {
    NogoMove listeCoups[MaxLegalMoves];
    while (true) {
      int nb = legalNogoMoves(joueur, listeCoups);
      if (nb == 0) {
        if (joueur == Black)
          return 1.0 / (length + 1);
        else
          return -1.0 / (length + 1);
        // if (joueur == Black)
        //   return 1.0 * (length + 1);
        // else
        //   return -1.0 * (length + 1);
      }
      int n = rand() % nb;
      play(listeCoups[n]);
      if (length >= maxLength) {
        return 0;
      }
      joueur = opponent(joueur);
    }
  }
};
