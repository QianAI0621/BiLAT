# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:37:40 2023

@author: Qian
"""

import os
import random
import re

from rdkit import Chem
from tqdm import tqdm

# import scaffoldgraph as sg


def randomize_smi(smi):
    random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))
    return random_equivalent_smiles


smi = "CC(C1=C(C2=CN=C(N=C2N(C1=O)C3CCCC3)NC4=NC=C(C=C4)N5CCNCC5)C)=O"

def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0,m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i,v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


mol = Chem.MolFromSmiles(smi)

Chem.MolToSmiles(mol,canonical = True)

print(randomSmiles(mol))