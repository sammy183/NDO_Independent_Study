# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:55:10 2025

XDSM diagram

@author: NASSAS
"""

from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=False)

x.add_system("opt", OPT, r"\text{Optimizer}")
x.add_system("func", FUNC, r"\text{Functions}")
x.add_system("res", FUNC, r"\text{Residuals}")

x.connect("opt", "func", r"x")
x.connect("opt", "res", "x")
x.add_output("opt", r"x^*", side=LEFT)
x.connect("func", "opt", "f, g")
x.connect("res", "opt", "r")

# x.add_output("opt", "x^*, z^*", side=LEFT)
# x.add_output("D1", "y_1^*", side=LEFT)
# x.add_output("D2", "y_2^*", side=LEFT)
# x.add_output("F", "f^*", side=LEFT)
# x.add_output("G", "g^*", side=LEFT)
x.write("SAND")
