#!/usr/bin/env python

from __future__ import print_function

import random

from pgen import pgen_opts, ProgGenerator
from pygen.cgen import CodeGenerator


if __name__ == "__main__":

    for i in range(1000):

        pgen = ProgGenerator(pgen_opts, random.Random())

        m = pgen.generate()

        cgen = CodeGenerator()

        program = cgen.generate(m)

        program = program.split('if __name__ == "__main__":')[0]
        program = program.split('from __future__ import print_function')[1]

        filename = "../embeddings/codes/functions/function_" + str(i) +".py"

        file = open(filename, "w")
        file.write(program)
        file.close()