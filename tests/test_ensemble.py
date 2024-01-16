from complex_network.hd5fensemble import generate_ensemble
from tests.demo import *

if __name__ == '__main__':
    filename = '../data/test_ensemble_data.h5'

    generator = atest_gen_mod
    processor = atest_proc_mod

    # dc = DemoClass()
    # generator = dc.atest_gen
    # processor = dc.atest_proc

    generator_args = {'j': 4}
    processor_args = {'removek': True}
    writemode = 'a'
    verbose = 1

    ivals0 = range(200)

    generate_ensemble(filename, ivals0,
                      generator, generator_args,
                      processor, processor_args,
                      writemode, verbose)
