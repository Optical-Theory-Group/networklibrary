import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Generator, Iterator, Union

import cloudpickle
import h5py

from complex_network._dict_hdf5 import recursively_save_dict_contents_to_group


# ------------------------------------------------------
# Core ensemble generator functions
# ------------------------------------------------------
def apply_cloudpickle(fn: callable, /, *args, **kwargs) -> callable:
    """
        Helper function to override standard pickling function used by concurrent.futures, with the aim to make the
        code more robust and general.
        Takes a pickled function, deserializes it and then calls it using specified arguments
    """
    fn = cloudpickle.loads(fn)
    return fn(*args, **kwargs)


class CloudpickleProcessPoolExecutor(cf.ProcessPoolExecutor):
    """
        Helper class to override standard pickling function used by concurrent.futures when jobs are submitted to 
        the pool, with the aim to make the code more robust and general.
    """

    def submit(self, fn: callable, /, *args, **kwargs) -> callable:
        """
            Overrides submit function, by first pickling the supplied function using cloudpickle, before it is
            submitted to the pool. This serialized function is deserialized via parallel calls to apply_cloudpickle as
            opposed to direct calls to fn.
        """
        return super().submit(apply_cloudpickle, cloudpickle.dumps(fn), *args, **kwargs)


def generator_worker(queue: 'mp.Manager.Queue', i: Any, generator: callable, **genargs) -> dict:
    """
        A worker function that runs a generator function for each input value and puts the resulting
        value into the data queue ready for writing. Also returns the same data. Returned values are wrapped in a
        dictionary identified by the realisation id, i.

        Parameters:
            queue: Manager.Queue object used for queue of data. Results are added to queue upon completion
            i: realisation id number used to distinguish different calculations
            generator: callable function that is submitted to worker. First argument must be i
            genargs: dictionary of keyword arguments to be supplied to generator upon calling.

        Returns:
            {i: generator(i, **genargs)}
    """
    # Call the generator function with the current input value and keyword arguments
    result = generator(i, **genargs)
    queue.put({i: result})
    return {i: result}


def writer_worker(q: 'mp.Manager.Queue', filename: str, writemode: str = 'w', processor: callable = None,
                  **prockwargs) -> None:
    """
        A worker function responsible for writing data that is placed onto a queue to a hdf5 file. Will abort when data
        entry on queue is explicitly set to None. An empty queue by itself does not cause worker to stop.

        Parameters:
            queue: Manager.Queue object used for queue of data. Results to write are read from the queue
            filename: path and filename of file to which to save data.
            writemode: str
                'w' to delete any existing datafile and store data in a new version
                'a' to append data to an existing datafile. Checks for repeated values of i (see generator) are made to
                    avoid repeated calculations. Note currently no checks on consistent data structures between new and
                    existing data are made.
            processor: func
                Function called on dictionary output from generator function before saving. processor should return a
                single dictionary consisting of data that should be saved in hdf5 file.
            prockwargs:
                Dictionary of keyword arguments used in call to processor function

    """
    if writemode == 'w' and os.path.exists(filename):
        print('\n Warning: Overwriting existing datafile.')
        os.remove(filename)  # delete existing datafile before beginning

    # open the HDF5 file for writing
    with h5py.File(filename, 'a', driver=None) as h5file:
        while True:
            if not q.empty():
                data = q.get()
                # Sentinel value to signal end of data
                if data is None:
                    print("Termination sentinel detected - closing writer...")
                    break

                # Process data with processor function if supplied
                if processor is not None:
                    key = list(data.keys())[0]
                    processed_data = {key: processor(data[key], **prockwargs)}
                else:
                    processed_data = data

                # Write data to file
                recursively_save_dict_contents_to_group(h5file, '/', processed_data)


def remove_existing(ivals: Union[Iterator, Generator, list], filename: str, verbose: Union[int, float, bool] = 1) -> (
        Generator, int):
    """
        Checks for the datafile specified and extracts the run id for any data already stored. These ids are removed
        from the input run ids if they exist to avoid repeated runs.

        Input
        _______
        ivals: generator, list, iterable:
            Describes all values of i for which generator function will be called.
        filename: str
            Filename of hdf5 file which should be checked for existing realisations
        verbose: int
            If >0 information on number of skipped realisations will be reported

        Returns
        _______
        ivals_filter:  generator, list, iterable:
            As the input, but with any values of i that have already been stored removed.
        skipped: int:
            Number of realisations that already exist and have been removed from iterable/generator

    """
    # get existing realisation ids from file
    skipped = 0
    print(filename,os.path.exists(filename))
    if os.path.exists(filename):
        with h5py.File(filename, 'r', driver=None) as h5file:
            existing_realisations = [int(k) for k in h5file.keys()]

            skipped = len(existing_realisations)
            # progresscount += skipped
            if (skipped > 0) and verbose:
                print('\n ---- Skipping {} realisations that already exist.'.format(skipped))

        ivals_filter = (i for i in ivals if i not in existing_realisations)
    else:
        ivals_filter = (i for i in ivals)

    return ivals_filter, skipped


def report_progress(progresscount: int, totalrealisations: int, starttime: datetime = None, filename: str = None,
                    verbose: Union[int, float, bool] = 1) -> int:
    """
        Displays progress bar for current progress.

        Input:
            progresscount: int
                Current counter recording progress.
            totalrealisations: int
                Total number of realisations to be calculation
            starttime: datetime
                Time at which simulation was started
            filename: str
                Path/filename of datafile to which data is being written
            verbose: int
                0: no output messages
                1: output progress bar with estimated run times
                2: output progress bar with estimated run times and data file size info

        Return:
            progresscount: int
                Input value incremented by 1.
    """
    # update progress trackers and inform user if they have so requested
    progresscount += 1
    if verbose:
        runtime = datetime.now() - starttime  # time.time() - starttime
        runtime -= timedelta(microseconds=runtime.microseconds)
        runtimesecs = runtime.total_seconds() if runtime.total_seconds() > 0 else .1  #
        remaintime = (runtime / progresscount) * (totalrealisations - progresscount)

        strmsg = '{}/{}' \
                 '   in   : {} ({}/s  eta: {}).'.format(progresscount, totalrealisations,
                                                        runtime, progresscount / runtimesecs, remaintime)
        if verbose > 1:
            if os.path.exists(filename):
                currentsize = os.path.getsize(filename) / (1024 * 1024)
            else:
                currentsize = 0
            predictedsize = (currentsize / progresscount) * (totalrealisations)

            strmsg = strmsg + '    Data  : {:.3f} MB. ({:.3f} kB/s  est.:  {:.3f} MB)  '.format(currentsize,
                                                                                                currentsize / runtimesecs * 1024,
                                                                                                predictedsize)

        update_progress(progresscount / totalrealisations, strmsg)
        return progresscount


def update_progress(progress: float, status: str = '', barlength: int = 20):
    """
    Prints a progress bar to console

    Parameters
    ----------
    progress : float
        Variable ranging from 0 to 1 indicating fractional progress.
    status : str, optional
        Status text to suffix progress bar. The default is ''.
    barlength : str, optional
        Controls width of progress bar in console. The default is 20.
    """

    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = 'error: progress var must be float\r\n'
    if progress < 0:
        progress = 0
        status = 'Halt...\r\n'
    if progress >= 1:
        progress = 1
        status += ' Done.\r\n'

    # count number of lines in status
    num_lines = status.count('\n')
    # include special characters to move cursor back to start of text to get updating versus new text
    prelim = '\r'
    for n in range(num_lines + 2):
        prelim = prelim + '\033[F'

    block = int(round(barlength * progress))
    text = prelim + '\rPercent: [{0}] {1:.2f}% {2}'.format('#' * block + '-' * (barlength - block), progress * 100,
                                                           status)
    sys.stdout.write(text)
    sys.stdout.flush()


def generate_ensemble(filename: str, ivals0: Union[list, Iterator, Generator],
                      generator: callable, generator_args: dict = None,
                      processor: callable = None, processor_args: dict = None,
                      writemode: str = 'a', verbose: int = 1):
    """
        Main public function designed to run a user specified function multiple times with distinct inputs
        in parallel. As data is returned from each function call it is written in tandem to a user specified hdf5
        formatted file. Optionally a data processing function can also be passed into this class upon initialisation for
        preprocessing of data returned by generator function before it is stored.

        Inputs
        --------
            filename: str
                Path to the file to which data will be stored. It will be created if it doesn't exist.
            ivals: generator, list, iterable:
                Describes all values of i for which generator function will be called. These should be unique and
                designate distinct realisations
            generator: func
                Function to be run in parallel. Call to function is of the form generator(i, **generator_args).
                The single input argument i varies between different runs (see documentation for
                ParallelHDF5Writer.run(...) below). Output from generator should be a single dictionary.
            generator_args: dict
                Dictionary of keyword arguments used in call to generator function
            processor: func
                Function called on dictionary output from generator function before saving. processor should return a
                single dictionary comprising of data that should be saved in hdf5 file.
            processor_args:
                Dictionary of keyword arguments used in call to processor function
            writemode: str
                'w' to delete any existing datafile and store data in a new version
                'a' to append data to an existing datafile. Checks for repeated values of i (see generator) are made to
                    avoid repeated calculations. Note currently no checks on consistent data structures between new and
                    existing data are made.
            verbose: int
                0: no output messages
                1: output progress bar with estimated run times
                2: output progress bar with estimated run times and data file size info

    """
    # define realisations to run
    ivals, skipped = remove_existing(ivals0, filename, verbose=verbose) if writemode == 'a' else (ivals0, 0)

    ## TO DO: fix number of workers.
    max_workers = mp.cpu_count()
    print("Maximum number of workers:", max_workers)
    with CloudpickleProcessPoolExecutor() as pool:
        with mp.Manager() as manager:
            # initialise progress tracking
            progresscount = skipped
            totalrealisations = sum(1 for _ in ivals0)
            starttime = datetime.now()

            # Start writer process
            print("Starting data writer...")
            q = manager.Queue()
            writer = pool.submit(writer_worker, q, filename, writemode, processor, **processor_args)

            # Start data generator processes
            print("Starting data generators...")
            computer = [pool.submit(generator_worker, q, i, generator, **generator_args) for i in ivals]

            for cfuture in cf.as_completed(computer):
                progresscount = report_progress(progresscount, totalrealisations, starttime, filename, verbose=verbose)
                if cfuture._exception is not None:
                    raise RuntimeError(cfuture._exception)

            cf.wait(computer)
            q.put(None)

            wresult = writer.result()
            if writer._exception is not None:
                raise RuntimeError(writer._exception)
            print("Data writing completed.")
