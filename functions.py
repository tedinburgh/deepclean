# ---------------------------------------------------------------------------
# MODULES

import numpy as np
import h5py
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from numpy.lib import recfunctions

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_h5(inputfilename, **kwargs):
    '''
    Reads in a hdf5 file, identifies datasets and performs checks.

    Reads in a hdf5 file, with certain requirements on the file
    structure, identifies the datasets within the file and checks that
    each dataset (and associated arrays) belongs to a group within the 
    file. It can also perform checks on the specific parts of the file
    to ensure the data format is as required (see the end of this 
    docstring for requirements on the structure of the file).

    Parameters
    ----------
    inputfilename: string
        The full path and filename of the hdf5 file. See end for 
        requirements of the structure of the file.

    Keyword arguments
    ----------
    verbose: boolean
        If False, then statements of progress/warnings are suppressed. 
        Note that some output is still necessarily printed. By default,
        True.
    checks: boolean
        If True, then a number of checks are performed to ensure the
        data is in the correct format. By default, True.
        The checks include checking that '.index' arrays are self-
        consistent, that '.index' and '.label' arrays are numpy
        ndarrays and that all data that should be numeric is numeric.

    Example usage
    ----------
    load_h5(inputfilename)
    load_h5(inputfilename, verbose = False, checks = False)

    
    Description of input file requirements
    ----------
    The input file must be a hdf5 file with certain requirements on the
    structure - note, hdf5 files from the ICU+ (this function has been 
    written with these files in mind) have this structure.
    It must include the following:
    ----------
    DataAvailable - attribute/list
        A list of datasets in the file. 
    [dataset] - arrays
        A dataset is an N x 1 array of y values and the corresponding 
        t values are encoded in an associated array, as described
        below.
        Each dataset listed must exist within some group of the hdf5 
        file. Note for the ICU+ data, some datasets exist within the
        different groups, such as 'waves' and 'numerics'.
    [dataset].index - arrays
        Each dataset must have an associated '.index' array. For
        example, if 'dataset1' is a dataset in the file, then
        'dataset1.index' must also exist within the same group of the
        file. The data is continuous and sampled at constant
        frequencies in a finite number of blocks or chunks, potentially
        with gaps between blocks (for example such as when a patient is
        moved into theatre for ICU+ data). The frequency of the data
        may also change between blocks.
        It lists the following:
            startidx - int
            starttime - int/float
            length - int
            frequency - float
        These correspond to the index of the dataset corresponding to 
        the new block of data, the start time of that block, the
        number of data points in that block and finally the frequency
        of the data. Note that the starttime may be given as an int in
        microseconds rather than a float in seconds. 
    ----------
    The file should preferably also include:
    ----------
    dataStartTimeUnix - attribute/float
        The start time of the data.
    dataEndTimeUnix - attribute/float
        The end time of the data. If these do not exist, then they can 
        be a proxy can be determined from the '.index' arrays.
    definitions/indexStruct - array
        Within the group 'definitions', the array 'indexStruct' will 
        define the variables used in each '.index' array more precisely
        than in the above, including an indication of the units of time
        used, which is needed to determine the time of each data point
        (for example, ICU+ data uses microseconds rather than seconds 
        in this array). 
    ----------
    The file may also include:
    ----------
    [dataset].label - arrays
        An associated label array, similar to the index array, may also
        be included if the data has been previously been labelled.
        It lists the following:
            startidx - int
            length - int
            label - int
        The label is a code assigned to a labelled section. The label
        codes should be negative integers for labels applicable to all
        datasets and positive integers for labels applicable to 
        specific datasets. The labels specific to a particular dataset, 
        identified by m, will be within 0-99 + m * 100.
        This array may also additionally include:
            starttime - int/float
        If this array exists, then the file should preferably also
        include a 'labelStruct' array and a 'labelRef' array inside the
        group 'definitions'. The former is as described above for 
        'indexStruct', the latter gives details of the label 
        corresponding to each label code.
    ----------
    Similar arrays may also exist for '.quality', which is a similar 
    identifier to '.label'. Other datasets and definitions may also
    exist that are useful for another analysis. See the hdf5 file 
    for more information.
    '''

    # ------------------------------------------------------------------------
    # MODULES

    # import numpy as np
    # import h5py

    # ------------------------------------------------------------------------
    # KWARGS

    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose = True

    if 'checks' in kwargs.keys():
        checks = kwargs['checks']
    else:
        checks = True

    # ------------------------------------------------------------------------
    # OPEN FILE

    n = np.size(inputfilename)
    if n != 1:
        err = 'Only one input file should be passed through this function'
        raise AssertionError(err)

    if verbose:
        print('--> Opening file' + inputfilename)

    # check that the file exists
    try:
        data = h5py.File(inputfilename, 'r')
    except IOError:
        raise FileNotFoundError
    # an alternative might be using
    ## with h5.File(inputFileName, 'r') as data:
    ##   # stuff involving data
    # apparently this can be cleaner and avoid memory leaks but not 
    #   sure if it's better

    # ------------------------------------------------------------------------
    # FILE CHECKS

    if verbose:
        print('---> Performing necessary checks')

    # datasets within the hdf5 file should be given as an attribute
    #   named DataAvailable
    try:
        datasetsUL = data.attrs['DataAvailable'].astype(str)[0]
    except KeyError:
        err = 'The datasets available must be given as an attribute DataAvailable'
        raise AssertionError(err)
    # remove opening and closing bracket and split string
    datasetsUL = datasetsUL.replace('[', '').replace(']', '').split(',')
    # need lower case for consistency but also the original for
    #   finding the right data.
    datasets = [x.lower() for x in datasetsUL]

    if verbose:
        print('---> Datasets are', ', '.join(datasets))

    # for each element in datasets, find the first group of the hdf5 
    #   file that it belongs to. if the dataset is missing, then raise 
    #   an exception, noting which datasets are missing
    # CHECK THIS: what if it exists in more than one group?
    group_data = [data[x] for x in data.keys()]
    try:
        datagroups_idx = [[i for i,x in enumerate(y in z for z in group_data) if x][0] \
                                                                    for y in datasetsUL]
    except IndexError:
        dataset_exists = [y in [x for sublist in group_data for x in sublist] \
                                                                    for y in datasetsUL]
        missing = np.array(datasetsUL)[np.invert(dataset_exists)]
        if len(missing) > 1:
            missing = ' '.join(missing)
            err = 'Datasets: ' + missing + ' are missing from the h5 file' 
            raise AssertionError(err)
        else:
            err = 'Dataset: ' + missing + ' is missing from the h5 file'
            raise AssertionError(err)
    # datagroups provides the group that each dataset belongs to so
    #   that datagroups + datasets provides the full path to the dataset
    datagroups = [list(data.keys())[x] + '/' for x in datagroups_idx]
    # clean up
    del datagroups_idx

    # next check that the '.index' array exists for each dataset, and 
    #   is in the same group as the dataset.
    index_exists = [y + '.index' in data[datagroups[datasetsUL.index(y)]] for y in datasetsUL]
    if not all(index_exists):
        missing = np.array([y + '.index' for y in datasetsUL])[np.invert(index_exists)]
        
        if len(missing) > 1:
            missing = ' '.join(missing)
            err = 'Index arrays: ' + missing + ' are missing from h5 groups'
            raise AssertionError(err)
        else:
            err = 'Index array: ' + missing + ' is missing from the h5 file'
            raise AssertionError(err)

    # additionally, '.index' array for each dataset must contain 
    #   specific keys, which allow the decoding of the time variable
    #   for the data.
    indexnames = ('startidx', 'starttime', 'length', 'frequency')

    def dataset_index(y):
        z = data[datagroups[datasetsUL.index(y)] + y + '.index']
        return(z)

    # note -  raising an exception like this allows all problematic
    #   datasets to be highlighted rather than looping over the
    #   datasets one by one - when the exception would raise only the
    #   first problematic dataset
    indexnames_exist = \
        [all(x in dataset_index(y).dtype.names for x in indexnames) for y in datasetsUL]   
    if not all(indexnames_exist):
        missing = np.array([y + '.index' for y in datasetsUL])[np.invert(indexnames_exists)]
        
        if len(missing) > 1:
            missing = ' '.join(missing)
            err = 'Index arrays: ' + missing + ' are missing keys'
            raise AssertionError(err)
        else:
            err = 'Index array: ' + missing + ' is missing keys'
            raise AssertionError(err)

    # start and end time of the data should be given as attributes in
    #   the file. if not, then determine these from the datasets.
    if 'dataStartTimeUnix' in list(data.attrs):
        data_starttime = data.attrs['dataStartTimeUnix'].astype(int)
    else:
        data_starttime = np.min([dataset_index(x)['starttime'][0] for x in datasetsUL])
    if 'dataEndTimeUnix' in list(data.attrs):
        data_endtime = data.attrs['dataEndTimeUnix'].astype(int)
    else:
        data_endtime = np.max([dataset_index(x)['starttime'][-1] \
                    + (dataset_index(x)['length'][-1] - 1) / dataset_index(x)['frequency'] \
                                                                    for x in datasetsUL])

    # determine if the units of time in the '.index' arrays are 
    #   documented in the file, inside the group 'definitions' and 
    #   field 'indexStruct'
    try:
        indexStruct = data['definitions/indexStruct'].value
        starttime_string = np.str(indexStruct['description'][np.where( \
                                        indexStruct['field'] == 'starttime')])
        # strip out non-alphanumeric characters (so prefix to seconds not lost)
        starttime_string_an = ''.join([s if s.isalnum() else ' ' for s in starttime_string])
        if 'microseconds' in np.str(starttime_string_an):
            index_tformat = 10 ** (-6)
        elif 'seconds' in np.str(starttime_string_an):
            index_tformat = 1
        else:
            print('Start times in index arrays are neither in seconds or microseconds:')
            print(starttime_string)
            print('Please input the units as converted to seconds \
                                            (i.e. milliseconds would be 10 ** (-3)):')
            index_tformat = input().astype(int)
    except KeyError:
        # if this indexStruct does not exist, make the assumption that 
        #   the units are microseconds if starttime is a float in all 
        #   index arrays and seconds if starttime is an int in all
        if verbose:
            print('WARNING: indexStruct does not exist in the group definitions')
        if all([isinstance(dataset_index(x)['starttime'][0], (np.int_, np.uint)) \
                                                                for x in datasetUL]):
            index_tformat = 1
            print('Assuming start times in index arrays are in seconds')
        elif all([isinstance(dataset_index(x)['starttime'][0], (np.float, np.float_)) \
                                                                for x in datasetUL]):
            index_tformat = 10 ** (-6)
            print('Assuming start times in index arrays are in microseconds')
        # otherwise raise an exception
        else:
            err = 'Unable to identify units of start times in index arrays'
            raise AssertionError(err)
        # allow the user to determine whether to continue
        selection = userInput('Continue', 'Exit')
        if selection == 2:
            err = 'Function terminated'
            raise AssertionError(err)

    # ------------------------------------------------------------------------
    # ADDITIONAL CHECKS

    if verbose and checks:
        print('---> Performing additional checks')

    def numeric_check(x):
        '''
        Check that x is numeric. Assumes numpy has been imported as np
        '''
        err = 'Data must be a numeric list or a numeric numpy.ndarray'
        try:
            # if x is a list, check if all elements of the list are int
            #    or float
            if isinstance(x, list):
                if not all([isinstance(y, (int, float)) for y in x]):
                    raise AssertionError(err)
            # if x is a numpy ndarray, check if all elements of the 
            #   array are np.int_ or np.float_
            elif isinstance(x, np.ndarray):
                if x.dtype.type in (np.str_, np.void):
                    # if a structured ndarray with different dtypes
                    #   then check each dtype
                    if any([x.dtype[y].type in (np.str_, np.void) \
                                        for y in np.arange(np.size(x.dtype))]):
                        raise AssertionError(err)
            # if x is a scalar, check if it is an int or a float
            elif not isinstance(x, (int, float)):
                raise AssertionError(err)
            else:
                raise AssertionError(err)
        except NameError as err:
            if 'np' in err.args[0]:
                raise AssertionError(err)

    if checks:
        # check that '.index' array is self consistent if 
        #   additional keys ('timestep', 'frequency') exist
        for x in datasetsUL:
            if 'timestep' in dataset_index(x).dtype.names:       
                if not all(dataset_index(x)['frequency'] == 1/dataset_index(x)['timestep']):
                    err = datasets[x] + '.index is not self-consistent'
                    raise AssertionError(err)
            if 'endtime' in dataset_index(x).dtype.names:
                if not all(dataset_index(x)['endtime'] == dataset_index(x)['starttime'] \
                            + (dataset_index(x)['length'] - 1) * dataset_index(x)['frequency']):
                    err = datasets[x] + '.index is not self-consistent'
                    raise AssertionError(err)

        # checks on '.label' arrays if any exist
        if any([x + '.label' in data[datagroups[datasetsUL.index(x)]] for x in datasetsUL]):
            # there exists some '.label' arrays
            # this definition makes things a bit less unwieldly later on
            def dataset_label(y): 
                z = data[datagroups[datasets.index(y)] + y + '.label']
                return(z)

            labelnames = ('startidx', 'length', 'label')
            # check if there are any '.label' arrays with missing keys
            labelnames_exist = \
                [all(x in dataset_index(y).dtype.names for x in indexnames) for y in datasetsUL]   
            if not all(labelnames_exist):
                missing = \
                    np.array([y + '.label' for y in datasetsUL])[np.invert(labelnames_exists)]
                if len(missing) > 1:
                    missing = ' '.join(missing)
                    err = 'Label arrays: ' + missing + ' are missing keys'
                    raise AssertionError(err)
                else:
                    err = 'Label array: ' + missing + ' is missing keys'
                    raise AssertionError(err)

        # check arrays are numpy ndarrays and all variables are numeric
        for x in datasetsUL:
            # check data is numeric
            numeric_check(data[datagroups[datasetsUL.index(x)] + x].value)
            # check that '.index' array is a numpy ndarray
            if not isinstance(dataset_index(x).value, np.ndarray):
                err = datasets[x] + '.index must be a numpy ndarray'
                raise AssertionError(err)
            # check '.index' array is numeric
            numeric_check(dataset_index(x).value)
            # if '.label' array exists for that dataset
            if x + '.label' in data[datagroups[datasetsUL.index(x)]]:
                # check that '.label' array is a numpy ndarray if it 
                #   exists
                if not isinstance(dataset_label(x).value, np.ndarray):
                    err = datasets[x] + '.label must be a numpy ndarray'
                    raise AssertionError(err)
                # check '.label' array is numeric
                numeric_check(dataset_label(x).value)

        # check data start and end time are numeric
        numeric_check(data_starttime)
        numeric_check(data_endtime)

    # ------------------------------------------------------------------------
    # RETURN

    # note, to access a specific dataset, e.g. 'abp', use the following
    ## if 'abp' in datasets:
    ##     data[datagroups[datasets.index('abp')] + 'abp'].value
    ##     data[datagroups[datasets.index('abp')] + 'abp.index'].value['startidx'] # etc
    ## else:
    ##     raise AssertionError('abp not in datasets')

    return ReturnValue(data = data, datasets = datasets, datasetsUL = datasetsUL, \
                        datagroups = datagroups, data_starttime = data_starttime, \
                        data_endtime = data_endtime, index_tformat = index_tformat)

        # function load_h5 end

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def plot_data(data, index, label, plot_starttime, plot_endtime, ax, **kwargs):
    '''
    Plots data of a particular format.

    Plots a section of labelled data where the t variable is given as 
    an array listing data blocks of constant frequency. Additional
    keyword arguments can specify features of the plot. 
    Note this is a stripped down version of the function h5_data_plot 
    and is designed for use within the function gui_h5. Most of the 
    checks of the original function aren't necessary and there are a 
    few shortcuts that take advantage of the fact the function is 
    repeatedly called with the GUI that will save some computational 
    time.

    Parameters
    ----------
    data: numpy array
        An N x 1 array of y values.
    index: numpy array
        An array encoding the t values. 
        The data is continuous and sampled at constant frequencies 
        in a finite number of blocks or chunks, potentially with gaps
        between blocks. The frequency of the data may also change 
        between blocks.
        It has the following dtype.names:
            startidx - int
            starttime - float
            length - int
            frequency - float
            timestep - float
            endtime - float
        These correspond to the index of the dataset corresponding to 
        the new block of data, the start time of that block (s), the
        number of data points in that block, the frequency of the 
        data (Hz), the time difference between successive data points 
        within the block (s) and finally the end time of the block (s).
    label: array
        An array encoding the data labels.
        It has the following dtype.names:
            startidx - int
            length - int
            label - int
        The label is a code assigned to a labelled section. The label
        codes should be negative integers for labels applicable to all
        datasets and positive integers for labels applicable to 
        specific datasets. The labels specific to a particular dataset, 
        identified by m, will be within 0-99 + m * 100.
    plot_starttime: float
        Starting time of plot window.
    plot_endtime: float
        End time of plot window. See also the keyword argument plot_pad.
    ax: matplotlib.axes object
        The axes on which the data is to be plotted.

    Keyword arguments
    ----------
    title: string
        Title of the plot. By default, left blank.
    ylim: array/list
        y limits for the plot. Note, this must be a monotonically 
        increasing numeric list or array of length 2. It is 
        recommended that it is given using knowledge of what a sensible
        range is for each dataset. By default, these are determined 
        from the maximum and minimum values of the dataset which may 
        not be sensible.
    ylabel: string
        y axis labels for the plot. By default, left blank.
        The xlabel is 's' as the units of the t variable are seconds.
    plot_pad: float
        A fraction indicating the amount of padding around plot_starttime
        and plot_endtime so there is a small overlap between successive
        plots if for example plot_starttime(kk + 1) = plot_endtime(kk) for
        some loop in kk. By default, no padding.
    cmap: list
        A list of colours corresponding to each label. These can be given
        either as RGB, RGBA or HEX. By default, no label is black and the
        remaining labels are given in standard python colors.

    Example usage
    ----------
    plot_h5(data, index, label, plot_starttime, plot_endtime, ax)
    plot_h5(data, index, label, plot_starttime, plot_endtime, ax,
            title = 'Dataset1', ylim = [ymin, ymax], plot_pad = 1/10)
    '''

    # WARNING - some of this is probably not very pythonic as it was 
    #   originally written in Matlab

    # ------------------------------------------------------------------------
    # MODULES

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    from matplotlib.lines import Line2D

    # ------------------------------------------------------------------------
    # PLOT INITIALISATION

    # adjust plot start and end times if outside of data times
    if plot_starttime < index['starttime'][0]:
        if plot_endtime < index['starttime'][0]:
            print('plot_starttime and plot_endtime outside of data')
            return
        plot_starttime = index['starttime'][0]
    data_endtime = index['starttime'][-1] + index['length'][-1] / index['frequency'][-1]
    if plot_endtime > data_endtime:
        if plot_starttime > data_endtime:
            print('plot_starttime and plot_endtime outside of data')
            return
        plot_endtime = data_endtime

    # pad around plot_starttime and plot_endtime if required
    #   this gives a small overlap between successive plots if for  
    #   example plot_starttime(kk + 1) = plot_endtime(kk)
    #   for some loop in kk outside of this function
    if 'plot_pad' in kwargs.keys():
        plot_starttime = plot_starttime - (plot_endtime - plot_starttime) * kwargs['plot_pad']
        plot_endtime = plot_endtime + (plot_endtime - plot_starttime) * kwargs['plot_pad']

    # hide the axis until everything is plotted and ready
    ax.set_visible(False)

    # set x limits to be plot_starttime and plot_endtime
    #   these are then changed in the code to the be equal to the
    #   closest (less than/greater than respectively) time point of the
    #   data
    ax.set_xlim([plot_starttime, plot_endtime])

    # if ylim has been given as a keyword argument, then set this now
    #   otherwise wait to determine it from the data
    if 'ylim' in kwargs.keys():
        try:
            ax.set_ylim(kwargs['ylim'])
        except:
            del kwargs['ylim']

    # set other plot features
    if not 'ylabel' in kwargs.keys():
        kwargs['ylabel'] = ''

    if not 'title' in kwargs.keys():
        kwargs['title'] = ''

    ax.set_ylabel(kwargs['ylabel'])
    ax.set_xlabel('s')
    ax.set_title(kwargs['title'])

    # ------------------------------------------------------------------------
    # INDEX DECODING

    timestep = 1 / index['frequency']
    index_endtime = index['starttime'] + (index['length'] - 1) * timestep

    # at this point, would be kind of simple to create a vector the 
    #   same length as data containing time points corresponding to 
    #   each data point and then pull out the part that lies inside the 
    #   plot window but kind of defeats the point of storing the data 
    #   so concisely and is a bit unnecessary what that means though is
    #   we need to be a bit smarter to unpack only the time points 
    #   needed and be wary of a plot window encompassing multiple data
    #   blocks as well

    # find the block that the starttime lies in and the block the end 
    #   time lies in and call these block_start, block_end.
    #   then find the approximate location within the block of the 
    #   starttime and select largest time point smaller than it, this 
    #   is the first point plotted
    #   note that the start time remains the same - xlim
    #   if it is before the first data block or between data blocks 
    #   then set it be the first element of the next data block
    # similarly for endtime
    # find also the indices involved - take care here!

    # it seems kind of pointless to compute the startblock, starttime 
    #   and startidx each time the function is called within the GUI, 
    #   rather than using the previous function call to update it - 
    #   maybe this could be changed to make use of that information but
    #   I don't see it as being very computationally intensive so
    #   probably not worth the effort of rewriting things
    if all(plot_starttime < index['starttime']):
        plot_startblock = 0
        plot_starttime = index['starttime'][plot_startblock]
        plot_startidx = index['startidx'][plot_startblock]
    else:
        plot_startblock = np.max(np.where(plot_starttime >= index['starttime']))
        
        if plot_starttime > index_endtime[plot_startblock]:
            plot_startblock += 1
            plot_starttime = index['starttime'][plot_startblock]
            plot_startidx = index['startidx'][plot_startblock]
        else:
            n = np.floor((plot_starttime - index['starttime'][plot_startblock]) \
                                                    / timestep[plot_startblock])
            plot_starttime = index['starttime'][plot_startblock] + n \
                                                    * timestep[plot_startblock]
            plot_startidx = index['startidx'][plot_startblock] + n

    if all(plot_endtime > index_endtime):
        # note as a result of how python deals with indices (e.g. 
        #   a[2:3] is the same as a[2]), plot_endtime and plot_endidx 
        #   are incremented one more than what they should really be so
        #   that data[plot_startidx:plot_endidx] includes 
        #   data[plot_endidx] etc.
        plot_endblock = np.size(index_endtime) - 1
        plot_endtime = index_endtime[plot_endblock]
        plot_endidx = index['startidx'][plot_endblock] \
                                + index['length'][plot_endblock]
    else:
        plot_endblock = np.min(np.where(plot_endtime <= index_endtime))
        # if the end of the plot window is between data blocks
        # then select the previous block
        if plot_endtime < index['starttime'][plot_endblock]:
            plot_endblock -= 1
            plot_endtime = index_endtime[plot_endblock]
            plot_endidx = index['startidx'][plot_endblock] \
                                + index['length'][plot_endblock]
        else:
            # note the + 1 is for the same reason as the previous 
            #   comment
            n = np.ceil((plot_endtime - index['starttime'][plot_endblock]) \
                                                    / timestep[plot_endblock])
            plot_endtime = index['starttime'][plot_endblock] + n \
                                                    * timestep[plot_endblock]
            plot_endidx = index['startidx'][plot_endblock] + n + 1

    # make sure indices are integers
    plot_startidx = plot_startidx.astype(int)
    plot_endidx = plot_endidx.astype(int)

    # check plot_endtime/plot_endidx is greater than plot_starttime/
    #   plot_endidx it is still worth including this check - though 
    #   might need to think about what is to happen if an exception 
    #   actually is raised
    if plot_endtime < plot_starttime or plot_endidx < plot_startidx:
        # if this is the case, then the plot window is outside of any 
        #   data blocks, so just return an empty plot
        
        # allow the axis to become visible again
        ax.set_visible(True)
        # add ylimits if they don't exist
        if not 'ylim' in kwargs.keys():
            ax.set_ylim([np.nanmin(data), np.nanmax(data)])
        return None

    # piece together time vector
    #   there's a lot going on here but essentially it's creating a 
    #   vector with all the time points between plot_starttime and
    #   plot_endtime even if the data of interest spans a couple of 
    #   data blocks
    plot_times = np.array([tuple((index['starttime'][x], index_endtime[x])) \
                                    for x in np.arange(np.size(index['starttime']))],
                            dtype = [('starttime', 'f8'), ('endtime', 'f8')])
    plot_times['starttime'][plot_startblock] = plot_starttime
    plot_times['endtime'][plot_endblock] = plot_endtime
    # this means we don't need to create a really long vector t 
    #   encompassing the whole of some very long data blocks, which is 
    #   computationally efficient and unnecessary
    # NOTE - the problem that can happen here is that due to 
    #   numerical precision, sometimes there is an additional element
    #   in np.arange when there shouldn't be timestep
    #   i.e. plot_times['starttime'] + m *  is less
    #   than plot_times['endtime'] + timestep when it should
    #   be equal
    # to counter this, subtract index['timeestep'][x]/2 from the end of
    #   np.arange, to give plot_times['endtime'] + timestep/2  
    t = np.hstack([np.arange(plot_times['starttime'][x], \
            plot_times['endtime'][x] + timestep[x]/2, timestep[x]) \
            for x in np.arange(plot_startblock, plot_endblock + 1)])

    # identify the data that lies within the plot window
    # CHECK THIS doesn't need a + 1 at plot_endidx
    data = data[np.arange(plot_startidx, plot_endidx)]

    # ------------------------------------------------------------------------
    # LABELS

    # list of unique labels in order 0, -ve decreasing, +ve increasing
    labels = np.unique(label['label'])
    labels = np.append(np.flip(labels[labels <= 0]), labels[labels > 0])

    # pick out the labels that lie within the plot window
    label = np.delete(label, np.where(label['startidx'] + label['length'] < plot_startidx))
    label = np.delete(label, np.where(label['startidx'] > plot_endidx))
    label['length'][0] = label['length'][0] - plot_startidx + label['startidx'][0]
    label['startidx'][0] = plot_startidx
    label['length'][-1] = plot_endidx + 1 - label['startidx'][-1]
    # convert labels from 0, -1, -2, 101, 102, 103, etc to 0, 1, 2, 3, 4, 5
    label_temp = label['label'].copy()
    for x in np.arange(np.size(labels)):
        label_temp[np.where(label['label'] == labels[x])] = x

    # expand structured array to a vector of labels, the same length as t/data
    label = np.hstack([np.repeat(label_temp[x], label['length'][x]) \
                                            for x in np.arange(np.shape(label)[0])])

    if 'cmap' in kwargs.keys():
        # needs to be RGB
        cmap = kwargs['cmap']
        if any([isinstance(x, str) for x in cmap]):
            for x in [y for y in cmap if isinstance(y, str)]:
                cmap[cmap.index(x)] = tuple(int(x.lstrip('#')[jj:jj + 2], 16) \
                                                    for jj in np.arange(0, 6, 2))
    else:
        if np.size(labels) > 1:
            cmap_temp = plt.get_cmap('tab10')
            cmap = np.vstack(((0, 0, 0), \
                [cmap_temp.colors[x] for x in np.mod(np.arange(np.size(labels) - 1), 10)]))
        else:
            cmap = np.array((0, 0, 0)).reshape(1, 3)

    # convert from RBG to RBGA
    if any([np.size(y) == 3 for y in cmap]):
        cmap = [list(x) for x in cmap]
        for x in [y for y in cmap if np.size(y) == 3]:
             cmap[cmap.index(x)].append(1)
        cmap = np.vstack(cmap)

    # ------------------------------------------------------------------------
    # STORE INDICES IN PLOT

    # plotting a hidden line containing the indices of data plotted 
    #   within the whole data file allows us to recover these indices 
    #   outside the function very easily by grabbing the ydata that is
    #   stored within the axes. add a label to make this easier
    idx = Line2D(t, np.arange(plot_startidx, plot_endidx + 1))
    idx.set_label('idx')
    idx.set_visible(False)

    # ------------------------------------------------------------------------
    # NANs

    # adding NaN (via numpy) between data blocks is a convenient way of 
    #   stopping a line segment being plotted between the last data 
    #   point of one block and the first of another, as would be done normally
    def insert_value(x, value = np.nan):
        '''Inserts value in between blocks'''
        if not x.dtype.type == float:
            x = np.array(x, dtype = float)
        pos = index['startidx'] - plot_startidx
        pos = [y for y in pos if y < np.size(x) and y > 0]
        x = np.insert(x, pos, np.repeat(value, np.size(pos)))
        return(x)

    data = insert_value(data)
    t = insert_value(t)
    label = insert_value(label, 0)

    # ------------------------------------------------------------------------
    # PLOT

    # method to plot multiple lines of different colours is adapted 
    #   from multicolored_line.py (see 
    #   https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html
    #   if link is still active). basically the idea is to collect a 
    #   line segments between successive points and assign
    #   colour to each depending on some criterion for two variables  
    #   data and t of length N, the object points will be a N x 1 x 2
    #   array with points[n, 0, :] = [data[n], t[n]] and the object 
    #   segments will be a (N - 1) x 2 x 2 array with 
    #   segments[n + 1, 0, :] = segments[n, 1, :] = points[n, 0, :].
    points = np.array([t, data]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # need to remove the last label as there are N - 1 line segments 
    #   and N labels i.e. the label assigned to point n becomes the 
    #   label between point n and point n + 1
    lc = LineCollection(segments, colors = cmap[label[:-1].astype(int)])

    # plot depending on label
    lines = ax.add_collection(lc)

    # add y limits if they don't already exist
    if not 'ylim' in kwargs.keys():
        ax.set_ylim([np.nanmin(data), np.nanmax(data)])

    # allow the axis to become visible again
    ax.set_visible(True)
    # function plot_data end

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def dtype_change(array, name, newtype):
    '''
    Change dtype of array element with dtype name 'name'. 
    '''
    dt = array.dtype
    dt = dt.descr
    pos = np.where(np.array(dt)[:, 0] == name)[0][0]
    dt[pos] = (dt[pos][0], newtype)
    dt = np.dtype(dt)
    array = array.astype(dt)
    return(array)
    # function dtype_change end

def preprocess_index(index):
    '''
    Preprocess index array such that the time variable is in seconds
    since the 00:00 on the day corresponding to the data start (as
    opposed to microseconds).
    Example usage:
    index = preprocess_index( \
            h5.data[datagroup + dataset + '.index'].value)
    '''
    index = dtype_change(index, 'starttime', 'f8')
    index['starttime'] = index['starttime'] / 10**6
    return(index)
    # function preprocess_index end

def preprocess_quality(quality, qualityRef, starttime):
    '''
    Preprocess quality array such that the time variable is in seconds
    since the 00:00 on the day corresponding to the data start (as in
    other arrays) and unidentified quality indicators are removed.
    Example usage:
    quality = preprocess_quality( \
            h5.data[datagroup + dataset + '.quality'].value, \
            h5.data['definitions/qualityRef'].value, \
            h5.data.attrs['dataStartTimeUnix'])
    '''
    quality = dtype_change(quality, 'time', 'f8')
    quality['time'] = quality['time'] / 10**6

    a = np.where([quality['value'][x] not in qualityRef['indicator'] \
        for x in np.arange(np.size(quality))])
    quality = np.delete(quality, a)

    # for some reason, the time variable here is actually time
    #   since 1/1/1970 (problem!) so need to adjust
    # should this be + dataStartTimeUnix not index['starttime'][0]?
    quality['time'] = quality['time'] - quality['time'][0] + starttime
    return(quality)
    # function preprocess_quality end

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def range_label(data, **kwargs):
    '''
    Creates label for data outside of a (optional) range and for data 
    marked as NaN.

    Parameters
    ----------
    data: numpy array
        An N x 1 array of y values.

    Keyword arguments
    ----------
    data_min: float
        Minimum value for the data. Data below this value is labelled.
    data_max: float
        Maximum value for the data. Data above this value is labelled.
    label: int
        Label code identifying that the region has been labelled with
        this function. By default, this is 1.
    '''
    invalid = np.isnan(data)

    if 'data_min' in kwargs.keys():
        invalid = np.logical_or(invalid, data < kwargs['data_min'])

    if 'data_max' in kwargs.keys():
        invalid = np.logical_or(invalid, data > kwargs['data_max'])

    groups = np.array([[k, sum(1 for k in g)] for k, g in groupby(invalid)])

    startidx = np.cumsum(np.append(0, groups[:-1, 1]))[groups[: ,0].astype(bool)]
    length = groups[groups[:, 0].astype(bool), 1]

    if 'label' in kwargs.keys():
        label = kwargs['label']
    else:
        label = 1

    label = np.array([tuple((startidx[x], length[x], label)) \
        for x in np.arange(np.size(startidx))],
            dtype = [('startidx', 'i8'), ('length', 'i8'), ('label', 'i4')])

    return label
    # function range_label end

def local_label(data, index, **kwargs):
    '''
    Creates labels for regions where the range of the data is either
    much smaller or much larger than it should be i.e. the data is 
    approximately stationary (no pulsatility) or is oscillating with 
    a much greater amplitude than expected. It assigns a label of 1
    to the former and of 2 to the latter.

    Parameters
    ----------
    data: numpy array
        An N x 1 array of y values.
    index: numpy array
        An array encoding the t values. 
        The data is continuous and sampled at constant frequencies 
        in a finite number of blocks or chunks, potentially with gaps
        between blocks. The frequency of the data may also change 
        between blocks.
        It has the following dtype.names:
            startidx - int
            starttime - float
            length - int
            frequency - float
        These correspond to the index of the dataset corresponding to 
        the new block of data, the start time of that block (s), the
        number of data points in that block, the frequency of the 
        data (Hz).

    Keyword arguments
    ----------
    n: int
        Min window length (number of elements) in which the data must be
        stationary (i.e. n successive elements must be identical) for it
        to be labelled. By default, this is 30.
    t: float
        Min window length (time, in s) in which the data must be 
        stationary for it to be labelled. If both n and t are give, then
        the window length (in terms of the number of elements) is set to 
        be the maximum of n and t * data frequency for each data block.
        This should depend on the frequency of oscillations in the data 
        (note this is not the frequency of the recording, which is the 
        frequency given in the index array), so by default, this is not 
        given.
    tol_min: float
        Minimum tolerance value (labelled if max - min < tol_min). By
        default, this is the difference between the 49th and 51st 
        percentiles of the data. However, this is highly dependent on
        the distribution of the data and so we recommend changing it.
    tol_max: float
        Maximum tolerance value (labelled if max - min > tol_max). By 
        default, this is the difference between the 5th and 95th 
        percentiles of the data (but see above).
    verbose: boolean
        If False, then statements of progress/warnings are suppressed. 
        Note that some output is still necessarily printed. By default,
        True.
    label: int
        Label code identifying that the region has been labelled with
        this function (needs two values, one for each part of the 
        function). By default, this is 1 and 2 respectively.
    '''
    nblocks = np.size(index)

    if 'n' in kwargs.keys():
        n = kwargs['n']
    else:
        n = 30

    if 't' in kwargs.keys():
        n = np.array([np.max((n, x)) for x in kwargs['t'] * index['frequency']])
    else:
        n = np.repeat(n, nblocks)

    if 'tol_min' in kwargs.keys():
        tol_min = kwargs['tol_min']
    else:
        tol_min = np.diff(np.percentile(data, [51, 49]))

    if 'tol_max' in kwargs.keys():
        tol_max = kwargs['tol_max']
    else:
        tol_max = np.diff(np.percentile(data, [95, 5]))

    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose = False

    if 'label' in kwargs.keys() and np.size(kwargs['label']) == 2:
        label_code = kwargs['label']
    else:
        label_code = np.array([1, 2])

    length = np.zeros(0)
    startidx = np.zeros(0)
    label = np.zeros(0)

    # idea:
    # so the idea is to look for regions of length n[ii] where the 
    #   difference between the min and max values of the data in this
    #   region is </> than some tolerance, tol. we know that if this is
    #   the case then we need to look at successive regions of length 
    #   n[ii]/2 and if any of these regions doesn't satisfy the tol 
    #   condition then we can immediately discard it. if a region does, 
    #   we need to expand that region to length n[ii] and if it 
    #   satisfies the condition still, then keep expanding the length of
    #   the region until it no longer satisfies the condition (this is 
    #   then a labelled region) - scanning through the data in regions 
    #   of length n[ii]/2 should speed up the code, especially as 
    #   stationary regions are likely to rare within the data

    # pseudocode:
    # (i) set s = 0
    # (ii) while max(data[s + 0:m]) - min(data[s + 0:m]) > tol,
    #   s = s + m  (where n may depend on the block of data and m
    #   is floor(n/2)
    # (iii) if max(data[s + 0:(n/2)]) - min(data[s + 0:(n/2)]) < tol, 
    #   find m1, m2 such that max - min of data[(s - m1):(s + m2 + n/2)]
    #   is < tol but does not hold if m1 += 1 or m2 += 1. then if 
    #   m1 + m2 > n/2, this region has length > n and so label it. set
    #   s = s + m2. note m1 < n/2 by construction but not necessarily 
    #   true for m2
    # (iv) repeat until s + n/2 > length of data

    for ii in np.arange(nblocks):
        if verbose:
            print('--> Starting block with startidx ' + np.str(index['startidx'][ii]))
        block = data[np.arange(index['startidx'][ii], \
            index['startidx'][ii] + index['length'][ii])]

        newstartidx = np.zeros(0)
        newlength = np.zeros(0)
        newlabel = np.zeros(0)

        s = 0
        valid = True
        m = np.floor(n[ii]/2).astype(int)
        s1 = 0

        while s < np.size(block) and valid is True:
            # numpy.ptp acts the same as numpy.max - numpy.min
            # brackets around (s + m) in the below are unnecessary for
            #   the code but imo in help readability
            while np.ptp(block[s:(s + m + 1)]) > tol_min and \
                    np.ptp(block[s:(s + m + 1)]) < tol_max and valid is True:
                if s + m < np.size(block):
                    s += m
                else:
                    # if the range is less than tol_min for s:np.size(block)
                    #   here then this region is smaller in length than
                    #   n[ii]
                    valid = False
            if np.ptp(block[s:(s + m + 1)]) <= tol_min:
                if s == 0:
                    m1 = 0
                else:
                    # if all entries of x are False, np.argmax(x) is 0
                    #   which is what we want here
                    m1 = np.argmax(np.max((np.flip(block[(s - m):s]) \
                            - np.min(block[s:(s + m + 1)]), \
                        np.max(block[s:(s + m + 1)]) \
                            - np.flip(block[(s - m):s])), \
                                axis = 0) > tol_min)
                    # BUG - sometimes if tol_min is some multiple of the 
                    #   precision of the data, then the labelled regions
                    #   overlap, try to prevent that here and deal with
                    #   it later
                    if np.size(newstartidx) > 1:
                        di = newstartidx[-1] + newlength[-1] - index['startidx'][ii]
                        if s - m1 < di:
                            m1 += s - di
                # in this while loop, rather than the whole region to
                #   being within tol_min (i.e. replace index with 
                #   s:(s + s1 + m + 1)), as long as m successive elements
                #   are within tol_min of each other, allow when this is not
                #   the case (i.e. (s + s1):(s + s1 + m + 1))
                while np.max(block[(s + s1):(s + s1 + m + 1)]) \
                        - np.min(block[(s + s1):(s + s1 + m + 1)]) <= tol_min \
                            and valid is True:
                    if s + s1 + m < np.size(block):
                        s1 += m
                    else:
                        # similar argument to above for terminating
                        #   while loop
                        valid = False
                m2 = np.argmax(np.max((block[(s + s1):(s + s1 + m + 1)] \
                        - np.min(block[s:(s + s1 + 1)]), \
                    np.max(block[s:(s + s1 + 1)]) \
                        - block[(s + s1):(s + s1 + m + 1)]), \
                            axis = 0) > tol_min)
                if s1 + m1 + m2 > n[ii]:
                    newlength = np.append(newlength, s1 + m1 + m2)
                    newstartidx = np.append(newstartidx, index['startidx'][ii] + s - m1)
                    newlabel = np.append(newlabel, label_code[0])
                s += s1 + m
                s1 = 0
            elif np.ptp(block[s:(s + m + 1)]) >= tol_max:
                if s == 0:
                    m1 = 0
                else:
                    # if all entries of x are False, np.argmax(x) is 0
                    #   which is what we want here
                    m1 = np.argmax(np.max((block[s:(s + m + 1)] \
                            - np.minimum.accumulate(block[s:(s + m + 1)]), \
                        np.maximum.accumulate(block[s:(s + m + 1)]) \
                            - block[s:(s + m + 1)]), \
                                axis = 0) >= tol_max) - 1
                    # m1 = np.argmin(np.min((block[s:(s + m)] - np.min(block[s:(s + m + 1)]), \
                    #                         np.max(block[s:(s + m + 1)]) - block[s:(s + m)]), \
                    #                         axis = 0) < tol_max)
                    # m1 = np.argmax(np.max((block[s:(s + m)] - np.min(block[s:(s + m + 1)]), \
                    #                         np.max(block[s:(s + m + 1)]) - block[s:(s + m)]), \
                    #                         axis = 0) >= tol_max)
                # in this while loop, rather than the whole region to
                #   being within tol_min (i.e. replace index with 
                #   s:(s + s1 + m + 1)), as long as m successive elements
                #   are within tol_min of each other, allow when this is not
                #   the case (i.e. (s + s1):(s + s1 + m + 1))
                while np.max(block[(s + s1):(s + s1 + m + 1)]) \
                        - np.min(block[(s + s1):(s + s1 + m + 1)]) >= tol_max \
                            and valid is True:
                    if s + s1 + m < np.size(block):
                        s1 += m
                    else:
                        # similar argument to above for terminating
                        #   while loop
                        valid = False
                # bit complicated: start at the end of the last region, 
                #   work backwards until the tol_max threshold is 
                #   reached with the elements included so far, here m2  
                #   eats into the region we've identified (s1) rather 
                #   than adds to it as in the tol_min case
                m2 = np.argmax(np.max(( \
                    np.flip(block[(s + s1 - m):(s + s1 + 1)]) \
                        - np.minimum.accumulate(np.flip(block[(s + s1 - m):(s + s1 + 1)])), \
                    np.maximum.accumulate(np.flip(block[(s + s1 - m):(s + s1 + 1)])) \
                        - np.flip(block[(s + s1 - m):(s + s1 + 1)])), \
                            axis = 0) >= tol_max)
                # m2 = np.argmax(np.max((np.flip(block[(s + s1 - m):(s + s1)]) \
                #                                 - np.min(block[s:(s + s1 + 1)]), \
                #                 np.max(block[s:(s + s1 + 1)]) \
                #                                 - np.flip(block[(s + s1 - m):(s + s1)])), \
                #                 axis = 0) < tol_max)
                if s1 - m1 - m2 > 0:
                    newlength = np.append(newlength, s1 - m1 - m2)
                    newstartidx = np.append(newstartidx, index['startidx'][ii] + s + m1)
                    newlabel = np.append(newlabel, label_code[1])
                s += s1 + m
                s1 = 0

        length = np.append(length, newlength).astype(int)
        startidx = np.append(startidx, newstartidx).astype(int)
        label = np.append(label, newlabel).astype(int)

    label = np.array([tuple((startidx[x], length[x], label[x])) \
        for x in np.arange(np.size(startidx))], \
            dtype = [('startidx', 'i8'), ('length', 'i8'), ('label', 'i4')])

    return label
    # function local_label end

def quality_label(quality, index, **kwargs):
    '''
    Creates a label array from the h5 quality array.

    Parameters
    ----------
    quality: numpy array
        An array encoding the quality score assigned to the data. 
        It has the following dtype.names:
            time: int
            value: int
        Note that time is given in microseconds. 
    index: numpy array
        An array encoding the t values. 
        The data is continuous and sampled at constant frequencies 
        in a finite number of blocks or chunks, potentially with gaps
        between blocks. The frequency of the data may also change 
        between blocks.
        It has the following dtype.names:
            startidx - int
            starttime - float
            length - int
            frequency - float
        These correspond to the index of the dataset corresponding to 
        the new block of data, the start time of that block (s), the
        number of data points in that block, the frequency of the 
        data (Hz).

    Keyword arguments
    ----------
    label: int
        Label code identifying that the region has been labelled with
        this function. Note that this needs the exact number of labels
        that exist within the quality array, otherwise it is ignored. 
        By default, this is 1 to N for each label in the array (by value
        increasing).
    '''
    if 'label' in kwargs.keys() \
            and np.size(kwargs['label']) == np.size(np.unique(quality['value'])):
        label_code = kwargs['label']
    else:
        label_code = np.arange(np.size(np.unique(quality['value'])))

    def time_to_index(index, t):
        '''
        Convert time to index of data vector using index array.
        '''

        timestep = 1 / index['frequency']
        index_endtime = index['starttime'] + (index['length'] - 1) * timestep

        if np.size(t) > 1:
            print('t must be of length 1, taking only first element of t')
            try:
                t = t.ravel()[0]
            except:
                return None
        if all(t < index['starttime']):
            block = 0
            t = index['starttime'][block]
            idx = index['startidx'][block]
        else:
            block = np.max(np.where(t >= index['starttime']))    
            if t > index_endtime[block]:
                block += 1
                t = index['starttime'][block]
                idx = index['startidx'][block]
            else:
                n = np.floor((t - index['starttime'][block]) / timestep[block])
                t = index['starttime'][block] + n * timestep[block]
                idx = index['startidx'][block] + n        
        return np.array([idx, t])

    label = np.zeros_like(quality['value'])
    for x in np.arange(np.size(np.unique(quality['value']))):
        label[np.where(quality['value'] == np.unique(quality['value'])[x])] = label_code[x]

    label = np.array([tuple((time_to_index(index, quality['time'][x])[0], 0, \
        label[x], quality['value'][x])) 
        for x in np.arange(np.size(quality))], \
            dtype = [('startidx', 'i8'), ('length', 'i8'), \
            ('label', 'i4'), ('label_original', 'i4')])
    label['length'][:-1] = label['startidx'][1:] - label['startidx'][:-1]
    label['length'][-1] = index['startidx'][-1] + index['length'][-1] - label['startidx'][-1]

    label = label[label['label'] != 0]

    return label
    # function quality_label end

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def join_label(label, index, **kwargs):
    '''
    Joins labels together when the the proportion of labelled to 
    non labelled in a particular region is high.

    Parameters
    ----------
    label: numpy array
        An array encoding the labelled regions - this can be output
        from a labelling function. 
        It has the following dtype.names:
            startidx - int
            length - int
            label - int
    index: numpy array
        An array encoding the t values. 
        The data is continuous and sampled at constant frequencies 
        in a finite number of blocks or chunks, potentially with gaps
        between blocks. The frequency of the data may also change 
        between blocks.
        It has the following dtype.names:
            startidx - int
            starttime - float
            length - int
            frequency - float
        These correspond to the index of the dataset corresponding to 
        the new block of data, the start time of that block (s), the
        number of data points in that block, the frequency of the 
        data (Hz).

    Keyword arguments
    ----------
    r: float
        Ratio of labelled to non labelled data above which the labels 
        are merged. This must be between 0 and 1 and by default is 0.7.
    '''
    if 'r' in kwargs.keys():
        r = kwargs['r']
    else:
        r = 0.7

    # within each block of data, the frequency is constant so fine to
    #   look at the ratio of labelled indices rather that the ratio of
    #   time that is labelled, but need to make sure we are within a
    #   block of data and not spanning more than one data block.

    x = 0

    while x + 1 < np.size(label):
        # the end index of a data block is the startidx of the next
        #   data block - 1. using the argmax in this way finds the 
        #   startidx of the next block.
        block_endidx = \
            index['startidx'][np.argmax(index['startidx'] > label['startidx'][x])] - 1
        # this is a bit tricky - if the ratio conditions holds for 
        #   d = d1 > 1, we don't actually care if it holds for all d in
        #   1,2,...,d1, only that it holds for d1. 
        # however, we do require that the label code is the same and 
        #   similarly that the label belongs to the same block for all
        #   d in 1,2,...,d1.
        # solution is to find the largest d, labelled d2, such that that
        #   second part holds (i.e. label same, block same), then find 
        #   the d1 such that the ratio condition holds
        d2 = np.argmin([label['label'][d] == label['label'][x] \
            and label['startidx'][d] < block_endidx \
            for d in np.arange(x + 1, np.size(label))])
        b = [np.sum(label['length'][x:(x + d + 1)]) > r * (label['startidx'][x + d] \
                + label['length'][x + d] - label['startidx'][x]) \
            for d in np.arange(1, d2 + 1)]
        if any(b):
            d1 = np.size(b) - np.argmax(np.flip(b))
        else:
            d1 = 0
        # z is a boolean vector, np.argmax returns array index of first 
        #   True, want the index of the last True value
        label['length'][x] = label['startidx'][x + d1] \
            + label['length'][x + d1] - label['startidx'][x]
        label = np.delete(label, np.arange(x + 1, x + d1 + 1))
        x += 1

    return label
    # function join_label end

def add_label(labels, index):
    '''
    Combine label arrays together or just add no label entries to an
    existing label structure.
    If more than one label array is given, combine the arrays into 
    a tuple before feeding through the function.
    '''

    # combine label arrays if there are multiple arrays
    if isinstance(labels, tuple):
        label = np.concatenate(labels)
        label = label[np.argsort(label['startidx'])]
    else:
        label = labels
    
    # remove 0 labels and add them back later
    label = label[label['label'] != 0]

    pos = [x for x in np.arange(np.size(label)) \
        if np.all(label['startidx'][x] > label['startidx'][:x] + label['length'][:x])]
    
    # if multiple labels overlap, it's a bit trickier to determine
    #   the start of a labelled region
    startidx = label['startidx'][pos]
    y = label['startidx'] + label['length']
    y = [np.max(y[pos[x]:pos[x + 1]]) for x in np.arange(np.size(pos) - 1)]
    y = np.append(y, label['startidx'][-1])
    length = np.array(y) - startidx

    # determine startidx and length of regions with no label
    nostartidx = np.insert(startidx + length, 0, 0)
    nolength = np.append(startidx - np.delete(nostartidx, -1), \
        index['startidx'][-1] + index['length'][-1] - nostartidx[-1])

    nolabel = np.array([tuple((nostartidx[x], nolength[x], 0)) \
        for x in np.arange(np.size(nostartidx))], \
            dtype = [('startidx', 'i4'), ('length', 'i8'), ('label', 'i8')])

    label = np.insert(label, np.append(pos, np.size(label)), nolabel[list(label.dtype.names)])

    return label
    # function add_label end

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def plot_index(index, **kwargs):
    '''
    Plots the data available, using index and label arrays.

    Gives a plot of the data available against time for each dataset, 
    including labelled data if it exists (or is given as a keyword
    argument).
    '''
    # ------------------------------------------------------------------------
    # LABEL

    if 'label' in kwargs.keys():
        label = kwargs['label']
    else:
        label = \
            np.array([(0, index['startidx'][-1] + index['length'][-1], 0)], \
                        dtype = [('label', 'i4'), ('length', 'i8'), ('startidx', 'i8')])

    def index_to_label_time(index, label):
        '''
        Recover the start and end times for the label array from the
        index array.
        '''
        m = np.shape(label)[0]
        time = np.zeros((m,), dtype = [('starttime', 'f8'), ('endtime', 'f8')])
        for x in np.arange(m):
            diff = label['startidx'][x] - index['startidx']
            index_block = np.max(np.where(diff >= 0))
            starttime = index['starttime'][index_block] \
                            + (diff[index_block] / index['frequency'][index_block])

            diff = label['startidx'][x] + label['length'][x] - index['startidx']
            index_block = np.max(np.where(diff > 0))
            endtime = index['starttime'][index_block] \
                            + (diff[index_block] / index['frequency'][index_block])
            time[x] = (starttime, endtime)

        return time
        # function index_to_label_time end

    if np.size(label) == 1:
        label = np.array([tuple((index['startidx'][x], index['length'][x], label['label'])) \
                                    for x in np.arange(np.size(index))], \
                            dtype = [('startidx', 'i8'), ('length', 'i8'), ('label','i8')])
    elif all(index['startidx'] < label['startidx'][-1]):
        # add index blocks within the labelled areas to label structure
        pos = [np.argmax(x < label['startidx']) for x in index['startidx'] \
                        if not all(x >= label['startidx']) and x not in label['startidx']]
        newstartidx = [x for x in index['startidx'] if not all(x > label['startidx']) \
                                                        and x not in label['startidx']]
        newlength = label['startidx'][pos] - newstartidx
        newlabel = label['label'][[x - 1 for x in pos]]

        newlabel = np.array([tuple((newstartidx[x], newlength[x], newlabel[x])) \
                                    for x in np.arange(np.size(newstartidx))], \
                            dtype = [('startidx', 'i8'), ('length', 'i8'), ('label','i8')])
        label = np.insert(label, pos, newlabel[list(label.dtype.names)])
        label['length'][pos + np.arange(np.size(pos)) - 1] = newstartidx \
            - label['startidx'][pos + np.arange(np.size(pos)) - 1]
    else:
        # add index blocks within the labelled areas to label structure
        m = np.argmax(index['startidx'] >= label['startidx'][-1])
        pos = [np.argmax(x < label['startidx']) for x in index['startidx'] \
                        if not all(x >= label['startidx']) and x not in label['startidx']]
        newstartidx = [x for x in index['startidx'] if not all(x > label['startidx']) \
                                                        and x not in label['startidx']]
        newlength = label['startidx'][pos] - newstartidx
        newlabel = label['label'][[x - 1 for x in pos]]
        newstartidx = np.append(newstartidx, index['startidx'][m:])
        newlength = np.append(newlength, index['length'][m:])
        newlabel = np.append(newlabel, np.repeat(label['label'][-1], np.size(index) - m))

        newlabel = np.array([tuple((newstartidx[x], newlength[x], newlabel[x])) \
                                    for x in np.arange(np.size(newstartidx))], \
                            dtype = [('startidx', 'i8'), ('length', 'i8'), ('label','i8')])
        pos = np.append(pos, np.repeat(np.size(label), np.size(index) - m)).astype(int)
        label = np.insert(label, pos, newlabel[list(label.dtype.names)])
        label['length'][pos + np.arange(np.size(pos)) - 1] = newstartidx \
            - label['startidx'][pos + np.arange(np.size(pos)) - 1]

    t = index_to_label_time(index, label)

    # ------------------------------------------------------------------------
    # CMAP

    # list of unique labels in order 0, -ve decreasing, +ve increasing
    labels = np.unique(label['label'])
    labels = np.append(np.flip(labels[labels <= 0]), labels[labels > 0])
    # convert labels from 0, -1, -2, 101, 102, 103, etc to 0, 1, 2, 3, 4, 5
    label_temp = label['label'].copy()
    for x in np.arange(np.size(labels)):
        label_temp[np.where(label['label'] == labels[x])] = x

    if 'cmap' in kwargs.keys():
        # needs to be RGB
        cmap = kwargs['cmap']
    else:
        if np.size(labels) > 1:
            cmap_temp = plt.get_cmap('tab10')
            cmap = np.vstack(((0, 0, 0), \
                [cmap_temp.colors[x] for x in np.mod(np.arange(np.size(labels) - 1), 10)]))
        else:
            cmap = np.array((0, 0, 0)).reshape(1, 3)

    # convert from RBG to RBGA
    if any([np.size(y) == 3 for y in cmap]):
        cmap = [list(x) for x in cmap]
        for x in [y for y in cmap if np.size(y) == 3]:
             cmap[cmap.index(x)].append(1)
        cmap = np.vstack(cmap)

    # ------------------------------------------------------------------------
    # PLOT
    
    y = 1

    y_vertical = 0.2

    dmin = y - 0.2 * y_vertical
    dmax = y

    verts = np.zeros((np.size(t) - 1, 4, 2))
    for ii in np.arange(np.size(t) - 1):
        ts = np.array([t['starttime'][ii], t['starttime'][ii + 1], \
            t['starttime'][ii + 1], t['starttime'][ii]])
        ys = np.array([dmin, dmin, dmax, dmax])
        verts[ii] = list(zip(ts, ys))

    from matplotlib.collections import PolyCollection
    poly = PolyCollection(verts, facecolors = cmap[label_temp], edgecolors = None)

    # poly.set_alpha(0.2)
    plt.gca().add_collection(poly)

    # segments = np.array([[[t['starttime'][x], y], [t['endtime'][x], y]] \
    #                                 for x in np.arange(np.size(t))])

    # lc = LineCollection(segments, colors = cmap[label_temp])
    # lc.set_linewidth(3)

    # if ax does not exists as a kwarg, then we need to create a new 
    #   axis object
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.axes()
        ax.set_yticks([])
        ax.set_xlim([t['starttime'][0], t['endtime'][-1]])

    # lines = ax.add_collection(lc)

    # add additional vertical lines up showing points where there is a 
    #   label (in case the labelled regions are small enough in 
    #   comparison to the data that they are not visible)
    segments_v = np.array([[[t['starttime'][x], y + y_vertical * 0.4], \
                            [t['starttime'][x], y + y_vertical * 0.8]] \
                                for x in np.arange(np.size(t)) if label_temp[x] != 0])

    lc_v = LineCollection(segments_v, colors = cmap[label_temp[label_temp != 0]])
    lc_v.set_linewidth(0.5)

    lines_vertical = ax.add_collection(lc_v)

    # also add start/end of vertical lines showing the start of each 
    #   data block
    segments_i = np.array([[[index['starttime'][x], y + y_vertical * 0.4], \
                                    [index['starttime'][x], y + y_vertical]] \
                                for x in np.arange(np.size(index))])

    lc_i = LineCollection(segments_i, colors = cmap[0])
    lc_i.set_linewidth(0.5)

    lines_index = ax.add_collection(lc_i)

    # set x and y limits based on existing x and y limits and new data
    xlim = ax.get_xlim()
    xlim = [np.min((xlim[0], t['starttime'][0])), np.max((xlim[1], t['endtime'][-1]))]
    ax.set_xlim(xlim)

    ax.set_ylim([0.8, 1.3])
    ax.set_yticks([])

    ax.set_xlabel('s')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # function plotindex_h5 end

def plot_block_section(data, index_block, label, starttime, endtime):
    startidx = (starttime - index_block['starttime']) * index_block['frequency'] \
        + index_block['startidx']
    endidx = (endtime - index_block['starttime']) * index_block['frequency'] \
        + index_block['startidx']

    startidx = np.floor(startidx).astype(int)
    endidx = np.ceil(endidx).astype(int)

    starttime = index_block['starttime'] + (startidx - index_block['startidx']) \
        / index_block['frequency']
    endtime = index_block['starttime'] + (endidx - index_block['startidx']) \
        / index_block['frequency']

    t = np.arange(starttime, endtime, 1 / index_block['frequency'])
    data = data[startidx:endidx]

    label = label[label['startidx'] > startidx]
    label = label[label['startidx'] < endidx]

    label = label[np.argsort(label['startidx'])]

    cmap = plt.get_cmap('tab10')
    cmap = np.vstack(((0, 0, 0), cmap.colors))

    for ii in np.arange(np.size(label)):
        length = label[ii]['length'] + 1
        start = label[ii]['startidx'] - startidx
        end = label[ii]['startidx'] - startidx + length
        plt.plot(t[start:end], data[start:end], \
            color = cmap[label[ii]['label']])

def index_to_time(index, idx):
    '''
    Go from index of data point to time of data point via index array
    '''
    block = np.argmin(idx >= index['startidx'])
    block += -1
    t = index['starttime'][block] + (idx - index['startidx'][block]) / index['frequency'][block]
    return t

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class ReturnValue(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
