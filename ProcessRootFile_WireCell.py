import time
import ROOT
import rootpy
import root_numpy
import numpy
import math
import sys
import os
import glob

sys.path.append(os.path.abspath("/opt/Wirecell_Root_Procssing/"))

from SparseTensorDataSet import *

from scipy import misc as m
from WireDataUtils import *
# from subprocess import call
# import multiprocessing


def preprocess(X):
    return X[:2, :, 0:4096]


def ProcessEvents(NEvents, infile, outfile2D, outfile3D, Offset=0):
    bins3D = [240, 240, 4096]
    # bins2D=[480,4096]

    # ReadEvents
    f = ROOT.TFile(infile)
    t = f.Get("wiredump/anatree")
    if NEvents <= 0:
        NEvents = t.GetEntries()

    EventList = range(0, NEvents)

    # Read one event.
    [example, Attributes] = ReadADCWire(t, [EventList[0]], NPlanes=2, samples=4500)

    X = preprocess(example[0])
    image_shape = X.shape
    assert image_shape == (2, 240, 4096)

    dtype = 'float16'  # 'float16' # Half-precision should be enough.
    compression = 'gzip'  # 'gzip'
    chunksize = 1  # We use chunksize=1 because we don't know how many events are in src file.
    chunkshape = (chunksize,) + image_shape

    h5FileName2D = outfile2D
    h5out2D = h5py.File(h5FileName2D + ".2d.h5", "w")
    h5FileName3D = outfile3D
    # h5out3D= h5py.File(h5FileName3D+".3d.h5","w")

    # file to save 3D images
    # fileout3D=os.path.basename(infile)+".3d.h5"
    f_3D = open_file(h5FileName3D + ".3d.h5", "w")
    # images3D.Writeh5(h5out3D,"images3D")
    # f.close()


    N = len(EventList)

    # Initialize data sets.
    dsets = {}

    # Each event contains the following attributes.
    attributes = Attributes[0].keys()
    for attr in attributes:
        dsets[attr] = h5out2D.create_dataset(attr, (N,), dtype='float32')

    # Each event is an image of image_shape.
    dsets['features'] = h5out2D.create_dataset('features', (N,) + image_shape, chunks=chunkshape, dtype=dtype,
                                               compression=compression)
    # getting 3D sparse matrix for 3D imaging
    images3D = SparseTensorDataSet(bins3D, unbinned=True)

    for EventI in range(0, N):

        [events, Attributes] = ReadADCWire(t, [EventList[EventI]], NPlanes=2, samples=4500)

        event = events[0]
        dsets['features'][EventI] = preprocess(event)

        for attr in Attributes[0]:
            dsets[attr][EventI] = Attributes[0][attr]

        images3D.C.append(np.array([t.simide_x, t.simide_y, t.simide_z]).transpose())
        images3D.V.append(np.array(t.simide_numElectrons))

        Cs = []
        Vs = []
        MaxSamples = 4096

    images3D.Writeh5(f_3D, "images3D")

    f.Close()

    h5out2D.close()
    f_3D.close()
    # h5out3D.close()

    return True


###################################################    

InputDir = sys.argv[1]

if len(sys.argv) > 2:
    OutputDir2D = sys.argv[2]
else:
    OutputDir2D = "./"

if OutputDir2D[:-1] != "/":
    OutputDir2D = OutputDir2D + "/"

if len(sys.argv) > 3:
    OutputDir3D = sys.argv[3]
else:
    OutputDir3D = "./"

if OutputDir3D[:-1] != "/":
    OutputDir3D = OutputDir3D + "/"

print("Reading Directory:", InputDir)
print("Output Dir 2d:", OutputDir2D)
print("Output Dir 3d:", OutputDir3D)

# Make the directories

if not os.path.exists(OutputDir2D):
    os.makedirs(OutputDir2D)
if not os.path.exists(OutputDir3D):
    os.makedirs(OutputDir3D)

Offset = 0

if len(sys.argv) > 4:
    Offset = max(long(sys.argv[4]), Offset)

NEvents = 0

if len(sys.argv) > 4:
    if long(sys.argv[4]) > 0:
        NEvents = long(sys.argv[4])

print("NEvents per file: ", NEvents)
print("Offset: ", Offset)

# files = glob.glob(InputDir + '/*/*/wire_dump*.root')
# print('Found %d files.' % len(files))


def wrapper(filename):
    basename = os.path.basename(filename)
    fout = '/' + basename.split(".")[-2]
    fout = fout.split("_")[-1]
    # Construct name from Docker Subdirectories
    dockername = filename.split("/")[-2]
    particlename = filename.split("/")[-3]
    fout2d = OutputDir2D + particlename + "_" + fout + "-" + dockername
    fout3d = OutputDir3D + particlename + "_" + fout + "-" + dockername

    print("2D File: ", fout2d, "3D File:  ", fout3d)

    if not os.path.isfile(fout + ".h5"):
        output = ProcessEvents(NEvents, filename, fout2d, fout3d,
                               Offset=Offset)
        pass
    else:
        print("Exists. Skipping.")

    print("Done.")
    return

wrapper(InputDir)

"""
num_threads=20

#wrapper(files[0])

tic = time.clock()
try:
    pool = multiprocessing.Pool(num_threads)
    pool.map(wrapper, files)
except:
    print "Error"
finally:
    pool.close()
    pool.join()
print time.clock() - tic
"""
