import time
import ROOT
import rootpy
import root_numpy
import numpy
import math
import sys
import os
import glob

from scipy import misc as m
from WireDataUtils import *
from subprocess import call
import multiprocessing 

def preprocess(X):
    return X[:2,:,0:4096]    

def ProcessEvents(NEvents,infile,outfile,Offset=0):
    # ReadEvents
    f=ROOT.TFile(infile)
    t=f.Get("wiredump/anatree")
    if NEvents<=0:
        NEvents=t.GetEntries()
    
    EventList=range(0,NEvents)

    #Read one event.
    [example,Attributes]=ReadADCWire(t,[EventList[0]],NPlanes=2,samples=4500)

    X=preprocess(example[0])
    image_shape= X.shape
    assert image_shape == (2, 240, 4096)

    dtype       = 'float16' # 'float16' # Half-precision should be enough.
    compression = 'gzip'  #'gzip'
    chunksize   = 1       # We use chunksize=1 because we don't know how many events are in src file.
    chunkshape  = (chunksize,) + image_shape

    h5FileName=outfile
    h5out= h5py.File(h5FileName+".h5","w")
    N=len(EventList)
    
    # Initialize data sets.
    dsets = {}

    # Each event contains the following attributes.
    attributes = Attributes[0].keys()
    for attr in attributes:
        dsets[attr] = h5out.create_dataset(attr, (N,), dtype='float32')

    # Each event is an image of image_shape.
    dsets['features'] = h5out.create_dataset('features', (N,)+image_shape, chunks=chunkshape, dtype=dtype, compression=compression)
    
    for EventI in xrange(0,N):    
        [events,Attributes]=ReadADCWire(t,[EventList[EventI]],NPlanes=2,samples=4500)
        
        event=events[0]
        dsets['features'][EventI] = preprocess(event)

        for attr in Attributes[0]:
            dsets[attr][EventI] = Attributes[0][attr]

    f.Close()

    h5out.close()

    return True


###################################################    

InputDir=sys.argv[1]

if len(sys.argv)>2:
    OutputDir=sys.argv[2]
else:
    OutputDir="./"

if OutputDir[:-1]!="/":
    OutputDir=OutputDir+"/"

print "Reading Directory:",InputDir
print "Output Dir:", OutputDir

#Make the directories

if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)

Offset=0

if len(sys.argv)>4:
    Offset=max(long(sys.argv[4]),Offset)

NEvents=0

if len(sys.argv)>3:
    if long(sys.argv[3])>0:
        NEvents=long(sys.argv[3])

print "NEvents per file: ",NEvents
print "Offset: ",Offset

files = glob.glob(InputDir + '/*/*/wire_dump*.root')
print 'Found %d files.' % len(files)


def wrapper(filename):
    basename   = os.path.basename(filename)
    fout       =  '/' + basename.split(".")[-2]
    fout = fout.split("_")[-1]
    # Construct name from Docker Subdirectories
    dockername       = filename.split("/")[-2]
    particlename = filename.split("/")[-3]
    fout = OutputDir +particlename+"_"+fout + "-" + dockername

    print fout,
    
    if not os.path.isfile(fout+".h5"): 
        output=ProcessEvents(NEvents,filename,fout,
                             Offset=Offset)
        pass
    else:
        print "Exists. Skipping.",

    print "Done."
    return

num_threads=48

#wrapper(files[0])

tic = time.clock()
try:
    pool = multiprocessing.Pool(num_threads)
    pool.map(wrapper, files)
except:
    print "Error____"
finally:
    pool.close()
    pool.join()
print time.clock() - tic
