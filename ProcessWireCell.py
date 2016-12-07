from ROOT import *
from root_numpy import hist2array
from SparseTensorDataSet import *
import os, sys


def Plot3DTrack(X, Y, Z, V):
    import matplotlib.pyplot as plt
    import plotly.plotly as py
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    fig, ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=V)
    fig.show()
    return ax


def PlotSparseTensor(T, i):
    return Plot3DTrack(T.C[i][0:100, 0],
                       T.C[i][0:100, 1],
                       T.C[i][0:100, 2],
                       T.V[i][0:100])


def ProcessRootFile(filein="WireDump_3D_electron_1462146937.root", fileout=False, N_Max=-1, Sparse2D=False):
    Rf = TFile.Open(filein)
    t = Rf.Get("wiredump/anatree")

    bins3D = [240, 240, 4096]
    bins2D = [480, 4096]

    N = t.GetEntries()

    if N_Max > 0:
        N = min(N, N_Max)

    images3D = SparseTensorDataSet(bins3D, unbinned=True)
    if Sparse2D:
        images2D = SparseTensorDataSet(bins2D, unbinned=False)
    else:
        d = np.zeros((N, 480, 4096))

    for i in range(N):
        t.GetEntry(i)
        print(i, ",")
        sys.stdout.flush()

        images3D.C.append(np.array([t.simide_x, t.simide_y, t.simide_z]).transpose())
        images3D.V.append(np.array(t.simide_numElectrons))

        Cs = []
        Vs = []

        MaxSamples = 4096

        # if Sparse2D:
        #     d=np.zeros((t.raw_nChannel,MaxSamples))
        #     for j in xrange(t.raw_nChannel):
        #         h=t.raw_wf[j]
        #         d[j]=hist2array(h)[:MaxSamples]

        #     images2D.AppendFromDense(d)
        # else:
        #     for j in xrange(t.raw_nChannel):
        #         h=t.raw_wf[j]
        #         d[i][j]=hist2array(h)[:MaxSamples]

    if not fileout:
        fileout2D = os.path.basename(filein) + ".2d.h5"
        fileout3D = os.path.basename(filein) + ".3d.h5"

    if Sparse2D:
        f = open_file(fileout2D, "w")
        images2D.Writeh5(f, "images2D")
        f.close()
    else:
        f = h5py.File(fileout2D, "w")
        dset = f.create_dataset("images2D", (N, 480, 4096), compression="gzip")
        dset[...] = d
        f.close()

    f = open_file(fileout3D, "w")
    images3D.Writeh5(f, "images3D")
    f.close()

    # return t,Rf,images2D,images3D


if __name__ == '__main__':
    # Main

    t = ProcessRootFile(Sparse2D=False, N_Max=10)

    # PlotSparseTensor(images,2)


    images3D = SparseTensorDataSet()

    #    f=h5py.File("celltree_SimChannel_Raw.root.3d.h5","r")
    #    images3D.Readh5(f,"images3D")
    #   f.close()
    # images2D=SparseTensorDataSet()
    # f=h5py.File("celltree_SimChannel_Raw.root.2d.h5","r")
    # images2D.Readh5(f,"images2D")
    #   f.close()


#    f=h5py.File("celltree_SimChannel_Raw.root.2d.h5","r")
#    images2D=np.array(f["images2D"])
#    f.close()

# hAll1=images1.histogramAll()

# NValues=np.prod(images.shape)*images.len()
# T=hAll[0]==hAll1[0]
# NGood=np.sum(T)
# print "Number of non-matching values",NValues-NGood ,"/",NValues

# print "The Difference:"
# print hAll[0][np.where(T==False)]-hAll1[0][np.where(T==False)]
# print "Average difference of mismatch terms:", np.sum( hAll[0][np.where(T==False)]-hAll1[0][np.where(T==False)])/(NValues-NGood)


#            for k in xrange(0,4096): #h.GetNbinsX()+1):
#                V=h.GetBinContent(k+1)
#                if V!=0.:
#                   Cs.append((j,k))
#                  Vs.append(V)

#        images2D.C.append(np.array(Cs))
#        images2D.V.append(np.array(Vs))
