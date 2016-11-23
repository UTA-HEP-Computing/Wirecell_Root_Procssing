from SparseNDArray import *
from tables import *
from scipy.sparse import find
import numpy as np
import h5py

class SparseTensorDataSet:
    def __init__(self,shape=(), default=0, unbinned=False,dtype="float32"):
        self.unbinned=unbinned
        self.shape=tuple(shape)
        self.__default = default #default value of non-assigned elements
        self.ndim = len(shape)
        self.dtype = dtype
        self.C = []  # This will hold the sparse ND arrays
        self.V = []  # This will hold the sparse ND arrays

    def append(self,Coordinates,Values):
        Cs=[]
        Vs=[]
        
        for C,V in zip(Coordinates,Values):
           Cs.append(tuple(C))
           Vs.append(V)
           
        self.C.append(Cs)
        self.V.append(Vs)

    def convertToBinned(self):
        ## Not completed... need to flatten, use find, and unflatten, and store in self.C,self.V. Requires reindexing.
        N=self.len()

        for i in xrange(B):
            out,binedges=self.histogram(i,bins)
            find(out)

        self.binedges=binedges
        
        
    def sparse(self,i):
        a=sparray(self.shape,default=self.__default, dtype=self.dtype)

        for C,V in zip(self.C[i],self.V[i]):
            a[C]=V

        return a

    def histogram(self,i,bins=False):
        if not (isinstance(bins,list) or isinstance(bins,tuple) ):
            bins=self.shape
        # returns histogram and bin edges
        return np.histogramdd(self.C[i],bins=list(bins),weights=self.V[i])

    def histogramAll(self,range=False,bins=False):
        if not (isinstance(bins,list) or isinstance(bins,tuple) ):
            bins=self.shape
        if range:
            N=range[1]-range[0]
        else:
            N=self.len()
        out=np.zeros((N,)+tuple(bins))

        if not range:
            range=xrange(N)
        else:
            range=xrange(range[0],range[1])

        for i in range:
            out[i],binedges=self.histogram(i,bins)
            
        return out,binedges
    
    # This only makes sense if the coordinates are integers
    def dense(self,i):
        if self.unbinned:
            return self.histogram(i)[0]
        
        a=np.zeros((1,)+self.shape,dtype=self.dtype)

        for C,V in zip(self.C[i],self.V[i]):
            a[tuple(C)]=V

        return a

    def denseAll(self,range=[]):
        if self.unbinned:
            return self.histogramAll(range)[0]

        if len(range)>0:
            Start=range[0]
            Stop=range[1]
        else:
            Start=0
            Stop=len(self.C)

        a=np.zeros((Stop-Start,)+tuple(self.shape),dtype=self.dtype)
        
        for i in xrange(Start,Stop):
            for C,V in zip(self.C[i],self.V[i]):
                try:
                    a[i-Start][tuple(C)]=V
                except:
                    print "Reached End of Sample."

        return a
            
        
    def Writeh5(self,h5file,name,range=[]):
        root = h5file.create_group("/",name,name)
        FILTERS = Filters(complib='zlib', complevel=5)

        if self.unbinned:
            CT=h5file.create_vlarray(root,"C", Float32Atom(shape=len(self.shape)),"",filters=FILTERS)
        else:
            CT=h5file.create_vlarray(root,"C", Int32Atom(shape=len(self.shape)),"",filters=FILTERS)
        VT=h5file.create_vlarray(root,"V", Float32Atom(),"",filters=FILTERS)        
        h5file.create_array(root,"shape",self.shape)
        h5file.create_array(root,"unbinned",[int(self.unbinned)])
        
        if len(range)>0:
            Start=range[0]
            Stop=range[1]
        else:
            Start=0
            Stop=len(self.C)

        for i in xrange(Start,Stop):
            CT.append(self.C[i])
            VT.append(self.V[i])

    def Readh5(self,f,name,range=[]):
        self.shape=np.array(f[name]["shape"])
        self.unbinned=bool(f[name]["unbinned"][0])
        if len(range)>0:
            Start=range[0]
            Stop=range[1]
            self.C=f[name]["C"][Start:Stop]
            self.V=f[name]["V"][Start:Stop]

        else:
            self.C=f[name]["C"]
            self.V=f[name]["V"]

    def Readh5Files(self,filelist,name):
        for filename in filelist:
            f=h5py.File(filename,"r")
            self.shape=np.array(f[name]["shape"])

            try:
                self.C=np.concatenate(self.C,f[name]["C"])
                #                self.V=np.concatenate(self.V,f[name]["V"])
                self.V+=list(self.V,f[name]["V"])
            except:
                self.C=np.array(f[name]["C"])
                self.V=list(f[name]["V"])
            f.close()

    def Readh5Files2(self,filelist,name):
        for filename in filelist:
            f=h5py.File(filename,"r")
            self.shape=np.array(f[name]["shape"])

            try:
                self.C=f[name]["C"]
                #                self.V=np.concatenate(self.V,f[name]["V"])
                self.V+=f[name]["V"]
            except:
                self.C=np.array(f[name]["C"])
                self.V=list(f[name]["V"])
            f.close()

            
    def FromDense(self,A,Clear=False):
        if Clear:
            self.C=[]
            self.V=[]
        for a in A:
            X,Y,V=find(a)
            C=np.array([X,Y])
            C=C.transpose()
            
            self.C.append(tuple(C))
            self.V.append(V)

    def AppendFromDense(self,a):

        X,Y,V=find(a)
        C=np.array([X,Y])
        C=C.transpose()
            
        self.C.append(tuple(C))
        self.V.append(V)


            
    def len(self):
        try:
            N=len(self.C)
        except:
            N=self.C.shape[0]
        return N

            
    def DenseGenerator(self,BatchSize,Wrap=True):
        Done=False
        N=self.len()
        while not Done:
            for i in xrange(0,N-BatchSize,BatchSize):  # May miss some Examples at end of file... need better logic
                yield self.denseAll([i,i+BatchSize])
                Done=not Wrap
                    
if __name__ == '__main__':
    # Main
    import h5py
    import time

    shape=(10000,10,360)
    density=0.1
    batchsize=100
        
    N_Examples=shape[0]
    N_Vals=np.prod(shape)
    
    print "Testing with tensor size", shape, " and density",density,"."

    # Generate Some Sparse Data
    start=time.time()
    Vals=np.array(np.random.random(int(N_Vals*density)),dtype="float32")
    Zs=np.zeros(N_Vals-Vals.shape[0],dtype="float32")
    Train_X=np.concatenate((Vals,Zs))
    np.random.shuffle(Train_X)
    Train_X=Train_X.reshape(shape)
    print "Time Generate Sparse Data (into a Dense Tensor):",time.time()-start

    # Now Create the sparse "Tensor"
    X=SparseTensorDataSet(Train_X.shape[1:])
    
    # Test making the data sparse. (Only works for 2d right now)
    start=time.time()
    X.FromDense(Train_X)
    print "Time to convert to Sparse:",time.time()-start

    print "Sparsity Achieved: ",float(sum(map(len,X.C)))/N_Vals

    # Write to File
    start=time.time()
    f=open_file("TestOut.h5","w")
    X.Writeh5(f,"Data")
    f.close()
    print "Time to write out:",time.time()-start
    
    # Read back
    start=time.time()
    XX=SparseTensorDataSet(Train_X.shape[1:])
    f=h5py.File("TestOut.h5","r")
    XX.Readh5(f,"Data")
    f.close()
    print "Time to read back:",time.time()-start
    
    # Try to reconstruct the original data
    start=time.time()
    XXX=XX.denseAll()
    print "Time to convert to Dense:",time.time()-start
    
    # Test
    print
    NValues=np.prod(Train_X.shape)
    T=Train_X==XXX
    NGood=np.sum(T)
    print "Number of non-matching values",NValues-NGood ,"/",NValues

    print "The Difference:"
    print XXX[np.where(T==False)]-Train_X[np.where(T==False)]
    print "Average difference of mismatch terms:", np.sum( XXX[np.where(T==False)]-Train_X[np.where(T==False)])/(NValues-NGood)

    start=time.time()
    print "Generator batchsize:",batchsize
    for D in XX.DenseGenerator(batchsize,False):
        print ".",

    print
    print "Time to run generator:",time.time()-start

