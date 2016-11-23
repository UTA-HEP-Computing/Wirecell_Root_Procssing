import numpy
import numpy as np
import h5py
import math
import ROOT

def DownSample(Data,factor,Nx,Ny,sumabs=False):
    if factor==0:
        return np.reshape(Data,[Nx,Ny]),Ny

    # Remove entries at the end so Down Sampling works
    NyNew=Ny-Ny%factor
    Data1=np.reshape(Data,[Nx,Ny])[:,0:NyNew]
    
    # DownSample 
    if sumabs:
        a=abs(Data1.reshape([Nx*NyNew/factor,factor])).sum(axis=1).reshape([Nx,NyNew/factor])
    else:
        a=Data1.reshape([Nx*NyNew/factor,factor]).sum(axis=1).reshape([Nx,NyNew/factor])

    return a,NyNew

def GetXWindow(Data,i,BoxSizeX):
    return Data[:,i:i+BoxSizeX]


def ScanWindow(Data,BoxSizeX=256,Nx=240,Ny=4096):

    NyNew=Ny
    #Scan the Window
    b=np.array([0]*(NyNew-BoxSizeX))

    for i in xrange(0,NyNew-BoxSizeX):
        b[i]=GetXWindow(Data,i,BoxSizeX).clip(0,99999999999).sum()

    #Find the Window with max Energy/Charge
    BoxStart=b.argmax()
    MaxSum=b[BoxStart]

    #Store the window
    Box=Data[:,BoxStart:BoxStart+BoxSizeX]

    return Box,BoxStart,MaxSum


def ReadADCWire(t,EventList=[0],NPlanes=2,samples=4500):

    Events=[]
    Attributes=[]

    for NEvent in EventList:

        if (NEvent>-1):
            t.GetEntry(NEvent)

            x=numpy.array(t.WireADC)
            a_size = t.geant_list_size 
            #print a_size
            if NPlanes<3:
                y=numpy.pad(x,(0,t.nWires/NPlanes*samples),mode='constant')
            else:
                y=x

            z=numpy.reshape(y,[NPlanes+1,t.nWires/NPlanes,samples])
            Events+=[z]
            #Creat Dictionary
            AA={}
	#getting information for the line equation
            #print "out"
#            print numpy.array(t.StartPointx)[0]
            #if (numpy.array(t.process_primary)[a_size] == 0) and (numpy.array(t.pdg)[a_size] == 11 or numpy.array(t.pdg)[a_size] == 13)
            for PrimaryI in xrange(0,a_size):
                if t.process_primary[PrimaryI]==1:
                    break

            #print "PrimaryI:", PrimaryI
     
            #print "xi  ",xi

            # [xi,yi,zi]= [t.StartPointx[PrimaryI]-t.EndPointx[PrimaryI],
            #              t.StartPointy[PrimaryI]-t.EndPointy[PrimaryI],
            #              t.StartPointz[PrimaryI]-t.EndPointz[PrimaryI]]
            
            AA["Track_length"]= numpy.array(t.Track_length)[PrimaryI] #<---- here is now computed in WireDump
            AA["pdg"] = numpy.array(t.pdg)[PrimaryI]
            AA["Eng"] = numpy.array(t.Eng)[PrimaryI]
             #Creat Dictionary
            #AA={}

#            AA["Foo"]=t.bar
	    AA["enu_truth"]=numpy.array(t.enu_truth)
            AA["lep_mom_truth"]=numpy.array(t.lep_mom_truth)
            AA["mode_truth"]=numpy.array(t.mode_truth)
            #AA["Track_length"]=math.sqrt(math.pow((xs-x_i),2)+math.pow((ys-y_i),2)+math.pow((zs-z_i),2))
            Attributes+=[AA]

    return [Events,Attributes]



#def LineEquation_point(



 
