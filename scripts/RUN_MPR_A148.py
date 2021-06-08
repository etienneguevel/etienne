"""Usage: example_run.py -G VALUE -n VALUE -o FILE  [-l LENGTH] [-c CUTIN] [-s SEED]

-G VALUE        global coupling scaling
-n VALUE        noise sigma
-o FILE         output file (npz)
-l LENGTH       simulation total length in ms/10 [default: 60e3]
-c CUTIN        first cut_in milliseconds of simulation to exclude [default: 1200]
-s SEED         random number generator seed [default: 42]
-h --help       show this
"""

from docopt import docopt
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
import time
from src import analysis, simulation  # Import analysis for fcd and clustering
from tvb.simulator.lab import *


if __name__ == '__main__':
    args      = docopt(__doc__)
    G         = float(args["-G"])
    nsigma    = float(args["-n"])
    out_path  = args["-o"]
    sim_len   = int(args["-l"])
    cut_in    = int(args["-c"])
    seed      = int(args["-s"])

    print('GOT INPUTS', flush=True)
    
    sys.stdout = open(f'{out_path}.log', 'w')
    # sys.stderr = sys.stdout
    print('GOT INPUTS2', flush=True)
    eta       =-4.6
    J         =14.5
    Delta     =0.7
    tau       =1.
    print('GOT INPUTS3', flush=True)
    Bperiod   =300
    Tperiod   =10.
    print('GOT INPUTS4', flush=True)
    T_len=int(sim_len*Tperiod)
    B_len=int(T_len/Bperiod)

    print('GOT INITIAL PARAMETERS', flush=True)

    #print(args)
    print(str(tau), flush=True)
    print(str(G), flush=True)

    #STRUCTURAL CONNECTIVITY
    path='/home/giovanni.rabuffo/fufo/data/connectivity/'
    A148_con = connectivity.Connectivity.from_file(path+'Allen_148.zip')
    nregions = len(A148_con.weights)     #number of regions
    A148_con.speed = np.asarray(np.inf)  #set the conduction speed
    np.fill_diagonal(A148_con.weights, 0.)
    A148_con.weights = A148_con.weights/np.max(A148_con.weights)
    A148_con.configure()
    A148_SC = A148_con.weights


    # Initialise the Simulator.

    # Connectivity
    #conn= connectivity.Connectivity(weights=A148_SC,region_labels=np.array(SC_labels,dtype='<U128'),tract_lengths=A148_con.tract_lengths,speed=np.array(2.,dtype=np.float),areas =np.zeros(np.shape(A148_SC)[0]),centres = A148_con.centres)
    conn               = connectivity.Connectivity(
        weights= A148_SC,
	    region_labels=np.array(ROIs,dtype='<U128'),
	    tract_lengths=A148_con.tract_lengths,
	    speed=np.array(2.,dtype=np.float),
	    areas =np.zeros(np.shape(A148_SC)[0]),
	    centres = A148_con.centres) 
   
    #LOCAL MODEL
    mpr = models.MontbrioPazoRoxin(
        J     = np.r_[J],
        Delta = np.r_[Delta],
        tau   = np.r_[tau],
        eta   = np.r_[eta],
    )

    con_coupling = coupling.Scaling(a=np.r_[G])     
    dt = 0.0025 #integration steps [ms]
    hiss = noise.Additive(nsig=np.array([nsigma,nsigma*2]))
    heunint = integrators.HeunStochastic(dt=dt, noise=hiss)


    sim = simulator.Simulator(
    	model=mpr,
    	connectivity=conn,
    	coupling=coupling.Scaling(a=np.r_[G]),
    	conduction_speed=np.Inf,
        integrator=integrators.HeunStochastic(dt=dt,
        	noise=noise.Additive(nsig=np.r_[nsigma/(tau), nsigma*2],noise_seed=seed)
        	),
        monitors=[monitors.TemporalAverage(period=1./Tperiod)]
    )

    sim.configure()


    print('SIMULATION CONFIGURED', flush=True)

    t0      = time.time()


    (Tavg_time, Tavg_data), = simulation.run_nbMPR_backend(sim, simulation_length=sim_len+cut_in)
    Tavg_time *= Tperiod # rescale time

    if True not in np.unique(np.isnan(Tavg_data)):
        print('NO NAN IN TAVG', flush=True)
        Bold_time, Bold_data = simulation.tavg_to_bold(Tavg_time, Tavg_data, tavg_period=1., connectivity=sim.connectivity, svar=0, decimate=Bperiod)
    
        Tavg_data      = Tavg_data[int(cut_in*Tperiod):,:,:,0]
        Bold_data      = Bold_data[int(cut_in*Tperiod/Bperiod):,:,:,0]


        np.savez(out_path, Bold_data = Bold_data)

    else:
        print('NAN IN TAVG', flush=True)


    CPU_TIME    = time.time() - t0
    print(f'CPU TIME {CPU_TIME}', flush=True)