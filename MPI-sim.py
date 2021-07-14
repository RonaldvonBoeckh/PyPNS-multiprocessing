#to run script use mpiexec -n {noThreads} python MPI-sim.py'
# can use --oversubscribe after the {nothreads} if you desire to force a higher thread count

from PyPNS import excitationMechanismClass
from mpi4py import MPI

from PyPNS.axonClass import *
import PyPNS.createGeometry as createGeometry

from PyPNS.takeTime import *
import PyPNS.constants as constants
import PyPNS.silencer as silencer

import numpy as np # for arrays managing
import time
import shutil
import copy

import matplotlib as mpl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from PyPNS.extracellularMechanismClass import precomputedFEM as precomputedFEM

from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

from PyPNS.nameSetters import *

import PyPNS
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
matplotlib.use('TkAgg')

#initialize the MPI object
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.Get_size()    #number of parallel processes

#let rank 0 do the initializing of the bundle and bcast relevant info to other ranks
if rank == 0:
    # ---------------------------------------------------------------------------
    # --------------------------------- DEFINITION ------------------------------
    # ---------------------------------------------------------------------------

    # ----------------------------- simulation params ---------------------------

    tStop=50
    dt=0.0025

    # ----------------------------- axon params ---------------------------

    # diameters enlarged for quicker execution
    myelinatedParameters = {'fiberD': {'distName': 'normal', 'params': (1.7, 0.4)}}
    unmyelinatedParameters = {'fiberD': {'distName': 'normal', 'params': (1.0, 0.2)}}

    segmentLengthAxon = 15
    rdc = 0.2 # random direction 

    # ----------------------------- bundle params -------------------------------

    # set length of bundle and number of axons
    bundleLength = 40000
    nAxons = 4
    # bundle guide
    bundleGuide = PyPNS.createGeometry.get_bundle_guide_straight(bundleLength, segmentLengthAxon)

    # ------------------------ intracellular stimulation params -----------------

    # parameters of signals for stimulation
    rectangularSignalParamsIntra = {'amplitude': 50., #50,  # Pulse amplitude (mA)
                                    'frequency': 20.,  # Frequency of the pulse (kHz)
                                    'dutyCycle': 0.5,  # Percentage stimulus is ON for one period (t_ON = duty_cyle*1/f)
                                    'stimDur': 0.05,  # Stimulus duration (ms)
                                    'waveform': 'MONOPHASIC',  # Type of waveform either "MONOPHASIC" or "BIPHASIC" symmetric
                                    'delay': 0.,  # ms
                                    # 'invert': True,
                                    # 'timeRes': timeRes,
                                    }

    intraParameters = {'stimulusSignal': PyPNS.signalGeneration.rectangular(**rectangularSignalParamsIntra)}

    # ------------------------- extracellular stimulation params -----------------

    rectangularSignalParamsExtra = {'amplitude': 3000, # Pulse amplitude (nA)
                                    'frequency': 1,  # Frequency of the pulse (kHz)
                                    'dutyCycle': 0.5, # Percentage stimulus is ON for one period (t_ON = duty_cyle*1/f)
                                    'stimDur': 1.,  # Stimulus duration (ms)
                                    'waveform': 'MONOPHASIC', # Type of waveform either "MONOPHASIC" or "BIPHASIC" symmetric
                                    'delay': 0.,  # ms
                                    # 'invert': True,
                                    # 'timeRes': timeRes,
                                    }

    elecPosStim = PyPNS.createGeometry.circular_electrode(bundleGuide, positionAlongBundle=12500, radius=235,
                                                        numberOfPoles=2, poleDistance=1000)
    extPotMechStim = PyPNS.Extracellular.precomputedFEM(bundleGuide) # , 'oil190Inner50Endoneurium')

    extraParameters = {'stimulusSignal': PyPNS.signalGeneration.rectangular(**rectangularSignalParamsExtra),
                    'electrodePositions': elecPosStim,
                    'extPotMech': extPotMechStim}

    # ----------------------------- recording params -------------------------------

    recordingParametersNew = {'bundleGuide': bundleGuide,
                            'radius': 220,
                            'positionAlongBundle': bundleLength*0.5,
                            'numberOfPoles': 1,
                            'poleDistance': 1000,
                            }

    electrodePoints = PyPNS.createGeometry.circular_electrode(**recordingParametersNew)

    extracellularMechs = []
    extracellularMechs.append(PyPNS.Extracellular.homogeneous(sigma=1))
    extracellularMechs.append(PyPNS.Extracellular.precomputedFEM(bundleGuide))
    extracellularMechs.append(PyPNS.Extracellular.analytic(bundleGuide))

    # ------------------------------------------------------------------------------
    # --------------------------- PyPNS object instantiation  -----------------------
    # ------------------------------------------------------------------------------

    # set all properties of the bundle
    bundleParameters = {'radius': 180,  #um Radius of the bundle (match carefully to extracellular mechanism)
                        'randomDirectionComponent': rdc,
                        'bundleGuide': bundleGuide,

                        'numberOfAxons': nAxons,  # Number of axons in the bundle
                        'pMyel': 1.,  # Percentage of myelinated fiber type A
                        'pUnmyel': 0.,  # Percentage of unmyelinated fiber type C
                        'paramsMyel': myelinatedParameters,  # parameters for fiber type A
                        'paramsUnmyel': unmyelinatedParameters,  # parameters for fiber type C

                        'tStop': tStop,
                        'timeRes': dt,

                        # 'saveI':True,
                        'saveV': False,

                        # 'numberOfSavedSegments': 50,  # number of segments of which the membrane potential is saved to disk
                        }

    # create the bundle with all properties of axons
    bundle = PyPNS.Bundle(**bundleParameters)

    # spiking through a single electrical stimulation
    bundle.add_excitation_mechanism(PyPNS.StimIntra(**intraParameters))
    bundle.add_excitation_mechanism(PyPNS.StimField(**extraParameters))


    #need to bcast required info to all processes
    #{
    # ABOUT BCAST:
    # bcast pickles the data and sends it to all processes. The two non-pickle-able objects
    # in the bundle are (1) recording mechanisms and (2) axons objects. These are therefore 
    # instantiated in the processes. Without these, the bundle object can be bcast to all processes
    # from the rank 0 which did the initializing}
    bcast_data = (bundle, recordingParametersNew)   #recordingParametersNew is to create rec mechanisms
#for all other ranks, set to None
else:
    bcast_data = None

#get the bcast data and set it to readable variables
bcast_data = comm.bcast(bcast_data, root=0)     #all ranks now have the required dat
bundle = bcast_data[0]
recordingParametersNew = bcast_data[1]

# #now we instantiate the recording mechansisms in every rank
electrodePoints = PyPNS.createGeometry.circular_electrode(**recordingParametersNew)
extracellularMechs = []
extracellularMechs.append(PyPNS.Extracellular.homogeneous(sigma=1))
extracellularMechs.append(PyPNS.Extracellular.precomputedFEM(recordingParametersNew['bundleGuide']))
extracellularMechs.append(PyPNS.Extracellular.analytic(recordingParametersNew['bundleGuide']))

for extracellularMech in extracellularMechs:
    bundle.add_recording_mechanism(PyPNS.RecordingMechanism(electrodePoints, extracellularMech))

#{create axon objects contianing the NEURON HocObject
#   this is a custom function I created which simply moves the axon instatiation
#   away from the class init() function into its own so the axons can be instantiated
#   when needed
# }
bundle.create_axon_objects()

#establishes which ranks simulate which axons. If 4 axons, 3 procs: rank0->a0,a3; rank1->a1, rank2->a2
axonsToSim = int(bundle.numberOfAxons/nprocs)
if rank < (bundle.numberOfAxons%nprocs):
    axonsToSim += 1

axonIndexList = []
for count in range(axonsToSim):
    axonIndexList.append(rank + count* nprocs)

#simulate the axons depending on the process rank using its axon sim list
#   this is also a modified function that simply simulates the desired axons instead of all axons
sim_times = bundle.simulate_axons(axonIndexList)

#TODO: Now we need to gather the CAP output of the simulations and sum them all. Not done yet
#   since the simulations still take longer, so need to fix that before

