'''
functions for model
'''
from   neuron           import h
import pandas as pd
import math as math
import numpy as np
import random 
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

def cell_build(cell_type='dspn', specs=None, addSpines=False, branch=False, spine_per_length=1.61, frequency=2000, d_lambda=0.05, verbose=True, dend2remove=None, neck_dynamics=False):
    model = specs[cell_type]['model']
    morphology = specs[cell_type]['morph']
    if model == 0:
        import MSN_builder0 as build
        if cell_type == 'dspn':
            params='params_dMSN0.json'
        elif cell_type == 'ispn':
            params='params_iMSN0.json'
        cell = build.MSN(params=params, morphology=morphology, variables=None)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if branch: branch_groups = get_root_groups(cell)
        if addSpines: spines = build.add_spines(cell, spines_per_sec=30)
    elif model == 1:
        import MSN_builder1 as build
        if cell_type == 'dspn':
            params='params_dMSN1.json'
        elif cell_type == 'ispn':
            params='params_iMSN1.json'
        cell = build.MSN(params=params, morphology=morphology, variables=None, freq=frequency, d_lambda=d_lambda)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if branch: branch_groups = get_root_groups(cell)
        if addSpines: spines = build.add_spines(cell=cell, spine_per_length=spine_per_length, verbose=verbose)
    elif model == 2:
        import MSN_builder2 as build
        if cell_type == 'dspn':
            params='params_dMSN2.json'
        elif cell_type == 'ispn':
            params='params_iMSN2.json'
        cell = build.MSN(params=params, morphology=morphology, variables=None)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if branch: branch_groups = get_root_groups(cell)
        if addSpines: spines = build.add_spines(params=params, cell=cell, spine_per_length=spine_per_length, verbose=verbose)
            
    elif model == 3:
        if neck_dynamics:
            # MSN_builder3a assumes channels in spine head are also in neck 
            # and have same distribution as the neighboring dendrite
            # in this model ONLY the spine head is unique
            import MSN_builder3a as build
        else:
            import MSN_builder3 as build
        if cell_type == 'dspn':
            params='params_dMSN3.json'
        elif cell_type == 'ispn':
            params='params_iMSN3.json'
        cell = build.MSN(params=params, morphology=morphology, variables=None, freq=frequency, d_lambda=d_lambda)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if branch: branch_groups = get_root_groups(cell)
        if addSpines: 
            if neck_dynamics:
                spines = build.add_spines(params=params, cell=cell, spine_per_length=spine_per_length, verbose=verbose)
            else:
                spines = build.add_spines(cell=cell, spine_per_length=spine_per_length, verbose=verbose)

    if addSpines and branch: return(cell, spines, dend_tree, branch_groups)
    elif addSpines and not branch: return(cell, spines, dend_tree)
    elif not addSpines and branch: return(cell, dend_tree, branch_groups)
    elif not addSpines and not branch: return(cell, dend_tree)   
    
def dendrite_removal(cell, dend_tree, dend2remove):
    target_list = []
    for target in dend2remove:
        for sec in cell.dendlist:
            if sec.name() == target:
                target_list.append(sec)
    
    objects_to_remove_set = set()
    for branch in dend_tree:
        for path in branch:
            if not isinstance(path, list):
                path = [path]  # Ensure path is always a list
            for target in target_list:
                if target in path:
                    # Find the index of the target
                    start_index = path.index(target)
                    # If not at the end, collect the target and all subsequent objects
                    if start_index < len(path) - 1:
                        objects_to_remove_set.update(path[start_index:])
                    else:
                        # If it's a terminal branch, simply remove it
                        objects_to_remove_set.add(target)

    # Convert the set to a list if needed
    dends2remove = list(objects_to_remove_set)
    return dends2remove

def CurrentClamp(sim_time, 
                    stim_time,
                    baseline,
                    glut, 
                    glut_frequency, 
                    glutamate_locations, 
                    glut_reversals,
                    glut_time, 
                    num_gluts, 
                    dend_glut, 
                    g_AMPA,
                    ratio,
                    gaba, 
                    gaba_frequency, 
                    gaba_locations,
                    gaba_reversals,
                    gaba_time, 
                    g_GABA, 
                    num_gabas, 
                    dend_gaba, 
                    specs, 
                    frequency=2000,
                    d_lambda=0.05,
                    dend2remove=None,
                    v_init=-84, 
                    AMPA=True,
                    NMDA=True,
                    method=1,
                    physiological=True,
                    timing_range=None, 
                    add_noise=False,
                    beta=0,                    
                    B=1,                       
                    add_sine=False, 
                    axoshaft=False,
                    cell_type='dspn',
                    current_step=False,
                    step_current=-200,
                    step_duration=500,
                    step_start = 300,
                    holding_current=0,
                    Cm=1,
                    Ra=200,
                    g_name_list=None,
                    g8_list=None,
                    spine_per_length=1.61,
                    spine_neck_diam=0.1,
                    spine_neck_len=1,
                    spine_head_diam=0.5,
                    neck_dynamics=False,
                    space_clamp=False,
                    record_dendrite=None, 
                    record_location=None, 
                    record_currents=False,     
                    record_branch=False,       
                    dend_glut2=None,
                    record_mechs=False,
                    record_path_dend=False,    
                    record_path_spines=False,  
                    record_all_path=True,      
                    record_3D=False,           
                    record_3D_impedance=False, 
                    freq=10,                   
                    record_3D_mechs=False,     
                    record_Ca=False,
                    record_3D_Ca=False,
                    tonic=False,
                    gbar_gaba=None,
                    rectification=False,       
                    distributed=False,         
                    gaba_params=None,
                    tonic_gaba_reversal=-60,
                    dt =0.025
                    ):

    # 1. initialize variables
    vspine = []; vdend = []; vsoma = []; zdend = []; ztransfer = []
    v_dend_tree = {}; v_spine_tree = {}; i_mechs_dend = {}; i_mechs_dend_tree = {};
    v_spine_activated = {}; v_dend_activated = {};
    i_mechs_spine_tree = {}; i_mechs_spine_tree = {}; v_branch = {} 
    v_all_3D = {}; imp_all_3D = {}; Ca_all_3D = {}; i_mechs_3D = {}
    Ca_spine = []; Ca_dend = []; Ca_soma = []

    start_time = min(stim_time, *timing_range)
    burn_time = start_time - baseline  # time allowed to reach clamped potential

    if current_step: cs = step_current/1e3        
    hc = holding_current/1e3

    # 2. build cell 
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove, neck_dynamics=neck_dynamics)

    # 3. set up tonic gaba
    if tonic:
        if distributed:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, d3=gaba_params['d3'], a4=gaba_params['a4'], a5=gaba_params['a5'], a6=gaba_params['a6'], a7=gaba_params['a7'], rectification=rectification)                
        else:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, rectification=rectification)

    # 4. change properties

    if Ra != 200:
        space_clamped(cell=cell, spines=spines, Ra=Ra)
        print(f"Ra is {Ra} ΜΩ")
    
    if Cm != 1:
        cap(cell=cell, spines=spines, cm = Cm)
        print(f"Cm is {Cm} μFcm\u207B\u00B2")

    if g_name_list is not None:
        for g_name, g8 in zip(g_name_list, g8_list):
            g_alter(cell=cell, spines=spines, g_name=g_name, g8=g8, specs=specs, cell_type=cell_type)
            if g_name in ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33']:
                print(f"permeability: '{g_name}' = {g8} cm/s")
            else:
                print(f"conductance: '{g_name}' = {g8} S/cm²")
        
    # change all spine neck diameters
    if spine_neck_diam != 0.1:
        spine_neck_diameter(cell=cell, spines=spines, diam=spine_neck_diam)
        print(f"spine neck diameter {spine_neck_diam} μm")
        
    # change all spine neck lengths
    if spine_neck_len != 1:
        spine_neck_length(cell=cell, spines=spines, length=spine_neck_len)
        print(f"spine neck length {spine_neck_len} μm")
        
    # change all spine head diameters
    if spine_head_diam != 0.5:
        spine_head_diameter(cell=cell, spines=spines, diam=spine_head_diam, length=spine_head_diam)
        print(f"spine_head_diameter {spine_head_diam} μm")

    capacitance = whole_cell_capacitance(cell, spines, Cm=Cm)
    
    # 5. prepare synaptic inputs
    dstep1 = int(1 / glut_frequency * 1e3)

    if gaba_frequency is not None:
        dstep2 = int(1 / gaba_frequency * 1e3)

    glut_secs = [sec for target_dend in dend_glut for sec in cell.dendlist if sec.name() == target_dend]
    if num_gluts > 1 and len(glut_secs) == 1:
        glut_secs = glut_secs * num_gluts

    if gaba:
        gaba_secs = [sec for target_dend in dend_gaba for sec in cell.allseclist if sec.name() == target_dend]
    else:
        gaba_secs = []

    if num_gabas > 1 and len(gaba_secs) == 1:
        gaba_secs = gaba_secs * num_gabas

    # phasic glut
    glut_onsets = list(range(glut_time, glut_time + num_gluts * dstep1, dstep1)) 

    # phasic gaba
    if 'rel_gaba_onsets' in globals():
        gaba_onsets = [x + gaba_time for x in rel_gaba_onsets]
    else:
        if gaba_frequency is None:
            gaba_onsets = [gaba_time] * len(gaba_secs)
        else:
            gaba_onsets = list(range(gaba_time, gaba_time + num_gabas * dstep2, dstep2)) * len(gaba_secs)

    # 6a. place glutamatergic synapses
    if num_gluts > 1 and len(glutamate_locations) == 1:
        glutamate_locations = glutamate_locations*num_gluts

    glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs = glut_place4(
        cell = cell,
        spines = spines,
        method = method, 
        physiological = physiological, 
        AMPA = AMPA, 
        g_AMPA = g_AMPA,
        NMDA = NMDA,
        ratio = ratio,
        glut_reversals = glut_reversals,
        glut = glut,
        glut_time = glut_time,
        glut_secs = glut_secs,
        glut_onsets = glut_onsets,
        glut_locs = glutamate_locations,
        num_gluts = num_gluts,
        return_currents = record_currents,
        axoshaft = axoshaft
    )
    glut_locs = []
    for spine in final_spines:
        glut_locs.append(spine.x)

    # 6b. place gabaergic synapses
    if num_gabas > 1 and len(gaba_locations) == 1:
        gaba_locations = gaba_locations*num_gabas

    gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs = gaba_place3(
        physiological = physiological,
        gaba_reversals = gaba_reversals,
        gaba_weight = g_GABA,
        gaba_time = gaba_time,
        gaba_secs = gaba_secs,
        gaba_onsets = gaba_onsets,
        gaba_locations = gaba_locations,
        num_gabas = num_gabas,
        return_currents = record_currents
    )   

    # 7. setup recording locations
    if record_location is None:
        if glut:
            # loc = sum(glut_locs) / len(glut_locs) # midpoint
            loc = glutamate_locations[0] # first location
        else:
            # loc = sum(gaba_locations) / len(gaba_locations) # midpoint
            loc = gaba_locations[0] # first location
    else:
        loc = record_location[0]
    spine_dist = []
    if record_dendrite is None: # will assume want 1st listed dendrite
        dendrite = glut_secs[0] if glut else glut_secs_orig[0] if gaba else None
        spine = final_spines[0] if glut else glut_secs_orig[0] if gaba else None
        spine_dist = h.distance(dendrite(glut_locs[0])) 
    else:
        for sec in cell.allseclist:
            if sec.name() == record_dendrite:
                dendrite = sec
        # only record from spine head IF only 1 glutamatergic input
        if num_gluts == 1:
            spine = final_spines[0]
            spine_dist = h.distance(dendrite(glut_locs[0]))
        else: 
            spine = None

    print('recording at {} with location {}'.format(dendrite, round(loc,4)))

    # 8. configure basic current clamp
    t = h.Vector().record(h._ref_t)
    iclamp1 = h.IClamp(cell.soma(0.5))
    iclamp1.dur = sim_time
    if add_noise:
        iclamp1.amp = 0 # nA
    else:
        iclamp1.amp = hc # nA

    # 9. add further types of stimulus
    # add coloured noise
    if add_noise:
        samples = int(sim_time/dt)  # number of samples to generate (time series extension)
        noise = B * cn.powerlaw_psd_gaussian(beta, samples) + hc
        noise_vector = h.Vector()
        noise_vector.from_python(noise)
        tvec = h.Vector(np.linspace(dt, sim_time, int(sim_time/dt)))
        noise_vector.play(iclamp1._ref_amp, tvec, True)

    # add sine wave
    if add_sine:
        time = np.linspace(dt, sim_time, int(sim_time/dt))
        tvec = h.Vector(time)
        sine_wave = amplitude/1e3 * np.sin(2*np.pi*frequency*time/1e3)
        sine_vector = h.Vector()
        sine_vector.from_python(sine_wave)
        sine_vector.play(iclamp1._ref_amp, tvec, True)

    # add current step
    if current_step:
        step_end = step_start + step_duration
        iclamp2 = h.IClamp(cell.soma(0.5))
        iclamp2.delay = step_start
        iclamp2.dur = step_duration
        iclamp2.amp = cs # nA

    # 10. set up vectors
    vsoma = h.Vector()
    vdend = h.Vector()
    vsoma.record(cell.soma(0.5)._ref_v)
    vdend.record(dendrite(loc)._ref_v)

    if num_gluts == 1:
        record_spine = True
        vspine = h.Vector()
        vspine.record(spine.head(0.5)._ref_v)
    else:
        record_spine = False 

    v_spine_activated, v_dend_activated, dists_spine_activated = record_all_activated_spine_v2(cell=cell, dendrite=dendrite, activated_spines=final_spines) 
    v_spine_activated = {
            'v': v_spine_activated,
            'dists': dists_spine_activated
        }  
    v_dend_activated = {
            'v': v_dend_activated,
            'dists': dists_spine_activated
        }
    if record_Ca:
        Ca_soma = h.Vector(); Ca_dend = h.Vector()
        Ca_soma.record(cell.soma(0.5)._ref_cai)        
        Ca_dend.record(dendrite(loc)._ref_cai)
        
        if record_spine:
            Ca_spine = h.Vector()
            Ca_spine.record(spine.head(0.5)._ref_cai)

    if record_path_dend:
        # all voltages in path in dendrite
        v_dend_tree, dists_tree, dends_v = record_all_path_secs_v2(cell=cell, dend_tree=dend_tree, dendrite=dendrite)
        v_dend_tree = {
            'v': v_dend_tree,
            'dists': dists_tree,
            'dendrites': dends_v
        }    

    if record_path_spines:
        # all voltages in path in a spine
        v_spine_tree, dists_tree, dends_spine = record_all_path_secs_spine_v2(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines)
        v_spine_tree = {
            'v': v_spine_tree,
            'dists': dists_tree,
            'dendrites': dends_spine
        } 

    if record_mechs:
        # i_mechs at recording site
        mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kcnq', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']
        i_mechs_dend = record_i_mechs(cell=cell, dend=dendrite, loc=loc, return_currents=record_currents, mechs=mechs)
        i_mechs_dend ={
            'i': i_mechs_dend,
            'mechs': mechs
        }
        if record_path_dend:
            # all i_mechs in path in dendrite
            i_mechs_dend_tree, dists_tree, dends_i = record_all_path_secs_i_mechs(cell=cell, dend_tree=dend_tree, dendrite=dendrite, mechs=mechs)
            i_mechs_dend_tree = {
                'i': i_mechs_dend_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': mechs
            } 

        if record_path_spines:
            # all i_mechs in path in spines
            spine_mechs = ['pas', 'kir', 'cal12', 'cal13', 'car', 'cav32', 'cav33', 'sk']
            i_mechs_spine_tree, dists_tree, dends_i = record_all_path_secs_spine_i_mechs(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines, spine_mechs=spine_mechs)
            i_mechs_spine_tree = {
                'i': i_mechs_spine_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': spine_mechs
            }
    
    if record_branch:
        for dend_name in dend_glut2:
            dendrite_branch = None
            # Find the dendrite in the cell's section list
            for dend in cell.allseclist:
                if dend.name() == dend_name:
                    dendrite_branch = dend
                    break
            # Collect the data
            v, _, _, dists = sec_all_v(section=dendrite_branch, all_v={}, i=0)
            v_branch[dend_name] = {
                'v': v,
                'dists': dists
            }
    
    # if want 3D heatmaps
    if record_3D:
        # all voltages at 3D coordinates
        all_v, cell_coordinates, dends3D, dists3D = record_all_3D(cell)
        v_all_3D = {
            'v': all_v,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   


    # if want 3D heatmaps
    if record_3D_Ca:
        # all voltages at 3D coordinates
        all_Ca, cell_coordinates, dends3D, dists3D = record_Ca_3D(cell)
        Ca_all_3D = {
            'Ca': all_Ca,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   

    if record_3D_mechs:
        # i_mechs at recording site
        mechs3D = ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33'] # ['cal12'] # ['cal12', 'cal13', 'cav32', 'cav33']
        all_i_mechs, cell_coordinates, dends3D, dists3D = record_mechs_3D(cell, mechs=mechs3D)
        i_mechs_3D ={
            'i': all_i_mechs,
            'cell_coordinates': cell_coordinates,
            'dists': dists3D,
            'dendrites': dends3D,
            'mechs': mechs3D
        }        

    # 11. run simulation
    h.dt = dt
    h.finitialize(v_init)
        
    if record_3D_impedance:
        impedance_locations, cell_coordinates, dends, dists = setup_impedance_measurements(cell)
        impedance_transfer_locations, _, _, _ = setup_impedance_measurements(cell)

#         n=2
#         impedance_locations = impedance_locations[:n]

        # Initialize a dictionary to store impedance vectors for each location
        impedance_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_transfer_vectors = {loc: h.Vector() for loc in impedance_locations}

        if record_spine:
            imp_spine = h.Vector()
        
        # Initialize Impedance object
        imp = h.Impedance()
        imp.loc(loc, sec=dendrite) # location of interest, impedance transfer is relative to this point 
        
        # Initialize a variable to track the next time to compute impedance
        next_impedance_time = 0 + burn_time # only start calculating impedance when burn_time is over
        ds_imp = 40 # downsample to ds_imp * h.dt
        
        # Simulation loop
        while h.t < sim_time:
            if h.t >= round(next_impedance_time,3):
                # Compute impedance at the specified frequency
                imp.compute(freq, 1)
                if num_gluts==1:
                    imp_spine.append(imp.input(0.5, sec=spine.head))
                # Record impedance at each location
                for (sec, loc) in impedance_locations:
                     # Append the impedance value to the corresponding vector
                    impedance_vectors[(sec, loc)].append(imp.input(loc, sec=sec))
                    impedance_transfer_vectors[(sec, loc)].append(imp.transfer(loc, sec=sec))
                # Schedule the next impedance computation
                next_impedance_time += dt * ds_imp

            # Advance the simulation by one time step
            h.fadvance()
    
        impedance_vector_list = list(impedance_vectors.values())
        impedance_transfer_vector_list = list(impedance_transfer_vectors.values())
        imp_all_3D = {
            'imp': impedance_vector_list,
            'imp transfer': impedance_transfer_vectors,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists,
            'dendrites': dends
        }
     
    else:
        while h.t < sim_time:
            h.fadvance()

    record_dist = h.distance(dendrite(loc))
    
    # make outputs as numpy arrays
    if record_spine:
        vspine = np.array(vspine)
        if record_3D_impedance:
            imp_spine=np.array(imp_spine)
    
    vdend = np.array(vdend); vsoma = np.array(vsoma)
    
    if record_Ca:  
        Ca_dend = np.array(Ca_dend); Ca_soma = np.array(Ca_soma)
        if record_spine:
            Ca_spine = np.array(Ca_spine)
            
    v_spine_activated['v'] = vec2np(v_spine_activated['v'])
    v_dend_activated['v'] = vec2np(v_dend_activated['v'])
    
    if record_path_dend:
        v_dend_tree['v'] = vec2np(v_dend_tree['v'])
    if record_path_spines:
        v_spine_tree['v'] = vec2np(v_spine_tree['v'])
    if record_mechs:
        i_mechs_dend['i'] = vec2np(i_mechs_dend['i'])

        if record_path_dend:        
            out = {}
            for name, data in i_mechs_dend_tree['i'].items():
                # initialize a list to hold NumPy arrays for each dendrite
                out[name] = vec2np(data)
            i_mechs_dend_tree['i'] = out

        if record_path_spines:    
            out = {}
            for name, data in i_mechs_spine_tree['i'].items():
                out[name] = vec2np(data)
            i_mechs_spine_tree['i'] = out

    
    if record_branch:
        # Convert NEURON Vectors in v_branch to NumPy arrays
        for dend_name, data in v_branch.items():
            # Initialize a list to hold NumPy arrays for the current dendrite
            v_branch_np_arrays = []

            # Iterate through each Vector in 'v' and convert to NumPy array
            for i, vector in data['v'].items():
                np_array = np.array(vector)
                v_branch_np_arrays.append(np_array)

            # Replace the original 'v' data with the list of NumPy arrays
            v_branch[dend_name]['v'] = vec2np(v_branch[dend_name]['v'])        
    
    if record_3D:
        v_all_3D['v'] = vec2np(v_all_3D['v'])
    
    if record_3D_Ca:
        Ca_all_3D['Ca'] = vec2np(Ca_all_3D['Ca'])

    if record_3D_impedance:
        imp_all_3D['imp'] = vec2np(imp_all_3D['imp'])
        imp_all_3D['imp transfer'] = vec2np(imp_all_3D['imp transfer'])
        if record_spine:
            imp_all_3D['imp spine'] = imp_spine

    if record_3D_mechs:        
        out = {}
        for name, data in i_mechs_3D['i'].items():
            # initialize a list to hold NumPy arrays for each dendrite
            out[name] = vec2np(data)
        i_mechs_3D['i'] = out
    
    if record_currents:
        ampa_currents = vec2np(ampa_currents)
        nmda_currents = vec2np(nmda_currents)
        gaba_currents = vec2np(gaba_currents)
        gaba_conductance = vec2np(gaba_conductances)

    return v_all_3D, Ca_all_3D, imp_all_3D, i_mechs_3D, vspine, v_spine_activated, vdend, v_dend_activated, vsoma, v_dend_tree, v_spine_tree, Ca_spine, Ca_dend, Ca_soma, i_mechs_dend, i_mechs_dend_tree, i_mechs_spine_tree, v_branch, zdend, ztransfer, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, record_dist, record_spine, spine_dist, capacitance, h.dt, burn_time, start_time

def syn_reversals(cell_type, specs, spine_per_length, frequency, d_lambda, dend_glut, glut_reversal, glutamate_locations, num_gluts, dend_gaba, gaba_reversal, gaba_locations, num_gabas, sim_time, dt=0.025, v_init=-84, dend2remove=None, neck_dynamics=False):
    
    print("calculating reversal potentials for all unique locations...")

    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=dend2remove, neck_dynamics=neck_dynamics)

    if gaba_reversal == 'Edend':

        if len(gaba_locations) != num_gabas and len(gaba_locations) == 1:
            gaba_locations = gaba_locations * num_gabas

        gaba_reversals = membrane_potentials(cell=cell, 
                                       dends=dend_gaba, 
                                       locs=gaba_locations,
                                       sim_time=sim_time,
                                       dt=dt,
                                       v_init=v_init
                                       )
    else:
        gaba_reversals = [gaba_reversal] * num_gabas

    if glut_reversal == 'Edend':

        if len(glutamate_locations) != num_gluts and len(glutamate_locations) == 1:
            glutamate_locations = glutamate_locations * num_gluts

        glut_reversals = membrane_potentials(cell=cell, 
                                       dends=dend_glut, 
                                       locs=glutamate_locations,
                                       sim_time=sim_time,
                                       dt=dt,
                                       v_init=v_init
                                       )
    else:
        glut_reversals = [glut_reversal] * num_gluts
        
    unique_gaba = pairs_in_order(dend_gaba, gaba_reversals)
    formatted_strs = ["{}: {:.2f} mV".format(d, round(rev, 2)) for d, rev in unique_gaba]
    formatted_output = "; ".join(formatted_strs)
    print("{} reversals: {}".format('GABA', formatted_output))
    
    unique_glut = pairs_in_order(dend_glut, glut_reversals)
    formatted_strs = ["{}: {:.2f} mV".format(d, round(rev, 2)) for d, rev in unique_glut]
    formatted_output = "; ".join(formatted_strs)
    print("{} reversals: {}".format('GLUT', formatted_output))
    
    return glut_reversals, gaba_reversals

def params_selector(cell_type, specs):
    model = specs[cell_type]['model']
    morphology = specs[cell_type]['morph']
    if cell_type == 'dspn':        
        if model == 0:
            params='params_dMSN0.json'
        elif model == 1:
            params='params_dMSN1.json'
        elif model == 2:
            params='params_dMSN2.json'
        elif model == 3:
            params='params_dMSN3.json'
            
    if cell_type == 'ispn':        
        if model == 0:
            params='params_iMSN0.json'
        elif model == 1:
            params='params_iMSN1.json'
        elif model == 2:
            params='params_iMSN2.json'
        elif model == 3:
            params='params_iMSN2.json'
    return(params)

def spines_per_dend(cell, spines):
    for sec in cell.dendlist:
        print(sec.name(), len(spines[sec.name()])) 
        
# function gives the distance of EVERY segment within a given dendrite from soma 
def seg_dist(cell, dend):
    dist = []
    for sec in cell.dendlist:
        if sec.name() == dend:
            for i,seg in enumerate(sec):
                dist.append(h.distance(seg))
    return(dist)

# get idxs for spines with UNIQUE locations on a particular dendrite 
# then use to find a unique spines in a given dendrite
def spine_idx(cell, spines, dend):
    for sec in cell.dendlist:
        if sec.name() == dend:
            Nseg = sec.nseg
    spine_locs = (2*np.linspace(1, Nseg, Nseg)-1)/Nseg/2
#     spine_loc = spine_locs[0]
#     spine_loc
    # Get possible spines from section
    candidate_spines = []
    sec_spines = list(spines[dend].items())

    for spine_i, spine_obj in sec_spines: 
        candidate_spines.append(spine_obj)

    # len(sec_spines)
    candidate_spines_locs = []
    for spine in candidate_spines:
        candidate_spines_locs.append(spine.x)
    # candidate_spines_locs
    # spine_idxs = []
    output = []
    for ii in range(Nseg):
        spine_loc = spine_locs[ii]
        a = abs(candidate_spines_locs - spine_loc)
        idx = np.argmin(a)
        # spine_idxs.append(idx)  
        output.append(candidate_spines[idx])
        # only return unique spines
        output = list(dict.fromkeys(output))
    return(output)

def calculate_dist(d3, dist, a4, a5,  a6,  a7, g8):
    '''
    Used for setting the maximal conductance of a segment.
    Scales the maximal conductance based on somatic distance and distribution type.

    Parameters:
    d3   = distribution type:
         0 linear, 
         1 sigmoidal, 
         2 exponential
         3 step function
    dist = somatic distance of segment
    a4-7 = distribution parameters 
    g8   = base conductance (similar to maximal conductance)
    '''

    if   d3 == 0: 
        value = a4 + a5*dist
    elif d3 == 1: 
        value = a4 + a5/(1 + np.exp((dist-a6)/a7) )
    elif d3 == 2: 
        value = a4 + a5*np.exp((dist-a6)/a7)
    elif d3 == 3:
        if (dist > a6) and (dist < a7):
            value = a4
        else:
            value = a5

    if value < 0:
        value = 0

    value = value*g8
    return value

# function to alter a particular conductance g_name in all spines 
def spine_alter(cell, spines, g_name, d3, a4, a5, a6, a7, g8, cell_type='dspn', model=1):

    if model == 2:
        if cell_type == 'dspn':
            par ='params_dMSN2.json'
        elif cell_type == 'dspn':
            par ='params_iMSN2.json'
    elif model == 3:
        if cell_type == 'dspn':
            par ='params_dMSN3.json'
        elif cell_type == 'dspn':
            par ='params_iMSN3.json'

    if g_name == 'kir':
        for sec in cell.dendlist:
            sec_spines = list(spines[sec.name()].items())
            for spine_i, spine_obj in sec_spines: 
                spine_obj.head.gbar_kir = g8
                if model not in [0,1]:
                    spine_obj.neck.gbar_kir = g8

    if g_name == 'cav32':
        if model == 2:
            for sec in cell.dendlist:
                sec_spines = list(spines[sec.name()].items())
                for spine_i, spine_obj in sec_spines: 
                    dist = h.distance(sec(spine_obj.x))
                    spine_obj.head.pbar_cav32 = calculate_dist(d3=d3, dist=dist, a4=a4, a5=a5, a6=a6, a7=a7, g8=g8)
                    spine_obj.neck.pbar_cav32 = calculate_dist(d3=d3, dist=dist, a4=a4, a5=a5, a6=a6, a7=a7, g8=g8)
        else:
            for sec in cell.dendlist:
                sec_spines = list(spines[sec.name()].items())
                for spine_i, spine_obj in sec_spines: 
                    spine_obj.head.pbar_cav32 = g8
                    spine_obj.neck.pbar_cav32 = g8


    if g_name == 'cav33':
        if model == 2:
            for sec in cell.dendlist:
                sec_spines = list(spines[sec.name()].items())
                for spine_i, spine_obj in sec_spines: 
                    dist = h.distance(sec(spine_obj.x))
                    spine_obj.head.pbar_cav33 = calculate_dist(1, dist, 0, 1.0, 100.0, -30.0, g8)
                    spine_obj.neck.pbar_cav33 = calculate_dist(1, dist, 0, 1.0, 100.0, -30.0, g8)
        else:
            for sec in cell.dendlist:
                sec_spines = list(spines[sec.name()].items())
                for spine_i, spine_obj in sec_spines: 
                    spine_obj.head.pbar_cav33 = g8
                    spine_obj.neck.pbar_cav33 = g8

    if g_name == 'car':
        for sec in cell.dendlist:
            sec_spines = list(spines[sec.name()].items())
            for spine_i, spine_obj in sec_spines: 
                spine_obj.head.pbar_car = g8

    if g_name == 'cal12':
        for sec in cell.dendlist:
            sec_spines = list(spines[sec.name()].items())
            for spine_i, spine_obj in sec_spines: 
                spine_obj.head.pbar_cal12 = g8


    if g_name == 'cal13':
        for sec in cell.dendlist:
            sec_spines = list(spines[sec.name()].items())
            for spine_i, spine_obj in sec_spines: 
                spine_obj.head.pbar_cal13 = g8


    if g_name == 'sk':
        for sec in cell.dendlist:
            sec_spines = list(spines[sec.name()].items())
            for spine_i, spine_obj in sec_spines: 
                spine_obj.head.gbar_sk = g8

    if g_name == 'bk':
        if model == 2:
            for sec in cell.dendlist:
                sec_spines = list(spines[sec.name()].items())
                for spine_i, spine_obj in sec_spines: 
                    spine_obj.head.gbar_bk = g8
                    
# finds dendrites with at least 3 spines
def dend_spine_selector(cell, spines, branch_groups, n=2):
    dends_with_spines = []
    # Make list of dendrite sections with at least 2 spines 
    for dend in cell.dendlist:
        sec_spines = list(spines[dend.name()].items())
        for group in branch_groups: # for each nrn dendrite sec, one plot per branch
            if dend in group:
                if len(group) > n:
                    if len(sec_spines) > n:
                        dends_with_spines.append(dend)
    return dends_with_spines

# function to alter a given conductance in any location (other than spine) in cell                
def conductance_alter(cell, g_name, d3, a4, a5, a6, a7, g8, g8_somatic_scale, g8_axonal_scale):
    if g_name in ['naf', 'nap', 'kaf', 'kas', 'kdr', 'kir', 'sk', 'bk']:
        gbar = 'gbar_{}'.format(g_name)
    else:
        gbar = 'pbar_{}'.format(g_name)        
    if g_name not in ['nap']:
        cell.distribute_channels('dend', gbar, d3, a4, a5, a6, a7, g8) 
    if g_name in ['naf', 'nap', 'kaf', 'kas', 'kdr', 'kir', 'cal12', 'cal13', 'can', 'car', 'sk', 'bk']:
        cell.distribute_channels('soma', gbar, 0, 1, 0, 0, 0, g8*g8_somatic_scale)    
    if g_name == 'naf':
        cell.distribute_channels('axon', gbar, 3, 1, 1.1, 30, 500, g8*g8_axonal_scale)
    if g_name == 'kas':
        cell.distribute_channels('axon', gbar, 0, 1, 0, 0, 0, g8*g8_axonal_scale)   

# this function will alter the relevant conductances in all sections and if present in spine heads and necks
def g_alter(cell, spines, g_name, g8, specs, cell_type='dspn'):
    g8_orig=g8
    model = specs[cell_type]['model'] 
    params =  params_selector(cell_type, specs)          
    # run once:
    [d3, a4, a5, a6, a7, g8, g8_somatic_scale, g8_axonal_scale] = scaling_factors(g_name=g_name, params = params, model=model) # maintains original ratios between dend, soma (and axon)
    [d3, a4, a5, a6, a7, g8, g8_somatic_scale, g8_axonal_scale]
    g8 = g8_orig
    conductance_alter(cell=cell, g_name=g_name, d3=d3, a4=a4, a5=a5, a6=a6, a7=a7, g8=g8, g8_somatic_scale=g8_somatic_scale, g8_axonal_scale=g8_axonal_scale)
    spine_alter(cell=cell, spines=spines, g_name=g_name, d3=d3, a4=a4, a5=a5, a6=a6, a7=a7, g8=g8, cell_type=cell_type, model=model)

# rectification is False then ohmic else rectification Pavlov 
def tonic_gaba(cell, gaba_reversal, gbar_gaba, d3=0, a4=1, a5=0, a6=0, a7=0, rectification=False):
    if rectification:
        for sec in cell.dendlist:
            sec.e_gaba2 = gaba_reversal
        for sec in cell.somalist:
            sec.e_gaba2 = gaba_reversal
        cell.distribute_channels('dend', 'gbar_gaba2', d3, a4, a5, a6, a7, gbar_gaba)
        
        g_name = 'gaba2'
        g = []
        gbar = 'gbar_{}'.format(g_name)  
        for sec in cell.dendlist:
            g.append((eval('sec.{}'.format(gbar))))        
        
        if g[0] < g[-1]:
            cell.distribute_channels('soma', 'gbar_gaba2', 0, 1, 0, 0, 0, g[0])
        else:
            cell.distribute_channels('soma', 'gbar_gaba2', 0, 1, 0, 0, 0, gbar_gaba)

    else:
        for sec in cell.dendlist:
            sec.e_gaba1 = gaba_reversal
        for sec in cell.somalist:
            sec.e_gaba1 = gaba_reversal
        cell.distribute_channels('dend', 'gbar_gaba1', d3, a4, a5, a6, a7, gbar_gaba)
        cell.distribute_channels('soma', 'gbar_gaba1', 0, 1, 0, 0, 0, gbar_gaba)

        g_name = 'gaba1'
        g = []
        gbar = 'gbar_{}'.format(g_name)  
        for sec in cell.dendlist:
            g.append((eval('sec.{}'.format(gbar))))        
        
        if g[0] < g[-1]:
            cell.distribute_channels('soma', 'gbar_gaba1', 0, 1, 0, 0, 0, g[0])
        else:
            cell.distribute_channels('soma', 'gbar_gaba1', 0, 1, 0, 0, 0, gbar_gaba)

# Get dendrite branches, list for each unique branch structure (TODO: there's probably a neuron func for this)
def get_children(dend, branch_list):
    branch_list.append(dend)
    branches = []

    for child in dend.children():
        branch_list_cpy = branch_list.copy()
        branches.append(get_children(child, branch_list_cpy))

    if len(branches) == 0:
        return branch_list
    else:
        return branches
    
# Parser helper func
def branch_parser_helper(tree):
    for branch in tree:
        if all(type(b) == list for b in branch):
            # need to keep parsing
            for b in branch:
                tree.append(b)
            tree.remove(branch)
        # done parsing branch
    branch_parser(tree)
        
# Parses children into list format
def branch_parser(tree):
    for branch in tree:
        if all(type(b) == list for b in branch):
            branch_parser_helper(tree)
    return

# Takes nrn cell and int for origin dendrite segment index that branches occur from
# Returns parsed list with each entry a list of each unique branch path from origin dendrite segment to termination
def get_dend_branches_from(cell, origin):
    i = 0 
    for dend in cell.dendlist:
        if i == origin: # origin dendrite number to get branches from
            dend_tree = []
            dend_tree = get_children(dend, dend_tree)
            branch_parser(dend_tree)
            return dend_tree
        i += 1
        
def get_root_branches(cell):
    sref_soma = h.SectionRef(sec=cell.soma)

    # Get sec roots (excluding axon)
    roots = []

    for child in sref_soma.child:
        roots.append(child)

    roots = roots[1:]

    # Get dend tree from all roots
    root_tree = []

    for root in roots:
        dend_branch = []
        branch = get_children(root, dend_branch)
        branch_parser(branch)
        root_tree.append(branch)
    
    return root_tree

# gets path from dend to soma
def path_finder2(cell, dend_tree, dend):
    dend_tree2 = [num for sublist in dend_tree for num in sublist]
    for XX in dend_tree2:
        if not isinstance(XX, list):
            XX = [XX]
        for XXX in XX:
            if XXX == dend:
                return XX

def include_upto(iterable, value):
    for it in iterable:
        yield it
        if it == value:
            return

def path_finder(cell, dend_tree, dend):               
    pathlist = []
    pathlist = path_finder2(cell=cell, dend_tree=dend_tree, dend=dend)
    pathlist =  [cell.soma] + pathlist
    return list(include_upto(pathlist, dend))      
            
# Takes cell
# Return the dendrites that are in a branch (with root dendrite first followed by children in that branch ordered by first instance in tree)
# Useful if you want to do something to all dendrites in a branch, without the ordered duplication of root branches 
def get_root_groups(cell):
    root_tree = get_root_branches(cell)
    branch_groups = []
    for branch in root_tree:
        dend_list = []

        for dend in branch:
            
            if isinstance(dend, list):
                for d in dend:
                    if d not in dend_list:
                        dend_list.append(d)
            else:
                if dend not in dend_list:
                    dend_list.append(dend)
                    
        branch_groups.append(dend_list)
    return branch_groups

def nsegs(cell):
    nsegs =[]
    for sec in cell.dendlist:
        nsegs.append(sec.nseg)
    N = sum(nsegs)
    return(N)

def extract(d):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    return y

def extract2(d):
    out = []
    for x in d:
        out.append(x)
    return out

def list2df(lst):
    df = pd.DataFrame() 
    df['time'] = extract2(lst[0])
    df['pas'] = extract2(lst[1])
    df['kdr'] = extract2(lst[2])
    df['naf'] = extract2(lst[3])
    df['kaf'] = extract2(lst[4])
    df['kas'] = extract2(lst[5])
    df['kir'] = extract2(lst[6])
    df['cal12'] = extract2(lst[7])
    df['cal13'] = extract2(lst[8])
    df['can'] = extract2(lst[9])
    df['car'] = extract2(lst[10])
    df['cav32'] = extract2(lst[11])
    df['cav33'] = extract2(lst[12])
    #     df['kcnq'] = extract2(lst[xx])
    df['sk'] = extract2(lst[13])
    df['bk'] = extract2(lst[14])  
    return df

def plot_mech(d, mech_name):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.title(mech_name)
    plt.plot(x, y)
    plt.show()
    
# return all dendritic inserted mechanisms
def mechanisms(cell):
    d_ = {}
    df = pd.DataFrame()
    # mechs = ['kdr', 'naf', 'kaf', 'kas', 'kdr', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kdr
    lists = sorted(d_.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    df['dist'] = x
    
    df['kdr'] = y

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_naf
    df['naf'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kaf
    df['kaf'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kas
    df['kas'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kdr
    df['kdr'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kir
    df['kir'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cal12
    df['cal12'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cal13
    df['cal13'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_can
    df['can'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_car
    df['car'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cav32
    df['cav32'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cav33
    df['cav33'] = extract(d_)


    # for sec in cell.dendlist:
    #      d_[h.distance(sec(0.5))] = sec.gbar_kcnq
    # mechanisms.append(extract(d_))
    for sec in cell.dendlist:      
         d_[h.distance(sec(0.5))] = sec.gbar_sk
    df['sk'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_bk
    df['bk'] = extract(d_)

    return(df)

def scaling_factors(g_name, params='params_dMSN.json', model=None):
    # Parameters:
    # d3   = distribution type:
    #      0 linear, 
    #      1 sigmoidal, 
    #      2 exponential
    #      3 step function
    # dist = somatic distance of segment
    # a4-7 = distribution parameters 
    # g8   = base conductance (similar to maximal conductance)

    import json
    with open(params) as file:
        par = json.load(file)

    # cell, SPINES, branch_groups, dend_tree = utils.make_cell()
    g8_axonal = 0
    g8_somatic = 0
    if g_name == 'naf':
        d3 = 1
        a4 = 0  # ALT
        a5 = 1  # SPREAD
        a6 = 50
        a7 = 10
        g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])
        g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])
        g8_axonal = float(par['gbar_{}_axonal'.format(g_name)]['Value'])

    if g_name == 'nap':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        g8_basal = float(par['gbar_{}_somatic'.format(g_name)]['Value'])
        g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'kaf':
        d3 = 1
        a4 = 0.5   # ALT
        a5 = 0.25  # SPREAD
        a6 = 120
        a7 = 30
        g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])
        g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'kas':
        d3 = 2
        a4 = 0.25  # ALT
        a5 = 5     # SPREAD
        a6 = 0
        a7 = -10
        g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])
        g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])
        g8_axonal = float(par['gbar_{}_axonal'.format(g_name)]['Value'])

    if g_name == 'kdr':
        d3 = 1
        a4 = 0.25  # ALT
        a5 = 1     # SPREAD
        a6 = 50
        a7 = 30
        g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])
        g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value']) # build.calculate_distribution(d3, -cell.soma.diam/2, a4, a5,  a6,  a7, g8_basal)

    if g_name == 'kir':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        if model == 0:
            g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])*2
            g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])*2
        else:
            g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])
            g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'sk':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])
        g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'bk':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        g8_basal = float(par['gbar_{}_basal'.format(g_name)]['Value'])
        g8_somatic = float(par['gbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'car':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        g8_basal = float(par['pbar_{}_basal'.format(g_name)]['Value'])
        g8_somatic = float(par['pbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'can':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        if model == 3:
            g8_basal = float(par['pbar_{}_dend'.format(g_name)]['Value'])
            g8_somatic = float(par['pbar_{}_somatic'.format(g_name)]['Value'])
        else:
            g8_basal = float(par['pbar_{}_basal'.format(g_name)]['Value'])
            g8_somatic = float(par['pbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'cal12':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        if model == 3:
            g8_basal = float(par['pbar_{}_dend'.format(g_name)]['Value'])
            g8_somatic = float(par['pbar_{}_somatic'.format(g_name)]['Value'])
        else:
            g8_basal = float(par['pbar_{}_basal'.format(g_name)]['Value'])
            g8_somatic = float(par['pbar_{}_somatic'.format(g_name)]['Value'])
        
    if g_name == 'cal13':
        d3 = 0
        a4 = 1  # ALT
        a5 = 0  # SPREAD
        a6 = 0
        a7 = 0
        if model == 3:
            g8_basal = float(par['pbar_{}_dend'.format(g_name)]['Value'])
            g8_somatic = float(par['pbar_{}_somatic'.format(g_name)]['Value'])
        else:
            g8_basal = float(par['pbar_{}_basal'.format(g_name)]['Value'])
            g8_somatic = float(par['pbar_{}_somatic'.format(g_name)]['Value'])

    if g_name == 'cav32':
        d3 = 1
        a4 = 0  # ALT
        a5 = 1  # SPREAD
        a6 = 100
        a7 = -30
        if params in ['params_dMSN.json3','params_iMSN.json3']:
            g8_basal = float(par['pbar_{}_dend'.format(g_name)]['Value'])
        else:
            g8_basal = float(par['pbar_{}_basal'.format(g_name)]['Value'])
 
    if g_name == 'cav33':
        d3 = 1
        a4 = 0  # ALT
        a5 = 1  # SPREAD
        a6 = 100
        a7 = -30
        if model == 3:
            g8_basal = float(par['pbar_{}_dend'.format(g_name)]['Value'])
        else:
            g8_basal = float(par['pbar_{}_basal'.format(g_name)]['Value'])
 
    g8_somatic_scale = g8_somatic / g8_basal
    g8_axonal_scale = g8_axonal / g8_basal
    
    out = [d3, a4, a5, a6, a7, g8_basal, g8_somatic_scale, g8_axonal_scale]
    return(out)

# Set up branch assignment
def branch_selection(cell, cell_type='dpsn'):
    branch1_dends = [None] * 2 
    branch2_dends = [None] * 2 
    branch3_dends = [None] * 2 
    branch4_dends = [None] * 2 
    branch5_dends = [None] * 2 

    # Define dendrite sections for each predefined branch (chosen as sterotypical primary dendrites)
    if cell_type == 'dspn':    
        for dend in cell.dendlist:
            if dend.name() == 'dend[28]':
                branch1_dends[-1] = dend
            if dend.name() == 'dend[25]':
                branch1_dends[0] = dend

            if dend.name() == 'dend[15]':
                branch2_dends[-1] = dend
            if dend.name() == 'dend[13]':
                branch2_dends[0] = dend

            if dend.name() == 'dend[46]':
                branch3_dends[-1] = dend
            if dend.name() == 'dend[43]':
                branch3_dends[0] = dend

            if dend.name() == 'dend[36]':
                branch4_dends[-1] = dend
            if dend.name() == 'dend[32]':
                branch4_dends[0] = dend

            if dend.name() == 'dend[57]':
                branch5_dends[-1] = dend
            if dend.name() == 'dend[55]':
                branch5_dends[0] = dend

    elif cell_type == 'ispn':
        for dend in cell.dendlist:
            if dend.name() == 'dend[29]':
                branch1_dends[-1] = dend
            if dend.name() == 'dend[27]':
                branch1_dends[0] = dend

            if dend.name() == 'dend[15]':
                branch2_dends[-1] = dend
            if dend.name() == 'dend[13]':
                branch2_dends[0] = dend

            if dend.name() == 'dend[17]':
                branch3_dends[-1] = dend
            if dend.name() == 'dend[12]':
                branch3_dends[0] = dend

            if dend.name() == 'dend[45]':
                branch4_dends[-1] = dend
            if dend.name() == 'dend[41]':
                branch4_dends[0] = dend

            if dend.name() == 'dend[36]':
                branch5_dends[-1] = dend
            if dend.name() == 'dend[32]':
                branch5_dends[0] = dend
    
    # For sparse plotting
    return [branch1_dends] + [branch2_dends] + [branch3_dends] + [branch4_dends] + [branch5_dends]

# change all spine neck diameters
def spine_neck_diameter(cell, spines, diam):
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.neck.diam = diam

def spine_neck_length(cell, spines, length):
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.neck.L = length
            
def spine_head_diameter(cell, spines, diam, length):
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.head.diam = diam
            spine_obj.head.L = length
            
# Set up branch assignment and add glutamate
def glut_add(cell=None,
               branch1_glut = False, 
               branch2_glut = True, 
               branch3_glut = False, 
               branch4_glut = False, 
               branch5_glut = False, 
               num_gluts = 15,
               glut_placement = 'distal',
               glut = True,
               cell_type='dspn'):
    [branch1_dends, branch2_dends, branch3_dends, branch4_dends, branch5_dends] = branch_selection(cell, cell_type=cell_type) 
    glut_secs = []
    glut_secs_orig = []
    # Define placement on dendritic branch (prox/dist)
    if 'proximal' in glut_placement:
        glut_site = 0
    else:
        glut_site = -1

    # Define branch for glutamate (multiple possible)
    if branch1_glut:
        glut_secs.append(branch1_dends[glut_site])
        glut_secs_orig.append(branch1_dends[glut_site])

    if branch2_glut:
        glut_secs.append(branch2_dends[glut_site])
        glut_secs_orig.append(branch2_dends[glut_site])

    if branch3_glut:
        glut_secs.append(branch3_dends[glut_site])
        glut_secs_orig.append(branch3_dends[glut_site])

    if branch4_glut:
        glut_secs.append(branch4_dends[glut_site])
        glut_secs_orig.append(branch4_dends[glut_site])

    if branch5_glut:
        glut_secs.append(branch5_dends[glut_site])
        glut_secs_orig.append(branch5_dends[glut_site])

    # Number of glutamatergic inputs per section is num_gluts
    glut_secs *= num_gluts     

    if glut:
        print("glut:{}".format(glut_secs))
    else:
        # No glutamate
        glut_secs = []
    return glut_secs, glut_secs_orig

def glut_place(spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               num_gluts=15,
               return_currents = True,
               model = 1):
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    if len(glut_secs) > 0: 
        glut_id = 0 # index used for glut_synapse list and printing
        final_spine_locs = []
        random.seed(42)
        for dend_glut in glut_secs:
            # Get possible spines from section
            candidate_spines = []
            sec_spines = list(spines[dend_glut.name()].items())

            if model in [1,2]:
            
                for spine_i, spine_obj in sec_spines: 
                    candidate_spines.append(spine_obj)

                if len(glut_secs) < len(sec_spines):
                    if method==1:
                        # reversed order so activate along dendrite towards soma
                        spine_idx = 2*len(candidate_spines)//3-1 # arbitrary start point at 2/3 of spines
                        spine = candidate_spines[spine_idx - glut_id] 
                    else:
                        spine_idx = 2*len(candidate_spines)//3 - num_gluts # arbitrary start point at 1/3 of spines
                        if spine_idx < 0:
                            if len(candidate_spines) >= num_gluts:
                                spine_idx = len(candidate_spines) - num_gluts
                            else:
                                spine_idx = 0        
                        spine = candidate_spines[spine_idx + glut_id] 
                else:
                    spine = random.choice(candidate_spines)
                    
            else:
            
                for spine_i, spine_obj in sec_spines: 
                    candidate_spines.append(spine_obj)
                if len(glut_secs) < len(sec_spines):
                    spine_idx = len(candidate_spines)//3-1 # arbitrary start point at 1/3 of spines
                    spine = candidate_spines[spine_idx + glut_id] 

                else:
                    spine = random.choice(candidate_spines)

            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 

            # Define glutamate syn 
            glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
            if physiological:
                if AMPA:
                    glut_synapses[glut_id].gmax_AMPA = g_AMPA
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0
                if NMDA:
                    glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                else:
                    glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                glut_synapses[glut_id].tau2_ampa = 4.8                
                # physiological kinetics for NMDA from Chapman et al. 2003, 
                # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                glut_synapses[glut_id].tau1_nmda = 5.52
                glut_synapses[glut_id].tau2_nmda = 231   
                # alpha and beta determine neg slope of Mg block for NMDA
                glut_synapses[glut_id].alpha = 0.096
                glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
            else:
                glut_synapses[glut_id].gmax_AMPA = 0.001 
                glut_synapses[glut_id].gmax_NMDA = 0.007
                # physiological kinetics for NMDA from Chapman et al. 2003, 
                # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                glut_synapses[glut_id].tau1_nmda = 5.52
                glut_synapses[glut_id].tau2_nmda = 231            

            # Stim to play back spike times as defined by onsets
            glut_stimulator[glut_id] = h.VecStim()
            glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

            # Connect stim and syn
            glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
            glut_connection[glut_id].weight[0] = 0.35

            if return_currents:
                # Record NMDA current for synapse
                nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

        print("# glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, final_spine_locs, glut_onsets))
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents

def glut_place_alt(spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               num_gluts=15,
               return_currents = True,
               model = 1):

    # Pairing each element in glut_secs with its corresponding onset in glut_onsets
    paired = list(zip(glut_secs, glut_onsets))

    # Sorting the pairs based on the dendrite element (the first item in each pair)
    sorted_pairs = sorted(paired, key=lambda x: str(x[0]))

    # Unpacking the sorted pairs back into separate lists
    glut_secs, glut_onsets = zip(*sorted_pairs)


    unique_glut_secs = list(set(glut_secs))

    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    if len(glut_secs) > 0: 
        glut_id = 0 # index used for glut_synapse list and printing
        final_spine_locs = []
        random.seed(42)

        for dend in unique_glut_secs:
            idx = [jj for jj, x in enumerate(glut_secs) if x.name() == dend.name()]
            selected_glut_secs = [glut_secs[i] for i in idx]
            id = 0
            for dend_glut in selected_glut_secs:
                # Get possible spines from section
                candidate_spines = []
                sec_spines = list(spines[dend_glut.name()].items())

                if model in [1,2]:

                    for spine_i, spine_obj in sec_spines: 
                        candidate_spines.append(spine_obj)

                    if len(glut_secs) < len(sec_spines):
                        if method==1:
                            # reversed order so activate along dendrite towards soma
                            spine_idx = 2*len(candidate_spines)//3-1 # arbitrary start point at 2/3 of spines
                            spine = candidate_spines[spine_idx - id] 
                        else:
                            spine_idx = 2*len(candidate_spines)//3 - num_gluts # arbitrary start point at 1/3 of spines
                            if spine_idx < 0:
                                if len(candidate_spines) >= num_gluts:
                                    spine_idx = len(candidate_spines) - num_gluts
                                else:
                                    spine_idx = 0        
                            spine = candidate_spines[spine_idx + id] 
                    else:
                        spine = random.choice(candidate_spines)

                else:

                    for spine_i, spine_obj in sec_spines: 
                        candidate_spines.append(spine_obj)
                    if len(glut_secs) < len(sec_spines):
                        spine_idx = len(candidate_spines)//3-1 # arbitrary start point at 1/3 of spines
                        spine = candidate_spines[spine_idx + id] 

                    else:
                        spine = random.choice(candidate_spines)

                spine_loc = spine.x
                spine_head = spine.head
                final_spine_locs.append(spine_loc) 

                # Define glutamate syn 
                glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

                glut_id += 1 # Increment glutamate counter
                id += 1

        rounded_locs = [round(value, 4) for value in final_spine_locs]
        print("# glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents

# finds distances of syanpses from soma
def synapse_dist(spines,
               method=0,
               glut_secs = None,
               num_gluts=15):
    if len(glut_secs) > 0: 
        glut_id = 0 # index used for glut_synapse list and printing
        final_spine_dists = []
        random.seed(42)
        for dend_glut in glut_secs:
            # Get possible spines from section
            candidate_spines = []
            sec_spines = list(spines[dend_glut.name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidate_spines.append(spine_obj)

            if len(glut_secs) < len(sec_spines):
                if method==1:
                    # reversed order so activate along dendrite towards soma
                    spine_idx = 2*len(candidate_spines)//3-1 # arbitrary start point at 2/3 of spines
                    spine = candidate_spines[spine_idx - glut_id] 
                else:
                    spine_idx = 2*len(candidate_spines)//3 - num_gluts # arbitrary start point at 1/3 of spines
                    if spine_idx < 0:
                        if len(candidate_spines) >= num_gluts:
                            spine_idx = len(candidate_spines) - num_gluts
                        else:
                            spine_idx = 0        
                    spine = candidate_spines[spine_idx + glut_id] 
            else:
                spine = random.choice(candidate_spines)

            spine_loc = spine.x
            final_spine_dists.append(h.distance(dend_glut(spine_loc))) 
            glut_id += 1 # Increment glutamate counter
    return final_spine_dists

def gaba_onset(gaba_time, num_gabas, num_branch2, model=1):
    if model == 0:
        gaba_onsets = list(range(gaba_time, gaba_time + int(num_gabas/3)+1)) * 3 * num_branch2
        gaba_onsets = gaba_onsets[:num_gabas]
    else:
        if num_branch2 in [0,1]:
            if (num_gabas < 4):
                gaba_onsets = list(range(gaba_time, gaba_time + num_gabas)) 
            else:
                if num_gabas % 3 == 0:
                    gaba_onsets = list(range(gaba_time, gaba_time + int(num_gabas/3))) * 3 * num_branch2
                else:
                    gaba_onsets = list(range(gaba_time, gaba_time + int(num_gabas/3)+1)) * 3 * num_branch2
            gaba_onsets = gaba_onsets[:num_gabas]
        else:
            onsets = list(range(gaba_time, gaba_time + num_gabas)) 
            gaba_onsets = [x for x in onsets for _ in range(num_branch2)]
    return gaba_onsets

def gaba_add(cell=None,
               gaba=True, 
               branch1_gaba = False, 
               branch2_gaba = True, 
               branch3_gaba = False, 
               branch4_gaba = False, 
               branch5_gaba = False, 
               gaba_placement = 'distal',
               num_gabas=15,
               show=True,
               cell_type='dspn'):

    if gaba > 0: 
        [branch1_dends, branch2_dends, branch3_dends, branch4_dends, branch5_dends] = branch_selection(cell, cell_type) 

        gaba_secs = []

        # Define gaba spatial placement 
        if 'soma' in gaba_placement:
            gaba_secs.append(cell.soma)

            gaba_secs *= num_gabas # need to duplicate sections to place synapses 

        elif 'everywhere' in gaba_placement: # append to every dendrite section
            for dend in cell.dendlist:
                gaba_secs.append(dend)

            gaba_secs *= num_gabas # need to duplicate sections to place synapses 


        elif 'distributed_branch' in gaba_placement: # append to specific branches

            # Define placement on dendritic branch (prox/dist)
            if 'proximal' in gaba_placement:
                gaba_site = 0
            else:
                gaba_site = -1

            # Define branch for gaba (multiple possible)
            if branch1_gaba:
                gaba_secs.append(branch1_dends[gaba_site])

            if branch2_gaba:
                gaba_secs.append(branch2_dends[gaba_site])

            if branch3_gaba:
                gaba_secs.append(branch3_dends[gaba_site])

            if branch4_gaba:
                gaba_secs.append(branch4_dends[gaba_site])

            if branch5_gaba:
                gaba_secs.append(branch5_dends[gaba_site])

            gaba_secs *= num_gabas # need to duplicate sections to place synapses 

    else:
        # No gaba
        gaba_secs = []

    if show:
        print("gaba:{}".format(gaba_secs))
    return gaba_secs

def gaba_place(physiological=True,
               gaba_reversal = -60,
               gaba_weight = 0.001,
               gaba_time = 200,
               gaba_secs = None,
               gaba_onsets=None,
               gaba_locations = None,
               num_gabas=15,
               return_currents = True,
               show=True):
    
    gaba_conductances = [0] * len(gaba_secs)
    gaba_currents = [0] * len(gaba_secs)
    gaba_synapses = [0]*len(gaba_secs) # list of gaba synapses
    gaba_stimulator = {}
    gaba_connection = {}
    if gaba_locations is None:
        gaba_locations = [0.5] * len(gaba_secs)
        
    # Place gabaergic synapses
    if len(gaba_secs) > 0:

        gaba_id = 0 # index used for gaba_synapse list and printing
        gaba_locs = []

        for dend_gaba in gaba_secs:

            # For now, just assign to middle of section instead of uniform random
            gaba_loc = gaba_locations[gaba_id]

            # Choose random location along section
    #                 gaba_loc = round(random.uniform(0, 1), 2)

            gaba_locs.append(gaba_loc)

            # Define gaba synapse
            gaba_synapses[gaba_id] = h.gabasynapse(dend_gaba(gaba_loc)) 
            if physiological:
                gaba_synapses[gaba_id].tau1 = 0.9 
                gaba_synapses[gaba_id].tau2 = 18
            else:
                gaba_synapses[gaba_id].tau2 = 0.9 # TODO: Tune tau2 further for accurate response 
            gaba_synapses[gaba_id].erev = gaba_reversal

            # Stim to play back spike times
            gaba_stimulator[gaba_id] = h.VecStim()

            # Use with deterministic onset times
            gaba_stimulator[gaba_id].play(h.Vector(1, gaba_onsets[gaba_id]))

            # Connect stim and syn
            gaba_connection[gaba_id] = h.NetCon(gaba_stimulator[gaba_id], gaba_synapses[gaba_id])
            gaba_connection[gaba_id].weight[0] = gaba_weight # Depending on desired EPSP response at soma, tune this

            if return_currents:
                # Measure conductance and current
                gaba_currents[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_i)
                gaba_conductances[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_g)

            gaba_id += 1 # increment gaba counter

        if show:
            print("# gaba synapses added:{} on:{} with locs:{} with timing onsets:{}".format(gaba_id, gaba_secs, gaba_locs, gaba_onsets))
    return gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances

def glut_place2(cell,
               spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut=True,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               glut_locs = None,
               num_gluts=15,
               return_currents = True):
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    final_spine_locs = []
    final_spines = []
    if num_gluts > 0: 
        glut_id = 0 # index used for glut_synapse list and printing

        for ii in list(range(0,num_gluts)):
            synapse_loc = glut_locs[ii]
            # Get candidate spines from section
            candidates = []
            sec_spines = list(spines[glut_secs[ii].name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidates.append(spine_obj)

            locs = []
            for spine in candidates:
                locs.append(spine.x)
            loc, idx = find_closest_value(locs, synapse_loc)
            spine = candidates[idx] # choose last spine

            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 
            final_spines.append(spine)
            if glut:
                # Define glutamate syn 
                glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

#         rounded_locs = [round(value, 4) for value in glut_locs]
        rounded_locs = [round(value, 4) for value in final_spine_locs]
        if glut:
            print("# glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs

def glut_place3(cell,
               spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut=True,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               glut_locs = None,
               num_gluts=15,
               return_currents = True,
               axoshaft=False):
    
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    final_spine_locs = []
    final_spines = []
    if num_gluts > 0: 
        glut_id = 0 # index used for glut_synapse list and printing

        for ii in list(range(0,num_gluts)):
            synapse_loc = glut_locs[ii]
            # Get candidate spines from section
            candidates = []
            sec_spines = list(spines[glut_secs[ii].name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidates.append(spine_obj)

            locs = []
            for spine in candidates:
                locs.append(spine.x)
            loc, idx = find_closest_value(locs, synapse_loc)
            spine = candidates[idx] # choose last spine

            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 
            final_spines.append(spine)
            if glut:
                # Define glutamate syn 
                if not axoshaft:
                    glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                else:
                    glut_synapses[glut_id] = h.glutsynapse(glut_secs[ii](glut_locs[ii]))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

#         rounded_locs = [round(value, 4) for value in glut_locs]
        rounded_locs = [round(value, 4) for value in final_spine_locs]
        if glut:
            if not axoshaft:
                print("# axospinous glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
            else:
                print("# axoshaft glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs

def glut_place4(cell,
               spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut_reversals=[0]*15, 
               glut=True,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               glut_locs = None,
               num_gluts=15,
               return_currents = True,
               axoshaft=False):
    
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    final_spine_locs = []
    final_spines = []
    if num_gluts > 0: 
        glut_id = 0 # index used for glut_synapse list and printing

        for ii in list(range(0,num_gluts)):
            synapse_loc = glut_locs[ii]
            # Get candidate spines from section
            candidates = []
            sec_spines = list(spines[glut_secs[ii].name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidates.append(spine_obj)

            locs = []
            for spine in candidates:
                locs.append(spine.x)
            loc, idx = find_closest_value(locs, synapse_loc)
            spine = candidates[idx] # choose last spine

            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 
            final_spines.append(spine)
            if glut:
                # Define glutamate syn 
                if not axoshaft:
                    glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                else:
                    glut_synapses[glut_id] = h.glutsynapse(glut_secs[ii](glut_locs[ii]))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            
               
                glut_synapses[glut_id].erev = glut_reversals[ii]

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

#         rounded_locs = [round(value, 4) for value in glut_locs]
        rounded_locs = [round(value, 4) for value in final_spine_locs]
        if glut:
            if not axoshaft:
                print("# axospinous glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
            else:
                print("# axoshaft glutamate added:{}, on sections:{}, with locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs

def gaba_place2(physiological=True,
               gaba_reversal = -60,
               gaba_weight = 0.001,
               gaba_time = 200,
               gaba_secs = None,
               gaba_onsets=None,
               gaba_locations = None,
               num_gabas=15,
               return_currents = True,
               show=True):
    
    gaba_conductances = [0] * len(gaba_secs)
    gaba_currents = [0] * len(gaba_secs)
    gaba_synapses = [0]*len(gaba_secs) # list of gaba synapses
    gaba_stimulator = {}
    gaba_connection = {}
    if gaba_locations is None:
        gaba_locations = [0.5] * len(gaba_secs)
    gaba_locs = []    
    # Place gabaergic synapses
    if len(gaba_secs) > 0:

        gaba_id = 0 # index used for gaba_synapse list and printing
        for dend_gaba in gaba_secs:

            # For now, just assign to middle of section instead of uniform random
            gaba_loc = gaba_locations[gaba_id]

            # Choose random location along section
    #                 gaba_loc = round(random.uniform(0, 1), 2)

            gaba_locs.append(gaba_loc)

            # Define gaba synapse
            gaba_synapses[gaba_id] = h.gabasynapse(dend_gaba(gaba_loc)) 
            if physiological:
                gaba_synapses[gaba_id].tau1 = 0.9 
                gaba_synapses[gaba_id].tau2 = 18
            else:
                gaba_synapses[gaba_id].tau2 = 0.9 # TODO: Tune tau2 further for accurate response 
            gaba_synapses[gaba_id].erev = gaba_reversal

            # Stim to play back spike times
            gaba_stimulator[gaba_id] = h.VecStim()

            # Use with deterministic onset times
            gaba_stimulator[gaba_id].play(h.Vector(1, gaba_onsets[gaba_id]))

            # Connect stim and syn
            gaba_connection[gaba_id] = h.NetCon(gaba_stimulator[gaba_id], gaba_synapses[gaba_id])
            gaba_connection[gaba_id].weight[0] = gaba_weight # Depending on desired EPSP response at soma, tune this

            if return_currents:
                # Measure conductance and current
                gaba_currents[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_i)
                gaba_conductances[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_g)

            gaba_id += 1 # increment gaba counter

        rounded_locs = [round(value, 4) for value in gaba_locs]
        print("# gaba synapses added:{} on:{} with locs:{} with timing onsets:{}".format(gaba_id, gaba_secs, gaba_locs, gaba_onsets))
    return gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs

def gaba_place3(physiological=True,
               gaba_reversals = [-60]*15,
               gaba_weight = 0.001,
               gaba_time = 200,
               gaba_secs = None,
               gaba_onsets=None,
               gaba_locations = None,
               num_gabas=15,
               return_currents = True,
               show=True):
    
    gaba_conductances = [0] * len(gaba_secs)
    gaba_currents = [0] * len(gaba_secs)
    gaba_synapses = [0]*len(gaba_secs) # list of gaba synapses
    gaba_stimulator = {}
    gaba_connection = {}
    if gaba_locations is None:
        gaba_locations = [0.5] * len(gaba_secs)
    gaba_locs = []    
    # Place gabaergic synapses
    if len(gaba_secs) > 0:

        gaba_id = 0 # index used for gaba_synapse list and printing
        for dend_gaba, gaba_reversal in zip(gaba_secs, gaba_reversals):

            # For now, just assign to middle of section instead of uniform random
            gaba_loc = gaba_locations[gaba_id]

            # Choose random location along section
    #                 gaba_loc = round(random.uniform(0, 1), 2)

            gaba_locs.append(gaba_loc)

            # Define gaba synapse
            gaba_synapses[gaba_id] = h.gabasynapse(dend_gaba(gaba_loc)) 
            if physiological:
                gaba_synapses[gaba_id].tau1 = 0.9 
                gaba_synapses[gaba_id].tau2 = 18
            else:
                gaba_synapses[gaba_id].tau2 = 0.9 # TODO: Tune tau2 further for accurate response 
            gaba_synapses[gaba_id].erev = gaba_reversal

            # Stim to play back spike times
            gaba_stimulator[gaba_id] = h.VecStim()

            # Use with deterministic onset times
            gaba_stimulator[gaba_id].play(h.Vector(1, gaba_onsets[gaba_id]))

            # Connect stim and syn
            gaba_connection[gaba_id] = h.NetCon(gaba_stimulator[gaba_id], gaba_synapses[gaba_id])
            gaba_connection[gaba_id].weight[0] = gaba_weight # Depending on desired EPSP response at soma, tune this

            if return_currents:
                # Measure conductance and current
                gaba_currents[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_i)
                gaba_conductances[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_g)

            gaba_id += 1 # increment gaba counter

        rounded_locs = [round(value, 4) for value in gaba_locs]
        print("# gaba synapses added:{} on:{} with locs:{} with timing onsets:{}".format(gaba_id, gaba_secs, rounded_locs, gaba_onsets))
    return gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs


def count_unique_dends(input_list):
    unique_names = set(input_list)
    count = len(unique_names)
    return count

# Records voltage across all sections
def record_all_v(cell, loc=0.4):
    all_v = {}
    for sec in cell.allseclist:
        all_v[sec.name()] = h.Vector()
        all_v[sec.name()].record(sec(loc)._ref_v) # given a sec with multiple seg, 
    return all_v

# Records voltage across selected sections
def record_v(cell, seclist, loc=0.4):
    all_v = {}
    for sec in seclist:
        all_v[sec.name()] = h.Vector()
        all_v[sec.name()].record(sec(loc)._ref_v) # given a sec with multiple seg, 
    return all_v


# function gets all unique locations on path to soma
def record_all_path_secs_v(cell, dend_tree, dendrite):
    all_v = {}
    dists = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
    # get path to soma
    if dendrite.name() != 'soma[0]':
        pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    else:
        pathlist = [dendrite]
    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        for seg in sec:
            dist = h.distance(seg)
            dists.append(dist)
            loc = seg.x
            all_v[i] = h.Vector()
            all_v[i].record(sec(loc)._ref_v) # given a sec with multiple seg
            i = i + 1
    return all_v, dists

def record_all_path_secs_v2(cell, dend_tree, dendrite):
    all_v = {}
    dists = []
    dends = []
    i = 0

    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend

    # Get path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)

    # Record data for each dendrite in the path
    for sec in pathlist:
        all_v, i, sec_dends, sec_dists = sec_all_v(sec, all_v, i)
        dends.extend(sec_dends)
        dists.extend(sec_dists)

    return all_v, dists, dends

def sec_all_v(section, all_v, i):
    """
    Records data from each segment in a given section.

    Args:
        section (object): The neuron section to record from.
        all_v (dict): Dictionary to store recorded vectors.
        i (int): Index for storing in the dictionary.

    Returns:
        tuple: Updated dictionary `all_v`, updated index `i`, list of section names, and list of distances.
    """
    dends = []
    dists = []

    for seg in section:
        dends.append(section.name())
        dist = h.distance(seg)
        dists.append(dist)
        loc = seg.x
        all_v[i] = h.Vector()
        all_v[i].record(section(loc)._ref_v)  # Record voltage at this segment
        i += 1

    return all_v, i, dends, dists

def unique_path_secs_v_old(cell_type, specs, dendrite, spine_per_length=1.61, frequency=2000, d_lambda=0.05, axospine=True, n=1, dend2remove=None, neck_dynamics=False):
    """
    Gets all unique locations on the path to soma for a given cell type and dendrite.
    
    Parameters:
        cell_type (str): The type of the cell.
        specs (Dict): The specifications for the cell.
        dendrite_name (str): The name of the dendrite.
        axospiny (bool): Whether to include dendrites with at least n spines.
        n (int): Minimum number of spines required for a dendrite to be included.
               
    Returns:
        three lists - dends, locs, and dists.
    """
        # Build cell
    cell, spines, dend_tree, branch_groups= cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, branch=True, dend2remove=dend2remove, neck_dynamics=neck_dynamics)
    
    # Identify the target dendrite
    dendrite = next((dend for dend in cell.allseclist if dend.name() == dendrite), None)
    if dendrite is None:
        raise ValueError(f"No dendrite found with name: {dendrite}")
    
    # Find path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    
    # determine which dendrites have at least n=1 spines
    if axospine:
        dends_required = dend_spine_selector(cell=cell, spines=spines, branch_groups=branch_groups, n=n)
    else:
        dends_required = pathlist
    
    dends, locs, dists = [], [], []
    
    # Iterate through each dendrite in the path and find unique locations corresponding to each segment
    for sec in pathlist:
        if sec in dends_required:
            for seg in sec:
                dist = h.distance(seg)
                dists.append(dist)
                locs.append(seg.x)
                dends.append(sec.name())
            
    return dends[::-1], locs[::-1], dists[::-1]

# returns locations of all spines        
def spine_locations(pathlist, dends_required, spines):
    spine_dends = []
    spine_locs = []
    spine_dists = []

    for sec in pathlist:
        if sec in dends_required:
            names = list(spines[sec.name()].keys())
            for name in names:
                spine_dends.append(sec.name())
                loc = spines[sec.name()][name].x
                spine_locs.append(loc)
                spine_dists.append(h.distance(sec(loc)))

    return spine_dends[::-1], spine_locs[::-1], spine_dists[::-1]

                
def shaft_locs(pathlist):
    dends = []
    locs = []
    dists = []
    # Iterate through each dendrite in the path and find unique locations corresponding to each segment
    for sec in pathlist:
        for seg in sec:
            dist = h.distance(seg)
            dists.append(dist)
            locs.append(seg.x)
            dends.append(sec.name())
    return dends[::-1], locs[::-1], dists[::-1]

def unique_path_secs_v(cell_type, specs, dendrite, spine_per_length=1.61, frequency=2000, d_lambda=0.05, axospine=True, n=1, dend2remove=None, neck_dynamics=False):
    """
    gets all unique locations on the path to soma for a given cell type and dendrite.
    
    parameters:
        cell_type (str): The type of the cell.
        specs (Dict): The specifications for the cell.
        dendrite_name (str): The name of the dendrite.
        axospiny (bool): Whether to include dendrites with at least n spines.
        n (int): Minimum number of spines required for a dendrite to be included.
               
    Returns:
        three lists - dends, locs, and dists.
    """
    # Build cell
    cell, spines, dend_tree, branch_groups= cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, branch=True, dend2remove=dend2remove, neck_dynamics=neck_dynamics)

    # Identify the target dendrite
    dendrite = next((dend for dend in cell.allseclist if dend.name() == dendrite), None)
    if dendrite is None:
        raise ValueError(f"No dendrite found with name: {dendrite}")

    # Find path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)

    # Find path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)

    # determine which dendrites have at least n=1 spines
    dends, locs, dists = shaft_locs(pathlist) # all unique locations

    if axospine:
        dends_required = dend_spine_selector(cell=cell, spines=spines, branch_groups=branch_groups, n=n)
        spine_dends, spine_locs, spine_dists = spine_locations(pathlist=pathlist, dends_required=dends_required, spines=spines)

        idxs = []
        # Using zip to loop through targets and dend_names in pairs
        for target, dend_name in zip(locs, dends):
            idx = find_closest_loc_index(spine_locs, spine_dends, target, dend_name)
            idxs.append(idx)


        idxs = [item for item in idxs if str(item).startswith('No locations') is False]

        # remove duplicates
        seen = set()
        idxs2 = []

        for idx in idxs:
            if idx not in seen:
                idxs2.append(idx)
                seen.add(idx)

        dends, locs, dist = [], [], []
        for idx in idxs2:
            locs.append(spine_locs[idx])
            dends.append(spine_dends[idx])
            dist.append(spine_dists[idx])

    return dends, locs, dists

def find_closest_loc_index(locs, dends, target, dend_name):
    # Filter the locs for the specified dend_name and keep track of the original indices
    filtered_locs = [(loc, index) for index, (loc, dend) in enumerate(zip(locs, dends)) if dend == dend_name]
    
    # If there are no matches, return a message
    if not filtered_locs:
        return f"No locations found for {dend_name}"
    
    # Find the index of the loc closest to the target
    closest_loc, closest_index = min(filtered_locs, key=lambda x: abs(x[0] - target))
    
    return closest_index

# find location in dend that is closest to dend_gaba with gaba_locations = [0.5] 
def find_closest_loc(locs, dends, target, dend_name):
    # Filter the locs for the specified dend_name
    filtered_locs = [loc for loc, dend in zip(locs, dends) if dend == dend_name]
    
    # If there are no matches, return a message
    if not filtered_locs:
        return f"No locations found for {dend_name}"
    
    # Find the loc closest to the target
    closest_loc = min(filtered_locs, key=lambda x: abs(x - target))
    
    return closest_loc

# Records Cai across selected sections
def record_cai(cell, seclist, loc=0.4, return_Ca=True):
    all_cai = {}
    if return_Ca:
        for sec in seclist:
            all_cai[sec.name()] = h.Vector()
            all_cai[sec.name()].record(sec(loc)._ref_cai) # given a sec with multiple seg, 
    return all_cai

# Returns vectors for impedance recording
def record_impedance(dend, loc=0.4):
    imp = h.Impedance()
    # imp.loc(0.5, sec=cell.soma) 
    # define location either for current stim or voltage measuring electrode
    # this is needed for the transfer impedance calculation
    imp.loc(loc, sec=dend) # location of interest; nb voltages are measured at 0.4 ; not necessary if computing imp.input()  
    zvec1 = h.Vector()  
    zvec1.append(0)
    zvec2 = h.Vector()  
    zvec2.append(0)
    return imp, zvec1, zvec2

def record_i_mechs(cell, dend, loc=0.4, return_currents=True, silent=False, mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kcnq', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']):
    i_mechs_out = []
    if return_currents:
        # Record time vector
        t = h.Vector().record(h._ref_t)
        i_mechs_out.append(t)

        if not silent: 
            print("i_mechanisms recorded in {}".format(dend))

        # Predefined dictionary for mechanism references
        mech_refs = {
            'pas': '_ref_i_pas',
            'kdr': '_ref_ik_kdr',
            'naf': '_ref_ina_naf',
            'kaf': '_ref_ik_kaf',
            'kas': '_ref_ik_kas',
            'kcnq': '_ref_ik_kcnq',
            'kir': '_ref_ik_kir',
            'cal12': '_ref_ical_cal12',
            'cal13': '_ref_ical_cal13',
            'can': '_ref_ica_can',
            'car': '_ref_ica_car',
            'cav32': '_ref_ical_cav32',
            'cav33': '_ref_ical_cav33',
            'sk': '_ref_ik_sk',
            'bk': '_ref_ik_bk'
        }

        # Loop over requested mechanisms
        for mech in mechs:
            ref_attr = mech_refs.get(mech)
            if ref_attr and hasattr(dend(loc), ref_attr):
                # Record the mechanism if it exists
                record_vector = h.Vector().record(getattr(dend(loc), ref_attr))
                i_mechs_out.append(record_vector)
            else:
                print(f"warning: mechanism '{mech}' not recognized or not present at the specified location")

        return i_mechs_out
    
# for plotting, returns all branch dendrites
def dend2plot(cell, cell_type='dspn'):
    [branch1_dends, branch2_dends, branch3_dends, branch4_dends, branch5_dends] = branch_selection(cell, cell_type) 
    branch_dends = [branch1_dends] + [branch2_dends] + [branch3_dends] + [branch4_dends] + [branch5_dends]
    branch_dends = [num for sublist in branch_dends for num in sublist]
    return [cell.soma] + branch_dends

def plot1(cell=None, dend=None, t=None, v=None, seclist=None, sparse=False, protocol=''):
    import plotly.graph_objects as go
    v_data = []
    if sparse:
        for group in seclist: # for each nrn dendrite sec, one plot per branch
            if dend in group: # Use if you want sparse plotting
                for sec in group:
                    v_data.append(go.Scatter(x=t, y=v[sec.name()], name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
                v_data.append(go.Scatter(x=t, y=v['soma[0]'], name='soma'))
    else:
        for sec in seclist: # for each nrn dendrite sec, one plot per branch
            v_data.append(go.Scatter(x=t, y=v[sec.name()], name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
    
    # Plot vdata
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(protocol),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title="V (mV)",
        legend_title="section")
    return fig   

def plot1_Ca(cell=None, dend=None, t=None, Ca=None, seclist=None, sparse=False, protocol=''):
    import plotly.graph_objects as go
    Ca_data = []
    if sparse:
        for group in seclist: # for each nrn dendrite sec, one plot per branch
            if dend in group: # Use if you want sparse plotting
                for sec in group:
                    Ca_data.append(go.Scatter(x=t, y=Ca[sec.name()]*1e3, name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
                Ca_data.append(go.Scatter(x=t, y=Ca['soma[0]']*1e3, name='soma'))
    else:
        for sec in seclist: # for each nrn dendrite sec, one plot per branch
            Ca_data.append(go.Scatter(x=t, y=Ca[sec.name()]*1e3, name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
    
    # Plot vdata
    fig = go.Figure(data=Ca_data)
    fig.update_layout(
        title="{}".format(protocol),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title="[Ca] (uM)",
        legend_title="section")
    return fig   


def plot2(soma_v_data, dend_v_data, glut_placement=None, yaxis='V (mV)'):
    # import plotly.graph_objects as go
    if yaxis=='V (mV)':
        title1='soma PSP'
        if glut_placement == 'distal':
            title2 = 'distal dendrite PSP'
        elif glut_placement == 'proximal':
            title2 = 'proximal dendrite PSP'
        else:
            title2 = 'dendrite PSP'
    else:
        title1='soma impedance'
        if glut_placement == 'distal':
            title2 = 'distal dendrite impedance'
        elif glut_placement == 'proximal':
            title2 = 'proximal dendrite impedance'
        else:
            title2 = 'dendrite impedance'
            
    fig_soma = go.Figure(data=soma_v_data)
    fig_soma.update_layout(
        title=title1,
        title_x=0.5,
        xaxis_title='time (ms)',
        yaxis_title=yaxis,
        legend_title='sim')
#     fig_soma.show() 

    fig_dend = go.Figure(data=dend_v_data)
    fig_dend.update_layout(
        title=title2,
        title_x=0.5,
        xaxis_title='time (ms)',
        yaxis_title=yaxis,
        legend_title='sim')
#         fig_dend.show() 
    return fig_soma, fig_dend  
 
def plot3(somaV, dendV, glut_placement=None, yaxis='V (mV)', yrange_soma=[-85,-60], yrange_dend=[-85,-30], stim_time = 150, sim_time=400, black_trace=0, gray_trace=None, err_bar=50, baseline=20, dt=0.025, width=500, height=500):

    def hex_palette(n):
        import seaborn as sns
        import matplotlib as mpl
        colors = ['#6A5ACD', '#CD5C5C', '#458B74', '#9932CC', '#FF8247'] # Set your custom color palette
        if n < len(colors):
            colors = colors[0:n]
        else:
            colors = sns.blend_palette(colors,n)
        cols = list(map(mpl.colors.rgb2hex, colors))
        return cols
    
    n = len(somaV)
    cols = hex_palette(n)
    # this routine places black and gray traces if required
    if black_trace == None and gray_trace == None:
        cols = hex_palette(n)
    elif black_trace is not None and gray_trace == None:
        cols = hex_palette(n-1)
        cols.insert(black_trace,'#000000')
    elif black_trace is not None and gray_trace is not None:  
        cols = hex_palette(n-2)
        if (black_trace>gray_trace):
            cols.insert(gray_trace,'#D3D3D3')
            cols.insert(black_trace,'#000000')
        elif (black_trace<gray_trace):
            cols.insert(black_trace,'#000000')
            cols.insert(gray_trace,'#D3D3D3')
            
    def update_layout(fig, main=None, yaxis=None, yrange=[-85,-60], width=500, height=500):
        font = 'Droid Sans'
        font_size = 18
        fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=main,
        title_x=0.45,
        title_font_family=font,
        title_font_size=font_size,
        xaxis=dict(showticklabels=False, titlefont=dict(size=font_size, family=font), tickfont=dict(size=font_size, family=font), showgrid=False), # title='time (ms)', 
        yaxis=dict(side='right', tick0 = yrange[0], dtick = (yrange[1]-yrange[0]), tickfont=dict(size=font_size, family=font), showgrid=False),
        legend = dict(title='sim', x=1.1, y=0.95)
    )

    def plot3_(somaV, dendV, glut_placement, yaxis='V (mV)', cols=None, yrange_soma=[-85,-60], yrange_dend=[-85,-30], err_bar=50, bl = 20):
        import plotly.graph_objects as go
        if yaxis=='V (mV)':
            title1='soma PSP'
            if glut_placement == 'distal':
                title2 = 'distal dendritic PSP'
            elif glut_placement == 'proximal':
                title2 = 'proximal dendritic PSP'
            else:
                title2 = 'dendritic PSP'
                
        else:
            title1='soma impedance'
            if glut_placement == 'distal':
                title2 = 'distal dendritic impedance'
            elif glut_placement == 'proximal':
                title2 = 'proximal dendritic impedance'
            else:
                title2 = 'dendritic impedance'

        figSoma = go.Figure()
        
        ind1 = 0
        ind2 = int((sim_time - stim_time + bl)/dt)
        ind3 = int((stim_time - bl)/dt)
        ind4 = int(sim_time/dt)
        
        for ii in range(len(somaV)):
            dat = somaV[ii]
            figSoma.add_trace( go.Scatter(x=dat['x'][ind1:ind2], y=dat['y'][ind3:ind4], mode='lines', line=dict(color=cols[ii])) )
        figSoma.add_hline(y=yrange_soma[0], line_width=2, line_dash="dot", line_color="gray")
        figSoma.add_hline(y=yrange_soma[1], line_width=2, line_dash="dot", line_color="gray")        
        figSoma.add_shape(
                    type='line',
                    x0=ind2*dt-err_bar,
                    y0=yrange_soma[0]+2,
                    x1=ind2*dt,
                    y1=yrange_soma[0]+2,
                    line=dict(color='black'),
                    xref='x',
                    yref='y'
        )
        update_layout(fig=figSoma, main=title1, yaxis=yaxis, yrange=yrange_soma, width=width, height=height)   
        
        figDend = go.Figure()
        for ii in range(len(dendV)):
            dat = dendV[ii]
            figDend.add_trace( go.Scatter(x=dat['x'][ind1:ind2], y=dat['y'][ind3:ind4], mode='lines', line=dict(color=cols[ii])) )
        figDend.add_hline(y=yrange_dend[0], line_width=2, line_dash="dot", line_color="gray")
        figDend.add_hline(y=yrange_dend[1], line_width=2, line_dash="dot", line_color="gray")
        update_layout(fig=figDend, main=title2, yaxis=yaxis, yrange=yrange_dend, width=width, height=height)
        
        return figSoma, figDend


    fig_soma_master, fig_dend_master =  plot3_(somaV=somaV, dendV=dendV, glut_placement=glut_placement, yaxis=yaxis, cols=cols, yrange_soma=yrange_soma, yrange_dend=yrange_dend, err_bar=err_bar, bl=baseline)    

    return fig_soma_master, fig_dend_master 

def save_fig2(soma_fig=None, dend_fig=None, cell_type='dspn', model=None, physiological=True, sim=None, g_name=None):
    import datetime 
    time = datetime.datetime.now()
    path_cell = "{}".format(cell_type)
    if not os.path.exists(path_cell):
        os.mkdir(path_cell)    
    path1 = "{}/model{}".format(path_cell, model)
    if not os.path.exists(path1):
        os.mkdir(path1)
    if physiological: 
        path2 = "{}/physiological".format(path1)
    else:
        path2 = "{}/nonphysiological".format(path1)
    if not os.path.exists(path2):
        os.mkdir(path2)  
    path3 = "{}/images".format(path2)
    if not os.path.exists(path3):
        os.mkdir(path3)

    image_dir = "{}/sim{}".format(path3, sim)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)    

    if (g_name is None):
        soma_fig.write_image("{}/soma_fig{}.svg".format(image_dir, time))
        dend_fig.write_image("{}/dend_fig{}.svg".format(image_dir, time))
        soma_fig.write_html("{}/soma_fig{}.html".format(image_dir, time))
        dend_fig.write_html("{}/dend_fig{}.html".format(image_dir, time))
    else:
        soma_fig.write_image("{}/{}_soma_fig{}.svg".format(image_dir, g_name, time))
        dend_fig.write_image("{}/{}_dend_fig{}.svg".format(image_dir, g_name, time))  
        soma_fig.write_html("{}/{}_soma_fig{}.html".format(image_dir, g_name, time))
        dend_fig.write_html("{}/{}_dend_fig{}.html".format(image_dir, g_name, time))

    
def convert2df(d, g_name):
    df = pd.DataFrame() 
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    df['dist'] = x
    df[g_name] = y
    return df

def dist_(cell, g_name):
    if g_name in ['naf', 'kaf', 'kas', 'kdr', 'kir', 'sk', 'bk', 'gaba1', 'gaba2']:
        gbar = 'gbar_{}'.format(g_name)
    else:
        gbar = 'pbar_{}'.format(g_name)        
    d__ = {}
    for sec in cell.dendlist:
         d__[h.distance(sec(0.5))] = eval('sec.{}'.format(gbar))
    return convert2df(d__, g_name)

def plot4(data, g_name): 
    import plotly.graph_objects as go
    y = data[0].y
    num = y.max()
    sig_fig = len(str(num)) - str(num).find('.') - 1
    y1 = 2*round(num, sig_fig)
    if (y1==0):
        y1 = 1e-4
    fig = go.Figure(data=data)
    fig.update_layout(
        title="{}".format(g_name),
        title_x=0.5,
        yaxis=dict(range=[0, y1]),
        xaxis_title="distance (um)",
        yaxis_title="conductance (S/cm2)",
        legend_title="cond")
    return fig

def plot5(X, dt, dists, xaxis_range=[0,150], yaxis_range=[0,8], normalised=True, title='', voltage=True):
    t2 = np.arange(0, len(X[0]), 1) * dt
    # import plotly.graph_objects as go
    v_data = []
    if normalised:
        yaxis_title="normalised amplitude"
    else:
        if voltage:
            yaxis_title="V (mV)" 
        else:
            yaxis_title="I (pA)" 
            
    for ii in list(range(len(X))):
        v_data.append(go.Scatter(x=t2, y=X[ii], name='{}'.format(round(dists[ii], 2))))
    # Plot vdata
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(title),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="distance (um)")
    return fig

def plot5a(X, dt, locs, xaxis_range=[0,150], yaxis_range=[0,-30], normalised=False, col=[], title=''):
    t2 = np.arange(0, len(X[0]), 1) * dt
    import plotly.graph_objects as go
    v_data = []
    if normalised:
        yaxis_title="normalised amplitude"
    else:
        yaxis_title="I (pA)"        
    for ii in list(range(len(X))):
        if len(col) == 0:
            v_data.append(go.Scatter(x=t2, y=X[ii], name='{}'.format(locs[ii])))
        else:
            v_data.append(go.Scatter(x=t2, y=X[ii], line=dict(color=col[ii]), name='{}'.format(locs[ii])))
    # Plot vdata
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(title),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="location")
    return fig

def plot5b(X, dt, locs, xaxis_range=[0,150], yaxis_range=[0,-30], normalised=False, dotted=False, col=[], title=''):
    t2 = np.arange(0, len(X[0]), 1) * dt
    import plotly.graph_objects as go
    v_data = []
    if normalised:
        yaxis_title="normalised amplitude"
    else:
        yaxis_title="V (mV)"        
    for ii in list(range(len(X))):
        if dotted:
            v_data.append(go.Scatter(x=t2, y=X[ii], line=dict(dash='dot', color='gray'), showlegend=False))
        else:
            if len(col) == 0: 
                v_data.append(go.Scatter(x=t2, y=X[ii], name='{}'.format(locs[ii])))
            else:
                v_data.append(go.Scatter(x=t2, y=X[ii], line=dict(color=col[ii]), name='{}'.format(locs[ii])))
    # Plot vdata
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(title),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="location")
    return fig

# remove offsets
def normalise(X, stim_time, burn_time, dt):    
    def mean(x):
        n = len(x)
        sum = 0
        for i in x:
            sum = sum + i
        return(sum/n)
    ind1 = int(burn_time/dt)
    ind2 = int(stim_time/dt)
    return(X[ind1:len(X)] - mean(X[ind1:ind2]) )

def plot6(y, x, xaxis_range=[200,0], yaxis_range=[0,1.01], normalised=True):
    import plotly.express as px
    if normalised:
        yaxis_title="normalised amplitude"
    else:
        yaxis_title="V (mV)"        
    fig2 = px.scatter(x=x, y=y)
    fig2.update_layout(
        title="{}".format(''),
        title_x=0.5,
        xaxis_title="distance (um)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="attenuation")
    return fig2

def plot6a(mat, x, xaxis_range=[200,0], yaxis_range=[0,1.01], normalised=True, col=[], current=True):
    import plotly.graph_objects as go
    i_data = []
    if normalised:
        if current:
            yaxis_title="normalised PSC"
        else:
            yaxis_title="normalised PSP"
    else:
        if current:
            yaxis_title="I (pA)" 
        else:
            yaxis_title="V (mV)" 

    rows, columns = mat.shape
    if rows == 3:
        names =['spine', 'dendrite', 'soma']
    else:
        names =['dendrite', 'soma']
    for ii in list(range(rows)):
        if len(col) == 0:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color='gray'), name='{}'.format(names[ii]), showlegend=True))
        else:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color=col[ii]), name='{}'.format(names[ii]), showlegend=True))

    fig2 = go.Figure(data=i_data)
    fig2.update_layout(
        title="{}".format(''),
        title_x=0.5,
        xaxis_title="distance (um)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="attenuation")
    return fig2

def plot6aa(mat, x, xaxis_range=[200,0], yaxis_range=[0,1.01], normalised=True, col=[], current=True):
    import plotly.graph_objects as go
    i_data = []
    if normalised:
        if current:
            yaxis_title="normalised PSC"
        else:
            yaxis_title="normalised PSP"
    else:
        if current:
            yaxis_title="I (pA)" 
        else:
            yaxis_title="V (mV)" 

    rows, columns = mat.shape
    if rows == 3:
        names =['spine', 'dendrite', 'soma']
    else:
        names =['dendrite', 'soma']
    for ii in list(range(rows)):
        if len(col) == 0:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color='gray'), name='{}'.format(names[ii]), showlegend=True))
        else:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color=col[ii]), name='{}'.format(names[ii]), showlegend=True))

    fig2 = go.Figure(data=i_data)
    fig2.update_layout(
        title="{}".format(''),
        title_x=0.5,
        xaxis_type='log',
        xaxis_title="series resistance (MOhm)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="attenuation")
    return fig2

# for plotting current steps
def plot9(x, ydict, yaxis='', xaxis='', current_step_range=None, yaxis_range=[-110,30], xaxis_range=[200, 1500], y_err_bar=10, x_err_bar=100, y_err_bar_shift=5, width=500, height=500):
    
    def hex_palette(n):
        import seaborn as sns
        import matplotlib as mpl
        colors = ['#6A5ACD', '#CD5C5C', '#458B74', '#9932CC', '#FF8247'] # Set custom color palette
        if n < len(colors):
            colors = colors[0:n]
        else:
            colors = sns.blend_palette(colors,n)
        cols = list(map(mpl.colors.rgb2hex, colors))
        return cols

    # Create the figure
    fig = go.Figure()

    # N traces to plot
    N = len(ydict)
    cols = hex_palette(N)
    
    # Add the line plot
    for ii in list(range(N)):
        if current_step_range is None:
            fig.add_trace(go.Scatter(x=x, y=ydict[ii], mode='lines', line=dict(color=cols[ii])))
        else:
            fig.add_trace(go.Scatter(x=x, y=ydict[ii], mode='lines', line=dict(color=cols[ii]), name=str(current_step_range[ii])))
        
    # Set the range for the x-axis
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=yaxis_range, showticklabels=False, title=yaxis),
        xaxis=dict(range=xaxis_range, showticklabels=False, title=xaxis),
        title='',
        xaxis_title=xaxis,
        yaxis_title=yaxis
    )
    
    fig.add_shape(
        type='line',
        x0=xaxis_range[1] - x_err_bar,
        y0=yaxis_range[0] + y_err_bar_shift,
        x1=xaxis_range[1],
        y1=yaxis_range[0] + y_err_bar_shift,
        line=dict(color='black'),
        xref='x',
        yref='y'
        )
    
    fig.add_shape(
        type='line',
        x0=xaxis_range[1] - x_err_bar,
        y0=yaxis_range[0] + y_err_bar_shift,
        x1=xaxis_range[1] - x_err_bar,
        y1=yaxis_range[0] + y_err_bar_shift + y_err_bar,
        line=dict(color='black'),
        xref='x',
        yref='y'
        )
    return fig


# Find all unique synapse locations in dendritic path 
# if GABA then want all unique locations
# if Glutamate then only where there is a spine
# start from end of particular dendrite and move towards soma

def gaba_idx(dend):
    locs = []
    for seg in dend:
        locs.append(seg.x)
    return locs

def all_synapses_tree(cell, dend_tree, dendrite, glut):
    candidate_list = []
    locs_list = []
    for dend in cell.dendlist:
        if dend.name() == dendrite:
            dendrite = dend
        # get path to soma
    pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    if glut:
        # get all unique spine locations to place glut synapses
        for sec in pathlist:
            dend_dist = h.distance(cell.soma(), sec(1)) # distance to end of that dendrite from middle of soma
            locs = []
            if dend_dist < 30:
                candidates = []
            else:
                candidates = spine_idx(cell=cell, spines=spines, dend=sec.name())
                for candidate in candidates:
                    locs.append(candidate.x)

            candidate_list.append(candidates)
            locs_list.append(locs)
    else:
        # get all unique gaba synapse locations
        locs_list = []
        for sec in pathlist:
            locs = gaba_idx(sec)
            locs_list.append(locs)

    return pathlist, locs_list, candidate_list

def space_clamped(cell, spines, Ra = 1.59e-10):
    for sec in cell.allseclist:
        sec.Ra = Ra
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.head.Ra = Ra
            spine_obj.neck.Ra = Ra

def cap(cell, spines, cm = 1):
    for sec in cell.allseclist:
        sec.cm = cm
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.head.cm = cm
            spine_obj.neck.cm = cm
            
def find_closest_value(test, target_value):
    import numpy as np
    test = np.array(test)
    distances = np.abs(test - target_value)
    closest_index = np.argmin(distances)
    closest_value = test[closest_index]
    return closest_value, closest_index

def rounded(number, n=10):
    if number >= 0:
        rounded = n*math.ceil(number/n)
    else:
        rounded = n*math.floor(number/n)
    return rounded


def IR(X, step_start, step_end, step):   
    ind1 = int((step_start-5)/dt)
    ind2 = int(step_start/dt)
    ind3 = int((step_end-5)/dt)
    ind4 = int(step_end/dt)

    def mean(x):
        n = len(x)
        sum = 0
        for i in x:
            sum = sum + i
        return(sum/n)
 
    return(1e3 *( mean(X[ind1:ind2]) - mean(X[ind3:ind4]) )  / -step) # MOhm

def whole_cell_capacitance(cell, spines=None, Cm=1):
    # for seg in sec.allseg():
    #     print(seg.area())

    # for sec in cell.dendlist:
    #     print(seg.area())
    areas = []
    for sec in cell.dendlist:
        for i,seg in enumerate(sec):
            areas.append(seg.area())

    if spines is None:
        AREA = sum(areas)
    else:
        spine_areas = []
        for sec in cell.dendlist:
            sec_spines = list(spines[sec.name()].items())
            for spine_i, spine_obj in sec_spines: # area of sphere + cylinder less areas that are connections to head and dendrite
                spine_areas.append(4 * math.pi * ((spine_obj.head.diam/2) ** 2) + 2 * math.pi * spine_obj.neck.L * spine_obj.neck.diam/2  - 2 * math.pi * ((spine_obj.neck.diam/2)**2) ) # diam

        AREA = sum(spine_areas) + sum(areas) +  4 * math.pi * cell.soma.diam/2 ** 2 # in um sq
    # AREA in um sq
    # Cm in uF/cm2
    # convert uF/cm2 to F/um2
    # To convert square centimeters (cm²) to square micrometers (μm²), you can use the following conversion factor:
    # 1 cm² = 1e8 μm²
    # convert uF to pF conversion factor
    # 1uF = 1e6 pF
    
    # AREA * Cm * 1e-8 * 1e6 # (cm2)
    cap = AREA * Cm * 1e-2 # pF
    return cap
 
def sampler(names, n, replacement=True):
    if replacement:
        return [random.choice(names) for _ in range(n)]
    else:
        return random.sample(names, n)
    
def uniform_values(n):
    return [random.uniform(0, 1) for _ in range(n)]

# function to determine what is varying
def variable_detector(xrange):
    vary = False
    differences = [abs(xrange[i] - xrange[i + 1]) for i in range(len(xrange) - 1)]
    sum_of_differences = sum(differences)
    if sum_of_differences > 0:
        vary = True
    return vary

def spine_locator(cell_type, specs, spine_per_length, frequency, d_lambda, dend_glut, num_gluts, method=0, rel_x=None, dend2remove=None):
    
    if rel_x is None:
        rel_x = 2/3
    
    # Create the cell and get the dendrite list
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=dend2remove)
    # Get target dendrites for glutamate synapses
    glut_secs = [sec for target_dend in dend_glut for sec in cell.dendlist if sec.name() == target_dend] * num_gluts

    final_spine_locs = []
    glut_id = 0 # Index used for glut_synapse list and printing

    for dend_glut in glut_secs:
        # Get candidate spines from section
        candidate_spines = []
        sec_spines = list(spines[dend_glut.name()].items())

        if sec_spines:
            for spine_i, spine_obj in sec_spines: 
                candidate_spines.append(spine_obj)

            if len(glut_secs) < len(sec_spines):
                if method==1:
                    # Reverse order so activate along dendrite towards soma
                    spine_idx = int(rel_x * len(candidate_spines)) - 1 # default arbitrary start point at 2/3 of spines
                    spine = candidate_spines[spine_idx - glut_id] 
                else:
                    spine_idx = int(rel_x * len(candidate_spines)) - num_gluts # default arbitrary start point at 1/3 of spines
                    if spine_idx < 0:
                        if len(candidate_spines) >= num_gluts:
                            spine_idx = len(candidate_spines) - num_gluts
                        else:
                            spine_idx = 0        
                    spine = candidate_spines[spine_idx + glut_id] 
            else:
                spine = random.choice(candidate_spines)

            spine_loc = spine.x
            final_spine_locs.append(spine_loc) 
            glut_id = glut_id + 1
    return final_spine_locs  

# relative version of gaba_onsets
def rel_gaba_onset(n, N):
    if N in [0,1]:
        if (n < 4):
            gaba_onsets = list(range(0, 0 + n)) 
        else:
            if n % 3 == 0:
                gaba_onsets = list(range(0, 0 + int(n/3))) * 3 * N
            else:
                gaba_onsets = list(range(0, 0 + int(n/3)+1)) * 3 * N
        gaba_onsets = gaba_onsets[:n]
    else:
        n1 = round(n / N)
        if n1 % 3 == 0:
            n1 = round(n / N)
            onsets = list(range(0, n1)) 
        else:
            onsets = list(range(0, n1+1)) 
        gaba_onsets = [x for x in onsets for _ in range(N)]
        gaba_onsets = gaba_onsets[:n]
    return gaba_onsets

def plt1(data_dict):
    # Unpack metadata
    metadata = data_dict['metadata']
    keys = ['sim_time', 'stim_time', 'model', 'cell_type', 'showplot', 'save', 'sim', 'physiological', 'dt']
    sim_time, stim_time, model, cell_type, showplot, save, sim, physiological, dt = (metadata[key] for key in keys)
    
    # Setup time axis
    N = len(data_dict['vsoma'][0])
    time_axis = np.linspace(0, sim_time, N)

    # Define voltage trace containers
    soma_v_traces = []
    dend_v_traces = []

    # Generate traces
    for vsoma, vdend in zip(data_dict['vsoma'], data_dict['vdend']):
        soma_v_traces.append(go.Scatter(x=time_axis, y=vsoma))
        dend_v_traces.append(go.Scatter(x=time_axis, y=vdend))

    # Set y-axis ranges based on model or cell type
    yrange_soma, yrange_dend = ([-85, -60], [-85, -30])  # Default ranges
    if model == 2 or cell_type == 'ispn':
        yrange_soma, yrange_dend = ([-85, -50], [-85, -20])

    # Generate plots
    fig_soma, fig_dend = plot3(somaV=soma_v_traces, dendV=dend_v_traces, glut_placement='', yaxis='V (mV)', yrange_soma=yrange_soma, yrange_dend=yrange_dend, stim_time=stim_time, sim_time=sim_time, black_trace=0, gray_trace=1, err_bar=50, baseline=20, dt=dt,width=600, height=400)

    # Show plots
    if showplot:
        fig_soma.show()
        fig_dend.show()

    # Save plots
    if save:
        base_dir = os.path.join(cell_type, f"model{model}", 'physiological' if physiological else 'nonphysiological', f"images/sim{sim}")
        os.makedirs(base_dir, exist_ok=True)
        for fig, name in zip([fig_soma, fig_dend], ['soma', 'dend']):
            fig_path = os.path.join(base_dir, f"fig1_{name}.svg")
            fig.write_image(fig_path)
            fig.write_html(fig_path.replace('.svg', '.html'))
        
def plt2(data_dict, sim_time, dt, model, cell_type, stim_time, showplot, save, physiological, sim, offset=40, spine=True
    ):
    
    time = np.arange(0, len(data_dict['vsoma'][0]) * dt, dt)
    
    idx1, idx2, idx3  = 0, int(sim_time/dt), int(stim_time/dt)
    soma_v_master, dend_v_master, spine_v_master = [], [], []
    peaks_v_soma, peaks_v_dend, peaks_v_spine = [], [], []
    mins_v_soma, mins_v_dend, mins_v_spine = [], [], []
    x=time[idx1:idx2]
    for ii, (soma_v, dend_v, spine_v) in enumerate(zip(data_dict['vsoma'], data_dict['vdend'], data_dict['vspine'])):
        
        y=extract2(soma_v)[idx1:idx2]
        soma_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_soma.append(max(soma_v))
        mins_v_soma.append(soma_v[idx3])
        
        y=extract2(dend_v)[idx1:idx2]
        dend_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_dend.append(max(dend_v))
        mins_v_dend.append(dend_v[idx3])
        
        y=extract2(spine_v)[idx1:idx2]
        spine_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_spine.append(max(spine_v))
        mins_v_spine.append(spine_v[idx3])
    ysoma_range = [0.1*math.floor(min(mins_v_soma)/0.1), 0.1*math.ceil(max(peaks_v_soma)/0.1)]
    ydend_range = [math.floor(min(mins_v_dend)), math.ceil(max(peaks_v_dend))]
    yspine_range = [math.floor(min(mins_v_spine)), math.ceil(max(peaks_v_spine))]

    figs = plot3a(somaV=soma_v_master, dendV=dend_v_master, spineV=spine_v_master, ysoma_range=ysoma_range, 
                  ydend_range=ydend_range, yspine_range=yspine_range, stim_time=stim_time, sim_time=sim_time, 
                  dt=dt, offset=offset,spine=spine)

    if showplot:
        for fig in figs:
            fig.show()
              
    if save:
        path_format = f'{cell_type}/model{model}/{{}}/images/sim{sim}'
        folder = path_format.format('physiological' if physiological else 'nonphysiological')
        os.makedirs(folder, exist_ok=True)

        plots_to_do = ['spine', 'dend', 'soma'] if spine else ['dend', 'soma']
        for idx, name in enumerate(plots_to_do):
            figs[idx].write_image(f'{folder}/fig1_{name}{time}.svg')
            figs[idx].write_html(f'{folder}/fig1_{name}{time}.html')

def hex_palette(n):
    colors = ['#6A5ACD', '#CD5C5C', '#458B74', '#9932CC', '#FF8247'] # Set your custom color palette
    if n < len(colors):
        colors = colors[0:n]
    else:
        colors = sns.blend_palette(colors,n)
    cols = list(map(mpl.colors.rgb2hex, colors))
    return cols

def update_layout(fig, title, yaxis, yrange, width, height):
    font = 'Droid Sans'
    font_size = 18
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        title_x=0.45,
        title_font_family=font,
        title_font_size=font_size,
        xaxis=dict(showticklabels=False, titlefont=dict(size=font_size, family=font), tickfont=dict(size=font_size, family=font), showgrid=False),
        yaxis=dict(side='right', tick0=yrange[0], dtick=yrange[1]-yrange[0], tickfont=dict(size=font_size, family=font), showgrid=False),
        legend=dict(title='sim', x=1.1, y=0.95)
    )

def plot_trace(fig, data_list, cols, dt, err_bar, yrange, stim_time, sim_time, offset):
    ind1, ind2 = 0, int((sim_time - stim_time + offset)/dt)
    ind3, ind4 = int((stim_time - offset)/dt), int(sim_time/dt)
    dy = (yrange[1] - yrange[0])*9/10
    for dat, color in zip(data_list, cols):
        fig.add_trace(go.Scatter(x=dat['x'][ind1:ind2], y=dat['y'][ind3:ind4], mode='lines', line=dict(color=color)))
    fig.add_hline(y=yrange[0], line_width=2, line_dash="dot", line_color="gray")
    fig.add_hline(y=yrange[1], line_width=2, line_dash="dot", line_color="gray")
    fig.add_shape(type='line', x0=ind2*dt-err_bar, y0=yrange[0]+dy, x1=ind2*dt, y1=yrange[0]+dy, line=dict(color='black'))

def plot3a(somaV, dendV, spineV, ysoma_range, ydend_range, yspine_range, stim_time, sim_time, dt, offset, width=800, height=400, controls=True, spine=True):
    n = len(somaV)
    if controls:
        cols = hex_palette(n-2)
        cols.insert(0,'#000000')
        cols.insert(1,'#D3D3D3')
    else:
        cols = hex_palette(n)
   
    titles = ['spine PSP', 'dendritic PSP', 'soma PSP'] if spine else ['dendritic PSP', 'soma PSP']
    yranges = [yspine_range, ydend_range, ysoma_range] if spine else [ydend_range, ysoma_range]
    data_list = [spineV, dendV, somaV] if spine else [dendV, somaV]
    
    figs = []
    for data, title, yrange in zip(data_list, titles, yranges):
        fig = go.Figure()
        plot_trace(fig, data, cols, dt, 25, yrange, stim_time, sim_time, offset)
        update_layout(fig, title, 'V (mV)', yrange, width, height)
        figs.append(fig)
    return figs

def check_sim(sim, sims):
    """
    Checks if sim starts with or is equivalent to any value in values_to_check.
    
    :param sim: Value to check (can be integer or string)
    :param values_to_check: List of values to check against (mix of integers and strings)
    :return: True if there's a match, False otherwise
    """
    strings_to_check = [str(val) for val in sims]  # Convert all values to strings

    sim_str = str(sim)  # Convert sim to a string regardless of its type

    return any(sim_str.startswith(s) or sim_str == s for s in strings_to_check)

def update_data_dict(data_dict, protocol, v_tree, v_tree_spine, v_branch, soma_v, dend_v, spine_v, timing, t, dists, dists_spine, dends_v, dends_spine, i_dend_mechs, i_mechs_all, i_mechs_all_spine, dists_i_mechs, dists_spine_i_mechs, dends_i_mechs, dends_spine_i_mechs, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, time, record_dist, impedance=False, return_currents=False, record_spine=False):
    
    data_dict['v_tree'][protocol] = v_tree
    data_dict['v_tree_spine'][protocol] = v_tree_spine
    data_dict['v_branch'][protocol] = v_branch
    data_dict['soma_v'].append(soma_v[0])
    data_dict['dend_v'].append(dend_v[0])
    if record_spine:
        data_dict['spine_v'].append(spine_v[0])
    data_dict['timing'].append(timing)
    data_dict['time'].append(np.asarray(t))
    data_dict['dists'].append(dists)
    data_dict['dists_spine'].append(dists_spine)
    data_dict['dendrites_v'].append(dends_v)
    data_dict['dendrites_spine'].append(dends_spine)
    data_dict['i_mechs'][protocol] = i_dend_mechs
    data_dict['i_mechs_all'][protocol] = i_mechs_all
    data_dict['i_mechs_all_spine'][protocol] = i_mechs_all_spine
    data_dict['dists_i_mechs'].append(dists_i_mechs)
    data_dict['dists_spine_i_mechs'].append(dists_spine_i_mechs)
    data_dict['dendrites_i_mechs'].append(dends_i_mechs)
    data_dict['dendrites_spine_i_mechs'].append(dends_spine_i_mechs)
    data_dict['record_dists'].append(record_dist)
    data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
    data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
    data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
    
    if impedance:
        data_dict['z_input'].append(z_input[0])
        data_dict['z_transfer'].append(z_transfer[0])
        
    if return_currents:
        data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
        data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
        data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
        data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()

    data_dict['timestamp'][protocol] = time
    
def plot_cumulative_frequency(cell_type, specs, spine_per_length=1.61, frequency=2000, d_lambda=0.05, dend2remove=None):
    """
    Plots the cumulative frequency of spine distances from the soma.

    Parameters:
    - cell_type: the type of cell
    - specs: cell specifications
    - spines: spine data

    Returns:
    - A plotly figure displaying the cumulative frequency
    """

    import plotly.graph_objects as go

    # Build the cell
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove)

    # Compute distances of each spine from the soma
    dist_spine = []
    for dend in cell.dendlist:
        sec_spines = list(spines[dend.name()].items())
        for spine in sec_spines:
            dist_spine.append(h.distance(dend(spine[1].x)))

    # Sort the data in ascending order
    sorted_data = sorted(dist_spine)

    # Calculate cumulative frequencies
    cumulative_freq = [i / len(sorted_data) for i in range(1, len(sorted_data) + 1)]

    # Create the cumulative frequency plot
    fig = go.Figure(data=go.Scatter(x=sorted_data, y=cumulative_freq, mode='lines'))

    # Set plot labels and title
    fig.update_layout(
        title={
            'text': 'cumulative frequency',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='distance (um)',
        yaxis_title='cumulative frequency',
        xaxis_range=[0, 300],  # Set x-axis range
    )

    # Display the plot
    fig.show()
    
def plot_spine_distance_histogram(cell_type, specs,  spine_per_length=1.61, frequency=2000, d_lambda=0.05, bin_size=10, dend2remove=None):
    """
    Plots the histogram of spine distances from the soma.

    Parameters:
    - cell_type: the type of cell
    - specs: cell specifications
    - bin_size: size of the bins for the histogram

    Returns:
    - A plotly figure displaying the histogram
    """

    import plotly.graph_objects as go

    # Build the cell
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove)

    # Compute distances of each spine from the soma
    dist_spine = []
    for dend in cell.dendlist:
        sec_spines = list(spines[dend.name()].items())
        for spine in sec_spines:
            dist_spine.append(h.distance(dend(spine[1].x)))

    # Create the histogram plot
    fig = go.Figure(data=go.Histogram(x=dist_spine, nbinsx=int(max(dist_spine) / bin_size)))

    # Set plot labels and title
    fig.update_layout(
        title={
            'text': 'Histogram of Spine Distances from Soma',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='distance (um)',
        yaxis_title='frequency',
        xaxis_range=[0, 300],  # Set x-axis range
    )

    # Display the plot
    fig.show()

def filter_master(v_master, filter_array):
    return [scatter for scatter, to_keep in zip(v_master, filter_array) if to_keep]

def plt3(data_dict, metadata, sim_time, dt, model, cell_type, stim_time, showplot, save, physiological, sim, offset=40, gaba=True, glut=True
    ):
    
    time = np.arange(0, len(data_dict['vsoma'][0]) * dt, dt)
    
    idx1, idx2, idx3  = 0, int(sim_time/dt), int(stim_time/dt)
    soma_v_master, dend_v_master, spine_v_master = [], [], []
    peaks_v_soma, peaks_v_dend, peaks_v_spine = [], [], []
    mins_v_soma, mins_v_dend, mins_v_spine = [], [], []
    x=time[idx1:idx2]
    
    for ii, (soma_v, dend_v, spine_v) in enumerate(zip(data_dict['vsoma'], data_dict['vdend'], data_dict['vspine'])):
        y=extract2(soma_v)[idx1:idx2]
        soma_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_soma.append(max(soma_v))
        mins_v_soma.append(soma_v[idx3])
        
        y=extract2(dend_v)[idx1:idx2]
        dend_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_dend.append(max(dend_v))
        mins_v_dend.append(dend_v[idx3])
        
        y=extract2(spine_v)[idx1:idx2]
        spine_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_spine.append(max(spine_v))
        mins_v_spine.append(spine_v[idx3])
    ysoma_range = [0.1*math.floor(min(mins_v_soma)/0.1), 0.1*math.ceil(max(peaks_v_soma)/0.1)]
    ydend_range = [math.floor(min(mins_v_dend)), math.ceil(max(peaks_v_dend))]
    yspine_range = [math.floor(min(mins_v_spine)), math.ceil(max(peaks_v_spine))]

    gaba_range = metadata['gaba_range']
    glut_range = metadata['glut_range']

    if gaba and glut:
        f = [f1 and f2 for f1, f2 in zip(gaba_range, glut_range)] # glut and gaba
    elif gaba and not glut:
        f = [f1 and not f2 for f1, f2 in zip(gaba_range, glut_range)] # gaba only
    elif glut and not gaba:
        f = [not f1 and f2 for f1, f2 in zip(gaba_range, glut_range)] # glut only
     
    spine_v_master = filter_master(spine_v_master, f)
    dend_v_master = filter_master(dend_v_master, f)
    soma_v_master = filter_master(soma_v_master, f)

    figs = plot3a(somaV=soma_v_master, dendV=dend_v_master, spineV=spine_v_master, ysoma_range=ysoma_range, 
                  ydend_range=ydend_range, yspine_range=yspine_range, stim_time=stim_time, sim_time=sim_time, 
                  dt=dt, offset=offset, controls=False)
    
    return figs

def plt4(title, figs1, figs2, figs3, index, sim, time, show_subtitles=False, showplot=True, save=True, cell_type='dspn', physiological=True, model=1):
    fig = make_subplots(rows=1, cols=3)
    fig1 = figs1[index]
    fig2 = figs2[index]
    fig3 = figs3[index]

    shapes = copy.deepcopy(fig1.layout.shapes)
    yaxis_range = [round(val, 1) for val in [shapes[0]['y0'], shapes[1]['y0']]]

    annotations = []
    subplot_titles = ['glut + gaba', 'glut only', 'gaba only']

    for col, subplot in enumerate([fig1, fig2, fig3], start=1):
        for trace in subplot.data:
            trace_copy = copy.deepcopy(trace)
            color = trace.line.color
            trace_copy.line.color = color
            fig.add_trace(trace_copy, row=1, col=col)

        fig.update_yaxes(
            range=yaxis_range,
            row=1,
            col=col,
            tickvals=yaxis_range,
            ticktext=[str(val) for val in yaxis_range] if col == 1 else [],
            showticklabels=(col == 1)
        )
        fig.update_xaxes(showticklabels=False, ticks="", row=1, col=col)

        if show_subtitles:
            annotations.append(dict(
                    xref='paper',  # This will refer to the entire width of the plotting area
                    yref='paper',
                    x = col / 2.5 - 0.3 ,  # Manually adjust to line up with your subplots
                    y=1.07,
                    text=subplot_titles[col - 1],
                    showarrow=False,
                    font=dict(size=12),
                ))

    fig.update_layout(
        shapes=shapes,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        annotations=annotations if show_subtitles else []
    )
    
    if showplot:
        fig.show()
              
    if save:
        path_format = f'{cell_type}/model{model}/{{}}/images/sim{sim}'
        folder = path_format.format('physiological' if physiological else 'nonphysiological')
        os.makedirs(folder, exist_ok=True)
        fig.write_image(f'{folder}/fig1 {title} {time}.svg')
        fig.write_html(f'{folder}/fig1 {title} {time}.html')       
        
# returns dists, dends and locs for all unique locations on a dendritic tree with dendrite being the most proximal
def path_secs(cell, dend_tree, dendrite):
    dists = []
    dends = []
    locs = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
    # get path to soma
    if dendrite.name() != 'soma[0]':
        pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    else:
        pathlist = [dendrite]
    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        for seg in sec:
            dist = h.distance(seg)
            dists.append(dist)
            loc = seg.x
            dends.append(sec.name())
            locs.append(loc)
            i = i + 1
    return dists[::-1], dends[::-1], locs[::-1]

# input comprises a list of dend names and their equivalent locations given in locs returns 
# returns record vectors for v in NEURON
def record_all_path(cell, dends, locs):
    
    pathlist = []
    for dend in dends:
        for sec in cell.allseclist:
            if sec.name() == dend:
                pathlist.append(sec)
        
    all_v = {}
    i=0
    for sec,loc in zip(pathlist,locs) :
        all_v[i] = h.Vector()
        all_v[i].record(sec(loc)._ref_v) # given a sec with multiple seg
        i = i + 1
    
    return all_v

# gives local membrane potential for dendds with their equivalenbt locations
def membrane_potentials(cell, 
                  dends, 
                  locs, 
                  sim_time=150, 
                  dt=0.025,
                  v_init=-85
                  ):
    t = h.Vector().record(h._ref_t)
    iclamp1 = h.IClamp(cell.soma(0.5))
    iclamp1.dur = sim_time

    v_path = record_all_path(cell=cell, dends=dends, locs=locs)

    # Initialize cell starting voltage
    h.finitialize(v_init)
    # Run simulation
    h.dt = dt

    while h.t < sim_time:
        h.fadvance()

    v_tree = []
    for ii in list(range(len(v_path))):
        v_tree.append(np.array(v_path[ii]))
        
    all_revs = []
    for v in v_tree:
        all_revs.append(v[int(sim_time/dt)])
        
    return all_revs
                    
def pairs_in_order(dend, reversals):
    seen = set()
    unique_pairs = []
    for d, rev in zip(dend, reversals):
        pair = (d, rev)  # No rounding here
        if pair not in seen:
            unique_pairs.append((d, rev))
            seen.add(pair)
    return unique_pairs

# function gets all unique locations on path to soma
def record_all_path_secs_i_mechs(cell, dend_tree, dendrite, mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']):
    all_i_mechs = {}
    dists = []
    dends = []
    
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
    # get path to soma
    if dendrite.name() != 'soma[0]':
        pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    else:
        pathlist = [dendrite]
    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        for seg in sec:
            dends.append(sec.name())
            dist = h.distance(seg)
            dists.append(dist)
            loc = seg.x
            all_i_mechs[i] = record_i_mechs(cell=cell, dend=sec, loc=loc, return_currents=True, silent=True, mechs=mechs)
            i = i + 1
    return all_i_mechs, dists, dends

# records from all activated spine locations
def record_all_activated_spine_v2(cell, dendrite, activated_spines):
    all_v_spines= {}
    all_v_dendrite = {}
    dists = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend

    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for spine in activated_spines:
        all_v_spines[i] = h.Vector()
        all_v_spines[i].record(spine.head(0.5)._ref_v)
        all_v_dendrite[i] = h.Vector()
        all_v_dendrite[i].record(dendrite(spine.x)._ref_v) 
        dists.append(h.distance(dendrite(spine.x)))
        i = i + 1

    return all_v_spines, all_v_dendrite, dists

# finds all unique spine locations on path to soma
# ignores spine if active
# records voltage changes in that spine to postsynaptic potential at a remote site
def record_all_path_secs_spine_v2(cell, spines, dend_tree, dendrite, activated_spines):
    all_v = {}
    dists = []
    dends = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
        # get path to soma
        if dendrite.name() != 'soma[0]':
            pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
            # Remove soma[0] from pathlist if it exists
            pathlist = [sec for sec in pathlist if sec.name() != 'soma[0]']
        else:
            pathlist = [dendrite]  

    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        sec_spines = list(spines[sec.name()].items())
        sec_spines_x = []
        for spine_i, spine_obj in sec_spines: 
            sec_spines_x.append(spine_obj.x)
        if sec_spines_x:
            for seg in sec:
                dends.append(sec.name())
                dist = h.distance(seg)
                dists.append(dist)
                loc = seg.x
                # Find the index of the value in sec_spines_x closest to loc
                idx = min(range(len(sec_spines_x)), key=lambda i: abs(sec_spines_x[i] - loc))
                spine_id, spine = sec_spines[idx]
                while spine in activated_spines:
                    idx += 1
                    spine_id, spine = sec_spines[idx]
                all_v[i] = h.Vector()
                all_v[i].record(spine.head(0.5)._ref_v) # given a sec with multiple seg
                i = i + 1

    return all_v, dists, dends

# finds all unique spine locations on path to soma
# ignores spine if active
# records current mechanisms specified in that spine to synaptic event at a remote site
def record_all_path_secs_spine_i_mechs(cell, spines, dend_tree, dendrite, activated_spines, spine_mechs=['pas', 'kir', 'cal12', 'cal13', 'car', 'cav32', 'cav33', 'sk']):
    all_i_mechs = {}
    dists = []
    dends = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
        # get path to soma
        if dendrite.name() != 'soma[0]':
            pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
            # Remove soma[0] from pathlist if it exists
            pathlist = [sec for sec in pathlist if sec.name() != 'soma[0]']
        else:
            pathlist = [dendrite]  

    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        sec_spines = list(spines[sec.name()].items())
        sec_spines_x = []
        for spine_i, spine_obj in sec_spines: 
            sec_spines_x.append(spine_obj.x)
        if sec_spines_x:
            for seg in sec:
                dends.append(sec.name())
                dist = h.distance(seg)
                dists.append(dist)
                loc = seg.x
                # Find the index of the value in sec_spines_x closest to loc
                idx = min(range(len(sec_spines_x)), key=lambda i: abs(sec_spines_x[i] - loc))
                spine_id, spine = sec_spines[idx]
                while spine in activated_spines:
                    idx += 1
                    spine_id, spine = sec_spines[idx]
                # for spines loc = 0.5 to record in the middle of the spine head
                all_i_mechs[i] = record_i_mechs(cell=cell, dend=spine.head, loc=0.5, return_currents=True, silent=True,
                    mechs=spine_mechs)
                i = i + 1

    return all_i_mechs, dists, dends

def extract_column(dict_df, column_name):
    """
    Extracts a specified column from each DataFrame in a dictionary.

    Parameters:
    dataframes (dict): A dictionary of DataFrames.
    column_name (str): The name of the column to extract.

    Returns:
    list: A list containing the extracted column from each DataFrame.
    """
    extracted_columns = []
    for key, df in dict_df.items():
        extracted_columns.append(df[column_name].values)
    return extracted_columns

# Iterate through each Vector in 'v' and convert to NumPy array
def vec2np(V):
    out = []
    # check if 'V' is a dictionary
    if isinstance(V, dict):
        # if 'V' is a dictionary, iterate over its values
        for vector in V.values():
            np_array = np.array(vector)
            out.append(np_array)
    # check if 'V' is a list
    elif isinstance(V, list):
        # If 'V' is a list, iterate directly over it
        for vector in V:
            np_array = np.array(vector)
            out.append(np_array)
    return out

def interpolate_3d(sec, seg_x):
    '''
    This function, interpolate_3d, interpolates the 3D coordinates and diameter for the center of a segment. 
    Use within record_all_3D to get the interpolated coordinates and diameter for each segment in cell.dendlist 
    and record the voltage at each segment's center.
    
    '''# Number of 3D points in the section
    n3d = int(h.n3d(sec=sec))
    
    # Segment's relative position along the section's total length
    seg_pos = seg_x * sec.L
    
    # Initialize variables to hold the interpolated values
    x, y, z, diam = None, None, None, None
    
    # Length along the section up to the current 3D point
    length = 0
    
    for i in range(n3d - 1):
        # Get the 3D coordinates and diameter of the current and next point
        x0, y0, z0, diam0 = h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec), h.diam3d(i, sec=sec)
        x1, y1, z1, diam1 = h.x3d(i + 1, sec=sec), h.y3d(i + 1, sec=sec), h.z3d(i + 1, sec=sec), h.diam3d(i + 1, sec=sec)
        
        # Calculate the distance between these two points
        point_dist = ((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)**0.5
        
        if length + point_dist >= seg_pos:
            # Segment's center falls between these two points, interpolate
            ratio = (seg_pos - length) / point_dist
            x = x0 + ratio * (x1 - x0)
            y = y0 + ratio * (y1 - y0)
            z = z0 + ratio * (z1 - z0)
            diam = diam0 + ratio * (diam1 - diam0)
            break
        
        length += point_dist
    
    return x, y, z, diam

def record_all_3D(cell):
    all_v = [] 
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Access the middle segment of the soma
        v_vec = h.Vector()
        v_vec.record(seg._ref_v)
        all_v.append(v_vec)

        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())  # You can distinguish soma from dendrites here if needed
        dists.append(h.distance(0.5, sec=sec))

    for sec in cell.dendlist:
        h('access ' + sec.name())
        sec.nseg = int(h.n3d())

        for seg in sec:
            v_vec = h.Vector()
            v_vec.record(seg._ref_v)
            all_v.append(v_vec)

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return all_v, cell_coordinates, dends, dists

def record_mechs_3D(cell, mechs):
    all_i_mechs = {}
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        out = record_i_mechs(cell=cell, dend=sec, loc=0.5, return_currents=True, silent=True, mechs=mechs)
        all_i_mechs[0] = out[1:]  # drops first vector (time)
        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())
        dists.append(h.distance(0.5, sec=sec))

    i = 1
    for sec in cell.dendlist:
        h('access ' + sec.name())
        sec.nseg = int(h.n3d())  # Ensure this is necessary and correct

        for seg in sec:
            loc = seg.x
            out = record_i_mechs(cell=cell, dend=sec, loc=loc, return_currents=True, silent=True, mechs=mechs)  # Update 'out' within the loop
            all_i_mechs[i] = out[1:]  # Append excluding the first element
            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
            i = i+1
    
    return all_i_mechs, cell_coordinates, dends, dists

def record_Ca_3D(cell):
    all_Ca = [] 
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Access the middle segment of the soma
        Ca_vec = h.Vector()
        Ca_vec.record(seg._ref_cai)
        all_Ca.append(Ca_vec)

        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())  # You can distinguish soma from dendrites here if needed
        dists.append(h.distance(0.5, sec=sec))

    for sec in cell.dendlist:
        h('access ' + sec.name())
        sec.nseg = int(h.n3d())

        for seg in sec:
            Ca_vec = h.Vector()
            Ca_vec.record(seg._ref_cai)
            all_Ca.append(Ca_vec)

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return all_Ca, cell_coordinates, dends, dists

def record_mechs_distr_3D(cell, mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']):
    
    dist_mechs_out = []
    cell_coordinates = []
    dists = []
    dends = []

    mech_refs = {
        'pas': 'g_pas',
        'kdr': 'gbar_kdr',
        'naf': 'gbar_naf',
        'kaf': 'gbar_kaf',
        'kas': 'gbar_kas',
        'kir': 'gbar_kir',
        'cal12': 'pbar_cal12',
        'cal13': 'pbar_cal13',
        'can': 'pbar_can',
        'car': 'pbar_car',
        'cav32': 'pbar_cav32',
        'cav33': 'pbar_cav33',
        'sk': 'gbar_sk',
        'bk': 'gbar_bk'
    }

    # Results dictionary to hold conductance/permeability values
    results = {}

    for sec in cell.somalist:
        h('access ' + sec.name())
        out = []
        for mech in mechs:
            ref_attr = mech_refs.get(mech)
            if ref_attr and hasattr(sec(0.5), ref_attr):
                # Record the mechanism if it exists
                out.append(getattr(sec(0.5), ref_attr))
            else:
                print(f"warning: mechanism '{mech}' not recognized or not present at the specified location")
        dist_mechs_out.append(out)
        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())
        dists.append(h.distance(0.5, sec=sec))

    i = 1
    for sec in cell.dendlist:
        h('access ' + sec.name())
        sec.nseg = int(h.n3d())  

        for seg in sec:
            loc = seg.x
            out = []
            for mech in mechs:
                ref_attr = mech_refs.get(mech)
                if ref_attr and hasattr(sec(loc), ref_attr):
                    # Record the mechanism if it exists
                    out.append(getattr(sec(loc), ref_attr))
                else:
                    print(f"warning: mechanism '{mech}' not recognized or not present at the specified location")
            dist_mechs_out.append(out)
            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))

        i = i+1
    
    df1 = pd.DataFrame(dist_mechs_out, columns=mechs)
    df2 = pd.DataFrame(cell_coordinates, columns=['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'])
        
    output = {
        'distributions': df1,
        'cell_coordinates': df2,
        'dists': dists,
        'dendrites': dends
    }          
     

    return output

def setup_impedance_measurements(cell):
    impedance_locations = []
    cell_coordinates = []
    dists = []
    dends = []

    # Setup for somatic sections
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Middle segment of the soma

        # Store information for impedance measurement
        impedance_locations.append((sec, seg.x))

        x, y, z, diam = interpolate_3d(sec, seg.x)
        cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
        dends.append(sec.name())  # Soma could be distinguished here if needed
        dists.append(h.distance(seg.x, sec=sec))

    # Setup for dendritic sections
    for sec in cell.dendlist:
        h('access ' + sec.name())
        sec.nseg = int(h.n3d())  # Ensure there's a segment at each 3D point

        for seg in sec:
            # Store information for impedance measurement
            impedance_locations.append((sec, seg.x))

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return impedance_locations, cell_coordinates, dends, dists

# routine to calculate spine_per_length_dict values below
def solve_spine_per_length(target_total_spines, cell_type, specs, frequency=2000, d_lambda=0.05, initial_tol=5, max_iterations=100, iteration_threshold=20, dend2remove=None):
    lower_bound = 0
    upper_bound = 4  # This might need to be adjusted based on your specific use case
    tolerance = initial_tol
    spine_per_length = (upper_bound + lower_bound) / 2
    iterations = 0
    iteration_since_last_adjustment = 0

    while iterations < max_iterations:
        _, spines, _ = cell_build(cell_type, specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=dend2remove)
        current_total_spines = sum(len(spine_list) for spine_list in spines.values())

        # Check if the current total is within the tolerance of the target
        if abs(current_total_spines - target_total_spines) <= tolerance:
            return spine_per_length  # Return the found spine_per_length value

        # Adjust spine_per_length based on whether the current total is less than or greater than the target
        elif current_total_spines < target_total_spines:
            lower_bound = spine_per_length
        else:
            upper_bound = spine_per_length

        spine_per_length = (upper_bound + lower_bound) / 2
        iterations += 1
        iteration_since_last_adjustment += 1

        # Increase tolerance if no solution is found within iteration_threshold
        if iteration_since_last_adjustment >= iteration_threshold:
            tolerance *= 2  # Double the tolerance
            iteration_since_last_adjustment = 0  # Reset the counter for adjustments

    # If the function reaches this point, it means no solution was found within max_iterations
    # You might want to return a special value or raise an exception here
    return None

def update_data_dictionary(data_dict, protocol, v_all_3D, Ca_all_3D, imp_all_3D, i_mechs_3D, vspine, v_spine_activated, vdend, v_dend_activated, vsoma, v_dend_tree, v_spine_tree, Ca_spine, Ca_dend, Ca_soma, timing, i_mechs_dend, i_mechs_dend_tree, i_mechs_spine_tree, v_branch, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, record_dist, time, record_currents=False, record_spine=False, spine_dist=None, cap=None):
    
    data_dict['v_all_3D'][protocol] = v_all_3D
    data_dict['Ca_all_3D'][protocol] = Ca_all_3D
    data_dict['imp_all_3D'][protocol] = imp_all_3D
    data_dict['i_mechs_3D'][protocol] = i_mechs_3D
    data_dict['v_dend_tree'][protocol] = v_dend_tree
    data_dict['v_spine_tree'][protocol] = v_spine_tree
    data_dict['v_branch'][protocol] = v_branch
    data_dict['vsoma'].append(vsoma)
    data_dict['vdend'].append(vdend)
    data_dict['v_dend_activated'][protocol] = v_dend_activated
    data_dict['vspine'].append(vspine)
    data_dict['v_spine_activated'][protocol] = v_spine_activated
    
    data_dict['Ca_soma'].append(Ca_soma)
    data_dict['Ca_dend'].append(Ca_dend)
    data_dict['Ca_spine'].append(Ca_spine)
    
    data_dict['timing'].append(timing)

    data_dict['i_mechs_dend'][protocol] = i_mechs_dend
    data_dict['i_mechs_dend_tree'][protocol] = i_mechs_dend_tree
    data_dict['i_mechs_spine_tree'][protocol] = i_mechs_spine_tree
    
    data_dict['record_dists'].append(record_dist)
    data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
    data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
    data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
    
    data_dict['spine_dist'].append(spine_dist)
    data_dict['capacitance'].append(cap)
        
    if record_currents:
        data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
        data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
        data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
        data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()

    data_dict['timestamp'][protocol] = time

def plots_return(v_tree, vspine, dists, spine_dist, num_gluts, start_time=150, burn_time=140, dt=0.025, xaxis_range=[0,100], Nsim_plot=False, Nsim_save=False, sim_image_path=None, time=None, width=1000, height=600):
    # only do if want to view each sim or save sim graphs
    if Nsim_plot or Nsim_save:
       # normalise tree data
        norm_v_all = []
        norm_v_tree = {}
        X = v_tree
        dists1 = dists
        for v in v_tree:
            norm_v_all.append( normalise(v, start_time, burn_time, dt) )
        if num_gluts == 1:
            norm_v_all.append(normalise(vspine, start_time, burn_time, dt)  )
            dists1.append(spine_dist)

        # find peak values
        peak_v = []
        for v in norm_v_all:
            peak_v.append(v.max())

        # isolate peak
        norm_peak_v = []
        for v in peak_v:
            norm_peak_v.append(v/max(peak_v))

        # only do if want to view each sim or save sim graphs
        if Nsim_plot or Nsim_save:
            # plot normalised potentials
            fig1 = plot5(X=norm_v_all, dt=dt, dists=dists1, xaxis_range=xaxis_range, yaxis_range=[-0.2, math.ceil(max(peak_v))], normalised=False)
            fig2 = plot6(y=norm_peak_v, x=dists1, xaxis_range=[300,0], yaxis_range=[0,1.01])

            fig1.update_layout(
                width=width,  
                height=height
            )
            fig2.update_layout(
                width=width,  
                height=height
            )

            if Nsim_plot:
                fig1.show()
                fig2.show()

            if Nsim_save:
                for fig, fig_name in [(fig1, 'fig1'), (fig2, 'fig2')]:
                    fig.write_html('{}/{}_{}.html'.format(sim_image_path, fig_name, time))

def setup_impedance_measurements(cell):
    impedance_locations = []
    cell_coordinates = []
    dists = []
    dends = []

    # Setup for somatic sections
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Middle segment of the soma

        # Store information for impedance measurement
        impedance_locations.append((sec, seg.x))

        x, y, z, diam = interpolate_3d(sec, seg.x)
        cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
        dends.append(sec.name())  # Soma could be distinguished here if needed
        dists.append(h.distance(seg.x, sec=sec))

    # Setup for dendritic sections
    for sec in cell.dendlist:
        h('access ' + sec.name())
        sec.nseg = int(h.n3d())  # Ensure there's a segment at each 3D point

        for seg in sec:
            # Store information for impedance measurement
            impedance_locations.append((sec, seg.x))

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return impedance_locations, cell_coordinates, dends, dists

def all_3D(cell):
    seg_x = [] 
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Access the middle segment of the soma
        seg_x.append(seg.x)

        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())  # You can distinguish soma from dendrites here if needed
        dists.append(h.distance(0.5, sec=sec))

    for sec in cell.dendlist:
        h('access ' + sec.name())
        sec.nseg = int(h.n3d())

        for seg in sec:
            seg_x.append(seg.x)
            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return seg_x, cell_coordinates, dends, dists
    
def get_(sim, _dict):
    letter = sim[-1]
    key_pattern = f'{letter}'
    return _dict.get(key_pattern, "Default Value")

def compare_last_digit(string, integer):
    """
    Compare the last digit in a string with an integer.

    :param string: The string to check the last digit of.
    :param integer: The integer to compare against the last digit of the string.
    :return: True if the last digit of the string matches the integer, False otherwise.
    """
    # Extract the last digit from the string
    last_digit_str = ''.join(filter(str.isdigit, string))[-1] if string else ''
    
    # Convert the last digit to an integer, if possible
    try:
        last_digit = int(last_digit_str)
    except ValueError:
        return False
    
    return last_digit == integer