## Repository for `NEURON` Model of Striatal Projection Neurons

This repository contains a `NEURON` + `Python` model of striatal projection neurons (or SPNs) designed to simulate the interaction between GABAergic and glutamatergic synaptic inputs. 

It also provides all the `R` code used to produce the graph output (as `svg`) from the resultant `NEURON` + `Python` output. 

The `NEURON` + `Python` model was built on top of the `striatal_SPN_lib` repository created by Lindroos and Kotaleski, 2020:

[Lindroos R, Kotaleski JH. Predicting complex spikes in striatal projection neurons of the direct pathway following neuromodulation by acetylcholine and dopamine. Eur J Neurosci. 2020](https://doi.org/10.1111/ejn.14891)

Their original model can be found here: [`modelDB`](https://modeldb.science/266775) or [`GitHub`](https://github.com/ModelDBRepository/266775)

A version of this modified code has been published by Day et al., 2024:

[Day M., Belal M., Surmeier W. C., Melendez A., Wokosin, D., Tkatch T., Clarke V. R. J. and Surmeier D. J.  GABAergic regulation of striatal spiny projection neurons depends upon their activity state (2024). PLoS Biol 22(1): e3002483](https://doi.org/10.1371/journal.pbio.3002483)

The above model can be found here: [`Zenodo`](https://doi.org/10.5281/zenodo.10162265) or [`GitHub`](https://github.com/vernonclarke/SPNfinal/releases/tag/v1.0)

The simulation provided in this repository was used to create Figure 6 of this manuscript:

**Zhai S., Otsuka S., Xu J., Clarke V. R. J., Tkatch T., Wokosin D., Xie Z., Tanimura A., Do H. T., Reidl C. T., Agarwal H. K., Ellis-Davies G. C. R., Silverman R. B., Contractor A. and Surmeier D. J.  Ca<sup>2+</sup>-dependent phosphodiesterase 1 regulates the plasticity of striatal spiny projection neuron glutamatergic synapses**

This model can also be found at [`Zenodo`](https://doi.org/10.5281/zenodo.12213216).

## Table of Contents
- [Initial Set Up](#Initial-Set-Up)
- [Running the Models](#running-the-models)
  - [Getting Started](#getting-started)
  - [Simulations](#running-simulations-in-jupyter-notebook)
- [Data Analysis](#data-analysis)
  - [Setting Up `R`](#setting-up-r)
  - [Analysis Using `R`](#using-r-to-analyse-a-simulation)
  - [Analysis Using `Jupyter Notebook`](#using-jupyter-notebook-to-analyse-a-simulation)
- [`Anaconda` vs `Miniconda`](#anaconda-vs-miniconda)
- [Virtual Environments](#virtual-environments)
- [`GitHub`](#github)
- [References](#references)
- [Contact](#contact)

## Initial Set Up

### Prerequisites
- [`Conda`](https://docs.conda.io/projects/conda/en/stable) official website
- [`NEURON`](https://www.neuron.yale.edu) official website (tested on versions 8.1 and 8.2) 

[Hines ML, Carnevale NT. The NEURON Simulation Environment. Neural Comput. 1997](https://doi.org/10.1162/neco.1997.9.6.1179)

- `Python` (tested using version 3.9.16)

### Steps
1. **Install [`Conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)** (Python package manager)
   
   `Conda` should include a version of `Python`. 

   `Conda` is simply a package manager and environment management system that is used to install, run and update packages and their dependencies.

   Whether to install [`Anaconda` or `Miniconda`](#anaconda-vs-miniconda) largely depends on personal preferences.

3. **Install [`Jupyter Notebook`](https://jupyter.org)**

   The simplest method to install `Jupyter Notebook` is via `Conda` using the command in `terminal`:

   ```bash
   conda install -c conda-forge notebook
   ```

4. **Install [`NEURON`](https://www.neuron.yale.edu)**

  Follow the guide at [`NEURON`](https://www.neuron.yale.edu/neuron/download) installation

* [Quickstart](https://www.neuron.yale.edu/ftp/neuron/2019umn/neuron-quickstart.pdf)

* [(Old) Mac instructions](https://www.neuron.yale.edu/ftp/neuron/nrn_mac_install.pdf)
  
## Running the Models

The following sections explain the initial set up required and instructions to create simulations subsequently used to generate figures.

### Getting Started

1. **Install `NEURON` with `Python` support** (see setup instructions)

2. **Open `terminal`**:
   - On MacOS: Press `cmd + space` to open spotlight search and type `terminal`.
   - On Linux: Search for `terminal` in your applications menu or press `ctrl + alt + T`.
   - On Windows: Search for `command prompt` or `PowerShell` in the start menu.

3. **Compile mechanisms**:

   Navigate to directory containing `NEURON` mechanisms.

   For instance if `SPNcell` is in documents folder in MacOS, mechanisms are located in `cd/documents/SPNcell/mechanisms/single`.

   Mechanisms are then compiled by entering `nrnivmodl` or `mknrndll`.

   On MacOS / Linux:
   ```bash
   cd documents/Repositories/SPNcell/mechanisms/single
   nrnivmodl
   ```

   On Windows (replace `YourUsername` with appropriate user name):
   ```bash
   cd C:\Users\YourUsername\documents\Repositories\SPNcell\mechanisms\single
   nrnivmodl
   ```

4. **Create a conda environment**

   There is a `YAML` file in the main directory called `environment.yml` for MacOS/Linux. This can be used to create a conda environment called `neuron` (not to be confused with the simulation enivronment `NEURON`). For further information see [Virtual Environments](#virtual-environments).

   Ensure make sure to navigate back to the main directory after step 3 above.

   Check installed correctly using `conda list`.

   On MacOS / Linux:
   ```bash
   cd ../.. 
   conda env create -f environment.yml
   conda list
   ```

   I have included a no_builds version `environment_no_builds.yml`. I had to use this `*.yml` on an older `Macbook Pro`. In addition, I also had to use older versions of two packages (`_pytorch_select=0.1` not `_pytorch_select=0.2` and `torch==2.0.1` not `torch==2.3.0`) in order to create the `neuron` environment without errors.

   On Windows:
   ```bash
   cd ..\.. 
   conda env create -f environment_pc.yml
   conda list
   ```
  
   Creating the environment on Windows is slightly different. `NEURON` cannot be installed via the `terminal` using `pip install neuron`. 
  
   Instead, `NEURON` must be installed via a downloaded setup.exe. 
  
   A separate `environment_pc.yml` is provided for Windows. Limited testing on a Windows laptop showed the simulations working but suggested that, despite reasonable specs, `Intel(R) Core(TM) i5-8350U CPU @ 1.7 GHz 1.9 GHz 32GB`, the code ran extremely slowly in the Windows environment (~ 9-fold slower) as compared to a `MacBook M2 pro 32GB`. In fact, it was slower (~ 4-fold) than a 2015 `MacBook Pro 2.7 GHz Dual-Core Intel core i5`.    
     
   
5. **Quit `terminal`**

   ```bash
   exit
   ```
   
### Running simulations in `Jupyter Notebook`

  The following steps 1-4 must be performed every time a new `Jupyter Notebook` session is started.

1. **Open `terminal`**:
   - On MacOS: Press `cmd + space` to open spotlight search and type `terminal`.
   - On Linux: Search for `terminal` in your applications menu or press `ctrl + alt + T`.
   - On Windows: Search for `command prompt`, `PowerShell` or the appropriate `Miniconda/Anaconda prompt` in the start menu.

2. **Activate conda environment `neuron`**

   Navigate back to the main directory

   MacOS / Linux:
   ```bash
   cd documents/Repositories/SPNcell
   conda activate neuron
   ```

   Windows (replace `YourUsername` with appropriate user name):
   ```bash
   cd C:\Users\YourUsername\Repositories\documents\SPNcell
   conda activate neuron
   ```
3. **Run `Jupyter Notebook`**

   Add `neuron` environment then open `Jupyter Notebook`
   ```bash
   python -m ipykernel install --user --name neuron --display-name "Python (neuron)"
   jupyter notebook
   ```

4. **Run a simulation**

   `Jupyter Notebook` should now be open in the default browser.

   Choose a notebook to open (by clicking on any notebook - `*.ipynb`).
   
   Kernel should already be set to `Python (neuron)`.

   From kernel dropdown menu choose `Restart Run All` (if running again then it's good practice to run `Restart and Clear Output` first).

   Code should run and generate raw data used to generate figures.

   If option `save = True` in the notebook then the raw figures and pickled data is stored in a subdirectory within the main one.

## Data Analysis

The final analysis and figures presented in the manuscript were generated using `R`. 

I have provided both the `R` and equivalent `Python` code to reproduce the figures in the manuscript

The analyses were conducted using `R` version 4.3.1 – `Beagle Scouts` which can be download [here](https://www.R-project.org/). 

In addition, the code was tested on the latest version of `R` version 4.4.1 – `Race for Your Life`.

Please refer to the `R analysis` directory for the code.

The equivalent `Python` analyses can be performed using provided `Jupyter Notebook` entitled `Figure 6 analysis.ipynb` which uses the underlying `Python` functions located in `analysis_functions.py`.

  ### Setting up R
  
  Only the `R console` was used for analysis. 
  
  If you prefer to work with `RStudio`, it can be downloaded [here](https://posit.co/products/open-source/rstudio/). The provided code should work although this has not been tested.
  
  In order for the `R` code to work, it is necessary to load various packages within the `R` environment.
  
  The following code should be executed in `R` prior to running any of this code. It checks if the required packages are present and, if they are not, it will install them.
  
  In addition, the first time the code is executed, it will install a `Miniconda` environment using the `reticulate` package in `R`: 

  The following steps 1-3 must be performed once.
  
  1. **Open `R` gui**
  2. **Run this code once**
  ```R
  rm( list=ls(all=TRUE ) )
  # load and install necessary packages
  load_required_packages <- function(packages) {
    new.packages <- packages[!(packages %in% installed.packages()[, 'Package'])]
    if (length(new.packages)) install.packages(new.packages)
    invisible(lapply(packages, library, character.only = TRUE))
  }

  required.packages <- c('reticulate', 'stringr', 'zoo', 'pracma')
  load_required_packages(required.packages)
  
  # install miniconda if necessary
  if (!reticulate::py_available(initialize = TRUE)) {
      # Load the reticulate library
      library(reticulate)

      # install miniconda through reticulate (if miniconda is not already installed)
      install_miniconda()
    
      # create and activate a new conda environment
      conda_create(envname = "myenv")
      use_condaenv("myenv", required = TRUE)
  
      # install pandas in the new environment
      py_install("pandas", envname = "myenv")
  }
  ```
  3. **Exit `R` session**

  **Once this code is run, it will perform any necessary installations.**
  
  A useful guide can be found [here](https://rstudio.github.io/reticulate/).
  
  ### Using `R` to analyse a simulation
  
  To run the analysis code, simply execute all the code for the relevant `R analysis` file.
   
  When the `Jupyter Notebook` is executed with `save = True`, the outputs are stored automatically in a subfolder using the relevant identifier (in this case `1100g`). 
  
  In this case, raw trace data is stored as pickled files in the subdirectory `dspn/model1/physiological/simulations/sim1100g`. 
  
  Any images generated are found in `dspn/model1/physiological/images/sim1100g`. 
  
  The `R` code to analyse the output from `Figure 6 sim.ipynb` is found in `Figure 6.R` in the `R analysis` directory. 
  
  1. **Open `R` gui**
  2. **Open `Figure 6.R` and run the code**
```R
# remove all objects from the environment
rm(list = ls(all = TRUE))

# load and install necessary packages
load_required_packages <- function(packages) {
  new.packages <- packages[!(packages %in% installed.packages()[, 'Package'])]
  if (length(new.packages)) install.packages(new.packages)
  invisible(lapply(packages, library, character.only = TRUE))
}

required.packages <- c('reticulate', 'stringr', 'zoo', 'pracma')
load_required_packages(required.packages)

# set up the `Python` environment
reticulate::use_condaenv('myenv', required = TRUE)

# import pandas
pd <- reticulate::import('pandas')
```

  3. **Insert the correct `username`**

```R
# insert your username here to define the correct path
username <- 'YourUsername'
```
  4. **Execute remaining code**
```R
# construct file path
file_path <- paste0('/Users/', username, '/Documents/Repositories/SPNcell')

# import custom functions
source(paste0(file_path, '/R analysis/functions.R'))

# to locate the created simulation directory
sim <- 'sim1100g'
spn <- 'dspn'

# set the working directory based on the value of spn and create the working directory path
wd <- paste0(file_path, '/', spn, '/model1/physiological/simulations/', sim)
setwd(wd)

# get list of files in directory
names <- list.files()

# initialize an empty list to store the data
data_list <- list()
filenames <- c()  

# load each pickle file and store in a list
for (file_name in names) {
  if (!grepl('\\.pickle$', file_name)) {
    next
  }
  key <- tools::file_path_sans_ext(file_name)
  tryCatch({
    data <- pd$read_pickle(file_name)
    data_list[[key]] <- data
    filenames <- c(filenames, key)  # Add the key to filenames if successful
  }, error = function(e) {
    message(paste('Error unpickling', file_name, ':', e$message))
  })
}

# unpack to create global variables
list2env(data_list, envir = .GlobalEnv)

# If metadata is present, unpack it as well
if ('metadata' %in% names(data_list)) {
  metadata <- data_list[['metadata']]
  for (key in names(metadata)) {
    if (key != 'sim') {
      assign(key, metadata[[key]], envir = .GlobalEnv)
    }
  }
}

# format data
dt <- metadata$dt
Vsoma <- do.call(cbind, vsoma)
Vdend <- do.call(cbind, vdend)

# reconstruct dend_tree; necessary for code to execute (note must be this format)
dend_tree <- list(
  c('dend[55]', 'dend[57]'),
  c('dend[55]', 'dend[56]'),
  c('dend[54]'),
  c('dend[49]', 'dend[53]'),
  c('dend[49]', 'dend[50]', 'dend[52]'),
  c('dend[49]', 'dend[50]', 'dend[51]'),
  c('dend[42]', 'dend[48]'),
  c('dend[42]', 'dend[43]', 'dend[47]'),
  c('dend[42]', 'dend[43]', 'dend[44]', 'dend[46]'),
  c('dend[42]', 'dend[43]', 'dend[44]', 'dend[45]'),
  c('dend[31]', 'dend[39]', 'dend[41]'),
  c('dend[31]', 'dend[39]', 'dend[40]'),
  c('dend[31]', 'dend[32]', 'dend[38]'),
  c('dend[31]', 'dend[32]', 'dend[33]', 'dend[37]'),
  c('dend[31]', 'dend[32]', 'dend[33]', 'dend[34]', 'dend[36]'),
  c('dend[31]', 'dend[32]', 'dend[33]', 'dend[34]', 'dend[35]'),
  c('dend[30]'),
  c('dend[19]', 'dend[23]', 'dend[24]'),
  c('dend[19]', 'dend[23]', 'dend[25]', 'dend[26]'),
  c('dend[19]', 'dend[23]', 'dend[25]', 'dend[27]', 'dend[29]'),
  c('dend[19]', 'dend[23]', 'dend[25]', 'dend[27]', 'dend[28]'),
  c('dend[19]', 'dend[20]', 'dend[22]'),
  c('dend[19]', 'dend[20]', 'dend[21]'),
  c('dend[0]', 'dend[6]', 'dend[16]', 'dend[18]'),
  c('dend[0]', 'dend[6]', 'dend[16]', 'dend[17]'),
  c('dend[0]', 'dend[1]', 'dend[5]'),
  c('dend[0]', 'dend[1]', 'dend[2]', 'dend[4]'),
  c('dend[0]', 'dend[1]', 'dend[2]', 'dend[3]'),
  c('dend[0]', 'dend[6]', 'dend[7]', 'dend[11]', 'dend[12]'),
  c('dend[0]', 'dend[6]', 'dend[7]', 'dend[11]', 'dend[13]', 'dend[15]'),
  c('dend[0]', 'dend[6]', 'dend[7]', 'dend[11]', 'dend[13]', 'dend[14]'),
  c('dend[0]', 'dend[6]', 'dend[7]', 'dend[8]', 'dend[10]'),
  c('dend[0]', 'dend[6]', 'dend[7]', 'dend[8]', 'dend[9]')
)

# create save directory; any plots will be dumped in svg format to the current set directory
# any plots will be saved in 'save_dir'

save_dir <- str_replace(wd, 'simulations', 'images')
# Create the save directory if it does not exist
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE)
}

# cell_coordinates
cell_coordinates <- mechanisms_3D_distribution$'cell_coordinates'
colnames(cell_coordinates) <- c('section', 'rel','x','y','z', 'dist', 'diam')
  
# morphology plot if required
morph = FALSE
if (morph){
  # set the working directory to the save directory
  setwd(save_dir)
  morphology_plot(cell_coordinates, dend_tree, smoothing=0.25)
  # return to original working directory
  setwd(wd)
}

# find section lengths if required
# section_lengths <- sapply(split(cell_coordinates[cell_coordinates$section != 'soma[0]', ], cell_coordinates$section[cell_coordinates$section != 'soma[0]']), function(df) max(df$dist) - min(df$dist))

# heat maps of peak membrane potential and charge transfer 
# heat maps are 2D representations of the data. As such, they represent the 2D view as defined by X and Y (i.e viewed as if observed by a viewer looking along the Z axis)
bl = 20        # required baseline
start_time=150 # stimulation time

# create matrix V3D of voltages at all coordinates 
V3D <- do.call(cbind, v_all_3D$Nsim0[[1]])

# use analysis() to return baseline adusted responses, maximum values and peaks
out <- analysis(V=V3D, bl=bl, start_time=start_time)
V3D0 <- out$V0
V3D0_normalised <- out$V0_norm
V3D_peaks <- out$Vpeaks
V3D_max <- out$Vmax

# V3D_max gives the maximum absolute membrane potential (i.e. 'peak') at that cell coordinate given by cell_coordinates

# Figure 6A: heatmap of peak voltages
setwd(save_dir)
heat2D(cell_coordinates, dend_tree, V3D_max,  title='', filename='figure6A_1', smoothing=0.25, upsample_factor=7, min_val=-84, max_val=-20, height=3.5, width=3.5)

# restricted heatmap of (majority of) dendritic tree for illustration
# plots everything that contains 'dend[7]' in path to soma
dend_name='dend[7]'
# filter_dend_tree(dend_tree=dend_tree, dend_name=dend_name)
heat2D(cell_coordinates, dend_tree, V3D_max, dend_name=dend_name,title='', filename='figure6A_2', smoothing=0.25, upsample_factor=7, min_val=-84, max_val=-20, height=4.15, width=4.15)

# Figure 6D: heatmap of total charge transfer
mechs <- i_mechs_3D$Nsim0$mechs
# mech of interest form recorded mechs
mech <- 'cal13'
idx <- match(mech, mechs)

i_all <- i_mechs_3D$Nsim0$i
out_i <- sapply(1:length(i_all), function(ii) i_all[[ii]][[idx]] )
  
out <- analysis(V=out_i, bl=bl, start_time=start_time, method='min')
i_mech_3D0 <- out$V0 *1e6 # current underlying mechanism in pA/cm2
perm1 <- out$Vmax*1e6 # returns peak current density

# # if required plot these as heatmaps
# heat2D(cell_coordinates, dend_tree, perm1, title='cal13_', smoothing=0.25, colorbar_reverse=TRUE, min_val=-250, max_val=0)
# heat2D(cell_coordinates, dend_tree, perm1, dend_name=dend_name, title='path_cal13_', smoothing=0.25, colorbar_reverse=TRUE, min_val=-250, max_val=0)

# get baseline adjusted currents for required mechanism 
i_mechs <- out$V0

# to obtain charge transfer (per cm2), find area under curve
# mA to pA 1e6, ms to secs 1e-3
x <- 0:dt:(dim(i_mechs)[1]-1)*dt
areas <- sapply(1:dim(i_mechs)[2], function(ii){
  y <- -i_mechs[,ii]
  trapz(x, y)*1e3
})

# areas gives the charge transfer/cm2 at that cell coordinate given by cell_coordinates

heat2D(cell_coordinates, dend_tree, areas, title='', filename='figure6D_1', smoothing=0.25, upsample_factor=7, min_val=0, max_val=12, height=3.5, width=3.5)
heat2D(cell_coordinates, dend_tree, areas, dend_name=dend_name, title='', filename='figure6D_2', smoothing=0.25, upsample_factor=7, min_val=0, max_val=12, height=4.15, width=4.15)

setwd(wd)

# Figure 6B and E
# dend_name gives the path to plot for figure
path2plot <-  filter_dend_tree(dend_tree=dend_tree, dend_name=dend_name)[[2]]; path2plot <- c('soma[0]', path2plot)
path_indices <- which(cell_coordinates$section %in% path2plot)
path2plot_coords <- cell_coordinates[path_indices,]

target_dend <- metadata$dend_glut
rel.x <- tail(metadata$glutamate_locations, n=1)

# plot response at idx corresponding to final location of the train (towards the soma)
idx <- which(path2plot_coords$section == target_dend)[which.min(abs(path2plot_coords$rel[path2plot_coords$section == target_dend] - rel.x))]
idx_start <- which(path2plot_coords$section == target_dend)[which.min(abs(path2plot_coords$rel[path2plot_coords$section == target_dend] - metadata$glutamate_locations[1]))]

# start and finish distance from soma
cell_coordinates[c(path_indices[idx_start], path_indices[idx]),]$dist
# [1] 156.3907 145.5985

# graph settings
lwd <- 0.8
traces <-  c(path_indices[1], path_indices[idx])
xlim <- c(0, 200)

width_egs <- 5.25
height_egs <- 8.25

setwd(save_dir)

svg(file=paste0('figure6B.svg'), width=width_egs, height=height_egs, pointsize=10, bg='transparent')  # Width and height in inches
traces2plot(V=V3D0, traces=traces, xlim=xlim, ylim=c(-5,75), lwd=lwd, xbar=50, ybar=20, dt=dt, reverse=FALSE)
dev.off()

svg(file=paste0('figure6E.svg'), width=width_egs, height=height_egs, pointsize=10, bg='transparent')  # Width and height in inches
traces2plot(V=i_mech_3D0, traces=traces, xlim=xlim, ylim=c(-255,5), lwd=lwd, xbar=50, ybar=50, dt=dt, reverse=FALSE)
dev.off()

# Figure 6C and F: plots of membrane potential change in dendritic path to soma and major branches (i.e as restricted dendritic tree)
x1 <- cell_coordinates[path_indices,]$dist; x1[1] <- 0 # somatic recording strictly at center of soma
y1 <- V3D_peaks[path_indices]

indices8_10 <- which(cell_coordinates$section %in% c('dend[8]', 'dend[10]'))
indices12 <- which(cell_coordinates$section %in% 'dend[12]')
indices14 <- which(cell_coordinates$section %in% 'dend[14]')

x2 <- cell_coordinates[indices8_10,]$dist
y2 <- V3D_peaks[indices8_10]

x3 <- cell_coordinates[indices12,]$dist
y3 <- V3D_peaks[indices12]

x4 <- cell_coordinates[indices14,]$dist
y4 <- V3D_peaks[indices14]

y_list <- list(y2, y3, y4, y1); x_list <- list(x2, x3, x4, x1)

colors <- c('#D3D3D3', '#808080', '#696969', '#CD5C5C')
filename = 'plot1.svg'
plot2(filename='figure6C.svg', x_list=x_list, y_list=y_list, lwd=lwd, colors=colors, height=3.25, width=2.75)

# charge transfer
y_list2 <- list(areas[indices8_10], areas[indices12], areas[indices14], areas[path_indices])
plot2(filename='figure6F.svg', x_list=x_list, y_list=y_list2, ylim=c(0, 12.5), ylab=expression(paste('charge transfer (pC/cm'^2, ')')), ymajor_tick=5, lwd=lwd, colors=colors, height=3.25, width=2.75)

setwd(wd)
```

  ## Summary
  1. **When executing this code, the correct user name is required.** This line should be the only one that it is necessary to change in order to execute the `R` code:
  
  ```R
  username <- 'YourUsername'
  ```

  2. **A warning occurs on my Windows laptop during the `R` installation of `Miniconda` / `pandas`.** It doesn't affect the subsequent code so its safe to ignore it.

  3. **Script assumes the `R` working directory aligns with the structure of the path.** The correct path occurs by default as `documents/Repositories/SPNcell/R analysis`. Adjust if needed.


### Using `Jupyter Notebook` to analyse a simulation

Open the terminal and run:

```bash
cd documents/Repositories/SPNcell
conda activate neuron
python -m ipykernel install --user --name neuron --display-name "Python (neuron)"
jupyter notebook
```

Then simply open the provided `Jupyter Notebook` entitled `Figure 6 analysis.ipynb` and execute the provided code. This code should work on both MacOS and Windows PC. The code should produce the same graphical outputs as the `R` code.

## `Anaconda` vs `Miniconda`

`Anaconda` and `Miniconda` are both popular distributions for `Python` and `R` programming in data science. They include the `Conda` package manager and aim to simplify package management and deployment.

**`Anaconda`** is a full-featured distribution that includes:

- `Python` and `R` language
- `Conda` package manager
- Over 1,500 pre-installed scientific packages
- Tools like `Jupyter Notebook`, `RStudio`, etc.

`Anaconda` provides an out-of-the-box setup for data science and scientific computing.

**`Miniconda`** offers a minimalistic approach:

- `Python` and `R` language
- `Conda` package manager
- No pre-installed packages

`Miniconda` provides a lightweight base to start with but packages must be installed if needed.

**Advantages of `Anaconda`**:

- Quick, easy setup with a comprehensive suite of scientific packages and tools.
- Wide array of data science tools readily available within a single application.

**Advantages of `Miniconda`**:

- Lightweight, minimal base installation.
- Control over which packages are installed.
- Requires limited disk space or bandwidth.
- Clean environment that only includes packages required.

**Installation**

- [`Anaconda`](https://www.anaconda.com/products/individual#Downloads) 
- [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html)

Follow the installation instructions provided on the respective download pages.

**Additional Resources**

- [`Anaconda`](https://docs.anaconda.com/) documentation
- [`Miniconda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) documentation
- [`Conda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) package management


## Virtual Environments

The `environment.yml` file is configured for `NEURON` 8.2.4 and `Python` 3.9.16. 

Use this file to create a consistent environment for running the models.

In brief:     

* **`YAML` Environment File**: The file `environment.yml` is a `YAML` file commonly used in `Conda`

  This file specifies the dependencies and settings for a particular virtual environment.

* **Environment Name** - name Key: In the `environment.yml` file, there is a key called name.

  The value associated with this key is the name of the `Conda` environment to be created.

  In this case, this name is `neuron`. This is the name that will subsequently be used

  to refer to the environment when activating it or installing additional packages into it.

* **Creating the Environment**: When the command `conda env create -f environment.yml` is executed

  `Conda` reads the `environment.yml` file and creates a new environment based on the specifications in that file.

  The environment will have the name given by the name key in the `YAML` file i.e. `neuron`.

## Updating `YAML` and conda environment
The `environment.yml` file can be updated to the latest packages using the following code. 

There may be compatibility issues with some package updates so it's always a good idea to retain the original `environment.yml` file in case a rollback is required / troubleshooting etc.

## Useful `terminal` code

To delete:
```bash
conda env remove --name neuron
```

To update:
```bash
cd documents/Repositories/SPNcell
conda activate neuron
conda update --all

python update_pip_packages.py

conda env export --name neuron --file updated_environment.yml
```

To remove the build information:
```bash
conda env export --no-builds > updated_environment.yml
```

When using Windows, I had an issue creating the `neuron` environment.

I tried to verify the active environment using the `powershell`:

```bash
conda activate neuron
conda info --envs
```

However, there was no asterix next to `neuron` indicating it had not been succesfully initiated. The solution was to run the following line once to initiate the `powershell`:

```bash
conda init powershell
```

I was then able to create and activate the `neuron` environment.

I expect this error would have been avoided if using the `Anaconda` prompt.

## `GitHub`

For beginners, the [`GitHub Desktop`](https://desktop.github.com/) is recommended. 

Instructions for cloning a repository using [`GitHub Desktop`](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop).

## References

[Day M., Belal M., Surmeier W. C., Melendez A., Wokosin, D., Tkatch T., Clarke V. R. J. and Surmeier D. J.  GABAergic regulation of striatal spiny projection neurons depends upon their activity state (2024). PLoS Biol 22(1): e3002483](https://doi.org/10.1371/journal.pbio.3002483)

[Du K, Wu Y-W, Lindroos R, Liu Y, Rózsa B, Katona G, et al. Cell-type–specific inhibition of the dendritic plateau potential in striatal spiny projection neurons. Proceedings of the National Academy of Sciences. 2017;114: E7612 E7621](https://doi.org/10.1073/pnas.1704893114)

[Hines ML, Carnevale NT. The NEURON Simulation Environment. Neural Comput. 1997;9: 1179–1209](https://doi.org/10.1162/neco.1997.9.6.1179)

[Lindroos R, Dorst MC, Du K, Filipović M, Keller D, Ketzef M, et al. Basal Ganglia Neuromodulation Over Multiple Temporal and Structural Scales-Simulations of Direct Pathway MSNs Investigate the Fast Onset of Dopaminergic Effects and Predict the Role of Kv4.2. Frontiers in neural circuits. 2018;12: 3](https://doi.org/10.3389/fncir.2018.00003) 

[Lindroos R, Kotaleski JH. Predicting complex spikes in striatal projection neurons of the direct pathway following neuromodulation by acetylcholine and dopamine. Eur J Neurosci. 2020](https://doi.org/10.1111/ejn.14891)



## Contact

The model was adapted from the publicly available code by Vernon Clarke.

The provided code was executed on a `MacBook M2 pro 32GB`. I have tried to ensure that the code works on other operating systems but it's certainly possible that some errors and bugs exist. 

In order to make this code accessible for publication, it is necessary to create a permanent public repository with a citable DOI using, for example using `Zenodo` to archive a version of this `GitHub` package.

If any bug fixes are necessary (most likely related to providing help on other operating systems), it will be provided as an update on the parent [`GitHub`](https://github.com/vernonclarke/SPNcell) page.

For queries related to this repository, please [open an issue](https://github.com/vernonclarke/SPNcell/issues) or [email](mailto:WOPR2@proton.me) directly. 

---


   

