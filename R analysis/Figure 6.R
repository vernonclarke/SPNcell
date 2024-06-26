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

# set up the Python environment
reticulate::use_condaenv('myenv', required = TRUE)

# import pandas
pd <- reticulate::import('pandas')

# insert your username here to define the correct path
username <- 'YourUsername'

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
  
# morpholog plot if required
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

# # alternative
# cell_coordinates <- do.call(rbind, lapply(v_all_3D$Nsim0[[2]], function(y) {
#   data.frame(
#     section = y[[1]],
#     rel = y[[2]],
#     x = y[[3]],
#     y = y[[4]],
#     z = y[[5]],
#     dist = y[[6]],
#     diam = y[[7]],
#     stringsAsFactors = FALSE
#   )
# }))

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


