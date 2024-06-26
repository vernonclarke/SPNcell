# Function to match coordinates based on section and closest distance
match_coordinates <- function(indexing_df, cell_coordinates_record_df) {
  matched_rows <- vector('list', nrow(indexing_df))
  
  for (i in 1:nrow(indexing_df)) {
    section <- indexing_df$section[i]
    dist <- as.numeric(indexing_df$dist[i])
    
    # Filter cell_coordinates_record_df for the same section
    section_rows <- cell_coordinates_record_df[cell_coordinates_record_df$section == section, ]
    
    # Ensure dist columns are numeric
    section_rows$dist <- as.numeric(section_rows$dist)
    
    # Find the row with the closest distance
    closest_row_index <- which.min(abs(section_rows$dist - dist))
    matched_rows[[i]] <- section_rows[closest_row_index, ]
  }
  
  # Combine the matched rows into a data frame
  matched_df <- do.call(rbind, matched_rows)
  return(matched_df)
}

morphology_plot1 <- function(cell_coordinates, dend_tree, title='morphology', lwd=0.8, color='black', xlim=c(-175, 175), ylim=c(-175, 175), xbar=50, height=10, width=10, filename='neuronal_morphology1.svg') {
  
  # convert columns to numeric if they are not already
  cell_coordinates$x <- as.numeric(cell_coordinates$x)
  cell_coordinates$y <- as.numeric(cell_coordinates$y)
  cell_coordinates$diam <- as.numeric(cell_coordinates$diam)
  
  # Open SVG device
  svg(file=filename, width=width, height=height, bg='transparent')
  
  # Prepare the plotting area
  plot(NA, xlim=xlim, ylim=ylim, xlab='', ylab='', asp=1, main=title,  xaxs='i', yaxs='i', xaxt='n', yaxt='n', bty='n')

  # Process connections
  connections <- list()
  for (branch in dend_tree) {
    if (length(branch) > 1) {
      for (i in 1:(length(branch) - 1)) {
        sec_name <- branch[[i]]
        next_sec_name <- branch[[i + 1]]
        
        if (is.null(connections[[sec_name]])) {
          connections[[sec_name]] <- character()
        }
        
        if (!(next_sec_name %in% connections[[sec_name]])) {
          connections[[sec_name]] <- c(connections[[sec_name]], next_sec_name)
        }
      }
    }
  }
  
  processed_sections <- character()
  
  # Plot soma
  soma_coords <- cell_coordinates[cell_coordinates$section == 'soma[0]', ]
  center_x <- mean(soma_coords$x, na.rm = TRUE)
  center_y <- mean(soma_coords$y, na.rm = TRUE)
  radius <- mean(soma_coords$diam, na.rm = TRUE) / 2
  
  symbols(center_x, center_y, circles=radius, inches=FALSE, add=TRUE, fg=color, lwd=lwd, bg=color)
  
  # Function to plot a line
  plot_line <- function(x_coords, y_coords, line_color, line_width) {
    lines(x_coords, y_coords, col=line_color, lwd=line_width)
  }
  
  # Function to add sections
  add_section <- function(section_name) {
    section_coords <- cell_coordinates[cell_coordinates$section == section_name, ]

    x_coords <- section_coords$x
    y_coords <- section_coords$y
    plot_line(x_coords, y_coords, color, lwd)

  }
  
  # Add primary sections from soma
  primary_sections <- unique(unlist(lapply(dend_tree, function(branch) {
    if (is.list(branch[[1]])) {
      return(lapply(branch, function(sub_branch) sub_branch[[1]]))
    } else {
      return(branch[[1]])
    }
  })))
  
  for (primary in primary_sections) {
    primary_coords <- cell_coordinates[cell_coordinates$section == primary, ]
    if (nrow(primary_coords) > 0) {
      start_x <- primary_coords[1, 'x']
      start_y <- primary_coords[1, 'y']
      
      angle <- atan2(start_y - center_y, start_x - center_x)
      perimeter_x <- center_x + radius * cos(angle)
      perimeter_y <- center_y + radius * sin(angle)
      
      plot_line(c(perimeter_x, start_x), c(perimeter_y, start_y), color, lwd)
      
      x_coords <- primary_coords$x
      y_coords <- primary_coords$y
      plot_line(x_coords, y_coords, color, lwd)
      
      processed_sections <- c(processed_sections, primary)
    }
  }
  
  # Plot dendrites
  for (parent in names(connections)) {
    if (!(parent %in% processed_sections)) {
      processed_sections <- c(processed_sections, parent)
      add_section(parent)
    }
    children <- connections[[parent]]
    for (child in children) {
      if (!(child %in% processed_sections)) {
        section_coords <- cell_coordinates[cell_coordinates$section == child, ]
        if (nrow(section_coords) > 0) {
          start_x <- section_coords[1, 'x']
          start_y <- section_coords[1, 'y']
          
          end_coords <- cell_coordinates[cell_coordinates$section == parent, ]
          end_x <- tail(end_coords$x, 1)
          end_y <- tail(end_coords$y, 1)
          
          plot_line(c(end_x, start_x), c(end_y, start_y), color, lwd)
          
          x_coords <- section_coords$x
          y_coords <- section_coords$y
          
          add_section(child)
          
          processed_sections <- c(processed_sections, child)
        }
      }

    }
  }
  
  # add x-axis scale bar in the lower right corner
  scale_bar_length <- xbar
  scale_bar_x_start <- xlim[2] - scale_bar_length - 10
  scale_bar_x_end <- xlim[2] - 10
  scale_bar_y <- ylim[1] + 15
  
  segments(x0 = scale_bar_x_start, y0 = scale_bar_y, x1 = scale_bar_x_end, y1 = scale_bar_y, col = 'black', lwd = lwd)
  text(x = (scale_bar_x_start + scale_bar_x_end) / 2, y = scale_bar_y - 5, labels = paste(scale_bar_length, '\u00B5m'), adj = c(0.5, 1))

# close the SVG device
  dev.off()
}

morphology_plot2 <- function(cell_coordinates, dend_tree, title='morphology', lwd=0.8, color='black', smoothing=0.4, xlim=c(-175, 175), ylim=c(-175, 175), xbar=50, height=10, width=10, filename='neuronal_morphology2.svg') {
  
  # convert columns to numeric if they are not already
  cell_coordinates$x <- as.numeric(cell_coordinates$x)
  cell_coordinates$y <- as.numeric(cell_coordinates$y)
  cell_coordinates$diam <- as.numeric(cell_coordinates$diam)
  
  # Open SVG device
  svg(file=filename, width=width, height=height, bg='transparent')
  
  # Prepare the plotting area
  plot(NA, xlim=xlim, ylim=ylim, xlab='', ylab='', asp=1, main=title, xaxs='i', yaxs='i', xaxt='n', yaxt='n', bty='n')
  
  # Process connections
  connections <- list()
  for (branch in dend_tree) {
    if (length(branch) > 1) {
      for (i in 1:(length(branch) - 1)) {
        sec_name <- branch[[i]]
        next_sec_name <- branch[[i + 1]]
        
        if (is.null(connections[[sec_name]])) {
          connections[[sec_name]] <- character()
        }
        
        if (!(next_sec_name %in% connections[[sec_name]])) {
          connections[[sec_name]] <- c(connections[[sec_name]], next_sec_name)
        }
      }
    }
  }
  
  processed_sections <- character()
  
  # Plot soma
  soma_coords <- cell_coordinates[cell_coordinates$section == 'soma[0]', ]
  center_x <- mean(soma_coords$x, na.rm = TRUE)
  center_y <- mean(soma_coords$y, na.rm = TRUE)
  radius <- mean(soma_coords$diam, na.rm = TRUE) / 2
  
  symbols(center_x, center_y, circles=radius, inches=FALSE, add=TRUE, fg=color, lwd=lwd, bg=color)
  
  # Function to plot a line
  plot_line <- function(x_coords, y_coords, line_color, line_width) {
    lines(x_coords, y_coords, col=line_color, lwd=line_width)
  }
  
  # Function to plot a line using splines
  plot_spline <- function(x_coords, y_coords, line_color, line_width) {
    # Remove NA and infinite values
    valid_indices <- which(!is.na(x_coords) & !is.na(y_coords) & is.finite(x_coords) & is.finite(y_coords))
    x_coords <- x_coords[valid_indices]
    y_coords <- y_coords[valid_indices]

    if (length(unique(x_coords)) > 3 && length(unique(y_coords)) > 3) {
      spline_fit_x <- smooth.spline(x_coords, spar=smoothing)
      spline_fit_y <- smooth.spline(y_coords, spar=smoothing)
      lines(spline_fit_x$y, spline_fit_y$y, col=line_color, lwd=line_width)
    } else {
      lines(x_coords, y_coords, col=line_color, lwd=line_width)
    }
  }


  # Function to add sections
  add_section <- function(section_name) {
    section_coords <- cell_coordinates[cell_coordinates$section == section_name, ]

    x_coords <- section_coords$x
    y_coords <- section_coords$y
    plot_spline(x_coords, y_coords, color, lwd)

  }
  
  # Add primary sections from soma
  primary_sections <- unique(unlist(lapply(dend_tree, function(branch) {
    if (is.list(branch[[1]])) {
      return(lapply(branch, function(sub_branch) sub_branch[[1]]))
    } else {
      return(branch[[1]])
    }
  })))
  
  for (primary in primary_sections) {
    primary_coords <- cell_coordinates[cell_coordinates$section == primary, ]
    if (nrow(primary_coords) > 0) {
      start_x <- primary_coords[1, 'x']
      start_y <- primary_coords[1, 'y']
      
      angle <- atan2(start_y - center_y, start_x - center_x)
      perimeter_x <- center_x + radius * cos(angle)
      perimeter_y <- center_y + radius * sin(angle)
      
      plot_line(c(perimeter_x, start_x), c(perimeter_y, start_y), color, lwd)
      
      x_coords <- primary_coords$x
      y_coords <- primary_coords$y
      plot_spline(x_coords, y_coords, color, lwd)
      
      processed_sections <- c(processed_sections, primary)
    }
  }
  
  # Plot dendrites
  for (parent in names(connections)) {
    if (!(parent %in% processed_sections)) {
      processed_sections <- c(processed_sections, parent)
      add_section(parent)
    }
    children <- connections[[parent]]
    for (child in children) {
      if (!(child %in% processed_sections)) {
        section_coords <- cell_coordinates[cell_coordinates$section == child, ]
        if (nrow(section_coords) > 0) {
          start_x <- section_coords[1, 'x']
          start_y <- section_coords[1, 'y']
          
          end_coords <- cell_coordinates[cell_coordinates$section == parent, ]
          end_x <- tail(end_coords$x, 1)
          end_y <- tail(end_coords$y, 1)
          
          plot_line(c(end_x, start_x), c(end_y, start_y), color, lwd)
          
          x_coords <- section_coords$x
          y_coords <- section_coords$y
          
          add_section(child)
          
          processed_sections <- c(processed_sections, child)
        }
      }

    }
  }
  
  # add x-axis scale bar in the lower right corner
  scale_bar_length <- xbar
  scale_bar_x_start <- xlim[2] - scale_bar_length - 10
  scale_bar_x_end <- xlim[2] - 10
  scale_bar_y <- ylim[1] + 15
  
  segments(x0 = scale_bar_x_start, y0 = scale_bar_y, x1 = scale_bar_x_end, y1 = scale_bar_y, col = 'black', lwd = lwd)
  text(x = (scale_bar_x_start + scale_bar_x_end) / 2, y = scale_bar_y - 5, labels = paste(scale_bar_length, '\u00B5m'), adj = c(0.5, 1))
 
  # close the SVG device
  dev.off()
}

morphology_plot <- function(cell_coordinates, dend_tree, title='morphology', lwd=0.8, color='darkgray', smoothing=NULL, xlim=c(-175, 175), ylim=c(-175, 175), xbar=50, height=10, width=10){

	if (is.null(smoothing)){
		filename <- paste0(gsub('[^a-zA-Z0-9]', '', title), '1.svg')
    morphology_plot1(cell_coordinates=cell_coordinates, dend_tree=dend_tree, title=title, lwd=lwd, color=color, xlim=xlim, ylim=ylim, xbar=xbar, height=height, width=width, filename=filename)
	}else{
		filename <- paste0(gsub('[^a-zA-Z0-9]', '', title), '2.svg')
    morphology_plot2(cell_coordinates=cell_coordinates, dend_tree=dend_tree, title=title, lwd=lwd, color=color, smoothing=smoothing, xlim=xlim, ylim=ylim, xbar=xbar, height=height, width=width, filename=filename)
	}
}

jet.colors <- function(n, alpha=1) {
  colors <- grDevices::colorRampPalette(c('#00007F', 'blue', '#007FFF', 'cyan', 
                                          '#7FFF7F', 'yellow', '#FF7F00', 'red', '#7F0000'))(n)
  colors <- sapply(colors, function(col) {
    rgb <- col2rgb(col)
    rgb(red = rgb[1], green = rgb[2], blue = rgb[3], alpha = alpha * 255, maxColorValue = 255)
  })
  return(colors)
}

filter_dend_tree <- function(dend_tree, dend_name) {
  result <- lapply(dend_tree, function(x) {
    if (any(dend_name %in% x)) {
      return(x)
    } else {
      return(NULL)
    }
  })
  result <- Filter(Negate(is.null), result)
  return(result)
}

heat2D_1 <- function(cell_coordinates, dend_tree, z, dend_name=NULL, title='', lwd=0.8, upsample_factor=10, alpha=0.6, xlim=c(-175, 175), ylim=c(-175, 175), xbar=50, height=10, width=10, colorbar_reverse=FALSE, filename='heatmap1.svg', min_val=NULL, max_val=NULL) {
  
  if (!is.null(dend_name)) dend_tree <-  filter_dend_tree(dend_tree=dend_tree, dend_name=dend_name)

  # Convert columns to numeric if they are not already
  cell_coordinates$x <- as.numeric(cell_coordinates$x)
  cell_coordinates$y <- as.numeric(cell_coordinates$y)
  cell_coordinates$diam <- as.numeric(cell_coordinates$diam)
  
  # color range
  if (is.null(min_val)) {
    min_val <- min(z, na.rm=TRUE)
  }
  if (is.null(max_val)) {
    max_val <- max(z, na.rm=TRUE)
  }
  
  # normalize z values to the range [0, 1] based on min_val and max_val
  z_normalized <- (z - min_val) / (max_val - min_val)
  z_normalized[z_normalized < 0] <- 0
  z_normalized[z_normalized > 1] <- 1
  
  # Create a color mapping based on normalized z values
  color_palette <- jet.colors(100, alpha=alpha)
  if (colorbar_reverse) color_palette <- rev(color_palette)
  color_indices <- cut(z_normalized, breaks=seq(0, 1, length.out=101), labels=FALSE, include.lowest=TRUE)
  point_colors <- color_palette[color_indices]
  
  # Open SVG device
  svg(file=filename, width=width, height=height, bg='transparent')  # Width and height in inches
  
  # Prepare the plotting area
  plot(NA, xlim=xlim, ylim=ylim, xlab='', ylab='', asp=1, main=title, xaxs='i', yaxs='i', xaxt='n', yaxt='n', bty='n')
  
  # Plot soma with the first value of z
  soma_coords <- cell_coordinates[cell_coordinates$section == 'soma[0]', ]
  center_x <- mean(soma_coords$x, na.rm = TRUE)
  center_y <- mean(soma_coords$y, na.rm = TRUE)
  radius <- mean(soma_coords$diam, na.rm = TRUE) / 2
  
  symbols(center_x, center_y, circles=radius, inches=FALSE, add=TRUE, fg=NA, lwd=lwd, bg=point_colors[1])
  
  # plot a segment with heatmap colors and flat ends
  plot_segment <- function(x_coords, y_coords, line_colors, line_width) {
  
    # Upsample the coordinates and colors
    if (length(x_coords) > 1) {
      t <- seq_along(x_coords)
      t_new <- seq(min(t), max(t), length.out = length(t) * upsample_factor)
      
      x_coords <- approx(t, x_coords, xout = t_new)$y
      y_coords <- approx(t, y_coords, xout = t_new)$y
      line_colors <- interpolate_colors_with_alpha(line_colors, length(x_coords))
    }  

    par(lend = 'butt')

    for (i in 1:(length(x_coords) - 1)) {
      lines(x_coords[i:(i + 1)], y_coords[i:(i + 1)], col = line_colors[i], lwd = line_width)
    }
  }
  
  # add sections
  add_section <- function(section_name) {
    section_coords <- cell_coordinates[cell_coordinates$section == section_name, ]
    section_z <- z[cell_coordinates$section == section_name]
    section_colors <- point_colors[cell_coordinates$section == section_name]
    if (nrow(section_coords) > 0) {
      plot_segment(section_coords$x, section_coords$y, section_colors, lwd)
    }
  }
  
  # add primary sections from soma
  primary_sections <- unique(unlist(lapply(dend_tree, function(branch) {
    if (is.list(branch[[1]])) {
      return(lapply(branch, function(sub_branch) sub_branch[[1]]))
    } else {
      return(branch[[1]])
    }
  })))
  
  processed_sections <- character()
  for (primary in primary_sections) {
    primary_coords <- cell_coordinates[cell_coordinates$section == primary, ]
    if (nrow(primary_coords) > 0) {
      start_x <- primary_coords[1, 'x']
      start_y <- primary_coords[1, 'y']
      
      angle <- atan2(start_y - center_y, start_x - center_x)
      perimeter_x <- center_x + radius * cos(angle)
      perimeter_y <- center_y + radius * sin(angle)
      
      plot_segment(c(perimeter_x, start_x), c(perimeter_y, start_y), point_colors[cell_coordinates$section == primary], lwd)
      
      x_coords <- primary_coords$x
      y_coords <- primary_coords$y
      plot_segment(x_coords, y_coords, point_colors[cell_coordinates$section == primary], lwd)

      processed_sections <- c(processed_sections, primary)
    }
  }
  
  # process connections
  connections <- list()
  for (branch in dend_tree) {
    if (length(branch) > 1) {
      for (i in 1:(length(branch) - 1)) {
        sec_name <- branch[[i]]
        next_sec_name <- branch[[i + 1]]
        
        if (is.null(connections[[sec_name]])) {
          connections[[sec_name]] <- character()
        }
        
        if (!(next_sec_name %in% connections[[sec_name]])) {
          connections[[sec_name]] <- c(connections[[sec_name]], next_sec_name)
        }
      }
    }
  }
  
  # plot dendrites

  for (parent in names(connections)) {
    if (!(parent %in% processed_sections)) {
      processed_sections <- c(processed_sections, parent)
      add_section(parent)
    }
    children <- connections[[parent]]
    for (child in children) {
      if (!(child %in% processed_sections)) {
        section_coords <- cell_coordinates[cell_coordinates$section == child, ]
        if (nrow(section_coords) > 0) {
          start_x <- section_coords[1, 'x']
          start_y <- section_coords[1, 'y']
          
          end_coords <- cell_coordinates[cell_coordinates$section == parent, ]
          end_x <- tail(end_coords$x, 1)
          end_y <- tail(end_coords$y, 1)
          
          parent_color <- tail(point_colors[cell_coordinates$section == parent], 1)
          segments(x0 = end_x, y0 = end_y, x1 = start_x, y1 = start_y, col = parent_color, lwd = lwd, lend = 'butt')
          
          x_coords <- section_coords$x
          y_coords <- section_coords$y
          
          add_section(child)
          
          processed_sections <- c(processed_sections, child)
        }
      }
    }
  }
  
  # plot color bar (top right corner)
  color_bar_x_start <- xlim[2] - 20    # start x coordinate for the color bar
  color_bar_x_end <- xlim[2] - 10      # end x coordinate for the color bar
  color_bar_y_start <- ylim[2] - 115   # start y coordinate for the color bar (20% of y axis)
  color_bar_y_end <- ylim[2] - 25      # end y coordinate for the color bar
  
  color_bar_z <- matrix(seq(min_val, max_val, length.out = 100), ncol = 1)
  image(x = seq(color_bar_x_start, color_bar_x_end, length = 2), 
        y = seq(color_bar_y_start, color_bar_y_end, length = 100), 
        z = t(color_bar_z), 
        col = color_palette, 
        add = TRUE)
  
  # add min and max labels to color bar
  text(x = color_bar_x_end + 2, y = color_bar_y_start, labels = round(min_val, 2), adj = c(0, 0.5))
  text(x = color_bar_x_end + 2, y = color_bar_y_end, labels = round(max_val, 2), adj = c(0, 0.5))
  
  # add x-axis scale bar in the lower right corner
  scale_bar_length <- xbar
  scale_bar_x_start <- xlim[2] - scale_bar_length - 10
  scale_bar_x_end <- xlim[2] - 10
  scale_bar_y <- ylim[1] + 15
  
  segments(x0 = scale_bar_x_start, y0 = scale_bar_y, x1 = scale_bar_x_end, y1 = scale_bar_y, col = 'black', lwd = lwd)
  text(x = (scale_bar_x_start + scale_bar_x_end) / 2, y = scale_bar_y - 5, labels = paste(scale_bar_length, '\u00B5m'), adj = c(0.5, 1))

  # close SVG device
  dev.off()
}

heat2D_2 <- function(cell_coordinates, dend_tree, z, dend_name=NULL, title='', lwd=0.8, smoothing=0.25, upsample_factor=10, alpha=0.6, xlim=c(-175, 175), ylim=c(-175, 175), xbar=50, height=10, width=10, colorbar_reverse=FALSE, filename='heatmap2.svg', min_val=NULL, max_val=NULL) {

  if (!is.null(dend_name)) dend_tree <-  filter_dend_tree(dend_tree=dend_tree, dend_name=dend_name)

  # Convert columns to numeric if they are not already
  cell_coordinates$x <- as.numeric(cell_coordinates$x)
  cell_coordinates$y <- as.numeric(cell_coordinates$y)
  cell_coordinates$diam <- as.numeric(cell_coordinates$diam)

  # Determine the color range
  if (is.null(min_val)) {
    min_val <- min(z, na.rm=TRUE)
  }
  if (is.null(max_val)) {
    max_val <- max(z, na.rm=TRUE)
  }

  # Normalize z values to the range [0, 1] based on min_val and max_val
  z_normalized <- (z - min_val) / (max_val - min_val)
  z_normalized[z_normalized < 0] <- 0
  z_normalized[z_normalized > 1] <- 1

  # Create a color mapping based on normalized z values
  color_palette <- jet.colors(100, alpha=alpha)
  if (colorbar_reverse) color_palette <- rev(color_palette)
  color_indices <- cut(z_normalized, breaks=seq(0, 1, length.out=101), labels=FALSE, include.lowest=TRUE)
  point_colors <- color_palette[color_indices]

  # Open SVG device
  svg(file=filename, width=width, height=height, bg='transparent')  # Width and height in inches

  # Prepare the plotting area
  plot(NA, xlim=xlim, ylim=ylim, xlab='', ylab='', asp=1, main=title, xaxs='i', yaxs='i', xaxt='n', yaxt='n', bty='n')

  # Plot soma with the first value of z
  soma_coords <- cell_coordinates[cell_coordinates$section == 'soma[0]', ]
  center_x <- mean(soma_coords$x, na.rm = TRUE)
  center_y <- mean(soma_coords$y, na.rm = TRUE)
  radius <- mean(soma_coords$diam, na.rm = TRUE) / 2

  symbols(center_x, center_y, circles=radius, inches=FALSE, add=TRUE, fg=NA, lwd=lwd, bg=point_colors[1])

  # Function to interpolate colors with alpha
  interpolate_colors_with_alpha <- function(colors, n) {
    rgb_colors <- col2rgb(colors, alpha = TRUE)
    interpolated_colors <- matrix(0, ncol = 4, nrow = n)
    
    for (i in 1:4) {
      interpolated_colors[, i] <- approx(seq_along(rgb_colors[i, ]), rgb_colors[i, ], n = n)$y
    }
    
    interpolated_colors <- apply(interpolated_colors, 1, function(channel) {
      sprintf('#%02X%02X%02X%02X', round(channel[1]), round(channel[2]), round(channel[3]), round(channel[4]))
    })
    
    return(interpolated_colors)
  }

  # Function to plot a line using splines with smooth color interpolation
  plot_segment <- function(x_coords, y_coords, line_colors, line_width, smoothing = 0.25, upsample_factor=10) {
    # Remove NA and infinite values
    valid_indices <- which(!is.na(x_coords) & !is.na(y_coords) & is.finite(x_coords) & is.finite(y_coords))
    x_coords <- x_coords[valid_indices]
    y_coords <- y_coords[valid_indices]
    
    # Upsample the coordinates and colors
    if (length(x_coords) > 1) {
      t <- seq_along(x_coords)
      t_new <- seq(min(t), max(t), length.out = length(t) * upsample_factor)
      
      x_coords <- approx(t, x_coords, xout = t_new)$y
      y_coords <- approx(t, y_coords, xout = t_new)$y
      line_colors <- interpolate_colors_with_alpha(line_colors, length(x_coords))
    }

    par(lend = 'butt')

    if (length(unique(x_coords)) > 3 && length(unique(y_coords)) > 3) {
      # Fit a smoothing spline
      spline_fit_x <- smooth.spline(x_coords, spar = smoothing)
      spline_fit_y <- smooth.spline(y_coords, spar = smoothing)
      
      # Extract the smoothed coordinates
      smoothed_x <- spline_fit_x$y
      smoothed_y <- spline_fit_y$y
      
      # Interpolate colors
      smoothed_colors <- interpolate_colors_with_alpha(line_colors, length(smoothed_x))
      
      # Plot the smoothed spline in segments with interpolated colors
      for (i in 1:(length(smoothed_x) - 1)) {
        lines(smoothed_x[i:(i + 1)], smoothed_y[i:(i + 1)], col = smoothed_colors[i], lwd = line_width)
      }
    } else {
      # If not enough unique points for spline, plot original line
      for (i in 1:(length(x_coords) - 1)) {
        lines(x_coords[i:(i + 1)], y_coords[i:(i + 1)], col = line_colors[i], lwd = line_width)
      }
    }
  }

  # Function to add sections
  add_section <- function(section_name) {
    section_coords <- cell_coordinates[cell_coordinates$section == section_name, ]
    section_z <- z[cell_coordinates$section == section_name]
    section_colors <- point_colors[cell_coordinates$section == section_name]
    if (nrow(section_coords) > 0) {
      plot_segment(section_coords$x, section_coords$y, section_colors, lwd, smoothing, upsample_factor)
    }
  }

  # define dend_tree2
  dend_tree2 <- dend_tree[order(sapply(dend_tree, function(x) x[1]), -sapply(dend_tree, length))]

  # Function to extract unique elements for each unique first entry
  extract_unique_paths <- function(dend_tree) {
    unique_first_entries <- unique(sapply(dend_tree, function(x) x[1]))
    result <- list()
    for (first_entry in unique_first_entries) {
      branches_with_first_entry <- dend_tree[sapply(dend_tree, function(x) x[1] == first_entry)]
      longest_branch <- branches_with_first_entry[[which.max(sapply(branches_with_first_entry, length))]]
      result[[length(result) + 1]] <- longest_branch
    }
    return(result)
  }

  # get unique dendrites to dend_tree2 for smooth plots
  unique_paths <- extract_unique_paths(dend_tree2)


  # check if a path is in unique_paths
  is_in_unique_paths <- function(path, unique_paths) {
    for (unique_path in unique_paths) {
      if (identical(path, unique_path)) {
        return(TRUE)
      }
    }
    return(FALSE)
  }

  # remove unique_paths from dend_tree2
  dend_tree3 <- dend_tree2[!sapply(dend_tree2, is_in_unique_paths, unique_paths = unique_paths)]

  # remove common elements from dend_tree3 based on unique_paths and include the last common one
  remove_common_elements_with_last <- function(dend_tree3, unique_paths) {
    result <- list()
    
    for (path in dend_tree3) {
      # Get the first element of the current path
      first_element <- path[1]
      
      # find the corresponding unique path with the same first element
      unique_path <- unique_paths[sapply(unique_paths, function(up) up[1] == first_element)]
      
      # If a corresponding unique path exists, remove the common elements and include the last common one
      if (length(unique_path) > 0) {
        unique_path <- unique_path[[1]]
        common_indices <- which(path %in% unique_path)
        last_common_index <- max(common_indices)
        filtered_path <- path[!(path %in% unique_path[-last_common_index])]
        if (length(filtered_path) > 0) {
          result <- append(result, list(filtered_path))
        }
      } else {
        result <- append(result, list(path))
      }
    }
    
    return(result)
  }

  # remaining paths to plot
  other_paths <- remove_common_elements_with_last(dend_tree3, unique_paths)


  get_path_coordinates <- function(cell_coordinates, path) {
    path_coords <- cell_coordinates[cell_coordinates$section %in% path, ]
    return(path_coords)
  }

  processed_sections <- c()

  for (ii in 1:length(unique_paths)) {
    path <- unique_paths[[ii]]
    
    # Add primary sections from soma
    primary <- path[1]

    primary_coords <- cell_coordinates[cell_coordinates$section == primary, ]

    start_x <- primary_coords[1, 'x']
    start_y <- primary_coords[1, 'y']
    
    angle <- atan2(start_y - center_y, start_x - center_x)
    perimeter_x <- center_x + radius * cos(angle)
    perimeter_y <- center_y + radius * sin(angle)

    plot_segment(c(perimeter_x, start_x), c(perimeter_y, start_y), point_colors[cell_coordinates$section == primary], lwd, smoothing, upsample_factor)

    # Function to get section coordinates for all elements of a path

    path_coords <- get_path_coordinates(cell_coordinates, path)
    
    x_coords <- path_coords$x
    y_coords <- path_coords$y

    # Get the relevant 'point_colors'
    path_indices <- which(cell_coordinates$section %in% path)

    colors <- point_colors[path_indices]

    plot_segment(x_coords, y_coords, colors, lwd, smoothing, upsample_factor)
    processed_sections <- c(processed_sections, path)
  }

  if (length(other_paths) != 0){
    for (ii in 1:length(other_paths)) {
      path <- other_paths[[ii]]
      parent_section <- path[which(!path %in% processed_sections)[1] - 1]
      path2 <- setdiff(path, processed_sections)

      parent_section_coords <- get_path_coordinates(cell_coordinates, parent_section)
      path2_coords <- get_path_coordinates(cell_coordinates, path2)

      x_coords <- c(tail(parent_section_coords$x[-1], 1), path2_coords$x)
      y_coords <- c(tail(parent_section_coords$y[-1], 1), path2_coords$y)

      # Get the relevant 'point_colors'
      path_indices <- c(tail(which(cell_coordinates$section %in% parent_section)[-1],1), which(cell_coordinates$section %in% path2))


      plot_segment(x_coords, y_coords, point_colors[path_indices], lwd, smoothing, upsample_factor)

      processed_sections <- c(processed_sections, path2)

    }

  }

  # plot color bar (top right corner)
  color_bar_x_start <- xlim[2] - 20  # Start x coordinate for the color bar
  color_bar_x_end <- xlim[2] - 10    # End x coordinate for the color bar
  color_bar_y_start <- ylim[2] - 115   # start y coordinate for the color bar (20% of y axis)
  color_bar_y_end <- ylim[2] - 25   # End y coordinate for the color bar

  color_bar_z <- matrix(seq(min_val, max_val, length.out = 100), ncol = 1)
  image(x = seq(color_bar_x_start, color_bar_x_end, length = 2), 
        y = seq(color_bar_y_start, color_bar_y_end, length = 100), 
        z = t(color_bar_z), 
        col = color_palette, 
        add = TRUE)

  # add min and max labels to color bar
  text(x = color_bar_x_end + 2, y = color_bar_y_start, labels = round(min_val, 2), adj = c(0, 0.5))
  text(x = color_bar_x_end + 2, y = color_bar_y_end, labels = round(max_val, 2), adj = c(0, 0.5))

  # add x-axis scale bar in the lower right corner
  scale_bar_length <- xbar
  scale_bar_x_start <- xlim[2] - scale_bar_length - 10
  scale_bar_x_end <- xlim[2] - 10
  scale_bar_y <- ylim[1] + 15

  segments(x0 = scale_bar_x_start, y0 = scale_bar_y, x1 = scale_bar_x_end, y1 = scale_bar_y, col = 'black', lwd = lwd)
  text(x = (scale_bar_x_start + scale_bar_x_end) / 2, y = scale_bar_y - 5, labels = paste(scale_bar_length, '\u00B5m'), adj = c(0.5, 1))

  # close SVG device
  dev.off()
}

heat2D <- function(cell_coordinates, dend_tree, z, dend_name=NULL, title='heatmap', lwd=0.8, smoothing=NULL, upsample_factor=10, alpha=0.6, xlim=c(-175, 175), ylim=c(-175, 175), xbar=50, height=10, width=10, colorbar_reverse=FALSE, filename='', min_val=NULL, max_val=NULL){
  if (is.null(smoothing)){
      filename <- paste0(filename, '.svg') # paste0(gsub('[^a-zA-Z0-9]', '', title), '.svg')
      heat2D_1(cell_coordinates=cell_coordinates, dend_tree=dend_tree, z=z, dend_name=dend_name, title=title, lwd=lwd, upsample_factor=upsample_factor, alpha=alpha, xlim=xlim, ylim=ylim, xbar=xbar, height=height, width=width, colorbar_reverse=colorbar_reverse, filename=filename, min_val=min_val, max_val=max_val)
    }else{
      filename <- paste0(filename, '.svg') # paste0(gsub('[^a-zA-Z0-9]', '', title), '.svg')
      heat2D_2(cell_coordinates=cell_coordinates, dend_tree=dend_tree, z=z, dend_name=dend_name,  title=title, lwd=lwd, smoothing=smoothing, upsample_factor=upsample_factor, alpha=alpha, xlim=xlim, ylim=ylim, xbar=xbar, height=height, width=width, colorbar_reverse=colorbar_reverse, filename=filename, min_val=min_val, max_val=max_val)
    }
}

analysis <- function(V, bl, start_time, method = 'max'){
  # Calculate the burn time
  burn_time = start_time - bl

  # Determine the index range for analysis
  idx1 <- burn_time/dt
  idx2 <- dim(V)[1]

  # Subset V based on the calculated indices
  V2 = V[idx1:idx2,]

  # Determine the baseline period
  idx3 <- bl/dt

  # Calculate the mean of the baseline period
  V2_bl <- apply(V2[1:idx3,], 2, mean)

  # Subtract the baseline mean from the rest of the data
  V3 <- sapply(1:dim(V2)[2], function(ii) V2[,ii] - V2_bl[ii])

  # Calculate the maximum and minimum for absolute max for V3
  V.amplitudes <- apply(V3, 2, function(x) {
    max_val <- max(x)
    min_val <- min(x)
    if (abs(max_val) >= abs(min_val)) {
      return(max_val)
    } else {
      return(min_val)
    }
  })

  # Calculate either the maximum or minimum for V2 based on method
  if (method == 'max') {
    V.max <- apply(V2, 2, max)
  } else if (method == 'min') {
    V.max <- apply(V2, 2, min)
  } else {
    stop("Invalid method. Choose either 'max' or 'min'")
  }

  # Normalize V3 by the calculated amplitudes
  V4 <- sapply(1:dim(V3)[2], function(ii) V3[,ii] / V.amplitudes[ii])

  # Return the results as a list
  return(list(V0 = V3, V0_norm = V4, Vpeaks = V.amplitudes, Vmax = V.max))
}

get_path_coordinates <- function(cell_coordinates, path) {
    path_coords <- cell_coordinates[cell_coordinates$section %in% path, ]
    return(path_coords)
  }

# Define the start and end colors of your palette Slate Blue to Indian Red
hex_palette <- function(n, color1='#6A5ACD', color2='#CD5C5C', reverse = FALSE) {
  
  # Create a sequence of colors
  colors <- colorRampPalette(c(color1, color2))(n)
  
  # reverse colors if reverse=TRUE
  if (reverse) {
    colors <- rev(colors)
  }
  
  return(colors)
}

traces2plot <- function(V, traces, xlim=NULL, ylim=NULL, lwd=1, xbar=20, ybar=0.2, reverse=TRUE, show_text=FALSE, normalise=FALSE, dt=0.05){
  
  n <- length(traces)
  colors <- hex_palette(n=n, reverse=reverse)
  if (reverse) traces = rev(traces)

  x = 0:dt:(dim(V)[1]-1)*dt

  if (is.null(xlim)) xlim <- c(min(x), max(x))
  if (is.null(ylim)) ylim <- c(0, max(apply(V[, traces],2,max)))

  y <- V[, traces[1]]
  plot(x, y, type='l', col=colors[1], xlim=xlim, ylim=ylim, bty='n', lwd=lwd, lty=1, axes=FALSE, frame=FALSE, xlab = '', ylab = '')

  # Loop through remaining traces and add them to the plot
  for (i in 2:n) {
    y <- V[, traces[i]]
    lines(x, y, col=colors[i], lwd=lwd, lty=1)
  }

  # Define scale bar lengths and ybar position
  ybar_start <- ylim[1] - (ylim[1] - ylim[2]) / 10

  # Add scale bars at the bottom right
  x_start <- max(xlim) - xbar
  y_start <- ybar_start
  x_end <- x_start + xbar
  y_end <- y_start + ybar

  # Draw the scale bars
  segments(x_start, y_start, x_end, y_start, lwd=lwd, col='black') # Horizontal scale bar
  if (!normalise){
    segments(x_start, y_start, x_start, y_end, lwd=lwd, col='black') # Vertical scale bar
  }

  # Add labels to the scale bars
  if (show_text){
    text(x = (x_start + x_end) / 2, y = y_start - ybar / 20, labels = paste(xbar, 'ms'), adj = c(0.5, 1))
    if (!normalise) text(x = x_start -xbar/4, y = (y_start + y_end) / 2, labels = paste(ybar, 'mV'), adj = c(0.5, 0.5), srt = 90)
  }

}

plot1 <- function(filename, x, y_list, xlim=c(0,275), ylim=c(0,8), lwd=1, xlab='distance from soma center (\u00B5m)', ylab='PSP amplitude (mV)', height=3, width=3, xmajor_tick=100, ymajor_tick=4) {
  # Open SVG device with transparent background
  n <- length(y_list)
  colors <- hex_palette(n=n, reverse=FALSE)

  svg(file=filename, width=width, height=height, bg='transparent')  # set width and height in inches

  # Create an empty plot with the correct axes
  plot(NA, type='n', xlim=xlim, ylim=ylim, bty='n', axes=FALSE, xaxs='i', yaxs='i', xlab=xlab, ylab=ylab)

  # Loop through the y_list and plot each smooth spline
  for (ii in seq_along(y_list)) {
    # Fit a smooth spline to the data
    spline_fit <- smooth.spline(x, y_list[[ii]])

    # Add the smooth spline to the plot
    lines(spline_fit, col=colors[ii], lwd=lwd, lty=3)
  }

  # Add custom x-axis ticks
  axis(1, at=seq(0, xlim[2], by=xmajor_tick), tcl=-0.2)  # Major ticks
  axis(1, at=seq(0, xlim[2], by=xmajor_tick / 2), labels=FALSE, tcl=-0.1)  # Minor ticks

  # Add custom y-axis ticks with rotated labels
  axis(2, at=seq(ylim[1], ylim[2], by=ymajor_tick), las=2, tcl=-0.2)  # Major ticks
  axis(2, at=seq(ylim[1], ylim[2], by=ymajor_tick / 2), labels=FALSE, tcl=-0.1)  # Minor ticks

  # Close SVG device
  dev.off()
}

# cex controls point size; ds mmust be integer 1,2... downsamples for points plot retaining first and last
# axes_offset extends y axis in both directions by ycorrect * 100% to show full points if close to axes limits

plot2 <- function(filename, x_list, y_list, colors=NULL, xlim=c(0,275), ylim=c(0,70), lwd=1, xlab='distance from soma center (\u00B5m)', ylab='PSP amplitude (mV)', height=3, width=3, xmajor_tick=50, ymajor_tick=20, ds=3, cex=0.3, xaxes_offset=0.025, yaxes_offset=0.02) {
  # Extend axis range by given % in either direction
  x_range <- diff(xlim)
  xlim <- c(xlim[1] - xaxes_offset * x_range, xlim[2] + xaxes_offset * x_range)
 
  y_range <- diff(ylim)
  ylim <- c(ylim[1] - yaxes_offset * y_range, ylim[2] + yaxes_offset * y_range)
  
  # Open SVG device with transparent background
  n <- length(y_list)
  if (is.null(colors)) colors <- hex_palette(n=n, reverse=FALSE)
  
  svg(file=filename, width=width, height=height, bg='transparent') # set width and height in inches
  
  # Create an empty plot with the correct axes
  plot(NA, type='n', xlim=xlim, ylim=ylim, bty='n', axes=FALSE, xaxs='i', yaxs='i', xlab=xlab, ylab=ylab)
  
  # Loop through the y_list and plot each smooth spline
  for (ii in seq_along(y_list)) {
    # Downsample points
    x_points <- x_list[[ii]]
    y_points <- y_list[[ii]]
    n_points <- length(x_points)
    
    # Determine indices to plot
    if (n_points > 2) {
      indices <- c(1, seq(2, n_points - 1, by=ds), n_points)
    } else {
      indices <- 1:n_points
    }
    
    # Plot the points in the same color
    points(x_points[indices], y_points[indices], col=colors[ii], pch=16, cex=cex)
    
    # Fit a smooth spline to the data
    spline_fit <- smooth.spline(x_list[[ii]], y_list[[ii]])
    
    # Add the smooth spline to the plot
    lines(spline_fit, col=colors[ii], lwd=lwd, lty=3)
  }
  
  # Add custom x-axis ticks
  axis(1, at=seq(xlim[1] + xaxes_offset * x_range, xlim[2] - xaxes_offset * x_range, by=xmajor_tick), tcl=-0.2) 
  axis(1, at=seq(xlim[1] + xaxes_offset * x_range, xlim[2] - xaxes_offset * x_range, by=xmajor_tick / 2), labels=FALSE, tcl=-0.1) 
 
  # Add custom y-axis ticks with rotated labels
  axis(2, at=seq(ylim[1] + yaxes_offset * y_range, ylim[2] - yaxes_offset * y_range, by=ymajor_tick), las=2, tcl=-0.2) 
  axis(2, at=seq(ylim[1] + yaxes_offset * y_range, ylim[2] - yaxes_offset * y_range, by=ymajor_tick / 2), labels=FALSE, tcl=-0.1) 
  
  # Close SVG device
  dev.off()
}

