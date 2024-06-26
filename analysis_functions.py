'''
analysis functions
'''
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots       
import datetime 
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splprep, splev, interp1d
from collections import OrderedDict
import seaborn as sns
import matplotlib as mpl
import ipywidgets as widgets
from IPython.display import clear_output, display
import copy

def Rneck(l=1, d=0.1, Ri=200):
    l = l/1e4 ; d=d/1e4
    return (4 * Ri * l/np.pi/d**2)/1e6

def plot_v(X, Y, titles, colors, yname='', xname='distance from soma (µm)', xrange=[-2, 275], yrange=[0, 70], ab1=None, smooth=1, width=1000, height=600, points=True, ignore_first=False):
    
    fig = go.Figure()

    for i, (x, y, title, color) in enumerate(zip(X, Y, titles, colors), start=1):
        if x is not None and y is not None:
            # Plot the original data points
            if points:
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{title}', marker=dict(color=color)))
            
            # Exclude the first point for spline fitting if ignore_first is True
            x_for_spline, y_for_spline = (x[1:], y[1:]) if ignore_first else (x, y)

            # Perform spline fitting on the adjusted data
            sorted_pairs = sorted(zip(x_for_spline, y_for_spline))
            x_for_spline, y_for_spline = zip(*sorted_pairs)
            s = UnivariateSpline(x_for_spline, y_for_spline, s=smooth)
            xnew = np.linspace(min(x_for_spline), max(x_for_spline), 1000)
            ynew = s(xnew)

            # Plot the spline fit
            fig.add_trace(go.Scatter(x=xnew, y=ynew, mode='lines', line=dict(dash='dot', color=color), name=f'{title} spline fit'))

    if ab1 is not None:
        fig.add_shape(type="line", x0=ab1, x1=ab1, y0=yrange[0], y1=yrange[1], line=dict(color="gray", width=1, dash="dot"))

    fig.update_layout(
        title={'text': '', 'x': 0.5, 'xanchor': 'center'}, 
        xaxis=dict(
            title=xname, 
            range=xrange, 
            showline=True, 
            linewidth=1,
            linecolor='black', 
            ticks='outside',
            showgrid=False
        ), 
        yaxis=dict(
            title=yname, 
            range=yrange, 
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            ticks='outside',
            showgrid=False
        ),
        width=width,
        height=height,
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="v", 
            yanchor="auto", 
            y=1, 
            xanchor="left", 
            x=1.05
        ),
        margin=dict(r=150)  # Adjust the right margin to ensure enough space for the legend
    )

    return fig

def heat2D(cell_coordinates, dend_tree, z, dend_name=None, alpha=0.6, lwd=0.8, show_bar=True, title='', zmin=None, zmax=None, scheme='Jet', colorbar_reverse=False, height=600, width=600, scale_title=u'\u0394V (mV)', exponent=1, s=None):
    if s is None:
        fig = heat2D_1(cell_coordinates=cell_coordinates, dend_tree=dend_tree, z=z, dend_name=dend_name, alpha=alpha, lwd=lwd, show_bar=show_bar, title=title, zmin=zmin, zmax=zmax, scheme=scheme, colorbar_reverse=colorbar_reverse, height=height, width=width, scale_title=scale_title, exponent=exponent,)
    else:
        fig = heat2D_2(cell_coordinates=cell_coordinates, dend_tree=dend_tree, z=z, dend_name=dend_name, alpha=alpha, lwd=lwd, show_bar=show_bar, title=title, zmin=zmin, zmax=zmax, scheme=scheme, colorbar_reverse=colorbar_reverse, height=height, width=width, scale_title=scale_title, exponent=exponent, s=s)
    return fig

def heat2D_1(cell_coordinates, dend_tree, z, dend_name=None, alpha=0.6, lwd=0.8, show_bar=True, title='', zmin=None, zmax=None, scheme='Jet', colorbar_reverse=False, height=600, width=600, scale_title=u'\u0394V (mV)', exponent=1):
    jet_rgb = [(0, 0, 127), (0, 0, 255), (0, 127, 255), (0, 255, 255), (127, 255, 127), (255, 255, 0), (255, 127, 0), (255, 0, 0), (127, 0, 0)]
    viridis_rgb = [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (41, 129, 142), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)]
    
    if scheme == 'Jet':
        cbar = jet_rgb
    elif scheme == 'Viridis':
        cbar = viridis_rgb
    
    if colorbar_reverse:
        cbar = cbar[::-1]
        scheme = scheme + '_r'
    
    def custom_transform(z, exponent=1):
        return np.power(z, exponent)

    if zmin is None: zmin = min(z)
    if zmax is None: zmax = max(z)
    z1 = custom_transform(np.array([zmin, zmax]), exponent)
    z_min, z_max = np.min(z1), np.max(z1)

    def get_color(value, colormap, alpha=0.6, min_value=None, max_value=None):
        scaled_value = (value - min_value) / (max_value - min_value)        
        max_idx = len(colormap) - 1
        scaled_value *= max_idx
        scaled_value = min(max(scaled_value, 0), max_idx)
        idx1 = int(scaled_value)
        idx2 = min(idx1 + 1, max_idx)
        fraction = scaled_value - idx1
        r = colormap[idx1][0] + fraction * (colormap[idx2][0] - colormap[idx1][0])
        g = colormap[idx1][1] + fraction * (colormap[idx2][1] - colormap[idx1][1])
        b = colormap[idx1][2] + fraction * (colormap[idx2][2] - colormap[idx1][2])
        return f'rgba({int(r)}, {int(g)}, {int(b)}, {alpha})'

    fig = go.Figure()

    simplified_dend_tree = [sub_branch for branch in dend_tree for sub_branch in (branch if isinstance(branch[0], list) else [branch])]

    def find_sublists_with_target(simplified_dend_tree, dend_name):
        result = []
        for path in simplified_dend_tree:
            if dend_name in path if isinstance(path, str) else dend_name in [str(d) for d in path]:  
                result.append(path)
        return result

    if dend_name is not None:
        simplified_dend_tree = find_sublists_with_target(simplified_dend_tree, dend_name)

    connections = {}
    for branch in simplified_dend_tree:
        for i in range(len(branch) - 1):
            sec_name = branch[i].name()
            next_sec_name = branch[i + 1].name()
            if sec_name not in connections:
                connections[sec_name] = set()
            connections[sec_name].add(next_sec_name)

    coord_to_z = {tuple(cell_coordinates[i]): z[i] for i in range(len(z))}
    processed_sections = set()

    soma_coords = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]']
    x_coords = [coord[2] for coord in soma_coords]
    y_coords = [coord[3] for coord in soma_coords]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    radius = sum([coord[6] for coord in soma_coords]) / len(soma_coords) / 2

    soma_z = coord_to_z.get(tuple(soma_coords[0]), 0)
    soma_color = get_color(soma_z, cbar, alpha, z_min, z_max)
    hover_text_soma = f'soma[0] dist: {soma_coords[0, 5]:.2f}, mV: {soma_z:.2f}'

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=center_x - radius, y0=center_y - radius,
                  x1=center_x + radius, y1=center_y + radius,
                  line=dict(color='rgba(0,0,0,0)', width=0),
                  fillcolor=soma_color
                  )
    
    fig.add_trace(go.Scatter(
        x=[center_x], y=[center_y],
        mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        text=hover_text_soma,
        hoverinfo='text',
        hoverlabel=dict(bgcolor="darkgrey", font=dict(color="white"))
    ))

    primaries = []
    for tree in simplified_dend_tree:
        primaries.append(tree[0].name())
    primaries = list(set(primaries))

    def point_on_circle(cx, cy, angle, radius):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        return x, y

    for primary in primaries:
        section_coords = cell_coordinates[cell_coordinates[:, 0] == primary]
        if section_coords.size > 0:
            start_x = section_coords[0, 2]
            start_y = section_coords[0, 3]

            angle = np.arctan2(start_y - center_y, start_x - center_x)
            perimeter_x, perimeter_y = point_on_circle(center_x, center_y, angle, radius)

            primary_index = np.where(cell_coordinates[:, 0] == primary)[0][0]
            primary_z = z[primary_index]
            primary_color = get_color(primary_z, cbar, alpha, z_min, z_max)
            hover_texts = f'{primary}, dist: {section_coords[0, 5]:.2f}, mV: {primary_z:.2f}'

            fig.add_trace(go.Scatter(
                x=[perimeter_x, start_x],
                y=[perimeter_y, start_y],
                mode='lines',
                name=primary,
                line=dict(color=primary_color, width=lwd),
                text=hover_texts,
                hoverinfo='text',
                hoverlabel=dict(bgcolor="darkgrey", font=dict(color="white"))
            ))

            x_coords = [coord[2] for coord in section_coords]
            y_coords = [coord[3] for coord in section_coords]
            hover_texts = [f'{primary} dist: {coord[5]:.2f}, mV: {coord_to_z[tuple(coord)]:.2f}' for coord in section_coords]
            for i in range(len(x_coords) - 1):
                color = get_color(coord_to_z[tuple(section_coords[i])], cbar, alpha, z_min, z_max)
                fig.add_trace(go.Scatter(
                    x=[x_coords[i], x_coords[i+1]], 
                    y=[y_coords[i], y_coords[i+1]], 
                    mode='lines',
                    name=primary, 
                    line=dict(color=color, width=lwd),
                    text=hover_texts[i],
                    hoverinfo='text',
                    hoverlabel=dict(bgcolor="darkgrey", font=dict(color="white"))
                ))

    for parent in connections.keys():
        section_coords = cell_coordinates[cell_coordinates[:, 0] == parent]
        if len(section_coords) == 0:
            continue
            
        parent_end_x = section_coords[-1, 2]
        parent_end_y = section_coords[-1, 3]
    
        children = connections[parent]
        for child in children:
            if child in processed_sections:
                continue
    
            section_coords = cell_coordinates[cell_coordinates[:, 0] == child]
            if len(section_coords) == 0:
                continue
            x_coords = [coord[2] for coord in section_coords]
            y_coords = [coord[3] for coord in section_coords]
            distances = [coord[5] for coord in section_coords]
            child_z = [coord_to_z.get(tuple(section), 0) for section in section_coords]
            hover_texts = [f'{child} dist: {dist:.2f}, mV: {z:.2f}' for dist, z in zip(distances, child_z)]
    
            start_x = section_coords[0, 2]
            start_y = section_coords[0, 3]
            color = get_color(child_z[0], cbar, alpha, z_min, z_max)
    
            fig.add_trace(go.Scatter(
                x=[parent_end_x, start_x],
                y=[parent_end_y, start_y],
                mode='lines',
                name=parent,
                line=dict(color=color, width=lwd),
                hoverlabel=dict(bgcolor="darkgrey", font=dict(color="white"))
            ))
    
            for i in range(len(section_coords) - 1):
                start_x = section_coords[i, 2]
                start_y = section_coords[i, 3]
                end_x = section_coords[i + 1, 2]
                end_y = section_coords[i + 1, 3]
                color = get_color(child_z[i], cbar, alpha, z_min, z_max)
                hover_text = hover_texts[i]
    
                fig.add_trace(go.Scatter(
                    x=[start_x, end_x],
                    y=[start_y, end_y],
                    mode='lines',
                    name=child,
                    line=dict(color=color, width=lwd),
                    text=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(bgcolor="darkgrey", font=dict(color="white"))
                ))
    
            processed_sections.add(child)
    
        processed_sections.add(parent)

    if show_bar:
        cbar_rgba = [(i / (len(cbar) - 1), get_color((i / (len(cbar) - 1)) * (z_max - z_min) + z_min, cbar, alpha, z_min, z_max)) for i in range(len(cbar))]
        colorbar_trace = go.Scatter(
            x=[None], 
            y=[None], 
            mode='markers',
            marker=dict(
                colorscale=cbar_rgba,
                cmin=z_min,
                cmax=z_max,
                color=z1,
                colorbar=dict(
                    title=scale_title,
                    thickness=10
                ),
                showscale=True
            ),
            hoverinfo='none'
        )
        fig.add_trace(colorbar_trace)
        
    fig.update_layout(
        title=title,
        title_x=0.5,
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(
            range=[-175,175],
            constrain='domain',  
            showline=False,
            zeroline=False,
            showticklabels=True,
            showgrid=True,
            dtick=50
        ),
        yaxis=dict(
            range=[-175,175],
            scaleanchor="x",
            scaleratio=1,
            showline=False,
            zeroline=False,
            showticklabels=True,
            showgrid=True, 
            dtick=50
        ),
        showlegend=False
    )
    return fig

def heat2D_2(cell_coordinates, dend_tree, z, dend_name=None, alpha=0.6, lwd=0.8, show_bar=True, title='', zmin=None, zmax=None, scheme='Jet', colorbar_reverse=False, height=600, width=600, scale_title=u'\u0394V (mV)', exponent=1, s=2):
    jet_rgb = [(0, 0, 127), (0, 0, 255), (0, 127, 255), (0, 255, 255), (127, 255, 127), (255, 255, 0), (255, 127, 0), (255, 0, 0), (127, 0, 0)]
    viridis_rgb = [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (41, 129, 142), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)]
    
    if scheme == 'Jet':
        cbar = jet_rgb
    elif scheme == 'Viridis':
        cbar = viridis_rgb
    
    if colorbar_reverse:
        cbar = cbar[::-1]
        scheme = scheme + '_r'
    
    def custom_transform(z, exponent=1):
        return np.power(z, exponent)

    if zmin is None: zmin = min(z)
    if zmax is None: zmax = max(z)
    z1 = custom_transform(np.array([zmin, zmax]), exponent)
    z_min, z_max = np.min(z1), np.max(z1)

    def get_color(value, colormap, alpha=0.6, min_value=None, max_value=None):
        scaled_value = (value - min_value) / (max_value - min_value)        
        max_idx = len(colormap) - 1
        scaled_value *= max_idx
        scaled_value = min(max(scaled_value, 0), max_idx)
        idx1 = int(scaled_value)
        idx2 = min(idx1 + 1, max_idx)
        fraction = scaled_value - idx1
        r = colormap[idx1][0] + fraction * (colormap[idx2][0] - colormap[idx1][0])
        g = colormap[idx1][1] + fraction * (colormap[idx2][1] - colormap[idx1][1])
        b = colormap[idx1][2] + fraction * (colormap[idx2][2] - colormap[idx1][2])
        return f'rgba({int(r)}, {int(g)}, {int(b)}, {alpha})'

    fig = go.Figure()

    simplified_dend_tree = [sub_branch for branch in dend_tree for sub_branch in (branch if isinstance(branch[0], list) else [branch])]

    def find_sublists_with_target(simplified_dend_tree, dend_name):
        result = []
        for path in simplified_dend_tree:
            if dend_name in path if isinstance(path, str) else dend_name in [str(d) for d in path]:  
                result.append(path)
        return result

    if dend_name is not None:
        simplified_dend_tree = find_sublists_with_target(simplified_dend_tree, dend_name)

    sorted_dend_tree = sorted(dend_tree, key=lambda sublist: (-len(sublist) if isinstance(sublist, list) else -1, [len(item) if isinstance(item, list) else 0 for item in sublist] if isinstance(sublist, list) else []))
    sorted_dend_tree = [sorted(sublist, key=lambda item: len(item) if isinstance(item, list) else 0, reverse=True) if isinstance(sublist, list) else sublist for sublist in sorted_dend_tree]

    def extract_dend_lists(dend_tree, dend_name):
        def contains_dend_name(subtree, name):
            if isinstance(subtree, list):
                return any(contains_dend_name(item, name) for item in subtree)
            return subtree.name() == name
        
        extracted_lists = []
        for sublist in dend_tree:
            filtered_paths = [path for path in sublist if contains_dend_name(path, dend_name)]
            if filtered_paths:
                extracted_lists.append(filtered_paths)
        
        return extracted_lists
    
    if dend_name is not None:
        sorted_dend_tree = extract_dend_lists(dend_tree=sorted_dend_tree, dend_name=dend_name)

    connections = {}
    for branch in simplified_dend_tree:
        for i in range(len(branch) - 1):
            sec_name = branch[i].name()
            next_sec_name = branch[i + 1].name()
            if sec_name not in connections:
                connections[sec_name] = set()
            connections[sec_name].add(next_sec_name)

    coord_to_z = {tuple(cell_coordinates[i]): z[i] for i in range(len(z))}
    processed_sections = set()

    soma_coords = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]']
    x_coords = [coord[2] for coord in soma_coords]
    y_coords = [coord[3] for coord in soma_coords]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    radius = sum([coord[6] for coord in soma_coords]) / len(soma_coords) / 2

    soma_z = coord_to_z.get(tuple(soma_coords[0]), 0)
    soma_color = get_color(soma_z, cbar, alpha, z_min, z_max)
    hover_text_soma = f'soma[0] dist: {soma_coords[0, 5]:.2f}, mV: {soma_z:.2f}'

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=center_x - radius, y0=center_y - radius,
                  x1=center_x + radius, y1=center_y + radius,
                  line=dict(color='rgba(0,0,0,0)', width=0),
                  fillcolor=soma_color
                  )
    
    fig.add_trace(go.Scatter(
        x=[center_x], y=[center_y],
        mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        text=hover_text_soma,
        hoverinfo='text',
        hoverlabel=dict(bgcolor="darkgrey", font=dict(color="white"))
    ))

    primaries = []
    for tree in simplified_dend_tree:
        primaries.append(tree[0].name())
    primaries = list(set(primaries))

    def point_on_circle(cx, cy, angle, radius):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        return x, y

    for primary in primaries:
        section_coords = cell_coordinates[cell_coordinates[:, 0] == primary]
        if section_coords.size > 0:
            start_x = section_coords[0, 2]
            start_y = section_coords[0, 3]

            angle = np.arctan2(start_y - center_y, start_x - center_x)
            perimeter_x, perimeter_y = point_on_circle(center_x, center_y, angle, radius)

            primary_index = np.where(cell_coordinates[:, 0] == primary)[0][0]
            primary_z = z[primary_index]
            primary_color = get_color(primary_z, cbar, alpha, z_min, z_max)
            hover_texts = f'{primary}, dist: {section_coords[0, 5]:.2f}, mV: {primary_z:.2f}'

            fig.add_trace(go.Scatter(
                x=[perimeter_x, start_x],
                y=[perimeter_y, start_y],
                mode='lines',
                name=primary,
                line=dict(color=primary_color, width=lwd),
                text=hover_texts,
                hoverinfo='text',
                hoverlabel=dict(bgcolor="darkgrey", font=dict(color="white"))
            ))
    
    def smooth_and_plot(coords, values):
        x_coords = [coord[2] for coord in coords]
        y_coords = [coord[3] for coord in coords]
        dists = [coord[5] for coord in coords]
        names = [coord[0] for coord in coords]
    
        if len(x_coords) > 3:
            # Interpolate coordinates
            tck, u = splprep([x_coords, y_coords], s=s)
            u_fine = np.linspace(0, 1, 100)
            x_smooth, y_smooth = splev(u_fine, tck)
    
            # Interpolate distances
            dist_tck, dist_u = splprep([dists], s=s)
            distances = splev(u_fine, dist_tck)[0]
    
            # Ensure values and u are of the same length
            values = np.array(values)
            if len(u) != len(values):
                u = np.linspace(0, 1, len(values))
    
            # Interpolate values using linear interpolation for z-values
            interpolated_values = np.interp(u_fine, u, values)
    
            # Map names to indices
            name_to_index = {name: idx for idx, name in enumerate(names)}
            indices = [name_to_index[name] for name in names]
    
            if len(u) != len(indices):
                u = np.linspace(0, 1, len(indices))
    
            # Interpolate names
            index_interpolator = interp1d(u, indices, kind='nearest', fill_value="extrapolate")
            interpolated_indices = index_interpolator(u_fine)
            interpolated_indices = np.round(interpolated_indices).astype(int)
            interpolated_names = [names[idx] for idx in interpolated_indices]
    
            # Prepare hover texts
            hover_texts = [f'{name}, dist: {dist:.2f}, mV: {value:.2f}' for name, dist, value in zip(interpolated_names, distances, interpolated_values)]
    
            # Plot the interpolated points as segments with corresponding colors
            for i in range(len(x_smooth) - 1):
                color = get_color(interpolated_values[i], cbar, alpha, z_min, z_max)
                fig.add_trace(go.Scatter(
                    x=[x_smooth[i], x_smooth[i + 1]],
                    y=[y_smooth[i], y_smooth[i + 1]],
                    mode='lines',
                    line=dict(color=color, width=lwd),
                    text=[hover_texts[i]],
                    hoverinfo='text'
                ))
        else:
            hover_texts = [f'{name}, dist: {dist:.2f}, mV: {value:.2f}' for name, dist, value in zip(names, dists, values)]
            for i in range(len(x_coords) - 1):
                color = get_color(values[i], cbar, alpha, z_min, z_max)
                fig.add_trace(go.Scatter(
                    x=[x_coords[i], x_coords[i + 1]],
                    y=[y_coords[i], y_coords[i + 1]],
                    mode='lines',
                    line=dict(color=color, width=lwd),
                    text=[hover_texts[i]],
                    hoverinfo='text'
                ))
                
    for path_group in sorted_dend_tree:
        seen_sections = set()
        new_paths = []

        for iii, path in enumerate(path_group):
            new_path = []
            if not isinstance(path, list):
                path = [path]

            for ii, sec in enumerate(path):
                if sec not in seen_sections:
                    if ii > 0:
                        new_path.append(path[ii - 1])
                    new_path.append(sec)
                    seen_sections.add(sec)

            if new_path:
                filtered_path = []
                last_sec = None
                for j, sec in enumerate(new_path):
                    if sec != last_sec:
                        filtered_path.append(sec)
                        last_sec = sec
                new_paths.append(filtered_path)

                if not isinstance(filtered_path, list):
                    filtered_path = [filtered_path]

                coords = []
                values = []
               
                # if not main path then selected the path for plotting includes the parent of that branch
                # it takes the last two point of the parent to 'join' the child to its parent 
                for sec in filtered_path:
                    name = sec.name()
                    cell_coords = cell_coordinates[cell_coordinates[:, 0] == name]
                    coords.append(cell_coords)
                    values.extend([coord_to_z[tuple(coord)] for coord in cell_coords])
                
                combined_coords = np.vstack(coords)
                combined_values = values.copy()
                
                if iii > 0 and len(filtered_path) > 1:
                    first_sec_name = filtered_path[0].name()
                    first_sec_index = np.where(combined_coords[:, 0] == first_sec_name)[0]
                    if len(first_sec_index) > 2:
                        combined_coords = np.vstack([combined_coords[first_sec_index[-2:]], combined_coords[combined_coords[:, 0] != first_sec_name]])
                        combined_values = [combined_values[i] for i in first_sec_index[-2:]] + [combined_values[i] for i in range(first_sec_index[-1] + 1, len(combined_values))]
                    else:
                        combined_coords = np.vstack([combined_coords[first_sec_index], combined_coords[combined_coords[:, 0] != first_sec_name]])
                        combined_values = [combined_values[i] for i in first_sec_index] + [combined_values[i] for i in range(first_sec_index[-1] + 1, len(combined_values))]
                                

                smooth_and_plot(combined_coords, combined_values)
    
    if show_bar:
        cbar_rgba = [(i / (len(cbar) - 1), get_color((i / (len(cbar) - 1)) * (z_max - z_min) + z_min, cbar, alpha, z_min, z_max)) for i in range(len(cbar))]
        colorbar_trace = go.Scatter(
            x=[None], 
            y=[None], 
            mode='markers',
            marker=dict(
                colorscale=cbar_rgba,
                cmin=z_min,
                cmax=z_max,
                color=z1,
                colorbar=dict(
                    title=scale_title,
                    thickness=10
                ),
                showscale=True
            ),
            hoverinfo='none'
        )
        fig.add_trace(colorbar_trace)
        
    fig.update_layout(
        title=title,
        title_x=0.5,
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(
            range=[-175, 175],
            constrain='domain',  
            showline=False,
            zeroline=False,
            showticklabels=True,
            showgrid=True,
            dtick=50
        ),
        yaxis=dict(
            range=[-175, 175],
            scaleanchor="x",
            scaleratio=1,
            showline=False,
            zeroline=False,
            showticklabels=True,
            showgrid=True, 
            dtick=50
        ),
        showlegend=False
    )
    return fig

def heatmap2D(dend_tree, cell_coordinates, z, dend_name=None, alpha=0.8, lwd=0.8, show_bar=True, title='', zmin=0, zmax=70, scheme='Jet', colorbar_reverse=False, width=600, height=600, scale_title=u'\u0394V (mV)', exponent=1, absolute=True, ):
    
    fig = go.Figure()
    
    # Initialize a flag to indicate presence of "soma" entries
    contains_soma = False

    # Loop through each entry in cell_coordinates
    for entry in cell_coordinates:
        # Check if the first element of the entry starts with "soma"
        if entry[0].startswith('soma'):
            contains_soma = True
            break  # Exit loop as soon as a "soma" entry is found

    # Define the 'Jet' colorscale as list of RGB tuples
    jet_rgb = [(0, 0, 127), (0, 0, 255), (0, 127, 255), (0, 255, 255), (127, 255, 127), (255, 255, 0), (255, 127, 0), (255, 0 , 0), (127, 0, 0)]
    # Define the 'viridis' colorscale as list of RGB tuples
    viridis_rgb = [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (41, 129, 142), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)]
    
    if scheme == 'Jet':
        cbar = jet_rgb
    elif scheme == 'Viridis':
        cbar = viridis_rgb
    
    if colorbar_reverse:
        cbar = cbar[::-1]
        scheme = scheme + '_r'
        
    def interpolate_color(val, val_min, val_max, colorscale, alpha=0.5):
        # Normalize val to [0, 1]
        norm_val = (val - val_min) / (val_max - val_min)
        norm_val = max(0, min(norm_val, 1))  # Clamp norm_val to the [0, 1] range to avoid extrapolation

        # Determine color index
        scale_len = len(colorscale) - 1
        idx_float = norm_val * scale_len
        idx_int = int(idx_float)

        # Interpolate between the two nearest colors
        color_low = np.array(colorscale[idx_int])
        color_high = np.array(colorscale[min(idx_int + 1, scale_len)])
        color = color_low + (color_high - color_low) * (idx_float - idx_int)

        # Clamp RGB values to the [0, 255] range to ensure valid colors
        color = np.clip(color, 0, 255)

        return f'rgba({int(color[0])}, {int(color[1])}, {int(color[2])}, {alpha})'

    # Determine min and max z values for the colorscale
    # exponents < 1 will increase scale sensitivity at higher voltages
    # if exponent = 1  then linear
    def custom_transform(z, exponent=1):
        return np.power(z, exponent)

    z1 = custom_transform(np.array([zmin, zmax]), exponent)
    z_min, z_max = np.min(z1), np.max(z1)

    def find_sublists_with_target(simplified_dend_tree, dend_name):
        # Initialize a list to hold sublists containing the target string
        result = []

        # Iterate through each sublist in simplified_dend_tree
        for path in simplified_dend_tree:
            if dend_name in path if isinstance(path, str) else dend_name in [str(d) for d in path]:  
                result.append(path)

        return result

    simplified_dend_tree = [sub_branch for branch in dend_tree for sub_branch in (branch if isinstance(branch[0], list) else [branch])]
    
    if dend_name is not None:
        simplified_dend_tree = find_sublists_with_target(simplified_dend_tree, dend_name)

    if dend_name is None:
        section_names = list(set(item[0] for item in cell_coordinates))
        section_names.reverse()  
    else:
        unique_vector = list(set(item for sublist in simplified_dend_tree for item in sublist))
        section_names = []
        for sec in unique_vector:
            section_names.append(sec.name())

    # Iterate through sections and smooth results using splines
    for sec_name in section_names:
        if 'dend' in sec_name:
            # Extract all coordinates for the current section
            section_coords = [item for item in cell_coordinates if item[0] == sec_name]
            x_coords = np.array([coord[2] for coord in section_coords])
            y_coords = np.array([coord[3] for coord in section_coords])

            # Check if the section has enough points for spline fitting
            if len(x_coords) > 3:  # Assuming cubic spline (k=3)
                try:
                    # Fit the spline to the x and y coordinates of the section
                    tck, u = splprep([x_coords, y_coords], s=2)  # s is the smoothing factor

                    # Evaluate the spline on a finer grid (more points means smoother)
                    new_points = splev(np.linspace(0, 1, 200), tck)

                    # Interpolate z values for the new points
                    new_z_values = np.interp(np.linspace(0, 1, 200), u, [z[cell_coordinates.index(coord)] for coord in section_coords])

                    # Draw the smooth line for the dendrite
                    for i in range(len(new_points[0]) - 1):
                        segment_color = interpolate_color(new_z_values[i], z_min, z_max, cbar, alpha=alpha)
                        fig.add_trace(go.Scatter(
                            x=new_points[0][i:i+2],
                            y=new_points[1][i:i+2],
                            mode='lines',
                            line=dict(color=segment_color, width=lwd),
                            hoverinfo='text',
                            text=f'{sec_name}<br>z: {new_z_values[i]:.2f}',
                            showlegend=False
                        ))
                except TypeError:
                    # catches errors from splprep when there are not enough points to create a spline
                    pass

            # Draw linear segments for sections with too few points or if spline fitting failed
            for i in range(len(x_coords) - 1):
                segment_color = interpolate_color(z[cell_coordinates.index(section_coords[i])], z_min, z_max, cbar, alpha=alpha)
                fig.add_trace(go.Scatter(
                    x=x_coords[i:i+2],
                    y=y_coords[i:i+2],
                    mode='lines',
                    line=dict(color=segment_color, width=lwd),
                    hoverinfo='text',
                    text=f'{sec_name}<br>z: {z[cell_coordinates.index(section_coords[i])]:.2f}',
                    showlegend=False
                ))


    def get_coords_and_z(dend_name, cell_coordinates, z_values, position='start'):
        # Filter coordinates for given dendrite name
        filtered_coords = [item for item in cell_coordinates if item[0] == dend_name]

        if not filtered_coords:
            return None, None, None  # Return None if no coordinates are found

        if position == 'end':
            chosen_coords = filtered_coords[-1]  # Get the last coordinates for the end
        else:
            chosen_coords = filtered_coords[0]   # Get the first coordinates for the start

        return chosen_coords[2], chosen_coords[3], z_values[cell_coordinates.index(chosen_coords)]


    def draw_connections(tree_map, cell_coordinates, z_values, fig, cbar, z_min, z_max, alpha=1.0, lwd=2):
        drawn_connections = set()  # track already drawn connections by names

        def draw_line(coords1, coords2, name1, name2, hover_text):
            connection_key = frozenset({name1, name2})

            if connection_key not in drawn_connections:  # Check if this connection has already been drawn
                z1, z2 = coords1[2], coords2[2]
                color = interpolate_color((z1 + z2) / 2, z_min, z_max, cbar, alpha)
                hover_texts = [hover_text] * 2
                fig.add_trace(go.Scatter(
                    x=[coords1[0], coords2[0]],
                    y=[coords1[1], coords2[1]],
                    mode='lines',
                    line=dict(color=color, width=lwd),
                    hoverinfo='text',
                    text=hover_texts,
                    showlegend=False
                ))
                drawn_connections.add(connection_key)  # marked as drawn

        def process_branch(branch, parent_coords, parent_name=None):
            for index, item in enumerate(branch):
                if isinstance(item, list):  # Sub-branch
                    for sub_item in item:
                        dend_name = f'{sub_item}'
                        # For the parent, get the 'end' coordinates; for the child, get the 'start'
                        child_coords = get_coords_and_z(dend_name, cell_coordinates, z_values, position='start')
                        if parent_coords and child_coords[0] is not None:
                            hover_text = f"{parent_name} to {dend_name}" if parent_name else ""
                            draw_line(parent_coords, child_coords, parent_name, dend_name, hover_text)
                        # Update parent_coords to the end of the current dendrite for the next iteration
                        parent_coords = get_coords_and_z(dend_name, cell_coordinates, z_values, position='end')
                        parent_name = dend_name

                else:  # Direct connection within the branch
                    dend_name = f'{item}'
                    child_coords = get_coords_and_z(dend_name, cell_coordinates, z_values, position='start')
                    if parent_coords and child_coords[0] is not None:
                        hover_text = f"{parent_name} to {dend_name}" if parent_name else ""
                        draw_line(parent_coords, child_coords, parent_name, dend_name, hover_text)
                    parent_coords = get_coords_and_z(dend_name, cell_coordinates, z_values, position='end')
                    parent_name = dend_name


        if contains_soma:
            soma_coords = get_coords_and_z('soma[0]', cell_coordinates, z_values)
            soma_name = 'soma[0]'
            for branch in tree_map:
                process_branch(branch, soma_coords, soma_name)

    draw_connections(simplified_dend_tree, cell_coordinates, z, fig, cbar, z_min, z_max, alpha=alpha, lwd=lwd)

    # Process the soma separately
    if contains_soma:
        soma_item = next((item for item in cell_coordinates if 'soma' in item[0]), None)
        if soma_item:
            soma_x, soma_y, soma_z, soma_radius = soma_item[2], soma_item[3], soma_item[4], soma_item[6] / 2
            soma_z_value = z[cell_coordinates.index(soma_item)]
            soma_color = interpolate_color(soma_z_value, z_min, z_max, cbar, alpha=1)

            # Add a circle for the soma
            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=soma_x - soma_radius, y0=soma_y - soma_radius,
                          x1=soma_x + soma_radius, y1=soma_y + soma_radius,
                          line=dict(color=soma_color, width=0),
                          fillcolor=soma_color)


    # dummy trace for color scale bar
    show_bar = True
    if show_bar:
        scale_title = scale_title
        # Create a dummy scatter trace for the color bar, using original z values
        colorbar_trace = go.Scatter(
            x=[None], 
            y=[None], 
            mode='markers',
            marker=dict(
                colorscale=scheme,  
                cmin=zmin,  
                cmax=zmax,  
                color=z1,  
                colorbar=dict(
                    title=scale_title,
                    thickness=10  # Adjust the thickness of the color bar here
                ),
                showscale=True
            ),
            hoverinfo='none'
        )

        fig.add_trace(colorbar_trace)

    # layout
    fig.update_layout(
    #     paper_bgcolor='rgba(0,0,0,0)',  
    #     plot_bgcolor='rgba(0,0,0,0)',  
        title=title,
        title_x=0.5,  # Centers the title
        autosize=False,
        width=width,  
        height=height,
        xaxis=dict(
            range=[-175,175],
            constrain='domain',  
            showline=False,
            zeroline=False,
            showticklabels=True,
            showgrid=True,
            dtick=50
        ),
        yaxis=dict(
            range=[-175,175],
            scaleanchor="x",
            scaleratio=1,
            showline=False,
            zeroline=False,
            showticklabels=True,
            showgrid=True, 
            dtick=50
        ),
        showlegend=False  
    )

    return fig

# remove offsets
def normalise2(X, stim_time, burn_time, dt):    
    def mean(x):
        n = len(x)
        sum = 0
        for i in x:
            sum = sum + i
        return(sum/n)
    ind1 = int(burn_time/dt)
    ind2 = int(stim_time/dt)
    bl = mean(X[ind1:ind2])
    Vpeak = X[ind1:len(X)]
    dVpeak = Vpeak - bl
    return Vpeak, dVpeak, bl

def peaks(zdata, start_time, burn_time, dt, peak_type='max'):
    # normalise tree data
    z_all = []
    dz_all = []
    baseline = []
    for z in zdata:  # Assuming zdata is the correct variable name
        zpeak, dzpeak, bl = normalise2(z, start_time, burn_time, dt) 
        z_all.append(zpeak)  # Fixed variable names from vpeak to zpeak
        dz_all.append(dzpeak)  # Fixed variable names from dvpeak to dzpeak
        baseline.append(bl)

    # find peak values
    peak_z = []; peak_dz = []
    for z in z_all:
        if peak_type == 'max':
            peak_z.append(z.max())
        elif peak_type == 'min':
            peak_z.append(z.min())

    for dz in dz_all:  # Fixed variable name from da_all to dz_all
        if peak_type == 'max':
            peak_dz.append(dz.max())
        elif peak_type == 'min':
            peak_dz.append(dz.min())

    return peak_z, peak_dz, baseline

def plot3D(x, y, df, dendrites=None, x_range=None, y_range=None, z_range=None, xaxis_title='time (ms)', 
           yaxis_title='distance (µm)', zaxis_title='E<sub>m</sub> (mV)', title='', colorscale='jet', 
           showscale=False, scene_camera=dict(eye=dict(x=2, y=-2, z=1)), width=1000, height=1000, 
           opacity=0.4, cmin=-85, cmax=-20, remove_text=False):
   
    y = np.array(y)
    # df is DataFrame, and x and y are vectors
    X, Y = np.meshgrid(x, y)  # create meshgrid
    Z = df.values.T  # transpose df to align with the meshgrid

    # Create a figure
    fig = go.Figure()
    # Add the surface plot
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=colorscale, showscale=showscale, opacity=opacity, cmin=cmin, cmax=cmax))

    # Add the line plots
    for i, column in enumerate(df.columns):
        fig.add_trace(go.Scatter3d(x=x, y=[y[i]]*len(x), z=df[column],
                                   mode='lines', name=f'Line {column}',
                                   line=dict(color='lightgrey', width=2)))

    # Prepare x1, y1, z1 arrays
    x1 = np.tile(round(max(x),2)+10, len(y))
    y1 = y 
    z1 = df.iloc[-1].values

    # if dendrites is provided then plot a scale to indicate different dendrite
    if dendrites is not None:
        # Get unique dendrites and sort them
        unique_dendrites = list(OrderedDict.fromkeys(dendrites))
        # Process each dendrite
        for i, dendrite in enumerate(unique_dendrites):
            # Determine color (alternating between two colors)
            color = 'lightgray' if i % 2 == 0 else 'white'

            # Filter the x, y, z values for the current dendrite
            indices = [j for j, d in enumerate(dendrites) if d == dendrite]
            x_filtered = [x1[j] for j in indices]
            y_filtered = [y1[j] for j in indices]
            z_filtered = [z1[j] for j in indices]

            # Interpolate to fill gaps
            if i < len(unique_dendrites) - 1:
                # Find the indices for the next dendrite
                next_dendrite = unique_dendrites[i + 1]
                next_indices = [j for j, d in enumerate(dendrites) if d == next_dendrite]

                # Calculate the midpoint of the gap
                midpoint_index_y = (y_filtered[-1] + y1[next_indices[0]]) / 2
                midpoint_index_z = (z_filtered[-1] + z1[next_indices[0]]) / 2

                # Extend current and next dendrite lines to the midpoint
                x_filtered.append(x1[next_indices[0]])
                y_filtered.append(midpoint_index_y)
                z_filtered.append(midpoint_index_z)

                # Adjust the beginning of the next dendrite l ine
                x1[next_indices[0]] = x_filtered[-1]
                y1[next_indices[0]] = y_filtered[-1]
                z1[next_indices[0]] = z_filtered[-1]

            # Add the trace for the current dendrite
            fig.add_trace(go.Scatter3d(x=x_filtered, y=y_filtered, z=z_filtered,
                                       mode='lines', name=dendrite,
                                       line=dict(color=color, width=12)))
        
    invisible_color = 'rgba(255, 255, 255, 0)' if remove_text else None  # Set to None if not removing text

    fig.update_layout(
        showlegend=False,
        title={
            'text': '' if remove_text else title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        scene=dict(
            xaxis=dict(
                title='' if remove_text else xaxis_title,
                range=x_range,
                title_font=dict(size=16),
                tickfont=dict(color=invisible_color)  # Make tick labels "invisible"
            ),
            yaxis=dict(
                title='' if remove_text else yaxis_title,
                range=y_range,
                title_font=dict(size=16),
                tickfont=dict(color=invisible_color)  # Make tick labels "invisible"
            ),
            zaxis=dict(
                title='' if remove_text else zaxis_title,
                range=z_range,
                title_font=dict(size=16),
                tickfont=dict(color=invisible_color)  # Make tick labels "invisible"
            ),
            aspectmode='cube'
        ),
        autosize=False,
        scene_camera=scene_camera,
        width=width,
        height=height,
        font=dict(
            size=12,
            family='Myriad Pro'
        )
    )

    return fig

def plot_mech(x, df, mech='', bl=20, stim_time=150, xlab = 'time (ms)', ylab = 'mA/cm2', cols=['slateblue', 'gray','indianred'], xrange = [0, 150], plot_height = 500, plot_width = 500, lwd=1):
    if ylab == 'mA/cm2': scale = 1
    elif ylab == 'uA/cm2': scale = 1000
    fig = go.Figure()
    fig = go.Figure()
    
    # Filter columns if 'mech' is specified
    columns_to_plot = all_y.columns if mech == '' else [c for c in all_y.columns if mech in c]

    for i, column in enumerate(columns_to_plot):
        # Filter x and y based on baseline and stim_time
        filtered_x = x[x > (stim_time - bl)] - (stim_time - bl)
        y = all_y[column][x > (stim_time - bl)].values * scale  # Scale y if necessary

        # Use modulo to cycle through colors if there are more traces than colors
        color = cols[i % len(cols)]

        fig.add_trace(go.Scatter(x=filtered_x, y=y*scale, mode='lines', name=column, line=dict(color=color, width=lwd)))
    
        
    # Update layout
    # Set layout, remove legend, set plot size, and customize axes
    fig.update_layout(
        xaxis=dict(
            title=xlab,
            range=xrange,
            showline=True,  
            linewidth=lwd,  
            linecolor='black',  
            mirror=False,  
            ticks='outside',  
            tickfont=dict(
                family='Calibri',
                size=12,
                color='black',
            ),
            showgrid=False  
        ),
        yaxis=dict(
            title=ylab,
            showline=True,  
            linewidth=lwd,  
            linecolor='black',  
            mirror=False,  
            ticks='outside',
            tickfont=dict(
                family='Calibri',
                size=12,
                color='black',
            ),
            showgrid=False  
        ),
        showlegend=False,
        width=plot_width,
        height=plot_height,
        plot_bgcolor='white' 
    )
    return fig

def save2svg(fig, fig_name, wd):
    image_dir = wd.replace('simulations', 'images')
    # Ensure the directory exists
    os.makedirs(image_dir, exist_ok=True)
    # Save the figure
    fig.write_image(f'{image_dir}/{fig_name}.svg', engine='kaleido')
    
def extract_mechs_3D(i_mechs_3D, mech, simulation):
    # Indices of mechanisms
    mechs_indices = i_mechs_3D[simulation]['mechs']

    # Find the index for the mechanism of interest
    mech_index = mechs_indices.index(mech)

    # Initialize a list to hold all arrays for the mechanism across all keys
    combined_mech_arrays = []

    # Iterate through the dictionary
    for key, mech_arrays in i_mechs_3D[simulation]['i'].items():
        # Append the array corresponding to the mechanism of interest
        combined_mech_arrays.append(mech_arrays[mech_index])
    
    return combined_mech_arrays

def extract_mechs(data_dict, mech, scale=1):
    # Initialize an empty DataFrame to store the consolidated data
    consolidated_df = pd.DataFrame()

    for key, df in data_dict.items():
        # Ensure the mechanism column exists in the DataFrame
        if mech in df.columns:
            # Create a new DataFrame with just the time and mechanism columns
            mech_data = df[['time', mech]].copy()
            # Multiply the mechanism column by the scale factor
            mech_data[mech] *= scale
            # Set the time column as the index to facilitate concatenation
            mech_data.set_index('time', inplace=True)
            # Rename the mechanism column to a numerical identifier based on its order in the dictionary
            mech_data.columns = [key]  # Using 'key' directly for numerical naming
            # Concatenate horizontally while keeping the time index aligned
            consolidated_df = pd.concat([consolidated_df, mech_data], axis=1)

    return consolidated_df

def mechs_np2df(X):
    out = {}
    for sim in X.keys():
        out1 = {}
        for key in X[sim]['i']:
            df = pd.DataFrame(X[sim]['i'][key])
            tdf = df.T  # T is the attribute for transpose
            tdf.columns = ['time'] + X[sim]['mechs']
            out1[key] = tdf
        out[sim] = out1
    return(out)

def on_button_clicked(b):
    # Display the next figure in the output widget, cycling through the list
    with fig_output:
        # Clear the previous figure
        fig_output.clear_output(wait=True)
        # Display the next figure
        figures[current_fig_index[0]].show()
        
    # Update the index for the next figure, wrapping around if at the end of the list
    current_fig_index[0] = (current_fig_index[0] + 1) % len(figures)
    
# function extracts the equivalent current densities that correspond to distances from v_branch from 3D all field simulations
def extract_current_densities(v_branch, i_mechs_3D, simulation, stim_time, bl, max_time, dend='dend[14]', mech='cal13', dt=0.025):
    target_dists = v_branch[simulation][dend]['dists']

    dendrites = i_mechs_3D[simulation]['dendrites']
    distances = i_mechs_3D[simulation]['dists']

    idx = [ii for ii, x in enumerate(dendrites) if x == dend]


    dists = [distances[i] for i in idx]
    idx2 = [np.abs(np.array(dists) - dist).argmin() for dist in target_dists]
    target_idx = [idx[ii] for ii in idx2]
    dists = [distances[ii] for ii in target_idx]

    df = pd.DataFrame(extract_mechs_3D(i_mechs_3D, mech='cal13', simulation=simulation)).transpose()
    df = df[target_idx]
    # rename columns
    df.columns = range(df.shape[1])

    x = np.arange(0, df.shape[0] * dt, dt)
    y = target_dists

    dendrites_dend = [dend] * len(y)

    idxs = (x > (stim_time-bl)) & (x < max_time)
    x = np.round(x[idxs] - (stim_time-bl), 10)
    df = df[idxs]

    return x, y, df, dendrites_dend

def morphology_plot(cell_coordinates, dend_tree, lwd=0.8, color='black', s=None, height=600, width=600):
    if s is None:
        fig = morphology_plot1(cell_coordinates=cell_coordinates, dend_tree=dend_tree, lwd=lwd, color=color, height=height, width=width)
    else:
        fig = morphology_plot2(cell_coordinates=cell_coordinates, dend_tree=dend_tree, s=s, lwd=lwd, color=color, height=height, width=width)
    return fig
    
def morphology_plot1(cell_coordinates, dend_tree, lwd=0.8, color='black', height=600, width=600):
    fig = go.Figure()

    simplified_dend_tree = [sub_branch for branch in dend_tree for sub_branch in (branch if isinstance(branch[0], list) else [branch])]

    connections = {}
    for branch in simplified_dend_tree:
        for i in range(len(branch) - 1):
            sec_name = branch[i].name()
            next_sec_name = branch[i + 1].name()

            if sec_name not in connections:
                connections[sec_name] = set()
            connections[sec_name].add(next_sec_name)

    processed_connections = set()
    processed_sections = set()

    section_coords = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]']
    x_coords = [coord[2] for coord in section_coords]
    y_coords = [coord[3] for coord in section_coords]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    radius = sum([coord[6] for coord in section_coords]) / len(section_coords) / 2

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=center_x - radius, y0=center_y - radius,
                  x1=center_x + radius, y1=center_y + radius,
                  line=dict(color=color, width=lwd),
                  fillcolor=color)

    primaries = []
    for tree in simplified_dend_tree:
        primaries.append(tree[0].name())
    primaries = list(set(primaries))

    def point_on_circle(cx, cy, angle, radius):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        return x, y

    for primary in primaries:
        section_coords = cell_coordinates[cell_coordinates[:, 0] == primary]
        if section_coords.size > 0:
            start_x = section_coords[0, 2]
            start_y = section_coords[0, 3]

            angle = np.arctan2(start_y - center_y, start_x - center_x)
            perimeter_x, perimeter_y = point_on_circle(center_x, center_y, angle, radius)

            fig.add_trace(go.Scatter(x=[perimeter_x, start_x],
                                     y=[perimeter_y, start_y],
                                     mode='lines',
                                     name=primary,
                                     line=dict(color=color, width=lwd),
                                     hoverinfo='none'))

            x_coords = [coord[2] for coord in section_coords]
            y_coords = [coord[3] for coord in section_coords]
            hover_texts = ['dist: {:.2f}'.format(coord[5]) for coord in section_coords]
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                name=primary,
                line=dict(color=color, width=lwd),
                text=hover_texts,
                hoverinfo='text+name'
            ))

    for parent in connections.keys():
        section_coords = cell_coordinates[cell_coordinates[:, 0] == parent]
        x_coords = [coord[2] for coord in section_coords]
        y_coords = [coord[3] for coord in section_coords]
        distances = [coord[5] for coord in section_coords]
        hover_texts = ['dist: {:.2f}'.format(dist) for dist in distances]

        end_x = section_coords[-1, 2]
        end_y = section_coords[-1, 3]

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            name=parent,
            line=dict(color=color, width=lwd),
            text=hover_texts,
            hoverinfo='text+name'
        ))
        processed_sections.add(parent)

        children = connections[parent]
        for child in children:
            if child in processed_sections:
                continue

            section_coords = cell_coordinates[cell_coordinates[:, 0] == child]
            x_coords = [coord[2] for coord in section_coords]
            y_coords = [coord[3] for coord in section_coords]
            distances = [coord[5] for coord in section_coords]
            hover_texts = ['dist: {:.2f}'.format(dist) for dist in distances]

            start_x = section_coords[0, 2]
            start_y = section_coords[0, 3]

            fig.add_trace(go.Scatter(x=[end_x, start_x],
                                     y=[end_y, start_y],
                                     mode='lines',
                                     name=parent,
                                     line=dict(color=color, width=lwd)))

            fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines',
                                     name=child, line=dict(color=color, width=lwd)))

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                name=child,
                line=dict(color=color, width=lwd),
                text=hover_texts,
                hoverinfo='text+name'
            ))

            processed_sections.add(child)

    fig.update_layout(
        title='neuronal morphology',
        title_x=0.5,
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(
            range=[-125, 175],
        ),
        yaxis=dict(
            range=[-150, 150],
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False
    )
    return fig
    
def morphology_plot2(cell_coordinates, dend_tree, lwd=0.8, color='black', s=2, height=600, width=600):

    def smooth_and_plot(coords):
        x_coords = [coord[2] for coord in coords]
        y_coords = [coord[3] for coord in coords]
        dists = [coord[5] for coord in coords]
        names = [coord[0] for coord in coords]
        
        if len(x_coords) > 3:
            tck, u = splprep([x_coords, y_coords], s=s)
            u_fine = np.linspace(0, 1, 1000)
            x_smooth, y_smooth = splev(u_fine, tck)

            # Interpolate distances
            dist_tck, dist_u = splprep([dists], s=s)
            distances = splev(u_fine, dist_tck)[0]

            # Map names to indices
            name_to_index = {name: idx for idx, name in enumerate(names)}
            indices = [name_to_index[name] for name in names]
    
            # Interpolate indices
            index_interpolator = interp1d(u, indices, kind='nearest', fill_value="extrapolate")
            interpolated_indices = index_interpolator(u_fine)
            interpolated_indices = np.round(interpolated_indices).astype(int)  # Round to nearest integer
    
            # Map interpolated indices back to names
            index_to_name = {idx: name for name, idx in name_to_index.items()}
            interpolated_names = [index_to_name[idx] for idx in interpolated_indices]
    
            hover_texts = ['{}, dist: {:.2f}'.format(name, dist) for name, dist in zip(interpolated_names, distances)]

            fig.add_trace(go.Scatter(
                x=x_smooth, 
                y=y_smooth, 
                mode='lines', 
                line=dict(color=color, width=lwd),
                text=hover_texts, 
                hoverinfo='text'
            ))
        else:
            hover_texts = ['{}, dist: {:.2f}'.format(coords[0][0], dist) for dist in dists]

            fig.add_trace(go.Scatter(
                x=x_coords, 
                y=y_coords, 
                mode='lines', 
                line=dict(color=color, width=lwd),
                text=hover_texts, 
                hoverinfo='text'
            ))
    
    def point_on_circle(cx, cy, angle, radius):
        """Calculate a point on the circle's perimeter given an angle and radius."""
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        return x, y
    
    fig = go.Figure()
    
    section_coords = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]']
    x_coords = [coord[2] for coord in section_coords]
    y_coords = [coord[3] for coord in section_coords]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    radius = sum([coord[6] for coord in section_coords]) / len(section_coords) / 2
    
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=center_x - radius, y0=center_y - radius,
                  x1=center_x + radius, y1=center_y + radius,
                  line=dict(color=color, width=lwd),
                  fillcolor=color)
    
    simplified_dend_tree = [sub_branch for branch in dend_tree for sub_branch in (branch if isinstance(branch[0], list) else [branch])]

    # visually best to smooth longest paths first so re-ordering dend_tree
    sorted_dend_tree = sorted(dend_tree, key=lambda sublist: (-len(sublist) if isinstance(sublist, list) else -1, [len(item) if isinstance(item, list) else 0 for item in sublist] if isinstance(sublist, list) else []))
    sorted_dend_tree = [sorted(sublist, key=lambda item: len(item) if isinstance(item, list) else 0, reverse=True) if isinstance(sublist, list) else sublist for sublist in sorted_dend_tree]
    
    for path_group in sorted_dend_tree:
    
        seen_sections = set()
        new_paths = []
        
        for iii, path in enumerate(path_group):
            new_path = []
            if not isinstance(path, list):  # Check if path are not a list
                path = [path]  # Ensure it is a list
        
            for ii, sec in enumerate(path):
                if sec not in seen_sections:
                    if ii > 0:
                        new_path.append(path[ii - 1])
                    new_path.append(sec)
                    seen_sections.add(sec)
            
            if new_path:
                # Remove consecutive duplicates and ensure predecessors are included
                filtered_path = []
                last_sec = None
                for j, sec in enumerate(new_path):
                    if sec != last_sec:
                        filtered_path.append(sec)
                        last_sec = sec
                new_paths.append(filtered_path)
        
                if not isinstance(filtered_path, list):  # Check if path are not a list
                    filtered_path = [filtered_path]  # Ensure it is a list
                    
                coords = []
                for sec in filtered_path:
                    name = sec.name()       
                    cell_coords = cell_coordinates[cell_coordinates[:, 0] == name]
                    coords.append(cell_coords)
                
                combined_coords = np.vstack(coords)
                # if iii > 0, exclude all but the last point from the first section in filtered_path
                if iii > 0 and len(filtered_path) > 1:
                    first_sec_coords = combined_coords[combined_coords[:, 0] == filtered_path[0].name()]
                    # combined_coords = np.vstack([first_sec_coords[-1:], combined_coords[combined_coords[:, 0] != filtered_path[0].name()]])
                    if len(first_sec_coords) > 2:
                        combined_coords = np.vstack([first_sec_coords[-2:], combined_coords[combined_coords[:, 0] != filtered_path[0].name()]])
                    else:
                        combined_coords = np.vstack([first_sec_coords, combined_coords[combined_coords[:, 0] != filtered_path[0].name()]])
        
                smooth_and_plot(combined_coords)
                    
                new_paths.append(new_path)
        
    # Draw lines from soma perimeter to primary dendrite starting points
    primaries = []
    for tree in simplified_dend_tree:
        primaries.append(tree[0].name())
    primaries = list(set(primaries))
    
    for primary in primaries:
        section_coords = cell_coordinates[cell_coordinates[:, 0] == primary]
        if section_coords.size > 0:
            start_x = section_coords[0, 2]
            start_y = section_coords[0, 3]
            
            angle = np.arctan2(start_y - center_y, start_x - center_x)
            perimeter_x, perimeter_y = point_on_circle(center_x, center_y, angle, radius)
            
            fig.add_trace(go.Scatter(x=[perimeter_x, start_x], 
                                     y=[perimeter_y, start_y], 
                                     mode='lines',
                                     line=dict(color=color, width=lwd),
                                     hoverinfo='none'))
    
    fig.update_layout(
        title='neuronal morphology',
        title_x=0.5,
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(
            range=[-125, 175],
        ),
        yaxis=dict(
            range=[-150, 150],
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False
    )
    
    return fig

def save2fig(fig, wd=None, out_name='fig', ext='svg', width=600, height=600, scale=1):
    """
    Save a plotly figure to a file.

    Parameters:
    fig (plotly.graph_objs.Figure): The plotly figure to save.
    wd (str, optional): The working directory where the file will be saved. 
                        Defaults to the current working directory.
    folder (str, optional): The name of the folder within the working directory to save the file. 
                            Defaults to 'example data'.
    filename (str, optional): The base name of the file (without extension). 
                              Defaults to 'your_filename'.
    ext (str, optional): The file extension (type of file to save as). 
                         Defaults to 'svg' for SVG file format.
    scale: increases the resolution of the image by multiplying the width and height by the scale factor. 
    """
    # If wd is None, use the current working directory
    wd = wd or os.getcwd()
    
    # Add the extension to the filename
    # This constructs the full filename including the chosen extension
    filename_with_ext = f"{out_name}.{ext}"

    # Construct the full file path by combining the working directory, folder, and filename
    file_path = os.path.join(wd, filename_with_ext)

    # Ensure the target folder exists, create it if it does not
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the Plotly figure to the specified file path
    fig.write_image(file_path, format=ext, scale=scale, width=width, height=height, validate=True, engine='kaleido')

def save_fig(fig, wd=None, out_name='fig', ext='svg', width=600, height=600):
    """
    Save a plotly figure to a file.

    Parameters:
    fig (plotly.graph_objs.Figure): The plotly figure to save.
    wd (str, optional): The working directory where the file will be saved. 
                        Defaults to the current working directory.
    folder (str, optional): The name of the folder within the working directory to save the file. 
                            Defaults to 'example data'.
    filename (str, optional): The base name of the file (without extension). 
                              Defaults to 'your_filename'.
    ext (str, optional): The file extension (type of file to save as). 
                         Defaults to 'svg' for SVG file format.
    """
    # If wd is None, use the current working directory
    wd = wd or os.getcwd()
    image_dir = wd.replace('simulations', 'images')
    
    # Add the extension to the filename
    # This constructs the full filename including the chosen extension
    filename_with_ext = f"{out_name}.{ext}"

    # Construct the full file path by combining the working directory, folder, and filename
    file_path = os.path.join(image_dir, filename_with_ext)

    # Ensure the target folder exists, create it if it does not
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the Plotly figure to the specified file path
    fig.write_image(file_path, format=ext, scale=None, width=width, height=height, validate=True, engine='kaleido')


# function extracts the measurements that correspond to distances from vdata from 3D all field simulations
def extract_data(vdata, zdata3D, simulation, dend=None, out='imp'):

    if dend is None:
        target_dists = vdata[simulation]['dists'] 
    else: 
        target_dists = vdata[simulation][dend]['dists']
    if dend is None:
        target_dendrites = vdata[simulation]['dendrites'] 
    else:
        target_dendrites = [dend] * len(target_dists)
    
    dendrites = zdata3D[simulation]['dendrites']
    distances = zdata3D[simulation]['dists']
    
    # Initialize lists to store the results
    idx = []
    final_dists = []
    final_dendrites = []

    used_distances = set()  # Initialize an empty set to keep track of used distances
    
    # Iterate over each target dendrite and its distance
    for target_dend, target_dist in zip(target_dendrites, target_dists):
        closest_diff = float('inf')
        closest_index = None
        closest_distance = None

        # Search for the closest match in zdata3D
        for i, (dend, dist) in enumerate(zip(dendrites, distances)):
            if dend == target_dend and dist not in used_distances:
                diff = abs(dist - target_dist)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_index = i
                    closest_distance = dist

        # Update the results if a match was found
        if closest_index is not None:
            idx.append(closest_index)
            final_dists.append(closest_distance)
            final_dendrites.append(target_dend)
            used_distances.add(closest_distance)  # Mark this distance as used

    # Select the relevant data from the DataFrame
    df = pd.DataFrame(zdata3D[simulation][out]).transpose()
    df_selected = df.iloc[:, idx]  # Use iloc to select columns by integer location
    df_selected.columns = range(df_selected.shape[1])  # Rename columns


    return df_selected, final_dists, final_dendrites


# Function to find the index in 'x' for each value in 'x2'
def resample_idx(x, x2):
    indices = []
    for value in x2:
        # Find the index where the difference between 'x' and the current 'x2' value is minimized
        index = np.argmin(np.abs(x - value))
        indices.append(index)
    return indices

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
    
def plot_summary(x, df, ident, title='', yaxis_title='membrane potential (mV)', 
                 xaxis_range=[0, 250], yaxis_range=[-90, -20], 
                 x_window = None, legend_title='log₁₀[spine volume]', lwd=1.5, 
                 yscalebar=10, xscalebar=25, y_ab1=-20, 
                 y_ab2=-85, x_offset=None, width=800, height=800):

    N = df.shape[1]
    v_data = []
    cols = hex_palette(N)

    for i in range(N):
        ii = N - i - 1  # Calculate the reverse index
    
        # Extracting the column at index 'ii' and converting it to a list
        y = df.iloc[:, ii].tolist()
        # Applying the x_offset for each subsequent plot
        if x_offset is not None:
            x1 = [x_val + ii * x_offset for x_val in x]  # Adjusting x values by the offset
        else:
            x1 = x
            
        if x_window is not None:
            dx = x[1] - x[0]
            ind = int(x_window/dx)
            y1 = y[0:ind]
        else:
            y1 = y
            
        # Creating a Scatter object for the current row and adding it to 'v_data'
        v_data.append(go.Scatter(x=x1, y=y1, mode='lines', 
                                 line=dict(color=cols[ii], width=lwd), 
                                 name='{}'.format(round(ident[ii], 3))))

    fig = go.Figure(data=v_data)
    # Update the figure layout
    if y_ab1 is not None:
        fig.add_hline(y=y_ab1, line_width=lwd, line_dash="dot", line_color="gray")
    if y_ab2 is not None:
        fig.add_hline(y=y_ab2, line_width=lwd, line_dash="dot", line_color="gray")

    # y and x scale bars

    # Assume y_start is an arbitrary starting point on your y-axis
    y_start = yaxis_range[0] + yscalebar  # You can adjust this based on your plot's y-axis range
    x_start = xaxis_range[1] - xscalebar
    # Add a black vertical line to represent a change of 10 units on the y-axis
    fig.add_shape(type="line",
                  x0=x_start,  # Use your x-axis range start as the x position
                  y0=y_start,  # Starting y value
                  x1=x_start,  # Use your x-axis range start as the x position to keep the line vertical
                  y1=y_start + yscalebar,  # Ending y value, 10 units away from the start
                  line=dict(color="black", width=lwd),  # Black line with a width of 2
                  )

    # Add a black horizontal line to represent 25 units on the x-axis
    fig.add_shape(type="line",
                  x0=x_start,  # Starting x value at 0 to represent the beginning of 25 units
                  y0=y_start,  # Use your y-axis range start as the y position
                  x1=x_start + xscalebar,  # Ending x value at 25 to represent 25 units on the x-axis
                  y1=y_start,  # Use your y-axis range start as the y position to keep the line horizontal
                  line=dict(color="black", width=lwd),  # Black line with a width of 2
                  )

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title='',
        yaxis_title='',
        xaxis_range=xaxis_range,
        yaxis_range=yaxis_range,
        legend_title=legend_title,
        legend=dict(
            x=1.05,  # Place legend to the right of the plot area
            xanchor='left',  # Anchor legend to the left side of its box
            y=1,  # Align the top of the legend with the top of the plot area
            yanchor='auto',  # Automatically adjust the vertical position
        ),
        width=width,  # Total width of the figure, including space for the legend
        height=height,  # Height of the figure
        xaxis=dict(
            showgrid=False,  # Remove x-axis grid lines
            zeroline=False,  # Optionally remove the x-axis zero line
            showticklabels=False,  # Remove x-axis tick labels
            tickvals=[]  # Remove x-axis tick marks
        ),
        yaxis=dict(
            showgrid=False,  # Remove y-axis grid lines
            zeroline=False,  # Optionally remove the y-axis zero line
            showticklabels=False,  # Remove y-axis tick labels
            tickvals=[]  # Remove y-axis tick marks
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Makes plot background transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Makes paper background transparent
        margin=dict(l=60, r=300, t=60, b=60)  # Adjust the right margin to create space for the legend
    )
    return fig

def path_names(cell, dend_tree, dendrite='dend[15]'):
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend

    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)

    target_dendrites = [sec.name() for sec in pathlist]   
    return target_dendrites

def extract3D(zdata3D, key='Nsim0', ident='imp', 
        target_dendrites=['soma[0]','dend[0]','dend[6]','dend[7]','dend[11]','dend[13]','dend[15]'],
        dt=1, start_time=40, burn_time=0):
    
    cell_coordinates = zdata3D[key]['cell_coordinates']
    dendrites = zdata3D[key]['dendrites']
    distances = zdata3D[key]['dists']

    idx = [i for i, item in enumerate(dendrites) if item in target_dendrites]

    extracted_dendrites = [dendrites[ii] for ii in idx]
    extracted_distances = [distances[ii] for ii in idx]
    extracted_distances = [0 if dendrite == 'soma[0]' else value for dendrite, value in zip(extracted_dendrites, extracted_distances)]

    extracted_zdata = [zdata3D[key][ident][ii] for ii in idx]
    _, _, extracted_Z = peaks(extracted_zdata, start_time=start_time, burn_time=burn_time, dt=dt, peak_type='min')
    # [np.mean(Z) for Z in extracted_zdata]
    
    return extracted_Z, extracted_dendrites, extracted_distances

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

# this function takes 2 inputs; the recorded cell_coordinates (cell_coordinates_record) and an index (indexing)
# it matches indexing to the nearest cell_coordinates and returns cell_coordinates_response
# cell_coordinates_response are the cell coordinates that actually match indexing
# this is necessary becayse glutamate_locations sometimes puts values in the incoorect place WHEN 
# part of that dendrite is < 30 um from soma
# also removes any duplicates

def cell_coordinates_match(cell_coordinates_record, indexing):
    
    """
    finds and returns the indices and details of records from `cell_coordinates_record` that closely match 
    the criteria specified in `indexing`, in particular identifying dendrite and distance from soma

    each record in `cell_coordinates_record` is expected to be a tuple or list where the first element is 
    a dendrite name and the second element is a distance value

    The `indexing` parameter should be an iterable where each item specifies search criteria, including a 
    dendrite name and a target distance. The structure for each item in `indexing` is expected to be:
    [_, [dendrite_name], [target_distance]], where _ can be any value

    Parameters:
    - cell_coordinates_record (list of tuples/lists): The dataset containing dendrite names and their respective distances
    - indexing (list of tuples/lists): The search criteria containing dendrite names and target distances

    Returns:
    - list of tuples: Each tuple contains the index of a matching record in `cell_coordinates_record` and the record itself
                      The matches are based on the closest distance to the target distance for the specified dendrite name
    """
    
    # Initialize an empty list to store the results
    cell_coordinates_response = []

    # Iterate through each item in indexing
    for index_item in indexing:
        dend_name = index_item[1][0]  # Dendrite name from indexing
        target_distance = index_item[2][0]  # Target distance from indexing

        # Filter cell_coordinates_record for matching dendrite name
        matching_records = [record for record in cell_coordinates_record if record[0] == dend_name]

        # Find the record with distance closest to the target_distance
        closest_record = min(matching_records, key=lambda record: abs(record[1] - target_distance))

        # Find the index of this closest record in the original cell_coordinates_record list
        closest_index = cell_coordinates_record.index(closest_record)

        # Append the result
        # Initialize an empty list to store the results
        cell_coordinates_response.append((closest_index, closest_record))
    
    return cell_coordinates_response
    
def cell_coordinates_remove_duplicates(cell_coordinates):
    """
    identifies the indices of duplicate items within a list of cell coordinates, excluding the first occurrence of each item

    this function is designed to help in data cleaning processes where duplicate entries, based on specific criteria, need to be identified
    and potentially removed. Each item in the `cell_coordinates` list is expected to be structured with an identifier followed by a list of 
    coordinates. For comparison purposes, each item's coordinates are converted to a tuple to ensure they are hashable and can be added to a set

    Parameters:
    - cell_coordinates (list of lists/tuples): The list containing the cell coordinates and their identifiers. Each item is expected to be in the format [identifier, [coordinate1, coordinate2, ...]]

    Returns:
    - list of int: A list of indices corresponding to duplicate items in the `cell_coordinates` list, excluding the index of their first occurrences
    """
    
    # Set to track seen items
    seen = set()

    # List to store indices of duplicates, excluding the first occurrence
    duplicate_indices = []

    for i, item in enumerate(cell_coordinates):
        # Convert the inner list of each item to a tuple for hashability
        item_key = (item[0], tuple(item[1]))

        if item_key in seen:
            # If the item is already seen, it's a duplicate
            # Save only the current index as it's a subsequent occurrence
            duplicate_indices.append(i)
        else:
            # Otherwise, mark it as seen
            seen.add(item_key)
    
    return duplicate_indices

# Function to get indices of cell_coordinates_Z based on the order of cell_coordinates_sorted
def get_matching_indices(cell_coordinates_sorted, cell_coordinates_Z):
    """
    finds the indices of items in `cell_coordinates_Z` that match the corresponding items in `cell_coordinates_sorted`.

    this function iterates over each item in `cell_coordinates_sorted` and searches for a matching item in `cell_coordinates_Z`.
    a match is determined by comparing the dendrite name and a unique identifier of each item. The function assumes that the first element
    of each item is the dendrite name and the second element is a unique identifier that can be used to establish a match.

    note: this function assumes each item in `cell_coordinates_sorted` has a corresponding match in `cell_coordinates_Z`.
    If no match is found for an item, it is skipped, and the function proceeds to the next item in `cell_coordinates_sorted`.

    Parameters:
    - cell_coordinates_sorted (list of lists/tuples): The list containing sorted cell coordinates and their identifiers, where each item is
                                                       expected to have at least a dendrite name and a unique identifier.
    - cell_coordinates_Z (list of lists/tuples): The list containing cell coordinates and their identifiers to be searched, structured similarly
                                                  to `cell_coordinates_sorted`.

    Returns:
    - list of int: A list of indices from `cell_coordinates_Z` corresponding to the matching items found in the order they appear in 
                   `cell_coordinates_sorted`. If no match is found for a particular item in `cell_coordinates_sorted`, it is not included in the result.

    """
    indices = []

    # Iterate over each item in cell_coordinates_sorted
    for sorted_item in cell_coordinates_sorted:
        # Extract identifying attributes, assuming first element is the dendrite name and the second is a unique identifier
        dend_name_sorted, unique_id_sorted = sorted_item[0], sorted_item[1]

        # Search for a match in cell_coordinates_Z
        for i, z_item in enumerate(cell_coordinates_Z):
            dend_name_Z, unique_id_Z = z_item[0], z_item[1]

            # If a matching item is found, record its index from cell_coordinates_Z
            if dend_name_sorted == dend_name_Z and unique_id_sorted == unique_id_Z:
                indices.append(i)
                break  # Move to the next item in cell_coordinates_sorted after finding a match

    return indices

def plot7(x, y, xrange, yrange, xname='', yname='', size=5, lwd=1, width=600, height=600):
    # Assuming AR1 and AR2 are defined and contain the data to plot
    data2plot = go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#D3D3D3', size=size), showlegend=False)

    # Create the figure object and assign it to fig1 instead of fig
    fig = go.Figure(data=data2plot)

    # Assuming AR1 and AR2 are your lists or arrays of x and y values respectively
    x1 = np.array(x)
    y1 = np.array(y)

    # Calculate the slope (m) and intercept (c) of the best-fit line
    m, c = np.polyfit(x1, y1, 1)

    # Generate y-values for the best-fit line using the slope and intercept
    y_fit = m * x1 + c

    # Calculate the correlation coefficient
    correlation_matrix = np.corrcoef(x1, y1)
    correlation_coefficient = correlation_matrix[0, 1]
    R_sqr = correlation_coefficient ** 2

    fit_line = go.Scatter(x=x1, y=y_fit, mode='lines', name=f'R\u00b2 = {np.round(R_sqr,5)}',
                          line=dict(color='#808080', dash='dash', width=lwd))  # Light gray color, finer 'dot' dash style, and line width of 2


    # Add the best-fit line to the figure object
    fig.add_trace(fit_line)


    # Configure layout of fig1
    fig.update_layout(
        title={'text': '', 'x': 0.5, 'xanchor': 'center'}, 
        xaxis=dict(
            title=xname, 
            range=xrange, 
            showline=True, 
            linewidth=1,
            linecolor='black', 
            ticks='outside',
            showgrid=False
        ), 
        yaxis=dict(
            title=yname, 
            range=yrange, 
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            ticks='outside',
            showgrid=False
        ),
        width=width,
        height=height,
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Optionally, you can also set the legend to appear outside the plot area
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    return fig

def fun_peaks(vdata, dt, stim_time, bl, max_time, dvspine=None, dvdend=None):

    df = pd.DataFrame(vdata['v']).transpose()

    x = np.arange(0, df.shape[0] * dt, dt)
    idxs = (x > (stim_time-bl)) & (x < max_time)
    x = np.round(x[idxs] - (stim_time-bl), 10)

    df = df[idxs]
    ind = int(bl/dt)
    dv = [df[col].max() - df[col].iloc[:ind].mean() for col in df.columns]
    
    if dvspine is None:
        AR = [v/dv[0] for v in dv]
    else:
        AR = [v/dvspine for v in dv]

    if dvdend is None:
        dv_dend = dv[1:]
        AR_dend = [v/max(dv_dend) for v in dv_dend]
        dends_dist = vdata['dists'][1:]
    else:
        AR_dend = [v/dvdend for v in dv]
        dends_dist = vdata['dists']
    vpeaks = {
        'dv': dv,
        'AR': AR,
        'distances': vdata['dists'],
        'AR_dend': AR_dend,
        'distances_dend': dends_dist
    }
    
    return vpeaks

def display_figures(figs):
    # Check if the input is a dictionary
    if isinstance(figs, dict):
        # Convert dictionary values to a list of figure objects
        figures = list(figs.values())
    elif isinstance(figs, list):
        # Use the list of figures directly
        figures = figs
    else:
        raise ValueError("input must be either a dictionary or a list of figures.")

    current_fig_index = [0]

    # Create an output widget to hold and display figures
    fig_output = widgets.Output()

    # Create a button widget
    button = widgets.Button(description="Next Figure")

    # Display the button and the output widget for figures
    display(button, fig_output)

    # Define the event handler for button click
    def on_button_clicked(b):
        # Display the next figure in the output widget, cycling through the list
        with fig_output:
            # Clear the previous figure
            fig_output.clear_output(wait=True)
            # Display the next figure
            figures[current_fig_index[0]].show()
        
        # Update the index for the next figure, wrapping around if at the end of the list
        current_fig_index[0] = (current_fig_index[0] + 1) % len(figures)

    # Attach the event handler to the button
    button.on_click(on_button_clicked)

def remove_all_text(fig):
    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='',
        annotations=[],
        showlegend=False
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

def figs2display(figures, width=1200, height=1200, save_dir=None, show_figs=True, save_figs=True, remove_text=False, scale=10):
    if remove_text:
        for fig in figures.values():
            remove_all_text(fig)
    
    if show_figs:
        display_figures(list(figures.values()))

    # set to True to save the figures
    if save_figs:
        for fig_name, fig in list(figures.items()):  
            save2fig(fig=fig, wd=save_dir, out_name=fig_name, ext='svg', width=width, height=height, scale=scale)
            

def match_3D(v_all_3D, imp_all_3D, vdend_tree, simulation, start_time=150, burn_time=130, dt=0.025, ds_imp=40, return_all = False):
    target_distances = vdend_tree[simulation]['dists']
    target_dendrites = vdend_tree[simulation]['dendrites']

    cell_coordinates = v_all_3D[simulation]['cell_coordinates']

    if return_all:
        idx = [i for i, coord in enumerate(cell_coordinates) if coord[0] in target_dendrites]
    else:
        idx = []
        for dend_name, dist in zip(target_dendrites, target_distances):

            # Filter cell_coordinates for entries matching the current dendrite name
            matching_dends = [dend for dend in cell_coordinates if dend[0] == dend_name]

            # Find the entry in matching_dends with a distance closest to the target distance
            closest_record = min(matching_dends, key=lambda dend: abs(dend[5] - dist))

            # Find the index of the closest entry in the original cell_coordinates list
            closest_index = cell_coordinates.index(closest_record)

            idx.append(closest_index)
            # idx gives indices of all entries of target dendrite that have distance closest to target distance


    _, dV1, _ = peaks(zdata=v_all_3D[simulation]['v'], start_time=start_time, burn_time=burn_time, dt=dt)

    _, _, Zbaseline = peaks(zdata=imp_all_3D[simulation]['imp'], start_time=(start_time-burn_time), burn_time=0, dt=(ds_imp*dt))

    _, _, Ztransfer = peaks(zdata=imp_all_3D[simulation]['imp transfer'], start_time=(start_time-burn_time), burn_time=0, dt=(ds_imp*dt))

    out = {} 
    dVout = [dV1[id] for id in idx]
    Zout = [Zbaseline[id] for id in idx]
    Ztransferout = [Ztransfer[id] for id in idx]
    dend_out = [cell_coordinates[id][0] for id in idx]
    dist_out = [cell_coordinates[id][5] for id in idx]
    
    out = {'dv':        dVout,
           'Zbaseline': Zout,
           'Ztransfer': Ztransferout,
           'dendrites': dend_out,
           'distances': dist_out}
    
    return out

def restructure_v_branch(vdict):
    new_v_branch = {'v': [], 'dists': [], 'dendrites': []}

    # Iterate over each dendrite in v_branch[sim]
    for dendrite, data in vdict.items():
        # Append the voltage arrays from this dendrite to the 'v' list in the new dictionary
        new_v_branch['v'].extend(data['v'])

        # Append the distances (if applicable, and assuming they match one-to-one with the 'v' entries)
        # This might need adjustment depending on the exact structure and meaning of your data
        if 'dists' in data:
            new_v_branch['dists'].extend(data['dists'])

        # Append the dendrite identifier to the 'dendrites' list, repeated for the number of 'v' entries
        # This assumes each 'v' entry in a dendrite corresponds to a unique position or measurement
        new_v_branch['dendrites'].extend([dendrite] * len(data['v']))
    
    return new_v_branch

def combine_vdicts(vdict1, vdict2):
    combined_vdict = {
        'v': vdict1['v'] + vdict2['v'],
        'dists': vdict1['dists'] + vdict2['dists'],
        'dendrites': vdict1['dendrites'] + vdict2['dendrites']
    }
    return combined_vdict

def match_3D_sim(v_3D, imp_3D, v_tree, start_time=150, burn_time=130, dt=0.025, ds_imp=40, return_all = False):
    target_distances = v_tree['dists']
    target_dendrites = v_tree['dendrites']

    cell_coordinates = v_3D['cell_coordinates']

    if return_all:
        idx = [i for i, coord in enumerate(cell_coordinates) if coord[0] in target_dendrites]
    else:
        idx = []
        for dend_name, dist in zip(target_dendrites, target_distances):

            # Filter cell_coordinates for entries matching the current dendrite name
            matching_dends = [dend for dend in cell_coordinates if dend[0] == dend_name]

            # Find the entry in matching_dends with a distance closest to the target distance
            closest_record = min(matching_dends, key=lambda dend: abs(dend[5] - dist))

            # Find the index of the closest entry in the original cell_coordinates list
            closest_index = cell_coordinates.index(closest_record)

            idx.append(closest_index)
            # idx gives indices of all entries of target dendrite that have distance closest to target distance


    _, dV1, _ = peaks(zdata=v_3D['v'], start_time=start_time, burn_time=burn_time, dt=dt)

    _, _, Zbaseline = peaks(zdata=imp_3D['imp'], start_time=(start_time-burn_time), burn_time=0, dt=(ds_imp*dt))

    dVout = [dV1[id] for id in idx]
    Zout = [Zbaseline[id] for id in idx]
    dend_out = [cell_coordinates[id][0] for id in idx]
    dist_out = [cell_coordinates[id][5] for id in idx]
    
    if 'imp transfer' in imp_3D:
        _, _, Ztransfer = peaks(zdata=imp_3D['imp transfer'], start_time=(start_time-burn_time), burn_time=0, dt=(ds_imp*dt))
        Ztransferout = [Ztransfer[id] for id in idx]
    else:
         Ztransferout = []
    out = {} 
    
    out = {'dv':        dVout,
           'Zbaseline': Zout,
           'Ztransfer': Ztransferout,
           'dendrites': dend_out,
           'distances': dist_out}
    
    return out

def XY(voutput, sim, add_spine=True):
    X = copy.deepcopy(voutput[sim]['distances'])
    Y = copy.deepcopy(voutput[sim]['dv'])
    if add_spine:
        # Insert values at the beginning of the lists
        X.insert(0, voutput[sim]['spine_dist'])
        Y.insert(0, voutput[sim]['dvspine'])

    return X, Y

def range_increase(_axis_range, percent=5):
    range_increase = percent / 100  * (_axis_range[1] - _axis_range[0])

    # Calculate the new y-axis range
    new_axis_range = [_axis_range[0] - range_increase, _axis_range[1] + range_increase]
    return new_axis_range

def PathFind(cell, dend_tree, dendrite='dend[15]'):
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend

    # Get path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    return [sec.name() for sec in pathlist]
    
def extractor(zdata3D, cell_coordinates, target_dendrites, start_time, burn_time, dt):
    # Find indices
    indices = [i for i, coord in enumerate(cell_coordinates) if coord[0] in target_dendrites]
    extracted_zdata = [zdata3D[ii] for ii in indices]
    df = pd.DataFrame(extracted_zdata).transpose()
    
    _, dZ, _ = peaks(extracted_zdata, start_time=start_time, burn_time=burn_time, dt=dt, peak_type='max')

    dists = [cell_coordinates[ii][5] for ii in indices]

    dendrites_v = [cell_coordinates[ii][0] for ii in indices]

    return df, dists, dendrites_v, dZ
