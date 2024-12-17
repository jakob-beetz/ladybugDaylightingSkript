from honeybee.room import Room
from honeybee.radiance.material.glass import Glass
from honeybee.radiance.sky.certainIlluminance import CertainIlluminanceLevel
from honeybee.radiance.recipe.pointintime.gridbased import GridBased
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# create a test room
def run_illuminance_simulation(
    room_width=4.2, 
    room_depth=6, 
    room_height=3.2, 
    rotation=20,
    # Back window parameters
    back_window_width=3,
    back_window_height=3,
    back_sill_height=0.1,
    # Right window parameters
    right_window_width=4,
    right_window_height=2.5,
    right_sill_height=0.2
):
    """Run the illuminance simulation for a room with parametric window configurations"""
    # Create room
    room = Room(origin=(0, 0, 3.2), width=room_width, depth=room_depth, height=room_height,
                rotation_angle=rotation)
    
    # Add back window with parameters
    room.add_fenestration_surface(
        wall_name='back', 
        width=back_window_width, 
        height=back_window_height, 
        sill_height=back_sill_height
    )
    
    # Add right window with parameters and glass material
    glass_60 = Glass.by_single_trans_value('tvis_0.6', 0.6)
    room.add_fenestration_surface(
        wall_name='right', 
        width=right_window_width, 
        height=right_window_height, 
        sill_height=right_sill_height,
        radiance_material=glass_60
    )

    # Setup and run simulation
    sky = CertainIlluminanceLevel(illuminance_value=2000)
    analysis_grid = room.generate_test_points(grid_size=0.1, height=0.75)
    rp = GridBased(sky=sky, analysis_grids=(analysis_grid,), simulation_type=0,
                   hb_objects=(room,))
    batch_file = rp.write(target_folder='.', project_name='room')
    rp.run(batch_file, debug=False)
    
    return rp.results()[0]

def create_illuminance_heatmap(result, filename):
    """Create basic illuminance heatmap visualization"""
    # Extract coordinates and values
    x_coords, y_coords, z_values = [], [], []
    points = list(result.points)
    for idx, value in enumerate(result.combined_value_by_id()):
        point = points[idx]
        x_coords.append(point.x)
        y_coords.append(point.y)
        z_values.append(value[0])

    # Setup grid and interpolation
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_grid = np.arange(x_min, x_max + 0.1, 0.1)
    y_grid = np.arange(y_min, y_max + 0.1, 0.1)
    xi, yi = np.meshgrid(x_grid, y_grid)
    z_grid = griddata((x_coords, y_coords), z_values, (xi, yi), method='cubic')

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.imshow(z_grid,
              interpolation='none',
              cmap='viridis',
              extent=[x_min, x_max, y_min, y_max],
              origin='lower')
    ax.set_aspect('equal')
    plt.colorbar(label='Illuminance (lux)')
    plt.title('Room Illuminance Analysis')
    plt.xlabel('Room Width (m)')
    plt.ylabel('Room Depth (m)')
    plt.grid(True, color='white', linestyle='-', linewidth=0.2)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

def create_comfort_zones_visualization(result, filename):
    """Create visualization with IES comfort zones"""
    # Extract coordinates and values
    x_coords, y_coords, z_values = [], [], []
    points = list(result.points)
    for idx, value in enumerate(result.combined_value_by_id()):
        point = points[idx]
        x_coords.append(point.x)
        y_coords.append(point.y)
        z_values.append(value[0])

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_grid = np.arange(x_min, x_max + 0.1, 0.1)
    y_grid = np.arange(y_min, y_max + 0.1, 0.1)
    xi, yi = np.meshgrid(x_grid, y_grid)

    # Interpolate values
    z_grid = griddata((x_coords, y_coords), z_values, (xi, yi), method='cubic')

    # Calculate cell area
    cell_area = 0.01
    total_area = z_grid.size * cell_area

    # Calculate areas for each zone
    insufficient_mask = z_grid < 100
    comfort_mask = (z_grid >= 100) & (z_grid <= 300)
    bright_mask = (z_grid > 300) & (z_grid <= 500)
    very_bright_mask = z_grid > 500

    zones = [insufficient_mask, comfort_mask, bright_mask, very_bright_mask]
    areas = [np.sum(mask) * cell_area for mask in zones]
    percentages = [(area / total_area) * 100 for area in areas]

    labels = [
        f'Insufficient (<100 lux): {areas[0]:.1f} m² ({percentages[0]:.1f}%)',
        f'Comfortable (100-300 lux): {areas[1]:.1f} m² ({percentages[1]:.1f}%)',
        f'Bright (300-500 lux): {areas[2]:.1f} m² ({percentages[2]:.1f}%)',
        f'Very Bright (>500 lux): {areas[3]:.1f} m² ({percentages[3]:.1f}%)'
    ]

    # Create figure with equal aspect ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    levels = [0, 150, 300, max(z_values)]

    colors = ['darkred', 'green', 'yellow']
    contour = ax.contourf(xi, yi, z_grid, levels=levels, colors=colors, alpha=1)
    ax.set_aspect('equal')  # Forces equal scaling on both axes

    # Add colorbar and labels
    cbar = plt.colorbar(contour)
    cbar.set_ticks([50, 200, 400, 600])
    cbar.set_ticklabels(labels)

    plt.title('Room Illuminance Analysis (IES Standards)')
    plt.xlabel('Room Width (m)')
    plt.ylabel('Room Depth (m)')
    plt.grid(True, color='white', linestyle='-', linewidth=0.2)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()



config = {
    'room_width': {'start': 4.0, 'end': 6.0, 'step': 0.5},
    'room_depth': {'start': 5.0, 'end': 8.0, 'step': 1.0},
    'back_window_width': {'start': 2.0, 'end': 3.0, 'step': 0.5},
    'right_window_width': {'start': 3.0, 'end': 4.0, 'step': 0.5},
}

def run_parametric_study():
    """Run parametric study with different combinations and save organized results"""
    results_data = []
    current_id = 1
    
    for room_width in np.arange(
        config['room_width']['start'],
        config['room_width']['end'] + config['room_width']['step'],
        config['room_width']['step']
    ):
        for room_depth in np.arange(
            config['room_depth']['start'],
            config['room_depth']['end'] + config['room_depth']['step'],
            config['room_depth']['step']
        ):
            for back_width in np.arange(
                config['back_window_width']['start'],
                config['back_window_width']['end'] + config['back_window_width']['step'],
                config['back_window_width']['step']
            ):
                for right_width in np.arange(
                    config['right_window_width']['start'],
                    config['right_window_width']['end'] + config['right_window_width']['step'],
                    config['right_window_width']['step']
                ):
                    study_id = f"S{current_id:03d}"
                    
                    result = run_illuminance_simulation(
                        room_width=room_width,
                        room_depth=room_depth,
                        back_window_width=back_width,
                        right_window_width=right_width
                    )
                    
                    create_illuminance_heatmap(result, f'heatmap_{study_id}.png')
                    create_comfort_zones_visualization(result, f'comfort_{study_id}.png')
                    
                    # Calculate metrics and store results
                    points = list(result.points)
                    values = [v[0] for v in result.combined_value_by_id()]
                    
                    insufficient = sum(1 for v in values if v < 100) / len(values) * 100
                    comfortable = sum(1 for v in values if 100 <= v <= 300) / len(values) * 100
                    bright = sum(1 for v in values if 300 < v <= 500) / len(values) * 100
                    very_bright = sum(1 for v in values if v > 500) / len(values) * 100
                    
                    results_data.append({
                        'Study ID': study_id,
                        'Room Width': room_width,
                        'Room Depth': room_depth,
                        'Back Window Width': back_width,
                        'Right Window Width': right_width,
                        'Insufficient Area %': insufficient,
                        'Comfortable Area %': comfortable,
                        'Bright Area %': bright,
                        'Very Bright Area %': very_bright
                    })
                    
                    current_id += 1
    
    results_df = pd.DataFrame(results_data)
    results_df.set_index('Study ID', inplace=True)
    results_df.to_csv('illuminance_results.csv')
    results_df.to_excel('illuminance_results.xlsx')
    
    return results_df
    # Setup grid and interpolation
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_grid = np.arange(x_min, x_max + 0.1, 0.1)
    y_grid = np.arange(y_min, y_max + 0.1, 0.1)
    xi, yi = np.meshgrid(x_grid, y_grid)
    z_grid = griddata((x_coords, y_coords), z_values, (xi, yi), method='cubic')
    
    # Replace any NaN values with the minimum value
    z_grid = np.nan_to_num(z_grid, nan=np.nanmin(z_grid))

    # Calculate areas
    cell_area = 0.01
    total_area = z_grid.size * cell_area
    zones = [z_grid < 100, 
             (z_grid >= 100) & (z_grid <= 300),
             (z_grid > 300) & (z_grid <= 500),
             z_grid > 500]
    areas = [np.sum(mask) * cell_area for mask in zones]
    percentages = [(area / total_area) * 100 for area in areas]

    # Create visualization with defined levels
    fig, ax = plt.subplots(figsize=(12, 8))
    min_val = np.min(z_grid)
    
    # z_grid = np.ma.masked_where((z_grid == 0) | np.isnan(z_grid), z_grid)
    max_val = np.max(z_grid)
    # Update levels to start from minimum valid value
    min_val = np.ma.min(z_grid)
    levels = [min_val, 100, 300, 500, max_val]
    colors = ['darkred', 'green', 'yellow', 'white'][:len(levels)-1]
    labels = [
        f'Insufficient (<100 lux): {areas[0]:.1f} m² ({percentages[0]:.1f}%)',
        f'Comfortable (100-300 lux): {areas[1]:.1f} m² ({percentages[1]:.1f}%)',
        f'Bright (300-500 lux): {areas[2]:.1f} m² ({percentages[2]:.1f}%)',
        f'Very Bright (>500 lux): {areas[3]:.1f} m² ({percentages[3]:.1f}%)'
    ]
    #  Create masked array excluding zeros and NaNs
    # z_grid = np.ma.masked_where((z_grid <10 ) | np.isnan(z_grid), z_grid)
    
    # Define strictly increasing levels with unique values
    min_val = np.ma.min(z_grid)
    max_val = np.ma.max(z_grid)
    base_levels = [100, 300, 500]
    levels = sorted(list(set([min_val] + base_levels + [max_val])))
    # levels =  sorted(list(set([min_val], 100, 300, 500, [max_val])))
    # Adjust colors to match number of regions
    colors = ['darkred', 'green', 'yellow', 'white'][:len(levels)-1]

    # Continue with contour plot
    contour = ax.contourf(xi, yi, z_grid, levels=levels, colors=colors, alpha=0.6)
    ax.set_aspect('equal')
    cbar = plt.colorbar(contour)
    cbar.set_ticks([50, 200, 400, 600])
    cbar.set_ticklabels(labels)
    plt.title('Room Illuminance Analysis (IES Standards)')
    plt.xlabel('Room Width (m)')
    plt.ylabel('Room Depth (m)')
    plt.grid(True, color='white', linestyle='-', linewidth=0.2)
    plt.savefig('illuminance_comfort_zones.png', dpi=600, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    # Run simulation and create visualizations
    # result = run_illuminance_simulation(
    #     room_width=5.0,
    #     room_depth=7.0,
    #     back_window_width=3.5,
    #     back_window_height=2.8,
    #     back_sill_height=0.15,
    #     right_window_width=4.2,
    #     right_window_height=2.3,
    #     right_sill_height=0.25
    # )
    # create_illuminance_heatmap(result)
    # create_comfort_zones_visualization(result)
    run_parametric_study()