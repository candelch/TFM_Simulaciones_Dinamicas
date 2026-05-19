import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer

# 1. Define the Data
evn_stations = [
    # (Code, X, Y, Z, SEFD_5)
    ("EF", 4033947.3, 486990.8, 4900431.0, 25.0),    # Effelsberg 
    ("IR", 3183649.3, 1276903.0, 5359264.7, 480.0),  # Irbene
    ("JB", 3822846.8, -153802.3, 5086285.9, 300.0),  # Jodrell Bank (Mrk 2)
    ("MC", 4461369.7, 919597.1, 4449559.4, 840.0),   # Medicina
    ("NT", 4934562.8, 1321201.5, 3806484.7, 1100.0), # Noto
    ("ON", 3370965.9, 711466.2, 5349664.2, 850.0),   # Onsala
    ("SR", 4865183.5, 791922.3, 4035136.0, 50.0),    # Sardinia
    ("TR", 3638558.5, 1221969.7, 5077036.8, 650.0),  # Torun
    ("WB", 3828767.3, 442446.2, 5064921.6, 1600.0),  # Westerbork 
    ("YS", 4848761.8, -261484.2, 4123085.1, 160.0)   # Yebes
]

africa_stations = [
    ("S5", 5109303.0031, 2065863.3087, -3230973.0000, 3.1),
    ("S004", 5109040.8523, 2063853.5134, -3238686.0441, 330.0),
    ("S008", 5108654.5126, 2064500.4184, -3239044.2096, 330.0),
    ("S133", 5104477.1687, 2073238.2514, -3242137.6044, 330.0),
    ("HH", 5084335.207, 2667653.767, -2768082.046, 650.0),
    ("KU", 6346222.481, -35250.52094, 634244.2061, 330.0),
    ("NA", 5632444.680, 1730320.594, -2433598.729, 330.0),
    ("BW", 5216975.743, 2534223.208, -2644711.076, 330.0),
    ("ML", 5127067.758, 3792660.53700, -97962.92579, 330.0)
]

# 2. Coordinate Conversion Setup
transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

def process_and_add_points(ax, stations_list, group_name, point_color, text_color):
    """Processes a list of stations, plots points and persistent labels."""
    processed_lons = []
    processed_lats = []
    
    # Text halo effect configuration
    halo_effect = [PathEffects.withStroke(linewidth=2.5, foreground='white')]

    for station in stations_list:
        label, x, y, z, _ = station
        
        # 1. Transform: ECEF X, Y, Z -> Longitude, Latitude
        lon, lat, alt = transformer.transform(x, y, z)
        processed_lons.append(lon)
        processed_lats.append(lat)
        
        # 2. Add Label
        txt = ax.text(lon + 1.2, lat, label, 
                      transform=ccrs.PlateCarree(),
                      fontsize=11, fontweight='bold', ha='left', va='center', 
                      color=text_color, zorder=10)
        
        # Apply the halo to the label
        txt.set_path_effects(halo_effect)

    # 3. Plot Points (scatter plot)
    ax.scatter(processed_lons, processed_lats, 
               color=point_color, marker='^', s=60, 
               transform=ccrs.PlateCarree(), zorder=9, 
               edgecolor='black', linewidth=1)

# 3. Globe and Figure Setup
# Fondo de la figura en blanco
fig = plt.figure(figsize=(12, 12), facecolor='white')

# Use Orthographic projection centered perfectly between Europe and Africa
center_lon = 15.0
center_lat = 10.0
globe_view = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)

# The ax is set to the Orthographic projection
ax = plt.axes(projection=globe_view)

# Fondo de los ejes en blanco
ax.set_facecolor('white')

# Add High-Resolution Realistic Imagery
ax.stock_img() 

# Enhance the globe's edge/limb
ax.set_global() 

# Add features with very fine, light lines
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#555555')
ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#777777', linestyle=':')

# Gridlines en gris suave
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                  alpha=0.3, color='gray', linewidth=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 4. Plot Stations and Persistent Labels
# Restaurados los colores originales de las etiquetas ('#000080' y '#006400')
process_and_add_points(ax, evn_stations, 'EVN', '#0000FF', '#000080') 
process_and_add_points(ax, africa_stations, 'Africa', '#00EE00', '#006400') 

# 5. Display
# Título en negro
plt.title("Spherical View of EVN and African Station Networks", 
          fontsize=16, fontweight='bold', color='black', pad=25)

# Adjust margins
plt.tight_layout()
plt.show()