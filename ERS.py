# =========================================================
# plot_earth_uv.py
# Ejecutar con: python plot_earth_uv.py
# =========================================================

import matplotlib
matplotlib.use('Agg') # Modo servidor sin pantalla

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from itertools import combinations
import ehtim as eh
import os

print("1. Configurando el array reducido (5 estaciones)...")

# 1. Seleccionamos solo 5 antenas clave para un trazado limpio
stations_subset = [
    ("EF",  4033947.3,   486990.8,  4900431.0, 25.0),    # Effelsberg (Alemania)
    ("ON",  3370965.9,   711466.2,  5349664.2, 850.0),   # Onsala (Suecia)
    ("NT",  4934562.8,  1321201.5,  3806484.7, 1100.0),  # Noto (Italia)
    ("HH",  5085442.8,  2668263.8, -2768696.8, 700.0),   # Hartebeesthoek (Sudáfrica)
    ("T6", -2826708.6,  4679237.1,  3274667.6, 26.0)     # Tianma (China)
]

temp_array_file = 'small_network.txt'
with open(temp_array_file, 'w') as f:
    f.write("# Site X Y Z SEFDR SEFDL FR_PAR_ANGLE FR_ELEV_ANGLE FR_OFFSET DR_RE DR_IM DL_RE DL_IM\n")
    for st in stations_subset:
        name, x, y, z, sefd = st
        f.write(f"{name} {x} {y} {z} {sefd} {sefd} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")

array_comb = eh.array.load_txt(temp_array_file)
os.remove(temp_array_file)

# ---------------------------------------------------------
# 2. Generar Datos UV (Simulando una observación fantasma)
# ---------------------------------------------------------
print("2. Calculando la física de rotación de la Tierra y plano UV...")

# Creamos una fuente puntual "falsa" en el cielo para que el telescopio la observe
im_dummy = eh.image.make_empty(32, 100*eh.RADPERUAS, ra=12.0, dec=12.0, rf=5e9)
im_dummy = im_dummy.add_gauss(1.0, [10*eh.RADPERUAS, 10*eh.RADPERUAS, 0, 0, 0])

obs = im_dummy.observe(array_comb, tint=120, tadv=600, tstart=0, tstop=8, 
                       bw=1e9, sgrscat=False, add_th_noise=False)

data = obs.data
u_giga = data['u'] / 1e9
v_giga = data['v'] / 1e9
times = data['time']
start_time = np.min(times)

# ---------------------------------------------------------
# 3. Preparar el Dashboard (2 Filas x 5 Columnas)
# ---------------------------------------------------------
print("3. Generando mapas y gráficas...")

# Horas de avance más lentas
snapshot_offsets = [0, 2, 4, 6, 8] 
n_snaps = len(snapshot_offsets)

fig = plt.figure(figsize=(4 * n_snaps, 8)) # Figura más alta para acomodar las dos filas

# Empezar sobre el Océano Índico. Esto empuja a Europa hacia el lado izquierdo del globo.
base_viewing_longitude = 60.0 

# Coordenadas geográficas
tarr = array_comb.tarr
x, y, z = tarr['x'], tarr['y'], tarr['z']
r = np.sqrt(x**2 + y**2 + z**2)
lat = np.degrees(np.arcsin(z / r))
lon = np.degrees(np.arctan2(y, x))

# ---------------------------------------------------------
# 4. Renderizar Filas
# ---------------------------------------------------------
for i, offset_hr in enumerate(snapshot_offsets):
    
    target_time = start_time + offset_hr
    
    # --- FILA SUPERIOR: EL GLOBO 3D ---
    current_lon = base_viewing_longitude - (offset_hr * 15.0)
    
    ax_globe = fig.add_subplot(2, n_snaps, i + 1, projection=ccrs.Orthographic(
        central_longitude=current_lon, 
        central_latitude=20.0 # Miramos un poco más hacia el hemisferio norte para ver Europa clara
    ))
    ax_globe.set_global()
    ax_globe.stock_img() 

    # Líneas de base
    for (lon1, lat1), (lon2, lat2) in combinations(zip(lon, lat), 2):
        ax_globe.plot([lon1, lon2], [lat1, lat2], color='red', linewidth=1.5, alpha=0.7, transform=ccrs.Geodetic())

    # Antenas
    ax_globe.scatter(lon, lat, color='red', s=60, edgecolor='white', linewidth=1, transform=ccrs.Geodetic(), zorder=5)
    ax_globe.set_title(f"+{offset_hr} Hrs", fontsize=18, fontweight='bold', pad=10)

    # --- FILA INFERIOR: COBERTURA UV ---
    ax_uv = fig.add_subplot(2, n_snaps, i + 1 + n_snaps)

    # 1. Pistas de fondo (todo el recorrido futuro en azul muy suave)
    ax_uv.scatter(u_giga, v_giga, color='lightsteelblue', s=3, alpha=0.2)
    ax_uv.scatter(-u_giga, -v_giga, color='lightsteelblue', s=3, alpha=0.2)

    # 2. Rastro acumulado (pinta de rojo lo que ya hemos observado hasta la hora actual)
    time_mask = times <= (target_time + 0.1) 
    u_acc = u_giga[time_mask]
    v_acc = v_giga[time_mask]
    
    ax_uv.scatter(u_acc, v_acc, color='red', s=12, alpha=0.7)
    ax_uv.scatter(-u_acc, -v_acc, color='red', s=12, alpha=0.7)

    # 3. Formato para que se vea como en un paper
    ax_uv.set_aspect('equal')
    ax_uv.set_xticks([])
    ax_uv.set_yticks([])
    ax_uv.set_xlabel("East-West Frequency (u)", fontsize=12)
    if i == 0:
        ax_uv.set_ylabel("North-South Frequency (v)", fontsize=12)

# ---------------------------------------------------------
# 5. Guardar el archivo final
# ---------------------------------------------------------
plt.tight_layout()
output_filename = "ERS_evolution_dashboard.png"
print(f"Guardando imagen de alta resolución en {output_filename}...")
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print("¡Completado! Ya puedes descargar la imagen.")