# =========================================================
# plot_uv_ss433.py (COLORES COHERENTES: AZUL Y VERDE)
# Ejecutar con: python plot_uv_ss433.py
# =========================================================

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import ehtim as eh
import os

print("1. Configurando entorno y arrays de SS 433...")

OUT_DIR = "Resultados_UV_SS433"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"   -> Todos los gráficos se guardarán en la carpeta: ./{OUT_DIR}/")

evn_stations = [
    ("EF",  4033947.3,   486990.8,  4900431.0, 25.0),
    ("IR",  3183649.3,  1276903.0,  5359264.7, 480.0),
    ("JB",  3822846.8,  -153802.3,  5086285.9, 300.0),
    ("MC",  4461369.7,   919597.1,  4449559.4, 840.0),
    ("NT",  4934562.8,  1321201.5,  3806484.7, 1100.0),
    ("ON",  3370965.9,   711466.2,  5349664.2, 850.0),
    ("SR",  4865183.5,   791922.3,  4035136.0, 50.0),
    ("TR",  3638558.5,  1221969.7,  5077036.8, 650.0),
    ("WB",  3828767.3,   442446.2,  5064921.6, 1600.0),
    ("YS",  4848761.8,  -261484.2,  4123085.1, 160.0)
]

africa_stations = [
    ("HH",   5085442.8,     2668263.8,  -2768696.8, 700.0), 
    ("S5",   5109303.0031,  2065863.3087, -3230973.0000, 3.1), 
    ("S004", 5109040.8523,  2063853.5134, -3238686.0441, 330.0),
    ("S008", 5108654.5126,  2064500.4184, -3239044.2096, 330.0), 
    ("S133", 5104477.1687,  2073238.2514, -3242137.6044, 330.0),
    ("KU",   6346222.481,    -35250.5209,   634244.2061, 330.0),
    ("NA",   5632444.680,   1730320.594,  -2433598.729,  330.0), 
    ("BW",   5216975.743,   2534223.208,  -2644711.076,  330.0),
    ("ML",   5127067.758,   3792660.5370,   -97962.9258, 330.0)
]

future_stations = evn_stations + africa_stations

def load_array_from_list(station_list, filename):
    with open(filename, 'w') as f:
        f.write("#NAME X Y Z SEFDR SEFDL FR_PAR FR_ELEV FR_OFF DR_RE DR_IM DL_RE DL_IM\n")
        for st in station_list:
            f.write(f"{st[0]:<8} {st[1]:<13.1f} {st[2]:<13.1f} {st[3]:<13.1f} {st[4]:<7.1f} {st[4]:<7.1f} 1 0 0 0 0 0 0\n")
    arr = eh.array.load_txt(filename)
    os.remove(filename)
    return arr

array_current = load_array_from_list(evn_stations, 'temp_current.txt')
array_future = load_array_from_list(future_stations, 'temp_future.txt')

print("2. Calculando ventana de observación para SS 433...")

def get_optimal_observation_window(mjd, ra_hr=19.19, duration=6.0):
    T = (mjd - 51544.5) / 36525.0
    gmst_0h = (6.697374558 + 2400.051336 * T + 0.000025862 * T**2) % 24.0
    lon_hr = 10.0 / 15.0 
    transit_utc = ((ra_hr - gmst_0h - lon_hr) % 24.0) * 0.99726958
    start_hr = transit_utc - (duration / 2.0)
    end_hr = transit_utc + (duration / 2.0)
    if end_hr >= 23.9:
        shift = end_hr - 23.9
        start_hr -= shift; end_hr -= shift
    elif start_hr <= 0.1:
        shift = 0.1 - start_hr
        start_hr += shift; end_hr += shift
    return start_hr, end_hr

obs_mjd = 59050.0
ra_source = 19.19
dec_source = 4.98
duration_hr = 6.0

start_hr, end_hr = get_optimal_observation_window(obs_mjd, ra_hr=ra_source, duration=duration_hr)

print("3. Generando visibilidades simuladas...")

im_dummy = eh.image.make_empty(32, 100*eh.RADPERUAS, ra=ra_source, dec=dec_source, rf=5.0e9)
im_dummy.mjd = obs_mjd 
im_dummy = im_dummy.add_gauss(1.0, [10*eh.RADPERUAS, 10*eh.RADPERUAS, 0, 0, 0])

obs_params = {
    'tint': 12.0,       
    'tadv': 240.0,      
    'tstart': start_hr, 
    'tstop': end_hr, 
    'bw': 4e9,          
    'elevmin': 10,      
    'ttype': 'nfft',
    'sgrscat': False, 
    'add_th_noise': False,
    'mjd': obs_mjd 
}

obs_current = im_dummy.observe(array_current, **obs_params)
obs_future = im_dummy.observe(array_future, **obs_params)

print("4. Extrayendo datos...")

u_curr = np.array(obs_current.data['u'], dtype=float) / 1e9
v_curr = np.array(obs_current.data['v'], dtype=float) / 1e9
t_curr = np.array(obs_current.data['time'], dtype=float)

u_fut = np.array(obs_future.data['u'], dtype=float) / 1e9
v_fut = np.array(obs_future.data['v'], dtype=float) / 1e9
t_fut = np.array(obs_future.data['time'], dtype=float)

t_curr_rel = t_curr - start_hr
t_fut_rel = t_fut - start_hr

limite_global = np.max([np.max(np.abs(u_fut)), np.max(np.abs(v_fut))]) * 1.05

# --- CREACIÓN DE DEGRADADOS (Sin Blancos) ---
cmap_azul = LinearSegmentedColormap.from_list('AzulSeguro', ['#00174A', '#005CE6', '#33CCFF'])
cmap_verde = LinearSegmentedColormap.from_list('VerdeSeguro', ['#003300', '#009900', '#66FF66'])

print("5. Generando gráficos en formato PDF dentro de la carpeta...")

def guardar_grafico_individual(u, v, t_rel, titulo, nombre_archivo, mapa_color):
    fig_ind, ax_ind = plt.subplots(figsize=(8, 7), facecolor='white')
    
    sc = ax_ind.scatter(u, v, c=t_rel, cmap=mapa_color, vmin=0, vmax=duration_hr, s=12, alpha=1.0, edgecolors='none', rasterized=True)
    ax_ind.scatter(-u, -v, c=t_rel, cmap=mapa_color, vmin=0, vmax=duration_hr, s=12, alpha=1.0, edgecolors='none', rasterized=True)
    
    ax_ind.set_title(titulo, fontweight='bold', fontsize=16, pad=15)
    ax_ind.set_xlabel("East-West Frequency (u) in G$\lambda$", fontsize=12)
    ax_ind.set_ylabel("North-South Frequency (v) in G$\lambda$", fontsize=12)
    
    ax_ind.set_xlim(limite_global, -limite_global) 
    ax_ind.set_ylim(-limite_global, limite_global)
    ax_ind.set_aspect('equal')
    ax_ind.grid(True, linestyle='--', alpha=0.3)
    
    divider = make_axes_locatable(ax_ind)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig_ind.colorbar(sc, cax=cax)
    cbar.set_ticks(np.linspace(0, duration_hr, 5))
    cbar.set_ticklabels([f"+{h:.1f}h" for h in np.linspace(0, duration_hr, 5)])
    cbar.set_label("Elapsed Observation Time", rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    ruta_completa = os.path.join(OUT_DIR, nombre_archivo)
    plt.savefig(ruta_completa, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig_ind) 
    print(f"   -> Guardado: {ruta_completa}")

# 1 y 2. Guardar Individuales (EVN en Azul, Futuro en Verde)
guardar_grafico_individual(u_curr, v_curr, t_curr_rel, "Current EVN Array", "01_uv_coverage_evn_only.pdf", cmap_azul)
guardar_grafico_individual(u_fut, v_fut, t_fut_rel, "EVN + Africa Array", "02_uv_coverage_future_network.pdf", cmap_verde)


# 3. PANEL DOBLE APILADO VERTICAL (Usando Azul y Verde)
fig_comb, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14), facecolor='white')

def plot_uv_safe(ax, u, v, t_rel, title, mapa_color):
    sc = ax.scatter(u, v, c=t_rel, cmap=mapa_color, vmin=0, vmax=duration_hr, s=8, alpha=1.0, edgecolors='none', rasterized=True)
    ax.scatter(-u, -v, c=t_rel, cmap=mapa_color, vmin=0, vmax=duration_hr, s=8, alpha=1.0, edgecolors='none', rasterized=True)
    
    ax.set_title(title, fontweight='bold', fontsize=15, pad=15)
    ax.set_xlabel("East-West Frequency (u) in G$\lambda$", fontsize=12)
    ax.set_ylabel("North-South Frequency (v) in G$\lambda$", fontsize=12)
    
    ax.set_xlim(limite_global, -limite_global) 
    ax.set_ylim(-limite_global, limite_global)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    return sc

plot_uv_safe(ax1, u_curr, v_curr, t_curr_rel, "EVN Array", cmap_azul)
sc_comb = plot_uv_safe(ax2, u_fut, v_fut, t_fut_rel, "EVN + Africa Array", cmap_verde)

divider_comb = make_axes_locatable(ax2)
cax_comb = divider_comb.append_axes("bottom", size="4%", pad=0.7)
cbar_comb = fig_comb.colorbar(sc_comb, cax=cax_comb, orientation='horizontal')
cbar_comb.set_ticks(np.linspace(0, duration_hr, 5))
cbar_comb.set_ticklabels([f"+{h:.1f}h" for h in np.linspace(0, duration_hr, 5)])
cbar_comb.set_label("Elapsed Observation Time", labelpad=10, fontsize=12, fontweight='bold')

ruta_comb = os.path.join(OUT_DIR, "03_uv_coverage_combined.pdf")
plt.savefig(ruta_comb, format='pdf', bbox_inches='tight', facecolor='white')
print(f"   -> Guardado: {ruta_comb}")
plt.close(fig_comb)


# 4. GRÁFICO SUPERPUESTO (EVN en Azul + Nuevas Líneas Africanas en Verde)
fig_overlay, ax_overlay = plt.subplots(figsize=(9, 8), facecolor='white')

# Dibujamos PRIMERO todo el array futuro (EVN + África) en VERDE
sc_fut = ax_overlay.scatter(u_fut, v_fut, c=t_fut_rel, cmap=cmap_verde, vmin=0, vmax=duration_hr, s=12, alpha=1.0, edgecolors='none', rasterized=True)
ax_overlay.scatter(-u_fut, -v_fut, c=t_fut_rel, cmap=cmap_verde, vmin=0, vmax=duration_hr, s=12, alpha=1.0, edgecolors='none', rasterized=True)

# Dibujamos ENCIMA las líneas base actuales (EVN) en AZUL
# Esto "tapa" la parte verde que coincide, dejando en verde solo las líneas nuevas
sc_curr = ax_overlay.scatter(u_curr, v_curr, c=t_curr_rel, cmap=cmap_azul, vmin=0, vmax=duration_hr, s=12, alpha=1.0, edgecolors='none', rasterized=True)
ax_overlay.scatter(-u_curr, -v_curr, c=t_curr_rel, cmap=cmap_azul, vmin=0, vmax=duration_hr, s=12, alpha=1.0, edgecolors='none', rasterized=True)

ax_overlay.set_title("Overlaid Coverage: Core EVN vs Added African Baselines", fontweight='bold', fontsize=16, pad=15)
ax_overlay.set_xlabel("East-West Frequency (u) in G$\lambda$", fontsize=12)
ax_overlay.set_ylabel("North-South Frequency (v) in G$\lambda$", fontsize=12)
ax_overlay.set_xlim(limite_global, -limite_global) 
ax_overlay.set_ylim(-limite_global, limite_global)
ax_overlay.set_aspect('equal')
ax_overlay.grid(True, linestyle='--', alpha=0.3)

divider_overlay = make_axes_locatable(ax_overlay)
cax1 = divider_overlay.append_axes("bottom", size="4%", pad=0.6)
cax2 = divider_overlay.append_axes("bottom", size="4%", pad=0.7) 

cbar1 = fig_overlay.colorbar(sc_curr, cax=cax1, orientation='horizontal')
cbar1.set_ticks(np.linspace(0, duration_hr, 5))
cbar1.set_ticklabels([f"+{h:.1f}h" for h in np.linspace(0, duration_hr, 5)])
cbar1.set_label("Elapsed Time: Core EVN Baselines (Blue Gradient)", labelpad=10, fontsize=12, fontweight='bold')

cbar2 = fig_overlay.colorbar(sc_fut, cax=cax2, orientation='horizontal')
cbar2.set_ticks(np.linspace(0, duration_hr, 5))
cbar2.set_ticklabels([f"+{h:.1f}h" for h in np.linspace(0, duration_hr, 5)])
cbar2.set_label("Elapsed Time: Added African Baselines (Green Gradient)", labelpad=10, fontsize=12, fontweight='bold')

ruta_overlay = os.path.join(OUT_DIR, "04_uv_coverage_overlaid.pdf")
plt.savefig(ruta_overlay, format='pdf', bbox_inches='tight', facecolor='white')
print(f"   -> Guardado: {ruta_overlay}")
plt.close(fig_overlay)

print(f"\n¡Listo! Tienes todo el material exportado de forma coherente en la carpeta '{OUT_DIR}'.")