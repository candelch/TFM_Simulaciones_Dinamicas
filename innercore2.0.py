import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. CONFIGURACIÓN VISUAL ---
fov = 150 * eh.RADPERUAS
npix = 256
freq_obs = 5e9

# --- 2. FÍSICA ---
total_frames = 250    
jet_velocity = 0.4
precession_period = 800.0  

opening_angle = 0.55
inclination_angle = np.radians(0)

# --- 3. CONFIGURACIÓN DE BLOBS ---
num_blobs = 20
spacing = 50.0

blob_core_size = 6 * eh.RADPERUAS
blob_halo_size = 14 * eh.RADPERUAS
trail_persistence = 0.85 

# --- 4. FUNCIÓN GENERADORA ---
def get_frame_image(frame_idx):
    im = eh.image.make_empty(npix, fov, 0, 0, rf=freq_obs)
    
    # Core Fijo: 600.0
    # Valor alto para que se mantenga en la zona "caliente" de la paleta
    im = im.add_gauss(600.0, (5*eh.RADPERUAS, 5*eh.RADPERUAS, 0, 0, 0))
    
    max_radius = fov/eh.RADPERUAS/2
    fade_start = max_radius * 0.85
    phase_offset = np.pi / 2.0

    for i in range(num_blobs):
        birth_time = (i * spacing)
        
        if birth_time > 205:
            continue
            
        age = frame_idx - birth_time
        
        if age < 0: continue 
        r = age * jet_velocity
        if r > max_radius: continue 
        
        # Cinemática
        phase = -2 * np.pi * (birth_time / precession_period) + phase_offset
        
        # Geometría
        x_loc = r
        y_loc = r * opening_angle * np.sin(phase) * 0.8
        
        pa = inclination_angle
        x1 = x_loc * np.cos(pa) - y_loc * np.sin(pa)
        y1 = x_loc * np.sin(pa) + y_loc * np.cos(pa)
        x2, y2 = -x1, -y1
        
        # Opacidad
        opacity = 1.0
        if r > fade_start:
            opacity = (max_radius - r) / (max_radius - fade_start)
        
        # Blob Nuevo: 1500.0 con decaimiento rápido (45.0)
        # Al nacer, sumado al core, tendremos 2100 de brillo.
        flux = 1500.0 * np.exp(-r / 45.0) * opacity
        
        if flux > 0.01:
            im = im.add_gauss(flux*0.2, (blob_halo_size, blob_halo_size, 0, x1*eh.RADPERUAS, y1*eh.RADPERUAS))
            im = im.add_gauss(flux*0.2, (blob_halo_size, blob_halo_size, 0, x2*eh.RADPERUAS, y2*eh.RADPERUAS))
            
            im = im.add_gauss(flux, (blob_core_size, blob_core_size, 0, x1*eh.RADPERUAS, y1*eh.RADPERUAS))
            im = im.add_gauss(flux, (blob_core_size, blob_core_size, 0, x2*eh.RADPERUAS, y2*eh.RADPERUAS))
            
    return im.imvec.reshape((npix, npix))

# --- 5. CALIBRACIÓN  ---
print("Calibrando brillo...")
test_data = get_frame_image(100)
max_val_real = np.max(test_data) 

vmax_calibrated = max_val_real * 0.4 

print(f"Max Real: {max_val_real:.2f}")
print(f"Vmax (Saturado): {vmax_calibrated:.4f}")

# --- 6. FIGURA ---
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

limit = fov/eh.RADPERUAS/2
extent = [-limit, limit, -limit, limit]

im_display = ax.imshow(np.zeros((npix, npix)), animated=True, cmap='jet', 
                       origin='lower', extent=extent, vmin=0, vmax=vmax_calibrated,
                       interpolation='bilinear')

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.axis('off')
ax.set_title("SS433: Saturated Intensity", color='white', fontsize=10, alpha=0.7)

trail_buffer = np.zeros((npix, npix))

# --- 7. BUCLE ---
def update(frame_idx):
    global trail_buffer
    
    current_frame = get_frame_image(frame_idx)
    
    trail_buffer = trail_buffer * trail_persistence
    trail_buffer = trail_buffer + (current_frame * 0.3)
    
    final_image = np.maximum(current_frame, trail_buffer)
    
    # Ajustamos la máscara para no mostrar ruido de fondo
    # (Como hemos bajado mucho el vmax, hay que subir un poco el umbral de corte)
    masked_image = np.ma.masked_less(final_image, vmax_calibrated * 0.05)
    
    im_display.set_data(masked_image)
    return [im_display]

print(f"Generando animación saturada ({total_frames} frames)...")
ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)
ani.save('ss433_saturated.gif', writer='pillow', fps=15)
print("¡GIF guardado!")
plt.close()
