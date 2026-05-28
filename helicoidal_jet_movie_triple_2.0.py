import matplotlib
matplotlib.use('Agg') 

import ehtim as eh
from ehtim.calibrating.self_cal import self_cal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import os
import imageio.v2 as imageio
import glob
import gc 
import warnings
import datetime 

# --- ESTILO CIENTÍFICO GLOBAL DE MATPLOTLIB ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Paleta de colores profesional (Colorbrewer Dark2)
C_EVN = '#1B9E77'   # Esmeralda oscuro
C_AFR = '#7570B3'   # Púrpura apagado
C_GLOBAL = ['#1B9E77', '#D95F02', '#7570B3'] # Esmeralda, Óxido, Púrpura para los 3 escenarios

# Intento de importación de scikit-image para SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("[!] Advertencia: la librería 'scikit-image' no está instalada.")
    print("[!] El cálculo de SSIM se guardará como 0.")

# --- 1. CONFIGURACIÓN E ID DE EJECUCIÓN ---
warnings.filterwarnings("ignore")
print("--- SS 433: BARRIDO CIENTÍFICO | MASTER PLOTS & PDF EXPORT ---")

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"sim_run_dual_{RUN_ID}"
print(f"-> Los resultados base se guardarán en: {OUT_DIR}/")

ESCENARIOS = {
    "01_Realista": 1.0,     
    "02_Exagerado": 5.0,    
    "03_Extremo": 20.0      
}

# =============================================================================
# 2. PARÁMETROS FÍSICOS Y DE TIEMPO
# =============================================================================
RF_HZ = 5.0e9 
N_DAYS = 43 
FRAMES_PER_DAY = 4.76 

PHYSICAL_SIZE_REF = 40000.0  
OBSERVATION_FOV = 60000.0    
COMMON_NPIX = 256            

TOY_FOV = 150.0
SCALE_FACTOR = PHYSICAL_SIZE_REF / TOY_FOV 

FLUX_SCALE = 1e-4 

BLOB_CORE_UAS = 6.0 * SCALE_FACTOR     
BLOB_HALO_UAS = 14.0 * SCALE_FACTOR    
DECAY_UAS = 100.0 * SCALE_FACTOR 

VELOCITY_FACTOR = 0.4 * SCALE_FACTOR 
PRECESSION_PERIOD = 800.0 
OPENING_ANGLE = 0.55 
INCLINATION = 0.0 
PHASE_OFFSET = np.pi / 2.0
SPACING_FRAMES = 52.0  

RA_SOURCE = 19.19
DEC_SOURCE = 4.98

MJD_START = 59050.0

# =============================================================================
# 3. GENERADOR DE MODELO Y FUNCIONES DE MÉTRICAS ROBUSTAS
# =============================================================================
def get_toy_model_frame(frame_idx, base_image):
    im = base_image.copy()
    im = im.add_gauss(3000.0 * FLUX_SCALE, (5*SCALE_FACTOR*eh.RADPERUAS, 5*SCALE_FACTOR*eh.RADPERUAS, 0, 0, 0))

    num_blobs = 20
    max_radius_uas = PHYSICAL_SIZE_REF / 1.6 
    fade_start_uas = max_radius_uas * 0.95 

    for i in range(num_blobs):
        birth_time = i * SPACING_FRAMES
        if birth_time > 205: continue
        age = frame_idx - birth_time
        if age < 0: continue
        r = age * VELOCITY_FACTOR
        if r > max_radius_uas: continue

        phase = -2 * np.pi * (birth_time / PRECESSION_PERIOD) + PHASE_OFFSET
        x_loc = r
        y_loc = r * OPENING_ANGLE * np.sin(phase) * 0.8
        pa = INCLINATION
        x1 = x_loc * np.cos(pa) - y_loc * np.sin(pa)
        y1 = x_loc * np.sin(pa) + y_loc * np.cos(pa)
        x2, y2 = -x1, -y1

        opacity = 1.0
        if r > fade_start_uas:
            opacity = (max_radius_uas - r) / (max_radius_uas - fade_start_uas)

        flux = 7500.0 * FLUX_SCALE * np.exp(-r / DECAY_UAS) * opacity

        if flux > (0.05 * FLUX_SCALE): 
            im = im.add_gauss(flux*0.2, (BLOB_HALO_UAS*eh.RADPERUAS, BLOB_HALO_UAS*eh.RADPERUAS, 0, x1*eh.RADPERUAS, y1*eh.RADPERUAS))
            im = im.add_gauss(flux,     (BLOB_CORE_UAS*eh.RADPERUAS, BLOB_CORE_UAS*eh.RADPERUAS, 0, x1*eh.RADPERUAS, y1*eh.RADPERUAS))
            im = im.add_gauss(flux*0.2, (BLOB_HALO_UAS*eh.RADPERUAS, BLOB_HALO_UAS*eh.RADPERUAS, 0, x2*eh.RADPERUAS, y2*eh.RADPERUAS))
            im = im.add_gauss(flux,     (BLOB_CORE_UAS*eh.RADPERUAS, BLOB_CORE_UAS*eh.RADPERUAS, 0, x2*eh.RADPERUAS, y2*eh.RADPERUAS))

    return im

def calculate_nxcorr(im_true, im_rec):
    d_true = np.nan_to_num(im_true.imarr())
    d_rec = np.nan_to_num(im_rec.imarr())
    norm = np.sqrt(np.sum(d_true**2) * np.sum(d_rec**2))
    if norm == 0: return 0
    return np.sum(d_true * d_rec) / norm

def calculate_chi2_amp(obs, im_rec):
    try:
        obs_sim = im_rec.observe_same_nonoise(obs, ttype='nfft')
        amp_obs = obs.unpack(['amp', 'sigma'])
        amp_sim = obs_sim.unpack(['amp'])
        mask = amp_obs['sigma'] > 0
        chi2 = np.sum(((amp_obs['amp'][mask] - amp_sim['amp'][mask]) / amp_obs['sigma'][mask])**2) / np.sum(mask)
        return chi2
    except: return np.nan

def calculate_chi2_cphase(obs_original, im_rec):
    try:
        if getattr(obs_original, 'cphase', None) is None or len(obs_original.cphase) == 0:
            obs_original.add_cphase()

        obs_sim = im_rec.observe_same_nonoise(obs_original, ttype='nfft')
        obs_sim.add_cphase()

        cp_obs = obs_original.cphase
        cp_sim = obs_sim.cphase

        if len(cp_obs) != len(cp_sim):
            return np.nan

        c1 = cp_obs['cphase']
        c2 = cp_sim['cphase']
        sig = cp_obs['sigmacp']

        mask = sig > 0
        if np.sum(mask) == 0: return np.nan

        diff_rad = np.angle(np.exp(1j * np.radians(c1)) * np.exp(-1j * np.radians(c2)))
        chi2 = np.sum((diff_rad[mask] / np.radians(sig[mask]))**2) / np.sum(mask)
        return chi2

    except Exception as e:
        return np.nan

def calculate_dynamic_range(im_rec):
    d_rec = np.nan_to_num(im_rec.imarr())
    peak = np.max(d_rec)
    background = d_rec[0:20, 0:20] 
    rms_noise = np.sqrt(np.mean(background**2))
    if rms_noise == 0: return 0
    return peak / rms_noise

def calculate_ssim(im_true, im_rec):
    if not HAS_SSIM: return 0.0
    d_true = np.nan_to_num(im_true.imarr())
    d_rec = np.nan_to_num(im_rec.imarr())
    v_max = np.max(d_true)
    return ssim(d_true, d_rec, data_range=v_max)

def save_scientific_pdf(im_arr, beamparams, filename, title, vmin=None, vmax=None, cmap='jet'):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    
    limit_mas = (OBSERVATION_FOV / 2.0) / 1000.0
    extent_mas = [limit_mas, -limit_mas, -limit_mas, limit_mas]
    
    im_arr_mJy = im_arr * 1000.0
    
    calc_vmin = vmin * 1000.0 if vmin is not None else np.min(im_arr_mJy)
    calc_vmax = vmax * 1000.0 if vmax is not None else np.max(im_arr_mJy)
    
    if calc_vmax <= calc_vmin: 
        calc_vmax = calc_vmin + 1e-5
        
    im = ax.imshow(im_arr_mJy, origin='lower', cmap=cmap, extent=extent_mas, vmin=calc_vmin, vmax=calc_vmax)
    
    if beamparams is not None:
        draw_beam_ellipse(ax, beamparams)
        
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel("Relative RA (mas)")
    ax.set_ylabel("Relative Dec (mas)")
    
    # Blindaje de ejes para evitar auto-zooms
    ax.set_xlim(limit_mas, -limit_mas)
    ax.set_ylim(-limit_mas, limit_mas)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity (mJy/pixel)")
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
# =============================================================================
# 4. MOTOR DE PELÍCULAS (CON CORRECCIÓN DE CENTRADO DE TIEMPO)
# =============================================================================
def observe_movie_track(base_frame_idx, base_image, array, obs_mjd, start_hr, end_hr, factor_exageracion, fps_sim=4):
    pad = 0.1 
    movie_start = start_hr - pad
    movie_end = end_hr + pad
    duration_hr = movie_end - movie_start
    total_frames = int(duration_hr * fps_sim) + 1 
    
    im_array_list = []
    times_utc = list(np.linspace(movie_start, movie_end, total_frames))
    
    im_truth = None
    mid_idx = total_frames // 2
    
    for i, t_utc in enumerate(times_utc):
        frac = i / (total_frames - 1)
        
        # CORRECCIÓN: Centrar el marco temporal (-0.5 a +0.5)
        centered_frac = frac - 0.5 
        fractional_advance = centered_frac * (duration_hr / 24.0) * FRAMES_PER_DAY * factor_exageracion
        current_frame = base_frame_idx + fractional_advance
        
        im = get_toy_model_frame(current_frame, base_image).center()
        im.mjd = obs_mjd 
        im_array_list.append(im.imarr()) 
        
        if i == mid_idx:
            im_truth = im.copy()
            
    movie = eh.movie.Movie(im_array_list, times=times_utc, psize=base_image.psize, 
                           ra=base_image.ra, dec=base_image.dec, rf=base_image.rf, mjd=obs_mjd)
    
    template_obs = im_truth.observe(array, tint=12.0, tadv=240.0, tstart=start_hr, tstop=end_hr, 
                                    bw=128e6, ttype='nfft', elevmin=10, ampcal=True)

    if len(template_obs.data) == 0: return None, None
        
    times_u, counts = np.unique(template_obs.data['time'], return_counts=True)
    valid_times = times_u[counts >= 3] 
    if len(valid_times) == 0: return None, None
        
    mask = np.isin(template_obs.data['time'], valid_times)
    template_obs.data = template_obs.data[mask]
    
    try:
        obs = movie.observe_same(template_obs, ttype='nfft', ampcal=True)
    except Exception as e:
        return None, None
        
    del im_array_list, movie
    gc.collect()
    
    return obs, im_truth

# =============================================================================
# 5. IMAGING PIPELINE 
# =============================================================================
def run_imaging_intensity(obs, fov_uas, npix, truth_image, previous_image=None, array_name=""):
    zbl_flux = np.sum(truth_image.imvec) 
    empty = eh.image.make_square(obs, npix, fov_uas * eh.RADPERUAS)
    sz = (fov_uas * 0.4) * eh.RADPERUAS 
    base_prior = empty.add_gauss(zbl_flux, (sz, sz, 0, 0, 0))
    
    try:
        beamparams = obs.fit_beam()
        res = obs.res()
        
        if previous_image is not None:
            init_im = previous_image.blur_circ(res * 0.4) 
            curr_flux = np.sum(init_im.imvec)
            if curr_flux > 1e-10 and not np.isnan(curr_flux):
                init_im.imvec *= (zbl_flux / curr_flux)
            else: init_im = base_prior
        else: init_im = base_prior

        reg_params = {'tv': 0.05, 'l1': 0.01}

        imgr1 = eh.imager.Imager(obs, init_im, init_im, None,
                                data_term={'cphase': 100, 'amp': 100},
                                reg_term=reg_params, maxit=80, ttype='nfft', show_updates=False)
        out1 = imgr1.make_image(clipfloor=1e-9)

        init2 = out1.blur_circ(res * 0.3) 
        imgr2 = eh.imager.Imager(obs, init2, init2, None,
                                data_term={'cphase': 100, 'amp': 100},
                                reg_term=reg_params, maxit=80, ttype='nfft', show_updates=False)
        out2 = imgr2.make_image(clipfloor=1e-9)

        obs_calibrated = self_cal(obs, out2, method='phase', ttype='nfft')

        init_3 = out2.blur_circ(res * 0.2)
        imgr3 = eh.imager.Imager(obs_calibrated, init_3, init_3, None,
                                data_term={'cphase': 100, 'amp': 100, 'vis': 30}, 
                                reg_term=reg_params, maxit=100, ttype='nfft', show_updates=False)
        out_final = imgr3.make_image(clipfloor=1e-9) 
        
        chi2_cp = calculate_chi2_cphase(obs, out_final)
        chi2_amp = calculate_chi2_amp(obs_calibrated, out_final)
        
        del imgr1, imgr2, imgr3, obs_calibrated
        
        aligned_tuple = truth_image.align_images([out_final])
        img_unblurred = aligned_tuple[0][0]
        img_blurred = img_unblurred.blur_gauss(beamparams, 0.4)
        return img_blurred, img_unblurred, beamparams, chi2_cp, chi2_amp
        
    except Exception as e:
        print(f"      [AVISO CRÍTICO] Fallback activado en {array_name}: {e}")
        aligned_fallback_tuple = truth_image.align_images([base_prior])
        img_unblurred = aligned_fallback_tuple[0][0]
        img_blurred = img_unblurred.blur_gauss(obs.fit_beam(), 0.4)
        return img_blurred, img_unblurred, obs.fit_beam(), np.nan, np.nan

# =============================================================================
# 6. ARRAYS Y GEOMETRÍA
# =============================================================================
header = "#NAME X Y Z SEFDR SEFDR FR_PAR FR_ELEV FR_OFF DR_RE DR_IM DL_RE DL_IM"

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

def draw_beam_ellipse(ax, beamparams):
    bmaj_rad, bmin_rad, bpa_rad = beamparams
    bmaj_uas = bmaj_rad / eh.RADPERUAS
    bmin_uas = bmin_rad / eh.RADPERUAS
    angle_mpl = np.degrees(bpa_rad) 
    xlims = ax.get_xlim() 
    ylims = ax.get_ylim()
    x_pos = xlims[0] + 0.1 * (xlims[1] - xlims[0])
    y_pos = ylims[0] + 0.1 * (ylims[1] - ylims[0])
    # Color de borde blanco y más grueso para resaltar sobre el fondo azul
    beam_ellipse = Ellipse(xy=(x_pos, y_pos), width=bmin_uas/1000.0, height=bmaj_uas/1000.0, angle=angle_mpl,
                           facecolor='none', edgecolor='white', linewidth=1.5, alpha=1.0)
    ax.add_patch(beam_ellipse)

if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
TXT_EVN = f"{OUT_DIR}/evn_only.txt"
TXT_AFRICA = f"{OUT_DIR}/evn_africa.txt"

with open(TXT_EVN, "w") as f:
    f.write(header + "\n")
    for st in evn_stations: f.write(f"{st[0]:<8} {st[1]:<13.1f} {st[2]:<13.1f} {st[3]:<13.1f} {st[4]:<7.1f} {st[4]:<7.1f} 1 0 0 0 0 0 0\n")

with open(TXT_AFRICA, "w") as f:
    f.write(header + "\n")
    for st in evn_stations + africa_stations: f.write(f"{st[0]:<8} {st[1]:<13.1f} {st[2]:<13.1f} {st[3]:<13.1f} {st[4]:<7.1f} {st[4]:<7.1f} 1 0 0 0 0 0 0\n")

array_evn = eh.array.load_txt(TXT_EVN)
array_africa = eh.array.load_txt(TXT_AFRICA)

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

# =============================================================================
# 7. BUCLE PRINCIPAL DE EXPERIMENTOS
# =============================================================================

global_metrics = {}

def save_single_frame(data_array, title, filename, beam=None, vmax=None, extent=None):
    fig_ind, ax_ind = plt.subplots(figsize=(6, 5), facecolor='white')
    im_ind = ax_ind.imshow(data_array, origin='lower', cmap='jet', vmin=0, vmax=vmax, extent=extent)
    ax_ind.set_title(title, pad=15, fontweight='bold')
    ax_ind.set_xlabel("Relative RA (mas)")
    ax_ind.set_ylabel("Relative Dec (mas)")
    if beam is not None: draw_beam_ellipse(ax_ind, beam)
    fig_ind.colorbar(im_ind, ax=ax_ind, fraction=0.046, pad=0.04).set_label("Intensity (mJy/pixel)")
    plt.tight_layout()
    fig_ind.savefig(filename, facecolor='white', bbox_inches='tight')
    plt.close(fig_ind)

for sim_name, current_factor in ESCENARIOS.items():
    print(f"\n" + "="*70)
    print(f"=== INICIANDO ESCENARIO: {sim_name} (Factor: {current_factor}x) ===")
    print("="*70)
    
    SIM_DIR = f"{OUT_DIR}/{sim_name}"
    S_TEMP = f"{SIM_DIR}/temp_frames"
    S_DIAG = f"{SIM_DIR}/diagnostics"
    S_PDFS = f"{SIM_DIR}/pdf_maps_sample" 
    
    S_UVFITS = f"{SIM_DIR}/uvfits"
    S_UVFITS_EVN = f"{S_UVFITS}/evn"
    S_UVFITS_AFR = f"{S_UVFITS}/afr"
    
    for fldr in [SIM_DIR, S_TEMP, S_DIAG, S_PDFS, S_UVFITS, S_UVFITS_EVN, S_UVFITS_AFR]:
        os.makedirs(fldr, exist_ok=True)
        
    for sub_dir in ["mod", "evn", "afr"]:
        os.makedirs(f"{S_TEMP}/{sub_dir}", exist_ok=True)
        
    PREV_IMG_EVN = None
    PREV_IMG_AFRICA = None
    VMAX_MJY = None
    
    limit_mas = (OBSERVATION_FOV / 2.0) / 1000.0
    extent_mas = [limit_mas, -limit_mas, -limit_mas, limit_mas]

    metrics = {
        'frames': [], 'rmse_evn': [], 'rmse_afr': [], 
        'nxcorr_evn': [], 'nxcorr_afr': [],
        'chi2_cp_evn': [], 'chi2_amp_evn': [],
        'chi2_cp_afr': [], 'chi2_amp_afr': [],
        'dr_evn': [], 'dr_afr': [],
        'ssim_evn': [], 'ssim_afr': []
    }

    successful_days = 0
    current_offset = 0  
    MAX_ATTEMPTS = 150  

    while successful_days < N_DAYS:
        if current_offset >= MAX_ATTEMPTS:
            print("\n[!] LÍMITE DE INTENTOS ALCANZADO.")
            break
            
        current_mjd = MJD_START + current_offset 
        base_frame_idx = 50 + (current_offset * FRAMES_PER_DAY)
        obs_mjd = MJD_START 
        
        print(f"\n--- {sim_name} | Día {successful_days+1}/{N_DAYS} ---")
        
        base = eh.image.make_empty(COMMON_NPIX, OBSERVATION_FOV * eh.RADPERUAS, ra=RA_SOURCE, dec=DEC_SOURCE, rf=RF_HZ)
        dinamic_start, dinamic_end = get_optimal_observation_window(obs_mjd, duration=6.0)
        
        obs_evn, im_model_truth = observe_movie_track(base_frame_idx, base, array_evn, obs_mjd, dinamic_start, dinamic_end, current_factor)
        obs_africa, _ = observe_movie_track(base_frame_idx, base, array_africa, obs_mjd, dinamic_start, dinamic_end, current_factor)

        if obs_evn is None or obs_africa is None:
            current_offset += 1
            continue

        obs_evn.mjd = current_mjd
        obs_africa.mjd = current_mjd

        uvfits_evn_path = f"{S_UVFITS_EVN}/obs_day_{successful_days+1:02d}.uvfits"
        uvfits_afr_path = f"{S_UVFITS_AFR}/obs_day_{successful_days+1:02d}.uvfits"
        obs_evn.save_uvfits(uvfits_evn_path)
        obs_africa.save_uvfits(uvfits_afr_path)

        print("    -> Imager EVN...")
        img_rec_evn, img_unb_evn, bp_evn, c2cp_evn, c2a_evn = run_imaging_intensity(obs_evn, OBSERVATION_FOV, COMMON_NPIX, im_model_truth, PREV_IMG_EVN, "EVN")
        PREV_IMG_EVN = img_rec_evn.copy() 
        
        print("    -> Imager EVN+AFRICA...")
        img_rec_afr, img_unb_afr, bp_afr, c2cp_afr, c2a_afr = run_imaging_intensity(obs_africa, OBSERVATION_FOV, COMMON_NPIX, im_model_truth, PREV_IMG_AFRICA, "AFRICA")
        PREV_IMG_AFRICA = img_rec_afr.copy()

        d_mod = np.nan_to_num(im_model_truth.imarr())
        d_evn = np.nan_to_num(img_rec_evn.imarr())
        d_afr = np.nan_to_num(img_rec_afr.imarr())
        
        if successful_days == N_DAYS // 2:
            print("    -> [Guardando Muestra de Beams/Images en PDF vectorial]")
            fov_rad = OBSERVATION_FOV * eh.RADPERUAS
            
            d_img_evn = obs_evn.dirtyimage(COMMON_NPIX, fov_rad)
            d_beam_evn = obs_evn.dirtybeam(COMMON_NPIX, fov_rad)
            cb_evn = eh.image.make_empty(COMMON_NPIX, fov_rad, ra=RA_SOURCE, dec=DEC_SOURCE, rf=RF_HZ)
            cb_evn = cb_evn.add_gauss(1.0, (bp_evn[0], bp_evn[1], bp_evn[2], 0, 0))
            
            save_scientific_pdf(d_img_evn.imarr(), None, f"{S_PDFS}/dirty_image_evn_midpoint.pdf", "Dirty Image (EVN)", cmap='inferno')
            save_scientific_pdf(d_beam_evn.imarr(), None, f"{S_PDFS}/dirty_beam_evn_midpoint.pdf", "Dirty Beam (EVN)", cmap='coolwarm')
            save_scientific_pdf(cb_evn.imarr(), None, f"{S_PDFS}/clean_beam_evn_midpoint.pdf", "Clean Beam (EVN)", cmap='inferno')
            save_scientific_pdf(img_rec_evn.imarr(), bp_evn, f"{S_PDFS}/clean_image_evn_midpoint.pdf", "Clean Image (EVN)")
            
            d_img_afr = obs_africa.dirtyimage(COMMON_NPIX, fov_rad)
            d_beam_afr = obs_africa.dirtybeam(COMMON_NPIX, fov_rad)
            cb_afr = eh.image.make_empty(COMMON_NPIX, fov_rad, ra=RA_SOURCE, dec=DEC_SOURCE, rf=RF_HZ)
            cb_afr = cb_afr.add_gauss(1.0, (bp_afr[0], bp_afr[1], bp_afr[2], 0, 0))
            
            save_scientific_pdf(d_img_afr.imarr(), None, f"{S_PDFS}/dirty_image_afr_midpoint.pdf", "Dirty Image (EVN+Africa)", cmap='inferno')
            save_scientific_pdf(d_beam_afr.imarr(), None, f"{S_PDFS}/dirty_beam_afr_midpoint.pdf", "Dirty Beam (EVN+Africa)", cmap='coolwarm')
            save_scientific_pdf(cb_afr.imarr(), None, f"{S_PDFS}/clean_beam_afr_midpoint.pdf", "Clean Beam (EVN+Africa)", cmap='inferno')
            save_scientific_pdf(img_rec_afr.imarr(), bp_afr, f"{S_PDFS}/clean_image_afr_midpoint.pdf", "Clean Image (EVN+Africa)")
            save_scientific_pdf(d_mod, None, f"{S_PDFS}/ground_truth_midpoint.pdf", "Ground Truth Model")

        metrics['frames'].append(successful_days + 1)
        metrics['rmse_evn'].append(np.sqrt(np.mean((d_mod - d_evn)**2)))
        metrics['rmse_afr'].append(np.sqrt(np.mean((d_mod - d_afr)**2)))
        metrics['nxcorr_evn'].append(calculate_nxcorr(im_model_truth, img_rec_evn))
        metrics['nxcorr_afr'].append(calculate_nxcorr(im_model_truth, img_rec_afr))
        metrics['chi2_cp_evn'].append(c2cp_evn)
        metrics['chi2_amp_evn'].append(c2a_evn)
        metrics['chi2_cp_afr'].append(c2cp_afr)
        metrics['chi2_amp_afr'].append(c2a_afr)
        metrics['dr_evn'].append(calculate_dynamic_range(img_rec_evn))
        metrics['dr_afr'].append(calculate_dynamic_range(img_rec_afr))
        metrics['ssim_evn'].append(calculate_ssim(im_model_truth, img_rec_evn))
        metrics['ssim_afr'].append(calculate_ssim(im_model_truth, img_rec_afr))

        d_mod_mJy = d_mod * 1000.0
        d_evn_mJy = d_evn * 1000.0
        d_afr_mJy = d_afr * 1000.0

        if VMAX_MJY is None: VMAX_MJY = np.max(d_mod_mJy) * 0.6

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
        fig.suptitle(f"{sim_name} | Frame {successful_days+1:02d} | Intensity (mJy/px)", color='black', fontsize=18, fontweight='bold', y=0.98)
        
        im1 = ax1.imshow(d_mod_mJy, origin='lower', cmap='jet', vmin=0, vmax=VMAX_MJY, extent=extent_mas)
        ax1.set_title("Ground Truth", color='black', pad=15)
        ax1.set_xlabel("Relative RA (mas)")
        ax1.set_ylabel("Relative Dec (mas)")
        
        im2 = ax2.imshow(d_evn_mJy, origin='lower', cmap='jet', vmin=0, vmax=VMAX_MJY, extent=extent_mas)
        ax2.set_title(f"EVN (RMSE: {metrics['rmse_evn'][-1]*1000:.2f} mJy)", color='black', pad=15)
        ax2.set_xlabel("Relative RA (mas)")
        draw_beam_ellipse(ax2, bp_evn)
        
        im3 = ax3.imshow(d_afr_mJy, origin='lower', cmap='jet', vmin=0, vmax=VMAX_MJY, extent=extent_mas)
        ax3.set_title(f"EVN+Afr (RMSE: {metrics['rmse_afr'][-1]*1000:.2f} mJy)", color='black', pad=15)
        ax3.set_xlabel("Relative RA (mas)")
        draw_beam_ellipse(ax3, bp_afr)
        
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('white')
            ax.tick_params(colors='black', direction='in', labelcolor='black')
            for spine in ax.spines.values(): spine.set_color('black'); spine.set_linewidth(1.0)
            
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im3, cax=cbar_ax)
        cbar.set_label("mJy/pixel", fontsize=12)
                
        plt.tight_layout(rect=[0, 0, 0.9, 0.93])
        plt.savefig(f"{S_TEMP}/f_{successful_days:03d}.png", facecolor='white', bbox_inches='tight')
        plt.close()
        
        save_single_frame(d_mod_mJy, "Ground Truth", f"{S_TEMP}/mod/f_{successful_days:03d}.png", vmax=VMAX_MJY, extent=extent_mas)
        save_single_frame(d_evn_mJy, "EVN", f"{S_TEMP}/evn/f_{successful_days:03d}.png", bp_evn, vmax=VMAX_MJY, extent=extent_mas)
        save_single_frame(d_afr_mJy, "EVN + África", f"{S_TEMP}/afr/f_{successful_days:03d}.png", bp_afr, vmax=VMAX_MJY, extent=extent_mas)

        gc.collect()
        
        successful_days += 1
        current_offset += 1

    global_metrics[sim_name] = metrics
    
    print(f"-> Compilando GIF Maestro de {sim_name}...")
    images = [imageio.imread(f) for f in sorted(glob.glob(f"{S_TEMP}/f_*.png"))]
    imageio.mimsave(f'{SIM_DIR}/Animation_{sim_name}.gif', images, fps=10)
    
    print(f"-> Compilando GIFs individuales de {sim_name}...")
    for sub_dir, prefix in [("mod", "Model"), ("evn", "EVN"), ("afr", "EVN_Africa")]:
        images_ind = [imageio.imread(f) for f in sorted(glob.glob(f"{S_TEMP}/{sub_dir}/f_*.png"))]
        imageio.mimsave(f'{SIM_DIR}/Animation_{prefix}_{sim_name}.gif', images_ind, fps=10)

# =============================================================================
# 9. ANÁLISIS GLOBAL INTER-ESCENARIOS
# =============================================================================
print("\n" + "="*70)
print("=== GENERANDO SÍNTESIS GLOBAL Y MASTER PLOTS ===")
print("="*70)

SUM_DIR = f"{OUT_DIR}/global_summary"
if not os.path.exists(SUM_DIR): os.makedirs(SUM_DIR)

labels_esc = list(ESCENARIOS.keys())

# --- PLOT 1: Closure Phase Chi-Squared ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
for idx, sim in enumerate(labels_esc):
    f_list = global_metrics[sim]['frames']
    color = C_GLOBAL[idx]
    factor = int(ESCENARIOS[sim])
    ax.plot(f_list, global_metrics[sim]['chi2_cp_evn'], color=color, linestyle='-', linewidth=2, label=f'EVN ({factor}x)')
    ax.plot(f_list, global_metrics[sim]['chi2_cp_afr'], color=color, linestyle='--', linewidth=2, label=f'EVN+Afr ({factor}x)')

ax.set_title(r"Global Convergence: Final Closure Phase $\chi^2$", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Observation Day")
ax.set_ylabel(r"$\chi^2_{CPhase}$")
ax.set_yscale('log')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='black')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig.savefig(f'{SUM_DIR}/01_global_chi2_cphase.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# --- PLOT 2: Amplitude Chi-Squared ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
for idx, sim in enumerate(labels_esc):
    f_list = global_metrics[sim]['frames']
    color = C_GLOBAL[idx]
    factor = int(ESCENARIOS[sim])
    ax.plot(f_list, global_metrics[sim]['chi2_amp_evn'], color=color, linestyle='-', linewidth=2, label=f'EVN ({factor}x)')
    ax.plot(f_list, global_metrics[sim]['chi2_amp_afr'], color=color, linestyle='--', linewidth=2, label=f'EVN+Afr ({factor}x)')

ax.set_title(r"Global Convergence: Final Amplitude $\chi^2$", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Observation Day")
ax.set_ylabel(r"$\chi^2_{Amp}$")
ax.set_yscale('log')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='black')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig.savefig(f'{SUM_DIR}/02_global_chi2_amp.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# --- PLOT 3: Pareto Scatter ---
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
for idx, sim in enumerate(labels_esc):
    factor = int(ESCENARIOS[sim])
    ax.scatter(global_metrics[sim]['dr_evn'], global_metrics[sim]['nxcorr_evn'], 
               color=C_GLOBAL[idx], marker='o', alpha=0.6, label=f"EVN ({factor}x)")
    ax.scatter(global_metrics[sim]['dr_afr'], global_metrics[sim]['nxcorr_afr'], 
               color=C_GLOBAL[idx], marker='s', alpha=0.8, edgecolor='black', label=f"EVN+Afr ({factor}x)")

ax.set_title("Pareto Front: Fidelity vs. Dynamic Range", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Dynamic Range (Peak / RMS)")
ax.set_ylabel("Structural Fidelity (NXCORR)")
ax.set_xscale('log')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(f'{SUM_DIR}/03_pareto_scatter.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# --- PLOT 4: Global RMSE Evolution ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
for idx, sim in enumerate(labels_esc):
    f_list = global_metrics[sim]['frames']
    color = C_GLOBAL[idx]
    factor = int(ESCENARIOS[sim])
    ax.plot(f_list, np.array(global_metrics[sim]['rmse_evn']) * 1000.0, color=color, linestyle='-', linewidth=2, label=f'EVN ({factor}x)')
    ax.plot(f_list, np.array(global_metrics[sim]['rmse_afr']) * 1000.0, color=color, linestyle='--', linewidth=2, label=f'EVN+Afr ({factor}x)')

ax.set_title("Global RMSE Evolution", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Observation Day")
ax.set_ylabel("RMSE (mJy)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='black')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig.savefig(f'{SUM_DIR}/04_global_rmse.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# --- PLOT 5: Global Structural Fidelity (NXCORR) ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
for idx, sim in enumerate(labels_esc):
    f_list = global_metrics[sim]['frames']
    color = C_GLOBAL[idx]
    factor = int(ESCENARIOS[sim])
    ax.plot(f_list, global_metrics[sim]['nxcorr_evn'], color=color, linestyle='-', linewidth=2, label=f'EVN ({factor}x)')
    ax.plot(f_list, global_metrics[sim]['nxcorr_afr'], color=color, linestyle='--', linewidth=2, label=f'EVN+Afr ({factor}x)')

ax.set_title("Global Structural Fidelity (NXCORR)", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Observation Day")
ax.set_ylabel("Correlation")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='black')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig.savefig(f'{SUM_DIR}/05_global_nxcorr.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# --- PLOT 6: Global Dynamic Range ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
for idx, sim in enumerate(labels_esc):
    f_list = global_metrics[sim]['frames']
    color = C_GLOBAL[idx]
    factor = int(ESCENARIOS[sim])
    ax.plot(f_list, global_metrics[sim]['dr_evn'], color=color, linestyle='-', linewidth=2, label=f'EVN ({factor}x)')
    ax.plot(f_list, global_metrics[sim]['dr_afr'], color=color, linestyle='--', linewidth=2, label=f'EVN+Afr ({factor}x)')

ax.set_title("Global Dynamic Range", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Observation Day")
ax.set_ylabel("DR (Peak / RMS Noise)")
ax.set_yscale('log')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='black')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig.savefig(f'{SUM_DIR}/06_global_dynamic_range.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

# --- PLOT 7: TRIPLE PLOT ESTILO ARTÍCULO CIENTÍFICO ---
fig_trip, (ax_rmse, ax_nxcorr, ax_dr) = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

for idx, sim in enumerate(labels_esc):
    f_list = global_metrics[sim]['frames']
    color = C_GLOBAL[idx]
    factor = int(ESCENARIOS[sim])
    label_evn = f'EVN ({factor}x)'
    label_afr = f'EVN+Afr ({factor}x)'
    
    # 1. RMSE
    rmse_evn = np.array(global_metrics[sim]['rmse_evn']) * 1000.0
    rmse_afr = np.array(global_metrics[sim]['rmse_afr']) * 1000.0
    ax_rmse.plot(f_list, rmse_evn, color=color, linestyle='-', linewidth=2, label=label_evn)
    ax_rmse.plot(f_list, rmse_afr, color=color, linestyle='--', linewidth=2, label=label_afr)
    
    # 2. NXCORR
    ax_nxcorr.plot(f_list, global_metrics[sim]['nxcorr_evn'], color=color, linestyle='-', linewidth=2)
    ax_nxcorr.plot(f_list, global_metrics[sim]['nxcorr_afr'], color=color, linestyle='--', linewidth=2)
    
    # 3. Dynamic Range
    ax_dr.plot(f_list, global_metrics[sim]['dr_evn'], color=color, linestyle='-', linewidth=2)
    ax_dr.plot(f_list, global_metrics[sim]['dr_afr'], color=color, linestyle='--', linewidth=2)

# Estilos subplots
ax_rmse.set_title("RMSE Evolution", fontweight='bold', pad=10)
ax_rmse.set_xlabel("Observation Day")
ax_rmse.set_ylabel("RMSE (mJy)")
ax_rmse.grid(True, linestyle='--', alpha=0.6)

ax_nxcorr.set_title("Structural Fidelity (NXCORR)", fontweight='bold', pad=10)
ax_nxcorr.set_xlabel("Observation Day")
ax_nxcorr.set_ylabel("Correlation")
ax_nxcorr.grid(True, linestyle='--', alpha=0.6)

ax_dr.set_title("Dynamic Range", fontweight='bold', pad=10)
ax_dr.set_xlabel("Observation Day")
ax_dr.set_ylabel("DR (Peak / RMS Noise)")
ax_dr.set_yscale('log')
ax_dr.grid(True, linestyle='--', alpha=0.6)

# Leyenda unificada en la parte superior
handles, labels = ax_rmse.get_legend_handles_labels()
fig_trip.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=True, edgecolor='black', fontsize=12)

plt.tight_layout()
fig_trip.savefig(f'{SUM_DIR}/07_global_triple_metrics.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_trip)

print(f"\n¡ESTUDIO COMPLETADO! Revisa la carpeta '{OUT_DIR}' para ver el análisis de síntesis en formato PDF y los UVFITS guardados.")