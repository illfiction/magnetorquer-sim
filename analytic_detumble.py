import json
import numpy as np


def calculate_analytical_detumble_time(config_file="config.json"):
    # 1. Load the configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # 2. Extract Satellite Parameters
    inertia_matrix = np.array(config['satellite']['moment_of_inertia_kgm2'])
    # Use the maximum principal moment of inertia for a conservative (worst-case) estimate
    I_max = np.max(np.diag(inertia_matrix))

    # Calculate initial and target spin magnitudes in rad/s
    w_init_deg = np.array(config['satellite']['initial_spin_deg_s'])
    w0 = np.radians(np.linalg.norm(w_init_deg))
    w_target = np.radians(config['satellite']['target_detumble_rate_deg_s'])

    # 3. Extract Magnetorquer Parameters
    mq = config['magnetorquer']
    m_max = (mq['number_of_turns'] * mq['max_current_amps'] * mq['coil_area_m2'] * mq['core_material'][
        'effective_relative_permeability'])

    # 4. Extract Orbit Parameters
    B_equator = config['orbit']['b_field_equator_tesla']
    # For a polar orbit, B varies from B_0 at equator to 2*B_0 at poles.
    # A standard time-averaged magnitude is roughly 1.5 * B_equator.
    B_avg = 1 * B_equator

    # 5. Physics Calculation (Linear Decay)
    # The efficiency factor accounts for the fact that the B-field isn't always
    # perpendicular to the optimal torque axis. 0.5 is a standard rule of thumb.
    efficiency_factor = 0.5

    tau_avg = m_max * B_avg * efficiency_factor

    # t = I * delta_w / tau
    if w0 <= w_target:
        time_seconds = 0.0
    else:
        time_seconds = (I_max * (w0 - w_target)) / tau_avg

    time_hours = time_seconds / 3600.0

    # 6. Output the Results
    print("========================================")
    print(" ANALYTICAL DETUMBLE ESTIMATE")
    print("========================================")
    print(f"Max Inertia (Worst Axis):  {I_max:.3f} kg*m^2")
    print(f"Initial Spin Magnitude:    {np.degrees(w0):.2f} deg/s")
    print(f"Target Spin Magnitude:     {np.degrees(w_target):.2f} deg/s")
    print(f"Max Dipole Moment (m_max): {m_max:.3f} A*m^2")
    print(f"Est. Average Torque:       {tau_avg:.2e} N*m")
    print("----------------------------------------")
    print(f"EXPECTED DETUMBLE TIME:    {time_hours:.2f} hours")
    print("========================================")


if __name__ == "__main__":
    calculate_analytical_detumble_time()