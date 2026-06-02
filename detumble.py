import json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ==========================================
# Math & Kinematics Helper Functions
# ==========================================
def quat_mult(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def quat_to_matrix(q):
    """Converts a quaternion to a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])


# ==========================================
# Main Simulation Class
# ==========================================
class DetumbleSimulation:
    def __init__(self, config_file="config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Parse Satellite Parameters
        self.inertia = np.array(self.config['satellite']['moment_of_inertia_kgm2'])
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.target_rate_rad = np.radians(self.config['satellite']['target_detumble_rate_deg_s'])

        # Parse Magnetorquer Parameters & Calculate Max Dipole Moment (m = N * I * A * mu_r)
        mq = self.config['magnetorquer']
        self.m_max = (mq['number_of_turns'] * mq['max_current_amps'] * mq['coil_area_m2'] * mq['core_material'][
            'effective_relative_permeability'])
        self.k_gain = mq['b_dot_gain']

        # Parse Orbit (Calculate orbital frequency for B-field approximation)
        alt_km = self.config['orbit']['altitude_km']
        mu_earth = 3.986e14  # m^3/s^2
        r_orbit = (6371 + alt_km) * 1000  # meters
        self.orbit_omega = np.sqrt(mu_earth / r_orbit ** 3)
        self.B_0 = self.config['orbit']['b_field_equator_tesla']

    def get_inertial_b_field(self, t):
        """Simple dipole model of Earth's B-field for a polar orbit."""
        # B-field vector rotates in the inertial frame as the satellite orbits
        return np.array([
            self.B_0 * np.cos(self.orbit_omega * t),
            0.0,
            2 * self.B_0 * np.sin(self.orbit_omega * t)
        ])

    def dynamics(self, t, state):
        """ODE function: calculates derivatives of quaternion and angular velocity."""
        q = state[0:4] / np.linalg.norm(state[0:4])  # Normalize quaternion to prevent drift
        w = state[4:7]

        # 1. Environment: Get local magnetic field and rotate to body frame
        B_inertial = self.get_inertial_b_field(t)
        R_inertial_to_body = quat_to_matrix(q)
        B_body = R_inertial_to_body @ B_inertial

        # 2. Sensor: Calculate rate of change of B-field in body frame (B-dot)
        # Assuming fast spin, B_dot is dominated by the cross product -w x B
        B_dot = -np.cross(w, B_body)

        # 3. Control: Calculate commanded dipole moment and clamp to hardware limits
        m_cmd = -self.k_gain * B_dot
        m_applied = np.clip(m_cmd, -self.m_max, self.m_max)

        # 4. Actuation: Resulting magnetic torque (tau = m x B)
        torque = np.cross(m_applied, B_body)

        # 5. Kinematics (dq/dt = 0.5 * q * w)
        dq = 0.5 * quat_mult(q, np.array([0, w[0], w[1], w[2]]))

        # 6. Dynamics (Euler's equations: dw/dt = I^-1 * (tau - w x (Iw)))
        dw = self.inv_inertia @ (torque - np.cross(w, self.inertia @ w))

        return np.concatenate((dq, dw))

    def detumbled_event(self, t, state):
        """Event function to stop integration when target rate is reached."""
        w = state[4:7]
        current_rate_rad = np.linalg.norm(w)
        return current_rate_rad - self.target_rate_rad

    # Required by scipy.integrate to terminate the loop
    detumbled_event.terminal = True
    detumbled_event.direction = -1

    # Add this inside your simulation class to extract detailed debug states post-run
    def extract_debug_metrics(self, sol):
        times = sol.t
        states = sol.y

        kinetic_energy = []
        saturation_flags = []

        for i in range(len(times)):
            q = states[0:4, i]
            w = states[4:7, i]

            # 1. Calculate Kinetic Energy: 0.5 * w . (I * w)
            E_k = 0.5 * np.dot(w, self.inertia @ w)
            kinetic_energy.append(E_k)

            # Reconstruct control step to check saturation
            B_inertial = self.get_inertial_b_field(times[i])
            B_body = quat_to_matrix(q) @ B_inertial
            B_dot = -np.cross(w, B_body)
            m_cmd = -self.k_gain * B_dot

            # 2. Check if any axis is saturated
            saturated = np.any(np.abs(m_cmd) >= self.m_max)
            saturation_flags.append(1.0 if saturated else 0.0)

        return np.array(kinetic_energy), np.array(saturation_flags)

    def run(self):
        """Executes the simulation and plots the diagnostic results."""
        w_init = np.radians(self.config['satellite']['initial_spin_deg_s'])
        q_init = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        state_init = np.concatenate((q_init, w_init))

        t_span = (0, self.config['simulation']['max_time_hours'] * 3600)

        print(f"Max Dipole Moment capacity calculated: {self.m_max:.3f} A*m^2")
        print("Starting integration... this may take a moment.")

        sol = solve_ivp(
            self.dynamics,
            t_span,
            state_init,
            events=self.detumbled_event,
            max_step=1.0  # Max step size in seconds to ensure we don't skip B-field variations
        )

        time_hours = sol.t / 3600
        w_deg_s = np.degrees(sol.y[4:7, :])
        norm_w = np.linalg.norm(w_deg_s, axis=0)

        if sol.status == 1:
            print(f"\n✅ Detumble Complete!")
            print(f"Time to detumble: {time_hours[-1]:.2f} hours")
        else:
            print(f"\n⚠️ Satellite did not reach target detumble rate within the maximum time.")

        # Extract metrics (No more printing raw arrays to the console)
        print("Generating diagnostic plots...")
        kinetic_energy, saturation_flags = self.extract_debug_metrics(sol)

        # Render the dashboard
        self.plot_diagnostics(time_hours, w_deg_s, norm_w, kinetic_energy, saturation_flags)

    def plot_diagnostics(self, time_hours, w_deg_s, norm_w, kinetic_energy, saturation_flags):
        """Creates a comprehensive 3-panel diagnostic dashboard."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle('Satellite Detumbling Diagnostic Dashboard', fontsize=16)

        # ==========================================
        # Panel 1: Angular Velocity
        # ==========================================
        axs[0].plot(time_hours, w_deg_s[0], label='X-axis', alpha=0.7)
        axs[0].plot(time_hours, w_deg_s[1], label='Y-axis', alpha=0.7)
        axs[0].plot(time_hours, w_deg_s[2], label='Z-axis', alpha=0.7)
        axs[0].plot(time_hours, norm_w, 'k--', label='Total Magnitude', linewidth=2)
        axs[0].axhline(self.config['satellite']['target_detumble_rate_deg_s'],
                       color='r', linestyle=':', label='Target Threshold')
        axs[0].set_ylabel('Angular Velocity (deg/s)')
        axs[0].set_title('1. Spin Rate Decay')
        axs[0].grid(True)
        axs[0].legend(loc='upper right')

        # ==========================================
        # Panel 2: Rotational Kinetic Energy
        # ==========================================
        axs[1].plot(time_hours, kinetic_energy, 'b-', linewidth=2)
        # Using a log scale because energy decays exponentially; makes it easier to spot positive feedback bugs
        # axs[1].set_yscale('log')
        axs[1].set_ylabel('Kinetic Energy (Joules)')
        axs[1].set_title('2. System Energy (Log Scale)')
        axs[1].grid(True, which="both", ls="--", alpha=0.5)

        # ==========================================
        # Panel 3: Actuator Saturation History
        # ==========================================
        # fill_between creates a solid block of color when the coils are maxed out
        axs[2].fill_between(time_hours, 0, saturation_flags, color='red', alpha=0.5, step="mid")
        axs[2].set_yticks([0, 1])
        axs[2].set_yticklabels(['Nominal\n(Linear)', 'Saturated\n(Max Capacity)'])
        axs[2].set_ylabel('Coil State')
        axs[2].set_xlabel('Time (Hours)')
        axs[2].set_title('3. Actuator Saturation Duty Cycle')
        axs[2].grid(True, axis='x')

        plt.tight_layout()
        # Adjust layout slightly to make room for the main title
        plt.subplots_adjust(top=0.93)
        plt.show()


if __name__ == "__main__":
    sim = DetumbleSimulation()
    sim.run()