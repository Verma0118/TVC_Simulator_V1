#!/usr/bin/env python3
"""
Basic 3D rigid-body TVC simulator.

State: pos(3), vel(3), quat(4), omega(3), mass(1)

Usage: python tvc3d.py
Produces `tvc3d.png` and `tvc3d_3d.png`.
"""
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

g0 = 9.80665


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conj(q):
    q = np.array(q)
    q[1:] *= -1
    return q


def quat_rotate(q, v):
    # rotate vector v (in body frame) to inertial: R(q) @ v
    qv = np.concatenate([[0.0], v])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_to_euler(q):
    # returns roll, pitch, yaw (rad)
    w, x, y, z = q
    # roll (x-axis rotation)
    t0 = 2.0*(w*x + y*z)
    t1 = 1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(t0, t1)
    # pitch (y-axis)
    t2 = 2.0*(w*y - z*x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    # yaw (z-axis)
    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


class TVC3DSim:
    def __init__(self,
                 mass0=100.0,
                 I=np.diag([20.0, 25.0, 15.0]),
                 T=20000.0,
                 r_gimbal=0.5,
                 max_gimbal=0.2,
                 Cd=0.5,
                 A=0.5,
                 rho=1.225,
                 Isp=250.0):
        self.mass0 = mass0
        self.I = I
        self.Iinv = np.linalg.inv(I)
        self.T = T
        self.r_gimbal = np.array([r_gimbal, 0.0, 0.0])
        self.max_gimbal = max_gimbal
        self.Cd = Cd
        self.A = A
        self.rho = rho
        self.Isp = Isp

        # mass flow (approx constant) mdot = T/(Isp*g0)
        self.mdot = - self.T / (self.Isp * g0)

    def dynamics(self, state, gimbal):
        # state: [pos(3), vel(3), quat(4), omega(3), mass]
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        mass = state[13]

        # thrust vector in body frame: assume nominal +z (0,0,1)
        # apply small gimbal around x and y -> rotation of thrust vector
        gx, gy = gimbal  # radians about body x and y axes (small)
        # rotation approx: thrust_body = R_gimbal * [0,0,T]
        tb = np.array([
            -math.sin(gx) * self.T,
            math.sin(gy) * self.T,
            math.cos(gx) * math.cos(gy) * self.T,
        ])

        # thrust in inertial
        thrust_inertial = quat_rotate(quat, tb)

        # gravity
        gravity = np.array([0.0, 0.0, -mass * g0])

        # aerodynamic drag (simple quadratic) using inertial vel
        v_rel = vel
        vnorm = np.linalg.norm(v_rel)
        drag = np.zeros(3)
        if vnorm > 1e-6:
            drag = -0.5 * self.rho * self.Cd * self.A * vnorm * v_rel

        force = thrust_inertial + gravity + drag
        acc = force / mass

        # torque in body frame: r_gimbal x thrust_body
        torque_body = np.cross(self.r_gimbal, tb)

        # angular accel (in body frame)
        omega_dot = self.Iinv.dot(torque_body - np.cross(omega, self.I.dot(omega)))

        # quaternion derivative: q_dot = 0.5 * q * [0; omega]
        omega_quat = np.concatenate([[0.0], omega])
        q_dot = 0.5 * quat_mul(quat, omega_quat)

        mass_dot = self.mdot

        deriv = np.zeros_like(state)
        deriv[0:3] = vel
        deriv[3:6] = acc
        deriv[6:10] = q_dot
        deriv[10:13] = omega_dot
        deriv[13] = mass_dot

        return deriv

    def rk4_step(self, state, gimbal, dt):
        k1 = self.dynamics(state, gimbal)
        k2 = self.dynamics(state + 0.5*dt*k1, gimbal)
        k3 = self.dynamics(state + 0.5*dt*k2, gimbal)
        k4 = self.dynamics(state + dt*k3, gimbal)
        new = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # renormalize quaternion
        new[6:10] /= np.linalg.norm(new[6:10])
        return new


def attitude_controller_pd(quat, omega, Kp=50.0, Kd=20.0):
    # simple attitude controller driving to identity quaternion (no rotation)
    # quaternion error: q_err = q_des * q_conj (q_des = [1,0,0,0]) -> q_err = q
    # For small angle, vector part ~ 0.5*angle
    q = quat
    q_vec = q[1:]
    # desired control torque in body frame
    torque = - (Kp * q_vec + Kd * omega)
    return torque


def attitude_controller_pid(quat, omega, euler_cmd=(0.0, 0.0, 0.0), Kp=(60.0, 60.0, 40.0), Kd=(30.0, 25.0, 20.0)):
    # PID-style PD controller operating on Euler angle errors (roll, pitch, yaw)
    # euler_cmd is desired (roll, pitch, yaw) in radians
    roll, pitch, yaw = quat_to_euler(quat)
    def ang_err(a, b):
        d = a - b
        # wrap to [-pi, pi]
        return (d + math.pi) % (2*math.pi) - math.pi

    err = np.array([ang_err(roll, euler_cmd[0]), ang_err(pitch, euler_cmd[1]), ang_err(yaw, euler_cmd[2])])
    Kp = np.array(Kp)
    Kd = np.array(Kd)
    torque = - (Kp * err + Kd * omega)
    return torque


def torque_to_gimbal(torque_body, T, r_gimbal, max_gimbal):
    # approximate mapping tau = r_gimbal x (0,0,T) * gimbal_vector_mag
    # For small gimballing, torque ≈ T * (r_gimbal cross gimbal_dir)
    # Solve for gimbal angles about x and y: tau_x ≈ T * r * gy, tau_y ≈ -T * r * gx
    rx = r_gimbal[0]
    if T * rx == 0:
        return np.array([0.0, 0.0])
    gx = - torque_body[1] / (T * rx)
    gy =   torque_body[0] / (T * rx)
    # clamp
    gx = max(-max_gimbal, min(max_gimbal, gx))
    gy = max(-max_gimbal, min(max_gimbal, gy))
    return np.array([gx, gy])


def run(duration=20.0, dt=0.01, out='tvc3d.png', out3='tvc3d_3d.png', show=False):
    sim = TVC3DSim()

    # initial state: slight tilt
    pos0 = np.array([0.0, 0.0, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    # small initial rotation around y (pitch)
    angle = math.radians(5.0)
    q0 = np.array([math.cos(angle/2), 0.0, math.sin(angle/2), 0.0])
    omega0 = np.array([0.0, 0.0, 0.0])
    mass0 = sim.mass0

    state = np.zeros(14)
    state[0:3] = pos0
    state[3:6] = vel0
    state[6:10] = q0
    state[10:13] = omega0
    state[13] = mass0

    steps = int(duration / dt)
    data = np.zeros((steps+1, len(state)))
    t = np.zeros(steps+1)
    gimbals = np.zeros((steps+1,2))
    torques = np.zeros((steps+1,3))

    data[0] = state

    for i in range(steps):
        quat = state[6:10]
        omega = state[10:13]
        # control
        torque_cmd = attitude_controller_pd(quat, omega)
        gimbal = torque_to_gimbal(torque_cmd, sim.T, sim.r_gimbal, sim.max_gimbal)
        # integrate
        state = sim.rk4_step(state, gimbal, dt)
        data[i+1] = state
        t[i+1] = (i+1)*dt
        gimbals[i+1] = gimbal
        torques[i+1] = torque_cmd

        # stop if crashes below ground
        if state[2] < -1.0:
            data = data[:i+2]
            t = t[:i+2]
            gimbals = gimbals[:i+2]
            torques = torques[:i+2]
            break

    # plotting
    pos = data[:,0:3]
    quat = data[:,6:10]
    mass = data[:,13]
    eulers = np.array([quat_to_euler(q) for q in quat])

    fig, axs = plt.subplots(4,1, figsize=(7,10))
    axs[0].plot(pos[:,0], pos[:,2])
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('z (m)')
    axs[0].set_title('X-Z Trajectory')
    axs[0].grid(True)

    axs[1].plot(t, np.degrees(eulers[:,0]), label='roll')
    axs[1].plot(t, np.degrees(eulers[:,1]), label='pitch')
    axs[1].plot(t, np.degrees(eulers[:,2]), label='yaw')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('angle (deg)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(t, np.degrees(gimbals[:,0]), label='gimbal_x (deg)')
    axs[2].plot(t, np.degrees(gimbals[:,1]), label='gimbal_y (deg)')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('gimbal (deg)')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(t, mass)
    axs[3].set_xlabel('t (s)')
    axs[3].set_ylabel('mass (kg)')
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(out)
    if show:
        plt.show()
    plt.close(fig)

    # 3D plot
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig2 = plt.figure(figsize=(6,6))
        ax3 = fig2.add_subplot(111, projection='3d')
        ax3.plot(pos[:,0], pos[:,1], pos[:,2])
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        ax3.set_title('3D Trajectory')
        plt.tight_layout()
        plt.savefig(out3)
        plt.close(fig2)
    except Exception:
        pass

    print(f"3D simulation complete — saved {out} and {out3}")
    return data, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=20.0)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--out', default='tvc3d.png')
    p.add_argument('--out3', default='tvc3d_3d.png')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()
    if getattr(args, 'staged', False):
        # example two-stage definition if none provided
        stages = [
            { 'T': 20000.0, 'Isp': 250.0, 'prop_mass': 60.0, 'dry_mass': 20.0, 'I': np.diag([20.0,25.0,15.0]) },
            { 'T': 5000.0, 'Isp': 300.0, 'prop_mass': 20.0, 'dry_mass': 10.0, 'I': np.diag([5.0,6.0,4.0]) },
        ]
        run_staged(stages=stages, duration=args.duration, dt=args.dt, out=args.out, out3=args.out3, show=args.show)
    else:
        run(duration=args.duration, dt=args.dt, out=args.out, out3=args.out3, show=args.show)


def run_staged(stages, duration=60.0, dt=0.02, out='tvc3d_staged.png', out3='tvc3d_staged_3d.png', show=False):
    """
    Run a multi-stage simulation. `stages` is a list of dicts with keys:
      - T: thrust (N)
      - Isp: specific impulse (s)
      - prop_mass: propellant mass in kg
      - dry_mass: structural dry mass that will be jettisoned at separation
      - I: inertia matrix (3x3) optional
    """
    # build initial totals
    total_prop = sum(s['prop_mass'] for s in stages)
    total_dry = sum(s.get('dry_mass', 0.0) for s in stages)
    total_mass = total_prop + total_dry

    # instantiate sim with first-stage properties
    first = stages[0]
    sim = TVC3DSim(mass0=total_mass, I=first.get('I', np.diag([20.0,25.0,15.0])), T=first['T'], Isp=first['Isp'])

    # initial state
    pos0 = np.array([0.0, 0.0, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    angle = math.radians(5.0)
    q0 = np.array([math.cos(angle/2), 0.0, math.sin(angle/2), 0.0])
    omega0 = np.array([0.0, 0.0, 0.0])
    state = np.zeros(14)
    state[0:3] = pos0
    state[3:6] = vel0
    state[6:10] = q0
    state[10:13] = omega0
    state[13] = total_mass

    steps = int(duration / dt)
    data = np.zeros((steps+1, len(state)))
    t = np.zeros(steps+1)
    gimbals = np.zeros((steps+1,2))
    torques = np.zeros((steps+1,3))
    stage_idx = 0
    prop_remaining = stages[0]['prop_mass']

    data[0] = state

    for i in range(steps):
        # guidance law: gravity turn -> desired pitch aligns with velocity flight path
        vel = state[3:6]
        speed = np.linalg.norm(vel)
        # avoid huge instantaneous pitch command when nearly stationary
        if speed > 1e-1:
            gamma = math.atan2(vel[2], vel[0])
        else:
            # use current pitch to prevent a large jump
            _, cur_pitch, _ = quat_to_euler(state[6:10])
            gamma = cur_pitch

        # desired Euler: keep roll=0, pitch = gamma, yaw=0
        euler_cmd = (0.0, gamma, 0.0)

        quat = state[6:10]
        omega = state[10:13]
        # PID attitude controller produces torque command
        torque_cmd = attitude_controller_pid(quat, omega, euler_cmd)

        # map torque to gimbal angles using current stage thrust
        cur_stage = stages[stage_idx]
        # limit commanded torque to what gimbaled thrust can realistically provide
        avail_tau = abs(cur_stage['T'] * sim.r_gimbal[0])
        tau_norm = np.linalg.norm(torque_cmd)
        if tau_norm > 1e-8 and tau_norm > avail_tau:
            torque_cmd = torque_cmd * (avail_tau / tau_norm)

        gimbal = torque_to_gimbal(torque_cmd, cur_stage['T'], np.array([sim.r_gimbal[0],0.0,0.0]), sim.max_gimbal)

        # integrate one step using sim settings
        # update sim thrust and mdot for current stage
        sim.T = cur_stage['T']
        sim.Isp = cur_stage['Isp']
        sim.mdot = - sim.T / (sim.Isp * g0)
        if 'I' in cur_stage:
            sim.I = cur_stage['I']
            sim.Iinv = np.linalg.inv(sim.I)

        state = sim.rk4_step(state, gimbal, dt)

        # consume propellant (approx mdot) but limit to remaining
        prop_used = min(-sim.mdot * dt, prop_remaining)
        prop_remaining -= prop_used
        state[13] -= prop_used

        data[i+1] = state
        t[i+1] = (i+1)*dt
        gimbals[i+1] = gimbal
        torques[i+1] = torque_cmd

        # check stage burnout
        if prop_remaining <= 1e-6:
            # perform staging: subtract dry mass and move to next stage
            drop = cur_stage.get('dry_mass', 0.0)
            state[13] -= drop
            stage_idx += 1
            if stage_idx >= len(stages):
                # no more stages -> engine cutoff
                sim.T = 0.0
                sim.mdot = 0.0
            else:
                prop_remaining = stages[stage_idx]['prop_mass']
                # update sim inertia if provided
                if 'I' in stages[stage_idx]:
                    sim.I = stages[stage_idx]['I']
                    sim.Iinv = np.linalg.inv(sim.I)

        # stop if below ground
        if state[2] < -1.0:
            data = data[:i+2]
            t = t[:i+2]
            gimbals = gimbals[:i+2]
            torques = torques[:i+2]
            break

    # plotting (reuse same layout)
    pos = data[:,0:3]
    quat = data[:,6:10]
    mass = data[:,13]
    eulers = np.array([quat_to_euler(q) for q in quat])

    fig, axs = plt.subplots(4,1, figsize=(7,10))
    axs[0].plot(pos[:,0], pos[:,2])
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('z (m)')
    axs[0].set_title('X-Z Trajectory (staged)')
    axs[0].grid(True)

    axs[1].plot(t, np.degrees(eulers[:,0]), label='roll')
    axs[1].plot(t, np.degrees(eulers[:,1]), label='pitch')
    axs[1].plot(t, np.degrees(eulers[:,2]), label='yaw')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('angle (deg)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(t, np.degrees(gimbals[:,0]), label='gimbal_x (deg)')
    axs[2].plot(t, np.degrees(gimbals[:,1]), label='gimbal_y (deg)')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('gimbal (deg)')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(t, mass)
    axs[3].set_xlabel('t (s)')
    axs[3].set_ylabel('mass (kg)')
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(out)
    if show:
        plt.show()
    plt.close(fig)

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig2 = plt.figure(figsize=(6,6))
        ax3 = fig2.add_subplot(111, projection='3d')
        ax3.plot(pos[:,0], pos[:,1], pos[:,2])
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        ax3.set_title('3D Trajectory (staged)')
        plt.tight_layout()
        plt.savefig(out3)
        plt.close(fig2)
    except Exception:
        pass

    print(f"Staged simulation complete — saved {out} and {out3}")
    return data, t


if __name__ == '__main__':
    main()
