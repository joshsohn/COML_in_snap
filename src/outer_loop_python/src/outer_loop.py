import numpy as np
from structs import AttCmdClass, ControlLogClass, GoalClass
from helpers import quaternion_multiply

class IntegratorClass:
    def __init__(self):
        self.value_ = 0

    def increment(self, inc, dt):
        self.value_ += inc * dt

    def reset(self):
        self.value_ = 0

    def value(self):
        return self.value_

class OuterLoop:
    def __init__(self, params):
        self.params_ = params
        self.GRAVITY = np.array([0, 0, -9.80665])

        self.Ix_ = IntegratorClass()
        self.Iy_ = IntegratorClass()
        self.Iz_ = IntegratorClass()

        self.log_ = ControlLogClass()
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0

        self.mode_xy_last_ = GoalClass.Mode.POS_CTRL
        self.mode_z_last_ = GoalClass.Mode.POS_CTRL

        self.reset()

    def reset(self):
        self.Ix_.reset()
        self.Iy_.reset()
        self.Iz_.reset()

        self.log_ = ControlLogClass()
        self.a_fb_last_ = np.zeros(3)
        self.j_fb_last_ = np.zeros(3)
        self.t_last_ = 0
    
    def update_log(self, state):
        self.log_ = ControlLogClass()
        self.log_.p = state.p
        self.log_.v = state.v
        self.log_.q = state.q
        self.log_.w = state.w

    def compute_attitude_command(self, t, state, goal):
        dt = 1e-2 if self.t_last_ == 0 else t - self.t_last_

        if dt > 0:
            self.t_last_ = t
        else:
            print("Warning: non-positive dt:", dt, "[s].")

        F_W = self.get_force(dt, state, goal)
        q_ref = self.get_attitude(state, goal, F_W)
        w_ref = self.get_rates(dt, state, goal, F_W, self.log_.a_fb, q_ref)

        cmd = AttCmdClass()
        # print('q_ref:', q_ref)
        cmd.q = q_ref
        cmd.w = w_ref
        cmd.F_W = F_W

        return cmd

    def get_force(self, dt, state, goal):
        e = goal.p - state.p
        edot = goal.v - state.v

        # Saturate error so it isn't too much for control gains
        e = np.minimum(np.maximum(e, -self.params_.maxPosErr), self.params_.maxPosErr)
        edot = np.minimum(np.maximum(edot, -self.params_.maxVelErr), self.params_.maxVelErr)

        # Manipulate error signals based on selected flight mode
        # Reset integrators on mode change
        if goal.mode_xy != self.mode_xy_last_:
            self.Ix_.reset()
            self.Iy_.reset()
            self.mode_xy_last_ = goal.mode_xy

        if goal.mode_z != self.mode_z_last_:
            self.Iz_.reset()
            self.mode_z_last_ = goal.mode_z

        # Check which control mode to use for x-y
        if goal.mode_xy == GoalClass.Mode.POS_CTRL:
            self.Ix_.increment(e[0], dt)
            self.Iy_.increment(e[1], dt)
        elif goal.mode_xy == GoalClass.Mode.VEL_CTRL:
            # Do not worry about position error---only vel error
            e[0] = e[1] = 0
        elif goal.mode_xy == GoalClass.Mode.ACC_CTRL:
            # Do not generate feedback accel---only control on goal accel
            e[0] = e[1] = 0
            edot[0] = edot[1] = 0

        # Check which control mode to use for z
        if goal.mode_z == GoalClass.Mode.POS_CTRL:
            self.Iz_.increment(e[2], dt)
        elif goal.mode_z == GoalClass.Mode.VEL_CTRL:
            # Do not worry about position error---only vel error
            e[2] = 0
        elif goal.mode_z == GoalClass.Mode.ACC_CTRL:
            # Do not generate feedback accel---only control on goal accel
            e[2] = 0
            edot[2] = 0

        # Compute feedback acceleration via PID, eq (2.9)
        eint = np.array([self.Ix_.value(), self.Iy_.value(), self.Iz_.value()])
        a_fb = np.multiply(self.params_.Kp, e) + np.multiply(self.params_.Ki, eint) + np.multiply(self.params_.Kd, edot)


        # Compute total desired force (expressed in world frame), eq (2.12)
        F_W = self.params_.mass * (goal.a + a_fb - self.GRAVITY)

        # Log control signals for debugging and inspection
        self.log_.p = state.p
        self.log_.p_ref = goal.p
        self.log_.p_err = e
        self.log_.p_err_int = eint
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = edot
        self.log_.a_ff = goal.a
        self.log_.a_fb = a_fb
        self.log_.F_W = F_W

        # Return total desired force expressed in world frame
        return F_W

    def get_attitude(self, state, goal, F_W):
        xi = F_W / self.params_.mass  # Eq. 26
        abc = xi / np.linalg.norm(xi)  # Eq. 19

        a, b, c = abc
        psi = goal.psi

        invsqrt21pc = 1 / np.sqrt(2 * (1 + c))

        quaternion0 = np.array([invsqrt21pc*(1+c), invsqrt21pc*(-b), invsqrt21pc*a, 0.0])
        quaternion1 = np.array([np.cos(psi/2.), 0.0, 0.0, np.sin(psi/2.)])

        # Construct the quaternion
        q_ref = quaternion_multiply(quaternion0, quaternion1)

        # Normalize quaternion
        q_ref = q_ref / np.linalg.norm(q_ref)

        # TODO: Implement the second fibration for the whole SO(3)
        # See Eq. 22, 23, and 24

        # Log control signals for debugging and inspection
        self.log_.q = state.q
        self.log_.q_ref = q_ref

        return q_ref

    def get_rates(self, dt, state, goal, F_W, a_fb, q_ref):
        # Generate feedback jerk by numerical derivative of feedback accel
        j_fb = np.zeros(3)
        if dt > 0:
            # Numeric derivative
            j_fb = (a_fb - self.a_fb_last_) / dt

            # Low-pass filter differentiation
            tau = 0.1
            alpha = dt / (tau + dt)
            j_fb = alpha * j_fb + (1 - alpha) * self.j_fb_last_
        else:
            # Re-use last value
            j_fb = self.j_fb_last_

        # Save for the next time
        self.a_fb_last_ = a_fb
        self.j_fb_last_ = j_fb

        # Construct angular rates consistent with trajectory dynamics
        Fdot_W = goal.j + j_fb
        xi = F_W / self.params_.mass  # Eq. 26
        abc = xi / np.linalg.norm(xi)  # Eq. 19
        xi_dot = Fdot_W / self.params_.mass
        I = np.eye(3)
        norm_xi = np.linalg.norm(xi)

        abcdot = ((norm_xi**2 * I - np.outer(xi, xi)) / norm_xi**3) @ xi_dot  # Eq. 20

        # Assert abc' * abcdot should be approximately 0.0
        assert np.allclose(np.dot(abc, abcdot), 0.0)

        a, b, c = abc
        adot, bdot, cdot = abcdot
        psi, psidot = goal.psi, goal.dpsi

        rates = np.zeros(3)
        rates[0] = np.sin(psi) * adot - np.cos(psi) * bdot - (a * np.sin(psi) - b * np.cos(psi)) * (cdot / (c + 1))
        rates[1] = np.cos(psi) * adot + np.sin(psi) * bdot - (a * np.cos(psi) + b * np.sin(psi)) * (cdot / (c + 1))
        rates[2] = (b * adot - a * bdot) / (1 + c) + psidot

        # Log control signals for debugging and inspection
        self.log_.p = state.p
        self.log_.p_ref = goal.p
        self.log_.p_err = goal.p - state.p
        self.log_.v = state.v
        self.log_.v_ref = goal.v
        self.log_.v_err = goal.v - state.v
        self.log_.a_ff = goal.j
        self.log_.a_fb = j_fb
        self.log_.F_W = F_W
        self.log_.j_ff = goal.j
        self.log_.j_fb = j_fb
        self.log_.w = state.w
        self.log_.w_ref = rates

        return rates