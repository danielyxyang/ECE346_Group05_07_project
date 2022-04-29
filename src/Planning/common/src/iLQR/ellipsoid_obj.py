from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import matplotlib

from .ellipsoid import Ellipsoid

class EllipsoidObj():

    def __init__(
            self, ellipsoid: Optional[Ellipsoid] = None,
            q: np.ndarray = np.array([]), Q: np.ndarray = np.array([[], []]),
            r=None, n_circ = None, center_L = None, major_axis = None,
            auto_sym: bool = True, psd_tol: float = 1e-15,
    ):
        """
        Approximate Ellipsoid object with `n_circ` many circles for faster computations

        Note that only either ellipsoid or (q, Q) are required to be specified to create an
        EllipsoidObj. The parameters `r` and `n_circ` can be used for fine-tuning to create
        more "conservative" EllipsoidObj. The remaining parameters are set automatically.

        Args:
            ellipsoid: Ellipsoid object to be approximated (alternatively specified by q and Q)
            q: (2,) array providing ellipsoid center
            Q: (2,2) matrix providing ellipsoid shape
            r: radius of circles (default: semi-minor axis)
            n_circ: number of circles used for ellipsoid (default: ceil(semi-major axis / r))
            center_L: (n_circ,) array providing circle centers (default: uniformly distributed)
            major_axis: (2,) array providing ellipsoid major axis (default: eigenvector of largest eigenvalue of Q)
        """
        
        def symmetricize(Q):
            if Q.shape[1] == 0:
                return Q
            else:
                return (Q + Q.T) / 2

        if ellipsoid is not None:
            self.q = ellipsoid.q
            self.Q = ellipsoid.Q
        else:
            self.q = q
            self.Q = symmetricize(Q)
        if q.shape[0] != 2:
            raise ValueError("The dimension should be 2 but get {}!".format(q.shape[0]))
        # super().__init__(q, Q, auto_sym, psd_tol)

        # assume 2d ellipsoid
        if r is None:
            eigVal, eigVec = np.linalg.eig(Q)
            eigVal = np.sqrt(eigVal)
            if eigVal[0] > eigVal[1]:
                a = eigVal[0]
                b = eigVal[1]
                self.major_axis = eigVec[:, 0]
            else:
                a = eigVal[1]
                b = eigVal[0]
                self.major_axis = eigVec[:, 1]
            self.r = b  # semi-minor axis
            self.n_circ = int(np.ceil(a / b))  # #circles to cover the ellipsoid.
            # Assume local coordinate's origin is at the center of ellipsoid,
            # and x is align with major axis. Locations of circle centers are on local
            # coordinate.
            self.center_L = np.linspace(-a + self.r, a - self.r, self.n_circ)
        else:
            self.r = r
            self.n_circ = n_circ
            self.center_L = center_L
            self.major_axis = major_axis

        # position of circle center on global coordinates
        self.center = np.einsum('n,a->an', self.center_L, self.major_axis) + self.q

        self.closest_pt = None
        self.slope = None
        self.width_L = None
        self.width_R = None

    def __matmul__(self, A: np.ndarray) -> EllipsoidObj:
        """
        Multiplication of the ellipsoid by a matrix or a scalar.
        If E(q,Q) is an ellipsoid, and A - matrix of suitable dimensions, then
            E(q, Q) @ A = E(Aq, AQA').
        """

        # t0 = time.time()
        q = A @ self.q
        Q = A @ self.Q @ A.T
        new_major_axis = A @ self.major_axis
        # print("matrix multiplication", time.time()-t0)
        return EllipsoidObj(
            q=q, Q=Q, r=self.r, n_circ=self.n_circ,
            center_L=self.center_L, major_axis=new_major_axis,
        )

    def __add__(self, b: np.ndarray) -> EllipsoidObj:
        """
        Computes vector addition with an ellipsoid:
        E + b = E(q + b, Q).

        Args:
            b (np.ndarray): a vector.
        """
        if b.shape != self.q.shape:
            raise ValueError("[ellReach] dimensions of b and q of the ellipsoid do not match.")
        return EllipsoidObj(
            q=self.q + b, Q=self.Q, r=self.r, n_circ = self.n_circ, 
            center_L = self.center_L, major_axis = self.major_axis,
        )

    def add(self, b: np.ndarray) -> None:
        """
        Computes vector addition with an ellipsoid:
            E + b = E(q + b, Q).

        Args:
            b (np.ndarray): a vector.
        """
        self.q = self.q + b
        self.center = self.center + b

    def __sub__(self, b: np.ndarray) -> EllipsoidObj:
        """
        Computes vector subtraction from an ellipsoid:
            E - b = E(q - b, Q).

        Args:
            b (np.ndarray): a vector.
        """
        if b.shape != self.q.shape:
            raise ValueError("[ellReach] dimensions of b and q of the ellipsoid do not match.")
        return EllipsoidObj(
            q=self.q - b, Q=self.Q, r=self.r, n_circ=self.n_circ,
            center_L=self.center_L, major_axis=self.major_axis,
        )

    def plot_circ(self, ax: matplotlib.axes.Axes) -> None:
        theta = np.linspace(0, 2 * np.pi, 100)
        x0 = self.r * np.cos(theta)
        y0 = self.r * np.sin(theta)
        for i in range(self.n_circ):
            x = self.center[0, i] + x0
            y = self.center[1, i] + y0
            ax.plot(x, y)

    def obstacle_cost(self, obs: EllipsoidObj, q1: float, q2: float) -> float:
        """Computes the barrier cost given an obstacle and hyperparameters.

        Args:
            obs (EllipsoidObj): obstacle.
            q1 (float): shape parameter.
            q2 (float): scale parameter.

        Returns:
            float: barrier cost.
        """
        # Broadcasts to (2, self.n_circ, obs.n_circ).
        d = (self.center.T[:, np.newaxis, :] - obs.center.T).T
        # Reshapes to (2, self.n_circ*obs.n_circ).
        d = np.reshape(d, (2, self.n_circ * obs.n_circ), "F")
        # Ignores and sets to -0.2 when self is far away from the obstacle.
        g = np.clip(self.r + obs.r - np.linalg.norm(d, axis=0), -0.2, None)
        b = np.sum(q1 * np.exp(q2 * g))  # barrier
        return b

    def obstacle_derivative(
        self, state: np.ndarray, center_L: float, obs: EllipsoidObj,
        q1: float, q2: float,
    ) -> Tuple(np.ndarray, np.ndarray):
        """
        TODO

        Args:
            state (np.ndarray): nominal state trajectory, of the shape
                (x_dim, N).
            center_L (float): displacement of vehicle origin along the major
                axis of ellipsoid
            obs (EllipsoidObj): obstacle (or its FRS) represented by ellipsoid
            q1 (float): barrier function parameters
            q2 (float): barrier function parameters

        Returns:
            np.ndarray: Jacobian of softconstraint of shape (x_dim, N)
            np.ndarray: Hessian of softconstraint of shape (x_dim, x_dim, N)
        """
        x = state[0]
        y = state[1]
        theta = state[3]
        ct = np.cos(theta)
        st = np.sin(theta)
        center_L = self.center_L + center_L

        c_x = np.zeros(4)
        c_xx = np.zeros((4, 4))

        for i in range(self.n_circ):
            c_x_temp = np.zeros((4, obs.n_circ))
            c_xx_temp = np.zeros((4, 4, obs.n_circ))
            center_i = self.center[:, i][:, np.newaxis]
            dxy = center_i - obs.center
            dx = dxy[0, :]
            dy = dxy[1, :]
            d = np.linalg.norm(dxy, axis=0)

            b = self.r + obs.r - d

            idx_ignore = b < -0.2
            c = q1 * np.exp(q2 * b)

            # jacobian
            c_x_temp[:2, :] = (-(q2 * c * dxy) / d)

            c_x_temp[3, :] = -(
                center_L[i] * q2 * c *
                (y*ct - obs.center[1, :] * ct - x*st + obs.center[0, :] * st)
            ) / d

            # hessian
            c_xx_temp[0, 0, :] = ((q2 * c * dx**2) / (d**3) - (q2 * c) / d + (q2**2 * c * dx**2) / (d**2))
            c_xx_temp[1, 1, :] = ((q2 * c * dy**2) / (d**3) - (q2 * c) / d + (q2**2 * c * dy**2) / (d**2))
            c_xx_temp[0, 1, :] = ((q2**2 * c * dx * dy) / (d**2) + (q2 * c * dx * dy) / (d**3))
            c_xx_temp[1, 0, :] = ((q2**2 * c * dx * dy) / (d**2) + (q2 * c * dx * dy) / (d**3))
            # theta terms
            # c_xx_temp[0, 3, :] = (center_L[i] * q2 * c * st) / d + (
            #     center_L[i] * q2**2 * c * dx *
            #     (y*ct - obs.center[1, :] * ct - x*st + obs.center[0, :] * st)
            # ) / (d**2) + (
            #     center_L[i] * q2 * c * dx *
            #     (y*ct - obs.center[1, :] * ct - x*st + obs.center[0, :] * st)
            # ) / (d**3)

            # c_xx_temp[1, 3, :] = (center_L[i] * q2**2 * c * dy) * (
            #     y*ct - obs.center[1, :] * ct - x*st + obs.center[0, :] * st
            # ) / (d**2) - (center_L[i] * q2 * c * ct) / d + (
            #     center_L[i] * q2 * c
            # ) * dy * (y*ct - obs.center[1, :] * ct - x*st
            #           + obs.center[0, :] * st) / (d**3)

            # c_xx_temp[3, 3, :] = (
            #     center_L[i] * q2 * c *
            #     (x*ct - obs.center[0, :] * ct + y*st - obs.center[1, :] * st)
            # ) / d + (
            #     center_L[i]**2 * q2**2 * c *
            #     (y*ct - obs.center[1, :] * ct - x*st + obs.center[0, :] * st)**2
            # ) / d**2 + (
            #     center_L[i]**2 * q2 * c *
            #     (y*ct - obs.center[1, :] * ct - x*st + obs.center[0, :] * st)**2
            # ) / d**3

            c_x_temp[:, idx_ignore] = 0
            c_xx_temp[:, :, idx_ignore] = 0

            c_x += c_x_temp.sum(axis=-1)
            c_xx += c_xx_temp.sum(axis=-1)
        return c_x, c_xx
