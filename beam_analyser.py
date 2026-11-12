#!/usr/bin/env python3
# =============================================================================
#  GENERAL BEAM ANALYSER  -  shear, bending moment, SLOPE and DEFLECTION
# =============================================================================
#
#  Solves a straight, prismatic (constant EI) beam of length L for ANY of:
#
#      * Cantilever                      (fixed at x = 0 or at x = L)
#      * Simply supported beam           (pins at x = 0 and x = L)
#      * Overhanging beam                (pins anywhere, e.g. at 0.6 m and 3 m)
#      * Propped cantilever              (fixed end + pin)
#      * Fixed-fixed beam                (built in at both ends)
#      * Continuous beams                (fixed/pins, any number of pins)
#
#  carrying any combination of
#
#      * concentrated (point) loads
#      * UDLs on the full span or on any part of it
#      * UVLs / trapezoidal loads (intensity w1 -> w2)
#      * applied couples (moments)
#
#  METHOD  (exact - no step size, no numerical integration error)
#  --------------------------------------------------------------
#   1. Every beam is first analysed as a "REFERENCE CANTILEVER" clamped at
#      x = 0.  Its moment, slope and deflection are exact piecewise
#      polynomials obtained by integrating  EI y'' = m(x)  (see class
#      ReferenceCantilever).
#   2. The real supports are replaced by unknown reactions (extra loads in the
#      reference cantilever) plus a rigid-body motion  y0 + theta0 * x.
#   3. The unknown reactions, y0 and theta0 are found from the support
#      conditions (y = 0 at every pin, y = theta = 0 at a fixed end, free-end
#      conditions at x = 0 if there is no support there).  Because everything is
#      linear this is a small linear system  A u = b  (see solve_beam).
#   4. The final shear, moment, slope and deflection are the reference
#      cantilever with the solved reactions + rigid-body motion added.
#
#  SIGN CONVENTIONS
#  ----------------
#      x             : measured from the LEFT end of the beam
#      Loads         : positive = DOWNWARD        (enter upward loads as negative)
#      Applied couple: positive = CLOCKWISE
#      Shear V       : positive when the left part pushes UP on the right part
#      Moment M      : positive = SAGGING (tension in the bottom fibre)
#      Slope, y      : positive = DOWNWARD / clockwise
#
#  INPUT UNITS:  m, mm (section), GPa, kN, kN/m, kN.m
#
#  HOW TO RUN
#  ----------
#      python beam_analyser.py                    interactive questions
#      python beam_analyser.py --config my.json   read a JSON file
#      python beam_analyser.py --example 5.2      run a built-in textbook example
#      python beam_analyser.py --list-examples
#      python beam_analyser.py --test             run the self-tests
#      extras:  --no-show   --outdir DIR   --csv   --npoints N
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORTS AND GLOBAL SETTINGS
# -----------------------------------------------------------------------------
import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial as Poly

TOL = 1e-9  # geometric tolerance (m) used when comparing positions


# -----------------------------------------------------------------------------
# 2. SMALL DATA CONTAINERS  (everything is stored internally in N and m)
# -----------------------------------------------------------------------------
@dataclass
class PointLoad:
    P: float  # force in N, positive downward
    a: float  # distance from left end, m


@dataclass
class DistLoad:
    w1: float     # intensity at 'start', N/m, positive downward
    w2: float     # intensity at 'end',   N/m  (w1 == w2 -> UDL, else UVL)
    start: float  # m
    end: float    # m


@dataclass
class Couple:
    M: float  # N.m, positive clockwise
    a: float  # m


@dataclass
class Support:
    kind: str  # "fixed"  (clamped, only allowed at x = 0 or x = L)
    x: float   # or "pin" (vertical restraint: pin or roller, anywhere)


@dataclass
class BeamModel:
    """Everything the solver needs, already converted to N and m."""
    L: float
    EI: float
    E: float                 # Pa  (nan if EI was given directly)
    I: float                 # m^4 (nan if EI was given directly)
    c: float                 # distance to extreme fibre, m (nan if unknown)
    section_name: str
    supports: list
    point_loads: list
    dist_loads: list
    couples: list
    query_points: list = field(default_factory=list)


# -----------------------------------------------------------------------------
# 3. CROSS-SECTION PROPERTIES
# -----------------------------------------------------------------------------
def section_properties(sec):
    """
    Return a dict with the section name, second moment of area I (m^4) and the
    extreme-fibre distance c (m).  If the section type is "EI", the flexural
    rigidity EI (N.m^2) is returned directly and no I / c exist.

    sec examples (JSON):
        {"type": "rect",   "b_mm": 150, "h_mm": 300}
        {"type": "I",      "B_mm": 200, "H_mm": 350, "tf_mm": 12, "tw_mm": 8}
        {"type": "custom", "I_cm4": 8300}            (optional "c_mm")
        {"type": "custom", "I_m4": 83e-6}            (optional "c_mm")
        {"type": "EI",     "EI_kNm2": 65000}         (or EI_MNm2 / EI_Nm2)
    """
    kind = str(sec["type"]).lower()

    # --- rectangle:  I = b h^3 / 12 ------------------------------------------
    if kind in ("rect", "rectangular", "1"):
        b, h = sec["b_mm"] / 1e3, sec["h_mm"] / 1e3
        if b <= 0 or h <= 0:
            raise ValueError("Rectangular section: b and h must be positive.")
        return dict(name=f"Rectangular {sec['b_mm']:g} x {sec['h_mm']:g} mm",
                    I=b * h**3 / 12, c=h / 2)

    # --- symmetric I-section: flanges + web (parallel-axis theorem) -------------
    if kind in ("i", "i-section", "isection", "2"):
        B, H = sec["B_mm"] / 1e3, sec["H_mm"] / 1e3
        tf, tw = sec["tf_mm"] / 1e3, sec["tw_mm"] / 1e3
        if min(B, H, tf, tw) <= 0:
            raise ValueError("I-section: all dimensions must be positive.")
        if 2 * tf >= H:
            raise ValueError("I-section: 2*tf must be less than H.")
        if tw > B:
            raise ValueError("I-section: web thickness cannot exceed flange width.")
        hw = H - 2 * tf  # clear web height
        I = 2 * (B * tf**3 / 12 + B * tf * (H / 2 - tf / 2) ** 2) + tw * hw**3 / 12
        return dict(name=(f"I-section B={sec['B_mm']:g}, H={sec['H_mm']:g}, "
                          f"tf={sec['tf_mm']:g}, tw={sec['tw_mm']:g} mm"),
                    I=I, c=H / 2)

    # --- section given only through its I --------------------------------------
    if kind in ("custom", "3"):
        if "I_m4" in sec:
            I = float(sec["I_m4"])
        else:
            I = float(sec["I_cm4"]) * 1e-8
        c = float(sec.get("c_mm", 0)) / 1e3 or float("nan")
        if I <= 0:
            raise ValueError("Custom section: I must be positive.")
        return dict(name=f"Custom (I = {I:.4e} m^4)", I=I, c=c)

    # --- EI given directly in the question ---------------------------------------
    if kind in ("ei", "4"):
        if "EI_kNm2" in sec:
            EI = float(sec["EI_kNm2"]) * 1e3
        elif "EI_MNm2" in sec:
            EI = float(sec["EI_MNm2"]) * 1e6
        elif "EI_Nm2" in sec:
            EI = float(sec["EI_Nm2"])
        else:
            raise ValueError('Section "EI" needs EI_kNm2, EI_MNm2 or EI_Nm2.')
        if EI <= 0:
            raise ValueError("EI must be positive.")
        return dict(name=f"Given EI = {EI/1e3:g} kN.m^2", I=float("nan"),
                    c=float("nan"), EI=EI)

    raise ValueError(f"Unknown section type '{sec['type']}'.")


# -----------------------------------------------------------------------------
# 4. BUILD + VALIDATE THE MODEL FROM A CONFIG DICTIONARY
# -----------------------------------------------------------------------------
def build_model(cfg):
    """Convert the user's dictionary (units: kN, m, ...) into a BeamModel (N, m)."""
    L = float(cfg["L"])
    if L <= 0:
        raise ValueError("Beam length must be positive.")

    # --- stiffness -------------------------------------------------------------
    sec = section_properties(cfg["section"])
    if "EI" in sec:                                   # EI given directly
        EI, E = sec["EI"], float("nan")
    else:                                             # EI = E * I
        E = float(cfg.get("E_GPa", 200.0)) * 1e9
        if E <= 0:
            raise ValueError("Young's modulus must be positive.")
        EI = E * sec["I"]

    # --- supports (default = cantilever fixed at x = 0) --------------------------
    raw = cfg.get("supports") or [dict(type="fixed", x=0.0)]
    supports = []
    for s in raw:
        kind = str(s["type"]).lower()
        kind = "fixed" if kind in ("fixed", "clamped", "built-in", "builtin") else \
               "pin" if kind in ("pin", "roller", "hinge", "simple", "support") else None
        if kind is None:
            raise ValueError(f"Unknown support type '{s['type']}' (use 'fixed' or 'pin').")
        x = float(s["x"])
        if not (-TOL <= x <= L + TOL):
            raise ValueError(f"Support at x = {x} lies outside the beam (0 .. {L}).")
        if kind == "fixed" and TOL < x < L - TOL:
            raise ValueError("A fixed (built-in) support is only allowed at x = 0 or x = L.")
        supports.append(Support(kind, min(max(x, 0.0), L)))
    supports.sort(key=lambda s: s.x)
    for s1, s2 in zip(supports, supports[1:]):
        if abs(s1.x - s2.x) < 1e-8:
            raise ValueError(f"Two supports at the same position x = {s1.x}.")

    # --- loads (kN -> N) -----------------------------------------------------------
    def inside(v, what):
        if not (-TOL <= v <= L + TOL):
            raise ValueError(f"{what} at x = {v} lies outside the beam (0 .. {L}).")
        return min(max(v, 0.0), L)

    pls = [PointLoad(float(p["P_kN"]) * 1e3, inside(float(p["a"]), "Point load"))
           for p in cfg.get("point_loads", [])]

    dls = []
    for d in cfg.get("distributed_loads", cfg.get("udls", [])):
        s, e = inside(float(d["start"]), "UDL start"), inside(float(d["end"]), "UDL end")
        if not s < e:
            raise ValueError(f"Distributed load needs start < end (got {s}, {e}).")
        w1 = float(d["w1_kNm"]) * 1e3
        w2 = float(d.get("w2_kNm", d["w1_kNm"])) * 1e3
        dls.append(DistLoad(w1, w2, s, e))

    cps = []
    for m in cfg.get("moments", []):
        a = inside(float(m["a"]), "Couple")
        if a <= TOL:
            raise ValueError("A couple exactly at x = 0 is not supported; use x = 0.001.")
        cps.append(Couple(float(m["M_kNm"]) * 1e3, a))

    q = [inside(float(v), "Query point") for v in cfg.get("query_points", [])]
    return BeamModel(L, EI, E, sec["I"], sec["c"], sec["name"], supports, pls, dls, cps, q)


# -----------------------------------------------------------------------------
# 5. THE EXACT ENGINE:  REFERENCE CANTILEVER (clamped at x = 0, free at x = L)
# -----------------------------------------------------------------------------
class ReferenceCantilever:
    """
    Exact analysis of a cantilever clamped at x = 0 under the given loads.

    Between two neighbouring "break points" (load positions, UDL ends, ends of
    the beam) every quantity is a polynomial in x, so we keep them as numpy
    Polynomial objects and integrate them exactly:

        V(x)  = total downward load to the RIGHT of x       (polynomial)
        m(x)  = hogging moment at x   (dm/dx = -V)          integrate from x = L
        th(x) = (1/EI) * integral of m                       from the clamp, th(0)=0
        y(x)  = integral of th                               from the clamp, y(0)=0

    Convention inside this class:  m is HOGGING-positive and y is DOWNWARD-
    positive (so a downward tip load gives m > 0, th > 0, y > 0).
    """

    def __init__(self, L, EI, point_loads, dist_loads, couples):
        self.L, self.EI = L, EI
        self.P, self.W, self.M = point_loads, dist_loads, couples

        # --- break points: segment boundaries ------------------------------------
        pts = [0.0, L] + [p.a for p in self.P] + [c.a for c in self.M]
        for d in self.W:
            pts += [d.start, d.end]
        self.breaks = np.unique(np.round(np.clip(pts, 0, L), 10))
        nseg = len(self.breaks) - 1

        # --- STEP A: shear polynomial V(x) in every segment -------------------------
        V = [None] * nseg
        for k in range(nseg):
            x0, x1 = self.breaks[k], self.breaks[k + 1]
            v = Poly([0.0])
            for p in self.P:                       # point loads lying to the right
                if p.a >= x1 - TOL:
                    v = v + p.P
            for d in self.W:                       # distributed loads
                if x0 >= d.end - TOL:              # completely to the left -> no effect
                    continue
                slope = (d.w2 - d.w1) / (d.end - d.start)
                w = Poly([d.w1 - slope * d.start, slope])   # w(s) = intensity at s
                W_int = w.integ()                           # antiderivative of w
                if x1 <= d.start + TOL:            # completely to the right of segment
                    v = v + (W_int(d.end) - W_int(d.start))
                else:                              # segment lies inside the load
                    v = v + (Poly([W_int(d.end)]) - W_int)  # load between x and 'end'
            V[k] = v

        # --- STEP B: hogging moment, integrated from the free end (x = L) back -------
        m = [None] * nseg
        running = 0.0                              # m just to the right of x = L is 0
        for k in range(nseg - 1, -1, -1):
            x0, x1 = self.breaks[k], self.breaks[k + 1]
            # an applied (clockwise) couple at x1 adds a jump to the moment on its left
            m_right_end = running + sum(c.M for c in self.M if abs(c.a - x1) < 1e-8)
            Vi = V[k].integ()
            m[k] = Poly([m_right_end + Vi(x1)]) - Vi     # m(x) = m(x1) + int_x^x1 V
            running = m[k](x0)

        # --- STEP C: slope and deflection, integrated from the clamp (x = 0) ----------
        th, y = [None] * nseg, [None] * nseg
        th0 = y0 = 0.0
        for k in range(nseg):
            x0, x1 = self.breaks[k], self.breaks[k + 1]
            Mi = m[k].integ() / EI
            th[k] = Mi + (th0 - Mi(x0))            # continuity of slope
            yi = th[k].integ()
            y[k] = yi + (y0 - yi(x0))              # continuity of deflection
            th0, y0 = th[k](x1), y[k](x1)

        self.V, self.m, self.th, self.y = V, m, th, y

    def at(self, x):
        """Values at one position (right-limit at inner break points, left-limit at x=L)."""
        k = int(np.clip(np.searchsorted(self.breaks, x, side="right") - 1,
                        0, len(self.breaks) - 2))
        return dict(V=self.V[k](x), m=self.m[k](x), theta=self.th[k](x), y=self.y[k](x))


# -----------------------------------------------------------------------------
# 6. SOLVING FOR THE SUPPORT REACTIONS  (the "general beam" part)
# -----------------------------------------------------------------------------
def solve_beam(model):
    """
    Replace the real supports by unknowns and solve  A u = b.

    UNKNOWN VECTOR u
        for every PIN at x > 0           : reaction R  (kN, upward)
        if FIXED at x = L                : reaction R (kN) and end moment Mr (kN.m)
        always                           : rigid-body translation y0 (m) and
                                           rotation th0 (rad) of the whole beam
        (a support at x = 0 needs no reaction unknown: the clamp of the
         reference cantilever supplies it, and we read it afterwards.)

    EQUATIONS  (one per unknown)
        fixed at 0     : y0 = 0 ,  th0 = 0
        pin at 0       : y0 = 0 ,  moment at x=0 = 0
        no support at 0: shear at x=0 = 0 (balancing any load at x=0) ,
                         moment at x=0 = 0         (free end)
        every pin x>0  : deflection = 0
        fixed at L     : deflection = 0 and slope = 0

    Everything is linear, so each column of A is simply the response of the
    reference cantilever to a UNIT value of that unknown.
    """
    L, EI = model.L, model.EI
    pins = [s.x for s in model.supports if s.kind == "pin"]
    fixed_left = any(s.kind == "fixed" and s.x <= TOL for s in model.supports)
    fixed_right = any(s.kind == "fixed" and s.x >= L - TOL for s in model.supports)
    pin_left = any(x <= TOL for x in pins)

    # ---- 6.1  list the unknown reactions --------------------------------------------
    unknowns = [("R", x) for x in pins if x > TOL]
    if fixed_right:
        unknowns += [("R", L), ("Mr", L)]
    n_react = len(unknowns)
    n = n_react + 2                      # + y0 and th0
    IY0, ITH0 = n_react, n_react + 1     # their positions in the unknown vector

    # ---- 6.2  list the support equations --------------------------------------------
    eqs = []
    if fixed_left:
        eqs += [("y0", None), ("th0", None)]
    elif pin_left:
        eqs += [("y0", None), ("M0", None)]
    else:
        eqs += [("V0", None), ("M0", None)]
    eqs += [("y", x) for x in pins if x > TOL]
    if fixed_right:
        eqs += [("y", L), ("th", L)]
    if len(eqs) != n:                    # cannot happen, kept as a safety net
        raise ValueError("Internal error: equations and unknowns differ.")

    # point loads sitting exactly at x = 0 go straight to a support there, or are
    # carried by the beam if x = 0 is a free end.  They do not enter V(0+).
    P0 = sum(p.P for p in model.point_loads if p.a <= TOL)

    def response(ref, eq, extra=0.0):
        """Value of one support-equation quantity for a reference-cantilever case."""
        name, x = eq
        if name == "V0":
            return ref.at(0.0)["V"] + extra
        if name == "M0":
            return ref.at(0.0)["m"]
        if name == "y":
            return ref.at(x)["y"]
        if name == "th":
            return ref.at(x)["theta"]
        return 0.0                       # "y0", "th0" act only on the rigid motion

    def rigid(eq):
        """How y0 and th0 enter each equation: (coefficient of y0, coefficient of th0)."""
        name, x = eq
        return {"y": (1.0, x), "th": (0.0, 1.0), "y0": (1.0, 0.0),
                "th0": (0.0, 1.0)}.get(name, (0.0, 0.0))

    # ---- 6.3  the base case (applied loads only) and one unit case per unknown ---------
    def case(pl=(), cp=()):
        return ReferenceCantilever(L, EI, list(pl), [], list(cp))

    base = ReferenceCantilever(L, EI, model.point_loads, model.dist_loads, model.couples)
    units = []
    for kind, x in unknowns:
        if kind == "R":                                   # 1 kN UPWARD at x
            units.append(case(pl=[PointLoad(-1e3, x)]))
        else:                                             # 1 kN.m clockwise couple at L
            units.append(case(cp=[Couple(1e3, x)]))

    # ---- 6.4  assemble and solve --------------------------------------------------------
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i, eq in enumerate(eqs):
        for j, ref in enumerate(units):
            A[i, j] = response(ref, eq)
        A[i, IY0], A[i, ITH0] = rigid(eq)
        b[i] = -response(base, eq, extra=P0)
        scale = np.max(np.abs(A[i])) or 1.0               # row scaling for accuracy
        A[i], b[i] = A[i] / scale, b[i] / scale
    if np.linalg.matrix_rank(A) < n:
        raise ValueError("The supports do not hold the beam stable (mechanism). "
                         "Add a fixed end or at least two pins.")
    u = np.linalg.solve(A, b)

    # ---- 6.5  final reference cantilever = applied loads + solved reactions -----------------
    pl = list(model.point_loads)
    cp = list(model.couples)
    for (kind, x), val in zip(unknowns, u[:n_react]):
        if kind == "R":
            pl.append(PointLoad(-val * 1e3, x))
        else:
            cp.append(Couple(val * 1e3, x))
    final = ReferenceCantilever(L, EI, pl, model.dist_loads, cp)
    return Solution(model, final, y0=u[IY0], th0=u[ITH0], P0=P0, unknowns=unknowns,
                    u=u[:n_react])


# -----------------------------------------------------------------------------
# 7. SOLUTION OBJECT: results at any x, sampling for plots, extrema, reactions
# -----------------------------------------------------------------------------
class Solution:
    def __init__(self, model, ref, y0, th0, P0, unknowns, u):
        self.model, self.ref, self.y0, self.th0 = model, ref, y0, th0
        self.breaks = ref.breaks
        # polynomial pieces of the FINAL answer (sign conventions of the header)
        n = len(self.breaks) - 1
        self.Vp = [ref.V[k] for k in range(n)]                       # shear
        self.Mp = [-ref.m[k] for k in range(n)]                      # sagging +
        self.thp = [ref.th[k] + th0 for k in range(n)]               # slope
        self.yp = [ref.y[k] + Poly([y0, th0]) for k in range(n)]     # deflection
        self.reactions = self._reactions(P0, unknowns, u)

    # ---- values at one point ---------------------------------------------------------------
    def at(self, x):
        k = int(np.clip(np.searchsorted(self.breaks, x, side="right") - 1,
                        0, len(self.breaks) - 2))
        return dict(V=self.Vp[k](x), M=self.Mp[k](x),
                    theta=self.thp[k](x), y=self.yp[k](x))

    # ---- reactions of every support ----------------------------------------------------------
    def _reactions(self, P0, unknowns, u):
        out, solved = [], {(kd, x): v for (kd, x), v in zip(unknowns, u)}
        for i, s in enumerate(self.model.supports):
            r = dict(label=chr(ord("A") + i), kind=s.kind, x=s.x, R=0.0, M=None)
            if s.x <= TOL:                      # left end: read it from the clamp
                r["R"] = (self.ref.at(0.0)["V"] + P0) / 1e3
            else:
                r["R"] = solved[("R", s.x)]
            if s.kind == "fixed":               # internal moment at the wall (sagging +)
                r["M"] = self.at(0.0)["M"] / 1e3 if s.x <= TOL else self.at(self.model.L)["M"] / 1e3
            out.append(r)
        return out

    def total_load(self):
        m = self.model
        tot = sum(p.P for p in m.point_loads)
        tot += sum((d.w1 + d.w2) / 2 * (d.end - d.start) for d in m.dist_loads)
        return tot / 1e3

    # ---- dense sampling for plots (break points appear twice: jumps plot vertically) ---------
    def sample(self, n_per_seg=400):
        xs, V, M, th, y = [], [], [], [], []
        for k in range(len(self.breaks) - 1):
            x0, x1 = self.breaks[k], self.breaks[k + 1]
            x = np.linspace(x0, x1, max(int(n_per_seg * (x1 - x0) / self.model.L), 25))
            xs.append(x), V.append(self.Vp[k](x)), M.append(self.Mp[k](x))
            th.append(self.thp[k](x)), y.append(self.yp[k](x))
        c = np.concatenate
        return dict(x=c(xs), V=c(V), M=c(M), theta=c(th), y=c(y))

    # ---- exact extreme values ---------------------------------------------------------------------
    @staticmethod
    def _roots_inside(d, x0, x1):
        """Real roots of polynomial d strictly inside (x0, x1)."""
        coef = np.array(d.coef, float)
        scale = np.max(np.abs(coef)) if coef.size else 0.0
        if scale == 0:
            return []
        d = Poly(coef).trim(tol=1e-12 * scale)
        if d.degree() < 1:
            return []
        return [z.real for z in d.roots()
                if abs(z.imag) < 1e-9 * max(1.0, abs(z.real)) and x0 < z.real < x1]

    def extremes(self, quantity):
        """
        Largest and smallest value of 'y', 'theta', 'M' or 'V' along the beam, with
        locations.  Candidates are the segment ends plus the roots of the derivative
        (e.g. maximum deflection is where the slope is zero).
        Returns ((x_max, v_max), (x_min, v_min)).
        """
        val, der = {
            "y":     (self.yp, self.thp),                          # y' = slope
            "theta": (self.thp, [-p for p in self.Mp]),            # slope' ~ hogging moment
            "M":     (self.Mp, self.Vp),                           # M' = V
            "V":     (self.Vp, [p.deriv() for p in self.Vp]),      # V' = -w
        }[quantity]
        cands = []
        for k in range(len(self.breaks) - 1):
            x0, x1 = self.breaks[k], self.breaks[k + 1]
            for x in [x0, x1] + self._roots_inside(der[k], x0, x1):
                cands.append((float(x), float(val[k](x))))
        return max(cands, key=lambda t: t[1]), min(cands, key=lambda t: t[1])


# -----------------------------------------------------------------------------
# 8. TEXT REPORT
# -----------------------------------------------------------------------------
def report(sol):
    m = sol.model
    line = "=" * 68
    print(f"\n{line}\n BEAM DATA\n{line}")
    print(f" Length L              : {m.L:.4f} m")
    print(f" Section / stiffness   : {m.section_name}")
    if not np.isnan(m.E):
        print(f" Young's modulus E     : {m.E/1e9:.2f} GPa")
        print(f" Moment of inertia I   : {m.I:.6e} m^4")
    print(f" Flexural rigidity EI  : {m.EI:.6e} N.m^2  ({m.EI/1e3:.6g} kN.m^2)")
    print(" Supports              : " + ", ".join(
        f"{chr(65+i)}: {s.kind} at x = {s.x:g} m" for i, s in enumerate(m.supports)))
    print(f"\n Point loads           : {len(m.point_loads)}")
    for p in m.point_loads:
        print(f"    {p.P/1e3:10.3f} kN  at x = {p.a:.3f} m")
    print(f" Distributed loads     : {len(m.dist_loads)}")
    for d in m.dist_loads:
        udl = abs(d.w1 - d.w2) < 1e-9
        txt = f"{d.w1/1e3:.3f} kN/m" if udl else f"{d.w1/1e3:.3f} -> {d.w2/1e3:.3f} kN/m"
        print(f"    {'UDL' if udl else 'UVL'}  {txt}  from x = {d.start:.3f} to {d.end:.3f} m")
    print(f" Applied couples       : {len(m.couples)}")
    for c in m.couples:
        print(f"    {c.M/1e3:10.3f} kN.m (clockwise +) at x = {c.a:.3f} m")

    print(f"\n{line}\n SUPPORT REACTIONS\n{line}")
    total_R = 0.0
    for r in sol.reactions:
        txt = f" {r['label']} ({r['kind']:5s}) at x = {r['x']:7.3f} m :  R = {r['R']:12.4f} kN (up)"
        if r["M"] is not None:
            txt += f",  end moment = {r['M']:.4f} kN.m ({'sagging' if r['M'] >= 0 else 'hogging'})"
        print(txt)
        total_R += r["R"]
    print(f" Equilibrium check     : sum R = {total_R:.4f} kN, total load = {sol.total_load():.4f} kN")

    print(f"\n{line}\n RESULTS\n{line}")
    (xM, Mmax), (xm, Mmin) = sol.extremes("M")
    (xV1, V1), (xV2, V2) = sol.extremes("V")
    Vabs = max((xV1, V1), (xV2, V2), key=lambda t: abs(t[1]))
    print(f" Max sagging moment    : {Mmax/1e3:12.4f} kN.m at x = {xM:.4f} m")
    if Mmin < -1e-6:
        print(f" Max hogging moment    : {Mmin/1e3:12.4f} kN.m at x = {xm:.4f} m")
    else:
        print(" Max hogging moment    :         none (beam sags everywhere)")
    print(f" Max |shear force|     : {abs(Vabs[1])/1e3:12.4f} kN   at x = {Vabs[0]:.4f} m")
    if not np.isnan(m.c) and not np.isnan(m.I):
        big = max(abs(Mmax), abs(Mmin))
        print(f" Max bending stress    : {big*m.c/m.I/1e6:12.3f} MPa  (sigma = M c / I)")
    else:
        print(" Max bending stress    :          n/a (section size not given)")

    (xyd, ydown), (xyu, yup) = sol.extremes("y")
    ymax = (xyd, ydown) if abs(ydown) >= abs(yup) else (xyu, yup)
    print(f" Max |deflection|      : {abs(ymax[1])*1e3:12.6f} mm at x = {ymax[0]:.4f} m "
          f"({'downward' if ymax[1] > 0 else 'upward'})")
    if ydown > 1e-12 and yup < -1e-12:
        print(f"    max downward       : {ydown*1e3:12.6f} mm at x = {xyd:.4f} m")
        print(f"    max upward         : {-yup*1e3:12.6f} mm at x = {xyu:.4f} m")
    (xt1, t1), (xt2, t2) = sol.extremes("theta")
    tmax = max((xt1, t1), (xt2, t2), key=lambda t: abs(t[1]))
    print(f" Max |slope|           : {abs(tmax[1]):12.6e} rad at x = {tmax[0]:.4f} m")
    clean = lambda v: 0.0 if abs(v) < 1e-12 else v      # avoid printing "-0.000000"
    for tag, xe in (("x = 0", 0.0), ("x = L", m.L)):
        r = sol.at(xe)
        print(f" Deflection at {tag}   : {clean(r['y'])*1e3:12.6f} mm,  slope {clean(r['theta']):.6e} rad")
    if abs(ymax[1]) > 0:
        print(f" Span / max deflection : {m.L/abs(ymax[1]):.1f}")

    if m.query_points:
        print(f"\n{line}\n RESULTS AT REQUESTED POINTS   (y +ve downward, slope +ve clockwise)\n{line}")
        print("   x (m)     shear (kN)   moment (kN.m)    slope (rad)     deflection (mm)")
        for x in m.query_points:
            r = sol.at(x)
            print(f" {x:7.3f}   {r['V']/1e3:11.4f}   {r['M']/1e3:13.4f}   {r['theta']:13.6e}   {r['y']*1e3:14.6f}")


# -----------------------------------------------------------------------------
# 9. PLOTS
# -----------------------------------------------------------------------------
def draw_loading_diagram(ax, sol):
    """Beam, supports, applied loads and the solved reactions."""
    from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
    m, L = sol.model, sol.model.L
    ax.plot([0, L], [0, 0], "k-", lw=5, solid_capstyle="butt", zorder=3)

    # --- supports ---
    for s in m.supports:
        if s.kind == "fixed":
            x0 = -0.03 * L if s.x <= TOL else L
            ax.add_patch(Rectangle((x0, -0.5), 0.03 * L, 1.0, fc="0.55", ec="k",
                                   hatch="////", zorder=2))
        else:
            ax.add_patch(Polygon([[s.x, 0], [s.x - 0.025 * L, -0.38], [s.x + 0.025 * L, -0.38]],
                                 closed=True, fc="0.8", ec="k", zorder=2))
            ax.plot([s.x - 0.04 * L, s.x + 0.04 * L], [-0.38, -0.38], "k-", lw=2)
    for r in sol.reactions:           # support letters
        ax.text(r["x"], 0.12, r["label"], ha="center", fontsize=10, fontweight="bold")

    Pmax = max([abs(p.P) for p in m.point_loads] + [1e-9])
    wmax = max([max(abs(d.w1), abs(d.w2)) for d in m.dist_loads] + [1e-9])

    # --- point loads ---
    for p in m.point_loads:
        h = 0.7 + 0.5 * abs(p.P) / Pmax
        y0, y1 = (h, 0.05) if p.P >= 0 else (-h, -0.05)
        ax.add_patch(FancyArrowPatch((p.a, y0), (p.a, y1), arrowstyle="-|>", mutation_scale=16,
                                     color="tab:red", lw=2, zorder=4))
        ax.text(p.a, y0 + (0.1 if p.P >= 0 else -0.22), f"{p.P/1e3:g} kN", ha="center",
                color="tab:red", fontsize=9, fontweight="bold")

    # --- distributed loads ---
    for d in m.dist_loads:
        hh = lambda w: 0.25 + 0.35 * abs(w) / wmax
        ax.fill([d.start, d.end, d.end, d.start],
                [0.05, 0.05, hh(d.w2) + 0.05, hh(d.w1) + 0.05], color="tab:blue", alpha=0.25)
        ax.plot([d.start, d.end], [hh(d.w1) + 0.05, hh(d.w2) + 0.05], color="tab:blue", lw=2)
        for xx in np.linspace(d.start, d.end, max(int(12 * (d.end - d.start) / L), 4)):
            w = d.w1 + (d.w2 - d.w1) * (xx - d.start) / (d.end - d.start)
            top = hh(w) + 0.05
            a0, a1 = (top, 0.05) if w >= 0 else (0.05, top)
            ax.add_patch(FancyArrowPatch((xx, a0), (xx, a1), arrowstyle="-|>",
                                         mutation_scale=9, color="tab:blue", lw=1))
        txt = (f"{d.w1/1e3:g} kN/m" if abs(d.w1 - d.w2) < 1e-9
               else f"{d.w1/1e3:g}->{d.w2/1e3:g} kN/m")
        ax.text((d.start + d.end) / 2, max(hh(d.w1), hh(d.w2)) + 0.15, txt, ha="center",
                color="tab:blue", fontsize=9, fontweight="bold")

    # --- couples ---
    for c in m.couples:
        rad = -0.9 if c.M > 0 else 0.9            # clockwise for positive couples
        ax.add_patch(FancyArrowPatch((c.a - 0.04 * L, 0.45), (c.a + 0.04 * L, 0.45),
                                     connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=15, color="tab:green", lw=2))
        ax.text(c.a, 0.8, f"{c.M/1e3:g} kN.m", ha="center", color="tab:green",
                fontsize=9, fontweight="bold")

    # --- reactions (solved) ---
    for r in sol.reactions:
        if abs(r["R"]) < 1e-9:
            continue
        a0, a1 = (-1.45, -0.45) if r["R"] >= 0 else (-0.45, -1.45)
        ax.add_patch(FancyArrowPatch((r["x"], a0), (r["x"], a1), arrowstyle="-|>",
                                     mutation_scale=14, color="tab:purple", lw=2))
        ax.text(r["x"], -1.75, f"R{r['label']} = {r['R']:.4g} kN", ha="center",
                color="tab:purple", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.08 * L, 1.08 * L)
    ax.set_ylim(-2.0, 2.2)
    ax.set_yticks([])
    ax.set_title("Loading diagram with support reactions")
    ax.grid(alpha=0.3)


def make_plots(sol, s, outdir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, x = sol.model, s["x"]
    fig, axs = plt.subplots(5, 1, figsize=(11, 17), sharex=True)
    draw_loading_diagram(axs[0], sol)

    def curve(ax, yv, label, color, title, invert=False, marks=()):
        ax.plot(x, yv, color=color, lw=2)
        ax.fill_between(x, 0, yv, color=color, alpha=0.15)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(label), ax.set_title(title), ax.grid(alpha=0.3)
        ax.margins(y=0.25)                         # head-room so labels stay inside the axes
        if invert:
            ax.invert_yaxis()
        i = np.argmax(abs(yv))
        ax.plot(x[i], yv[i], "ko")
        # put the label on whichever side of the beam has room, away from the plot edge
        right_side = x[i] < 0.6 * x[-1]
        ax.annotate(f"max {yv[i]:.4g} @ x={x[i]:.3f} m", (x[i], yv[i]), textcoords="offset points",
                    xytext=(12 if right_side else -12, -22 if yv[i] > 0 else 22),
                    ha="left" if right_side else "right", fontsize=9,
                    bbox=dict(boxstyle="round", fc="w", ec="0.6"))
        for xq, vq in marks:                       # requested points
            ax.plot(xq, vq, "s", color="k", ms=6)
            ax.annotate(f"x={xq:g}: {vq:.4g}", (xq, vq), textcoords="offset points",
                        xytext=(6, -16 if not invert else 12), fontsize=8)

    q = m.query_points
    curve(axs[1], s["V"] / 1e3, "V (kN)", "tab:purple", "Shear force diagram",
          marks=[(xq, sol.at(xq)["V"] / 1e3) for xq in q])
    curve(axs[2], s["M"] / 1e3, "M (kN.m)", "tab:orange", "Bending moment diagram (sagging +)",
          marks=[(xq, sol.at(xq)["M"] / 1e3) for xq in q])
    curve(axs[3], s["theta"], "slope (rad)", "tab:green", "Slope variation (clockwise +)",
          marks=[(xq, sol.at(xq)["theta"]) for xq in q])
    curve(axs[4], s["y"] * 1e3, "y (mm)", "tab:red", "Deflection variation (downward drawn down)",
          invert=True, marks=[(xq, sol.at(xq)["y"] * 1e3) for xq in q])
    axs[4].set_xlabel("Distance from left end x (m)")
    head = f"EI = {m.EI/1e3:.6g} kN.m^2" if np.isnan(m.E) else f"{m.section_name}, EI = {m.EI/1e3:.6g} kN.m^2"
    fig.suptitle(f"Beam analysis - L = {m.L:g} m, {head}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "beam_results.png")
    fig.savefig(path, dpi=150)
    print(f"\n Plot saved to: {path}")

    # separate slope and deflection figures
    for key, lab, col, fname, ttl, scale, inv in [
            ("theta", "Slope (rad)", "tab:green", "slope.png", "Slope variation along the beam", 1, False),
            ("y", "Deflection y (mm)", "tab:red", "deflection.png", "Deflection variation along the beam", 1e3, True)]:
        f2, a2 = plt.subplots(figsize=(10, 4.5))
        a2.plot(x, s[key] * scale, color=col, lw=2)
        a2.fill_between(x, 0, s[key] * scale, color=col, alpha=0.15)
        a2.axhline(0, color="k", lw=0.8)
        a2.set_xlabel("Distance from left end x (m)"), a2.set_ylabel(lab)
        a2.set_title(ttl), a2.grid(alpha=0.3)
        if inv:
            a2.invert_yaxis()
        f2.tight_layout()
        f2.savefig(os.path.join(outdir, fname), dpi=150)
        if not show:
            plt.close(f2)
    if show:
        plt.show()
    else:
        plt.close("all")


def export_csv(s, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "beam_results.csv")
    data = np.column_stack([s["x"], s["V"] / 1e3, s["M"] / 1e3, s["theta"], s["y"] * 1e3])
    np.savetxt(path, data, delimiter=",", comments="",
               header="x_m,shear_kN,moment_kNm_sagging,slope_rad,deflection_mm_down")
    print(f" CSV saved to: {path}")


# -----------------------------------------------------------------------------
# 10. BUILT-IN TEXTBOOK EXAMPLES  (also used by the self-tests)
# -----------------------------------------------------------------------------
def examples():
    return {
        "5.1a": dict(  # 4 m cantilever, 40 kN at 3 m from the wall, EI = 65 MN.m^2
            L=4.0, section=dict(type="EI", EI_MNm2=65),
            supports=[dict(type="fixed", x=0)],
            point_loads=[dict(P_kN=40, a=3.0)], query_points=[3.0, 4.0]),
        "5.1b": dict(  # same total load (40 kN) spread uniformly over the first 3 m
            L=4.0, section=dict(type="EI", EI_MNm2=65),
            supports=[dict(type="fixed", x=0)],
            distributed_loads=[dict(w1_kNm=40 / 3, start=0, end=3.0)], query_points=[3.0, 4.0]),
        "5.2": dict(   # simply supported 5 m beam; E = 200 GN/m^2, I = 83e-6 m^4
            L=5.0, E_GPa=200, section=dict(type="custom", I_m4=83e-6),
            supports=[dict(type="pin", x=0), dict(type="pin", x=5.0)],
            point_loads=[dict(P_kN=20, a=1.0), dict(P_kN=50, a=3.0)],
            distributed_loads=[dict(w1_kNm=60, start=3.0, end=5.0)], query_points=[3.0]),
        "5.3": dict(   # overhang at the left: supports at 0.6 m (A) and 3 m (B), EI = 0.65 MN.m^2
            L=3.0, section=dict(type="EI", EI_MNm2=0.65),
            supports=[dict(type="pin", x=0.6), dict(type="pin", x=3.0)],
            point_loads=[dict(P_kN=20, a=0.0), dict(P_kN=20, a=1.8)],
            distributed_loads=[dict(w1_kNm=30, start=0.6, end=1.8)], query_points=[1.0]),
    }


# -----------------------------------------------------------------------------
# 11. INTERACTIVE INPUT
# -----------------------------------------------------------------------------
def ask(prompt, default=None, cast=float, check=None, msg=""):
    """Prompt until a valid value is typed.  Empty input accepts the default."""
    while True:
        raw = input(prompt + (f" [{default}]" if default is not None else "") + ": ").strip()
        try:
            v = default if (raw == "" and default is not None) else cast(raw)
            if check is None or check(v):
                return v
            print("  ! Invalid value. " + msg)
        except ValueError:
            print("  ! Please enter a valid value.")


def ask_list(prompt, L, allow_empty=True):
    """Comma separated positions (all between 0 and L)."""
    while True:
        raw = input(prompt + ": ").strip()
        if raw == "" and allow_empty:
            return []
        try:
            vals = [float(t) for t in raw.replace(";", ",").split(",") if t.strip()]
            if all(0 <= v <= L for v in vals) and (vals or allow_empty):
                return vals
            print(f"  ! Values must lie between 0 and {L:g} m.")
        except ValueError:
            print("  ! Enter numbers separated by commas.")


def interactive_config():
    print("\n================ GENERAL BEAM ANALYSER ================")
    print("Loads +ve DOWNWARD | couples +ve CLOCKWISE | x measured from the LEFT end\n")
    L = ask("Beam length L (m)", check=lambda v: v > 0, msg="Must be > 0.")

    # ---- supports -------------------------------------------------------------------
    print("\n--------- SUPPORTS ---------")
    print(" 1. Cantilever, fixed at LEFT end (x = 0)")
    print(" 2. Cantilever, fixed at RIGHT end (x = L)")
    print(" 3. Simply supported (pins at x = 0 and x = L)")
    print(" 4. Overhanging / any pins (you give the pin positions, two or more)")
    print(" 5. Propped cantilever (fixed at x = 0 + pin(s) you choose)")
    print(" 6. Fixed at both ends")
    ch = ask("Select (1-6)", 1, cast=int, check=lambda v: 1 <= v <= 6, msg="Enter 1-6.")
    if ch == 1:
        sup = [dict(type="fixed", x=0)]
    elif ch == 2:
        sup = [dict(type="fixed", x=L)]
    elif ch == 3:
        sup = [dict(type="pin", x=0), dict(type="pin", x=L)]
    elif ch == 4:
        pos = ask_list("  Pin positions, comma separated (m)", L, allow_empty=False)
        sup = [dict(type="pin", x=p) for p in pos]
    elif ch == 5:
        pos = ask_list("  Pin positions, comma separated (m)", L, allow_empty=False)
        sup = [dict(type="fixed", x=0)] + [dict(type="pin", x=p) for p in pos]
    else:
        sup = [dict(type="fixed", x=0), dict(type="fixed", x=L)]

    # ---- stiffness --------------------------------------------------------------------
    print("\n--------- CROSS SECTION / STIFFNESS ---------")
    print(" 1. Rectangular     2. I-section     3. Custom (enter I)     4. EI given directly")
    sc = ask("Select (1-4)", 1, cast=int, check=lambda v: 1 <= v <= 4, msg="Enter 1-4.")
    pos_ = lambda v: v > 0
    E = 200.0
    if sc != 4:
        E = ask("Young's modulus E (GPa)  (steel 200, aluminium 70, concrete ~30)", 200.0, check=pos_)
    if sc == 1:
        sec = dict(type="rect", b_mm=ask("  Width b (mm)", 150.0, check=pos_),
                   h_mm=ask("  Depth h (mm)", 300.0, check=pos_))
    elif sc == 2:
        sec = dict(type="I", B_mm=ask("  Flange width B (mm)", 200.0, check=pos_),
                   H_mm=ask("  Overall depth H (mm)", 350.0, check=pos_),
                   tf_mm=ask("  Flange thickness tf (mm)", 12.0, check=pos_),
                   tw_mm=ask("  Web thickness tw (mm)", 8.0, check=pos_))
    elif sc == 3:
        sec = dict(type="custom", I_cm4=ask("  I (cm^4)   [1 m^4 = 1e8 cm^4]", check=pos_),
                   c_mm=ask("  Extreme-fibre distance c (mm), 0 if unknown", 0.0, check=lambda v: v >= 0))
    else:
        print("  Unit of EI:  1 = kN.m^2    2 = MN.m^2    3 = N.m^2")
        u = ask("  Choose unit", 1, cast=int, check=lambda v: v in (1, 2, 3))
        key = {1: "EI_kNm2", 2: "EI_MNm2", 3: "EI_Nm2"}[u]
        sec = {"type": "EI", key: ask("  Flexural rigidity EI", check=pos_)}

    # ---- loads --------------------------------------------------------------------------
    inside = lambda v: 0 <= v <= L
    n = ask("\nNumber of POINT loads (assignment asks for at least 3; 0 = none)", 3, cast=int,
            check=lambda v: v >= 0)
    pl = []
    for i in range(n):
        print(f" Point load {i+1}")
        pl.append(dict(P_kN=ask("   Magnitude (kN, +ve down)"),
                       a=ask(f"   Distance from LEFT end (0..{L:g} m)", check=inside, msg="Outside beam.")))

    n = ask("\nNumber of distributed loads (UDL / UVL)", 1, cast=int, check=lambda v: v >= 0)
    dl = []
    for i in range(n):
        print(f" Distributed load {i+1}   (equal w1 and w2 = UDL)")
        s = ask(f"   Start x (0..{L:g} m)", 0.0, check=inside, msg="Outside beam.")
        e = ask(f"   End x ({s:g}..{L:g} m)", L, check=lambda v: s < v <= L, msg="Must be > start, <= L.")
        w1 = ask("   Intensity at start w1 (kN/m, +ve down)")
        w2 = ask("   Intensity at end   w2 (kN/m)", w1)
        dl.append(dict(w1_kNm=w1, w2_kNm=w2, start=s, end=e))

    n = ask("\nNumber of applied couples (moments)", 0, cast=int, check=lambda v: v >= 0)
    mo = []
    for i in range(n):
        print(f" Couple {i+1}")
        mo.append(dict(M_kNm=ask("   Magnitude (kN.m, +ve clockwise)"),
                       a=ask(f"   Position (0.001..{L:g} m)", check=lambda v: 0 < v <= L)))

    q = ask_list("\nPositions where you want slope & deflection (comma separated, blank = none)", L)
    return dict(L=L, E_GPa=E, section=sec, supports=sup, point_loads=pl,
                distributed_loads=dl, moments=mo, query_points=q)


# -----------------------------------------------------------------------------
# 12. SELF-TESTS: closed-form formulas + an INDEPENDENT Macaulay calculation
# -----------------------------------------------------------------------------
def _macaulay(terms, EI, zero_at):
    """
    Independent textbook check.  terms = [(c, a, n), ...] describes the SAGGING
    moment  M(x) = sum c <x-a>^n  (kN, m).  Integrating twice,
        EI*y_up = sum c <x-a>^(n+2)/((n+1)(n+2)) + C1 x + C2,
    with C1, C2 from  y = 0  at the two x-values in 'zero_at'.
    Returns a function giving the DOWNWARD deflection (m) and slope (rad).
    """
    br = lambda x, a, p: max(x - a, 0.0) ** p
    S = lambda x: sum(c * br(x, a, n + 2) / ((n + 1) * (n + 2)) for c, a, n in terms)
    A = np.array([[xz, 1.0] for xz in zero_at])
    C1, C2 = np.linalg.solve(A, [-S(xz) for xz in zero_at])
    y_down = lambda x: -(S(x) + C1 * x + C2) / EI          # EI in kN.m^2 -> y in m
    th = lambda x: -(sum(c * br(x, a, n + 1) / (n + 1) for c, a, n in terms) + C1) / EI
    return y_down, th


def run_tests():
    ok = True
    rows = []

    def check(name, got, exp, tol=1e-8):
        nonlocal ok
        rel = abs(got - exp) / (abs(exp) + 1e-30)
        ok &= rel < tol
        rows.append(f" {'PASS' if rel < tol else 'FAIL'}  {name:<58s} rel.err = {rel:.1e}")

    def beam(L, sup, EI_kNm2, P=(), W=(), M=()):
        cfg = dict(L=L, section=dict(type="EI", EI_kNm2=EI_kNm2), supports=sup,
                   point_loads=[dict(P_kN=p, a=a) for p, a in P],
                   distributed_loads=[dict(w1_kNm=w1, w2_kNm=w2, start=s, end=e) for w1, w2, s, e in W],
                   moments=[dict(M_kNm=m, a=a) for m, a in M])
        return solve_beam(build_model(cfg))

    L, EI = 4.0, 20000.0           # kN.m^2
    EIn = EI * 1e3                 # N.m^2
    fix0, fixL = dict(type="fixed", x=0), dict(type="fixed", x=L)
    pin0, pinL = dict(type="pin", x=0), dict(type="pin", x=L)

    # ---- cantilever -------------------------------------------------------------------------
    s = beam(L, [fix0], EI, P=[(10, L)])
    check("Cantilever tip load: y = PL^3/3EI", s.at(L)["y"], 10e3 * L**3 / (3 * EIn))
    s = beam(L, [fix0], EI, W=[(8, 8, 0, L)])
    check("Cantilever full UDL: y = wL^4/8EI", s.at(L)["y"], 8e3 * L**4 / (8 * EIn))
    s = beam(L, [fix0], EI, W=[(8, 0, 0, L)])
    check("Cantilever triangular load: y = w0L^4/30EI", s.at(L)["y"], 8e3 * L**4 / (30 * EIn))
    s = beam(L, [fix0], EI, M=[(5, L)])
    check("Cantilever tip couple: slope = ML/EI", s.at(L)["theta"], 5e3 * L / EIn)
    # mirror image: clamped at the RIGHT end, load at the free LEFT end
    s = beam(L, [fixL], EI, P=[(10, 0.0)])
    check("Cantilever fixed at RIGHT end: y(0) = PL^3/3EI", s.at(0.0)["y"], 10e3 * L**3 / (3 * EIn))

    # ---- simply supported ---------------------------------------------------------------------
    s = beam(L, [pin0, pinL], EI, P=[(12, L / 2)])
    check("Simply supported, central load: y = PL^3/48EI", s.at(L / 2)["y"], 12e3 * L**3 / (48 * EIn))
    check("Simply supported, central load: end slope = PL^2/16EI", s.at(0)["theta"], 12e3 * L**2 / (16 * EIn))
    check("Simply supported, central load: R = P/2", s.reactions[0]["R"], 6.0)
    check("Max deflection location is mid-span", s.extremes("y")[0][0], L / 2)
    s = beam(L, [pin0, pinL], EI, W=[(8, 8, 0, L)])
    check("Simply supported UDL: y = 5wL^4/384EI", s.at(L / 2)["y"], 5 * 8e3 * L**4 / (384 * EIn))
    check("Simply supported UDL: end slope = wL^3/24EI", s.at(0)["theta"], 8e3 * L**3 / (24 * EIn))

    # ---- fixed-fixed and propped cantilever ---------------------------------------------------
    s = beam(L, [fix0, fixL], EI, W=[(8, 8, 0, L)])
    check("Fixed-fixed UDL: y = wL^4/384EI", s.at(L / 2)["y"], 8e3 * L**4 / (384 * EIn))
    check("Fixed-fixed UDL: end moment = wL^2/12 (hogging)", s.at(0)["M"], -8e3 * L**2 / 12)
    s = beam(L, [fix0, fixL], EI, P=[(12, L / 2)])
    check("Fixed-fixed central load: y = PL^3/192EI", s.at(L / 2)["y"], 12e3 * L**3 / (192 * EIn))
    s = beam(L, [fix0, pinL], EI, W=[(8, 8, 0, L)])
    check("Propped cantilever UDL: prop reaction = 3wL/8", s.reactions[1]["R"], 3 * 8 * L / 8)
    check("Propped cantilever UDL: fixed-end moment = wL^2/8", s.at(0)["M"], -8e3 * L**2 / 8)

    # ---- the three textbook examples, vs an independent Macaulay solution ------------------------
    ex = examples()
    s = solve_beam(build_model(ex["5.1a"]))
    check("Example 5.1(a): deflection at load = WL^3/3EI (5.538 mm)", s.at(3.0)["y"], 40e3 * 27 / (3 * 65e6))
    check("Example 5.1(a): free-end deflection", s.at(4.0)["y"], 40e3 * 27 / (3 * 65e6) + 40e3 * 9 / (2 * 65e6))

    s = solve_beam(build_model(ex["5.2"]))
    EI52 = 200e6 * 83e-6                                    # kN.m^2
    y_ref, th_ref = _macaulay([(60, 0, 1), (-20, 1, 1), (-50, 3, 1), (-30, 3, 2)], EI52, [0, 5])
    check("Example 5.2: R_A = 60 kN", s.reactions[0]["R"], 60.0)
    check("Example 5.2: R_B = 130 kN", s.reactions[1]["R"], 130.0)
    check("Example 5.2: deflection under 50 kN vs Macaulay", s.at(3.0)["y"], y_ref(3.0))
    check("Example 5.2: slope under 50 kN vs Macaulay", s.at(3.0)["theta"], th_ref(3.0))
    xmax, ymax = s.extremes("y")[0]
    check("Example 5.2: max deflection vs Macaulay (same x)", ymax, y_ref(xmax))
    check("Example 5.2: slope at max-deflection point is ~0 (vs 50 kN slope)",
          abs(s.at(xmax)["theta"]) + abs(s.at(3.0)["theta"]), abs(s.at(3.0)["theta"]), tol=1e-3)

    s = solve_beam(build_model(ex["5.3"]))
    y_ref, th_ref = _macaulay([(-20, 0, 1), (62, 0.6, 1), (-15, 0.6, 2), (15, 1.8, 2), (-20, 1.8, 1)],
                              650.0, [0.6, 3.0])
    check("Example 5.3: R_A = 62 kN", s.reactions[0]["R"], 62.0)
    check("Example 5.3: R_B = 14 kN", s.reactions[1]["R"], 14.0)
    check("Example 5.3: deflection 1 m from left end vs Macaulay", s.at(1.0)["y"], y_ref(1.0))
    check("Example 5.3: slope 1 m from left end vs Macaulay", s.at(1.0)["theta"], th_ref(1.0))

    # ---- superposition & direct-EI consistency -----------------------------------------------------
    cfg = dict(L=L, E_GPa=200, section=dict(type="rect", b_mm=100, h_mm=200), supports=[fix0],
               point_loads=[dict(P_kN=12, a=1.2), dict(P_kN=7, a=2.2), dict(P_kN=-3, a=L)],
               distributed_loads=[dict(w1_kNm=8, start=0.4, end=2.0)],
               moments=[dict(M_kNm=5, a=2.5)])
    tot = solve_beam(build_model(cfg)).at(L)["y"]
    parts = 0.0
    for key in ("point_loads", "distributed_loads", "moments"):
        for item in cfg[key]:
            c = {k: v for k, v in cfg.items() if k not in ("point_loads", "distributed_loads", "moments")}
            c[key] = [item]
            parts += solve_beam(build_model(c)).at(L)["y"]
    check("Superposition of 5 mixed loads on a cantilever", tot, parts)
    cfg2 = dict(cfg, section=dict(type="EI", EI_Nm2=200e9 * 0.1 * 0.2**3 / 12))
    check("EI given directly equals E x I", solve_beam(build_model(cfg2)).at(L)["y"], tot)

    print("\n SELF-TESTS\n " + "-" * 82)
    print("\n".join(rows))
    print(" " + "-" * 82)
    print(" ALL TESTS PASSED" if ok else " SOME TESTS FAILED")
    return ok


# -----------------------------------------------------------------------------
# 13. MAIN PROGRAM
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="General beam slope & deflection analyser")
    ap.add_argument("--config", help="JSON file describing the beam")
    ap.add_argument("--example", help="built-in textbook example: 5.1a, 5.1b, 5.2, 5.3")
    ap.add_argument("--list-examples", action="store_true")
    ap.add_argument("--test", action="store_true", help="run the self-tests")
    ap.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    ap.add_argument("--outdir", default="beam_output", help="output folder")
    ap.add_argument("--npoints", type=int, default=1000, help="plot resolution")
    ap.add_argument("--csv", action="store_true", help="also export a CSV table")
    args = ap.parse_args()

    if args.test:
        sys.exit(0 if run_tests() else 1)
    if args.list_examples:
        print("Built-in examples:", ", ".join(examples()))
        return

    try:
        if args.config:
            with open(args.config) as f:
                cfg = json.load(f)
        elif args.example:
            cfg = examples()[args.example]
        else:
            cfg = interactive_config()
        sol = solve_beam(build_model(cfg))
    except KeyError as exc:
        sys.exit(f"Input error: missing or unknown key {exc}")
    except (ValueError, OSError, json.JSONDecodeError, np.linalg.LinAlgError) as exc:
        sys.exit(f"Input error: {exc}")

    report(sol)
    s = sol.sample(args.npoints)
    if args.csv:
        export_csv(s, args.outdir)
    make_plots(sol, s, args.outdir, show=not args.no_show)


if __name__ == "__main__":
    main()
