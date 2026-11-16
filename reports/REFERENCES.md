# Research References and Method Validation

Every entry below was checked against Crossref or OpenAlex; titles, authors, year and DOI are
the registry values, not paraphrases. Where a formulation is attributed to a paper, the
attribution is limited to what the paper's abstract/method states. **Our implementation is an
adaptation, not a reproduction** — the distinctions are stated explicitly.

---

## A. Datasets

**A1. FPV-DB (2026).** *FPV-DB: open dataset of FPV drone component specifications.* Zenodo.
DOI: [10.5281/zenodo.23613812](https://doi.org/10.5281/zenodo.23613812). License CC BY 4.0.
Canonical copy: https://github.com/fpvdb/fpv-db-data.
**Used for:** all masses, thrust-bench curves, cells, capacities, C-ratings.
**Role:** primary empirical basis of the entire drone component.

**A2. NTNU UAV Database (2020).** Hann, R.; Wallisch, J. *UAV Database.* DataverseNO.
DOI: [10.18710/L41IGQ](https://doi.org/10.18710/L41IGQ). License CC0 1.0.
93 fixed-wing UAVs with MTOW, payload, wingspan, endurance, range.
**Evaluated and not used:** fixed-wing, no component masses, different vehicle class. Recorded here
so the choice is auditable.

---

## B. Weight build-up and sizing

**B1. Ong, W.; Srigrarom, S.; Hesse, H. (2019).** *Design Methodology for Heavy-Lift Unmanned
Aerial Vehicles with Coaxial Rotors.* AIAA SciTech 2019. DOI:
[10.2514/6.2019-2095](https://doi.org/10.2514/6.2019-2095).
*Formulation used there:* mass sizing as
$W_o = W_{str} + W_m + W_{sys} + W_{pl} + W_b + W_{fix}$, with empty and battery mass fractions
taken from a market survey; thrust must balance total weight.
*Relation to ours:* our objective is the same additive structure with $W_{str}+W_{sys}$ lumped into
$W_{\text{airframe}}$, $W_m$ expanded to 4 motors, and $W_b$ taken from a real pack catalogue rather
than a fraction. **Difference:** they use class-level mass fractions; we use per-component
catalogue values and an explicit constraint set.

**B2. Rand, O.; Khromov, V. (2021).** *Mission Oriented Multi-Prop UAV Analysis Using Statistical
Design Trends.* Aerospace 8(11):321. DOI: [10.3390/aerospace8110321](https://doi.org/10.3390/aerospace8110321).
*Method:* a database of multi-prop UAVs is mined for design trends in geometry, component weight,
power and performance; a mission requirement is then optimised for minimum weight or minimum
dimensions.
*Relation to ours:* same philosophy — empirical trends + constrained minimisation. **Difference:**
their design variables are geometric/mission parameters; ours are discrete catalogue parts, and
our exact optimum is certified by enumeration rather than a trend-based sizing model.

**B3. Diogo, C. M. A.; Fernandes, E. C. (2024).** *A Mass, Fuel, and Energy Perspective on
Fixed-Wing Unmanned Aerial Vehicle Scaling.* Drones 8(8):396. DOI:
[10.3390/drones8080396](https://doi.org/10.3390/drones8080396).
*Method:* mass/energy scaling laws across a UAV dataset, with convex-hull envelopes per class.
*Relation to ours:* motivates the allometric (log–log) analysis in §3 of the methodology.
**Difference:** fixed-wing and fuel-based; we apply the scaling diagnostic only, not their
performance model.

**B4. Rubrichi, L.; Gnoni, M. G.; Tornese, F. (2025).** *Data Driven Decision Support Tool for
Assessing the Environmental Impact of Unmanned Aerial Vehicles in Last Mile Logistics.*
IFAC-PapersOnLine 59(10):1666–1670. DOI:
[10.1016/j.ifacol.2025.09.280](https://doi.org/10.1016/j.ifacol.2025.09.280).
*Method:* 90 commercial UAVs; chained regressions predict MTOW, frame, motor, propeller and
battery mass from payload; KNN imputation; Monte-Carlo robustness.
*Relation to ours:* closest published analogue to our **prediction** stage, and direct evidence
that component masses are predictable from design inputs. **Difference:** they predict one
component from another in a chain (payload → MTOW → …); we benchmark ten regressors on total
weight with leak-free CV, then optimise on measured catalogue values rather than on predicted
masses, so surrogate error does not propagate into the optimum.

---

## C. Propulsion, battery and endurance trade-offs

**C1. de Angelis, E. L.; Giulietti, F.; Rossetti, G.; Bellani, G. (2021).** *Performance analysis
and optimal sizing of electric multirotors.* Aerospace Science and Technology 118:107057. DOI:
[10.1016/j.ast.2021.107057](https://doi.org/10.1016/j.ast.2021.107057).
*Method:* analytical hover-power model with a figure of merit derived from blade parameters;
constant-power battery discharge; closed-form optimal battery capacity and endurance.
*Relation to ours:* provides the theoretical justification for a **non-monotonic** mass–endurance
trade-off: adding battery mass raises energy but also raises required power. Our optimizer
rediscovers this numerically (endurance sensitivity is convex and steepens above ~12 min).
**Difference:** they solve for battery capacity analytically for a fixed platform; we solve a
discrete mixed-component problem with measured thrust curves and a TWR constraint.

**C2. Hwang, M.-H.; Cha, H.-R.; Jung, S. Y. (2018).** *Practical Endurance Estimation for
Minimizing Energy Consumption of Multirotor Unmanned Aerial Vehicles.* Energies 11(9):2221. DOI:
[10.3390/en11092221](https://doi.org/10.3390/en11092221).
*Method:* thrust from momentum theory; required power from propeller/motor efficiency; battery
discharge model; validated experimentally (average error 2.3 %).
*Relation to ours:* our endurance constraint (C2) follows the same logic — thrust from weight,
power from the propulsion map, endurance from usable energy. **Difference:** we replace their
efficiency assumptions with **measured** current at the interpolated hover thrust from 6,786 bench
rows, so no figure of merit is assumed.

**C3. Bershadsky, D.; Haviland, S.; Johnson, E. N. (2016).** *Electric Multirotor UAV Propulsion
System Sizing for Performance Prediction and Design Optimization.* AIAA 2016-0581. DOI:
[10.2514/6.2016-0581](https://doi.org/10.2514/6.2016-0581).
*Method:* propulsion sizing from motor/prop maps for performance prediction and design.
*Relation to ours:* justifies using measured motor+prop maps as the propulsion model.
**Difference:** their maps are characterised in-house; we use the public bench tables and add a
commercial-component selection layer.

**C4. Zhu, H.; Wei, Y.; Nie, H.; Wei, X. (2024).** *Optimization Design and Propulsion System
Sizing Methodology of Double-Layer Staggered Octocopter.* International Journal of Aerospace
Engineering 2024:5562139. DOI: [10.1155/2024/5562139](https://doi.org/10.1155/2024/5562139).
*Method:* surrogate thrust model + multi-objective evolutionary algorithm for structure, followed
by propulsion sizing from **non-customised (COTS) product data** to minimise weight and maximise
efficiency.
*Relation to ours:* the closest methodological match to our two-layer structure (search over a
design space, then resolve to catalogue parts). **Difference:** they optimise rotor layout with a
surrogate first; we search the COTS catalogue directly and certify optimality exhaustively.

**C5. Dai, X.; Quan, Q.; Cai, K.-Y. (2021).** *Design Automation and Optimization Methodology for
Electric Multicopter Unmanned Aerial Robots.* IEEE Transactions on Automation Science and
Engineering. DOI: [10.1109/TASE.2021.3081122](https://doi.org/10.1109/TASE.2021.3081122).
*Method:* automated selection/sequencing of propulsion components for multicopters.
*Relation to ours:* supports component-selection (rather than continuous sizing) as a legitimate
design formulation. **Difference:** their procedure is a deterministic selection sequence; ours is
a constrained combinatorial optimisation with an explicit objective and feasibility proof.

**C6. May, M. J. S.; Braga, M. A. R.; Ferguson, P. A. (2022).** *SOFAR and Auto-link: software
tools that bridge the gap between multirotor drone design optimization and CAD.* Drone Systems and
Applications. DOI: [10.1139/dsa-2021-0051](https://doi.org/10.1139/dsa-2021-0051).
*Method:* given payload and flight time, select COTS frames, batteries, motors, ESCs and
propellers, minimising parts cost, then generate CAD.
*Relation to ours:* the same COTS-selection concept, and the source of our scenario framing
(payload + endurance as requirements). **Difference:** their objective is **cost**; ours is
**mass**, with a measured-thrust TWR constraint and C-rate limit that their formulation does not
report.

---

## D. Multidisciplinary design optimization (MDO)

**D1. Martins, J. R. R. A.; Lambe, A. B. (2013).** *Multidisciplinary Design Optimization: A
Survey of Architectures.* AIAA Journal 51(9):2049–2075. DOI:
[10.2514/1.J051895](https://doi.org/10.2514/1.J051895).
*Role:* taxonomy of MDO architectures (MDF, IDF, CO, …).
*Relation:* our problem is a monolithic MDF-style formulation — one objective, all constraints
evaluated simultaneously. **Not used:** no decomposition architecture, because the catalogue
evaluation is cheap and the exact optimum is tractable.

**D2. Hosseini, M.; Nosratollahi, M.; Sadati, H. (2017).** *Multidisciplinary Design Optimization
of Unmanned Aerial Vehicle under Uncertainty.* Journal of Aerospace Technology and Management
9(2):160–169. DOI: [10.5028/jatm.v9i2.725](https://doi.org/10.5028/jatm.v9i2.725).
*Method:* 37 design parameters, 33 constraints, overall mass as the objective; GA + SQP with
Monte-Carlo uncertainty.
*Relation to ours:* same objective (minimum overall mass under many constraints) and the same
GA-based search idea. **Difference:** fixed-wing with aerodynamic/stability disciplines; we have
no wing, and we use an exact enumerator rather than GA+SQP.

**D3. Liscouët, J.; Pollet, F.; Jézégou, J.; Budinger, M. (2022).** *A methodology to integrate
reliability into the conceptual design of safety-critical multirotor unmanned aerial vehicles.*
Aerospace Science and Technology 127:107681. DOI:
[10.1016/j.ast.2022.107681](https://doi.org/10.1016/j.ast.2022.107681).
*Method:* conceptual multirotor sizing with reliability constraints.
*Relation to ours:* shows that multirotor sizing is a constrained mass problem with safety
margins. **Difference:** reliability/redundancy is out of scope here; our safety margin is the
TWR ≥ 2 requirement and the C-rate limit.

**D4. Pollet, F.; Budinger, M.; Delbecq, S.; Moschetta, J.-M. (2023).** *Quantifying and
mitigating uncertainties in design optimization including off-the-shelf components: Application to
an electric multirotor UAV.* Aerospace Science and Technology 141:108179. DOI:
[10.1016/j.ast.2023.108179](https://doi.org/10.1016/j.ast.2023.108179).
*Method:* design optimisation with COTS components under uncertainty, uncertainty propagation to
the optimum.
*Relation to ours:* motivates our honest treatment of the frame-mass input and the ±30 %
sensitivity. **Difference:** they propagate full component uncertainty distributions; we provide a
deterministic optimum plus one-at-a-time sensitivity because the catalogue lacks uncertainty
intervals.

---

## E. Optimization algorithms

**E1. Storn, R.; Price, K. (1997).** *Differential Evolution – A Simple and Efficient Heuristic for
Global Optimization over Continuous Spaces.* Journal of Global Optimization 11:341–359. DOI:
[10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328).
**Used for:** the DE baseline (`scipy.optimize.differential_evolution`, integer encoding).

**E2. Kennedy, J.; Eberhart, R. (1995).** *Particle Swarm Optimization.* Proc. ICNN'95. DOI:
[10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).
**Used for:** the PSO baseline ($w=0.72$, $c_1=c_2=1.5$).

**E3. Deb, K.; Pratap, A.; Agarwal, S.; Meyarivan, T. (2002).** *A fast and elitist multiobjective
genetic algorithm: NSGA-II.* IEEE Transactions on Evolutionary Computation 6(2):182–197. DOI:
[10.1109/4235.996017](https://doi.org/10.1109/4235.996017).
**Used for:** the GA operators (tournament selection, elitism). **Not used:** the multi-objective
non-dominated sorting, because our problem is single-objective with constraints.

**E4. Shahriari, B.; Swersky, K.; Wang, Z.; Adams, R. P.; de Freitas, N. (2016).** *Taking the
Human Out of the Loop: A Review of Bayesian Optimization.* Proceedings of the IEEE 104(1):148–175.
DOI: [10.1109/JPROC.2015.2494218](https://doi.org/10.1109/JPROC.2015.2494218).
**Used for:** the Bayesian baseline (TPE surrogate, as implemented by Optuna).

**E5. Babaei, A. R.; Setayandeh, M. R.; Farrokhfal, H. (2018).** *Aircraft robust multidisciplinary
design optimization methodology based on fuzzy preference function.* Chinese Journal of
Aeronautics 31(12):2248–2259. DOI: [10.1016/j.cja.2018.04.018](https://doi.org/10.1016/j.cja.2018.04.018).
*Method:* NSGA-based minimisation of take-off weight and drag for the Predator MQ-1, with
robustness under cruise uncertainty.
*Relation to ours:* published precedent that take-off-weight minimisation with an evolutionary
algorithm is a valid UAV design objective. **Difference:** fixed-wing, multi-objective, surrogate-
free aerodynamic analysis; ours is a discrete multirotor catalogue problem.

**E6. Saib, S.; Achour, Y.; Ghennam, T.; Marouani, K.; Rizoug, N. (2023).** *Design optimization
and power management of a fuel cell-battery fixed-wing electric UAV.* Proc. IMechE Part G: Journal
of Aerospace Engineering 238(1). DOI: [10.1177/09544100231212651](https://doi.org/10.1177/09544100231212651).
*Method:* PSO minimises the combined mass of fuel cell and battery subject to power, load-sharing
and state-of-charge constraints; reports an **8 %** weight reduction.
*Relation to ours:* a directly comparable PSO weight-minimisation result and a sanity anchor for
the magnitude of realistic gains (single-digit to low-double-digit percent). Our larger reductions
are explained by the deliberately over-provisioned median baseline and are reported alongside the
much smaller difference against real aircraft.

---

## F. What is original in this component

To keep the boundary clear:

1. The **verified component-sum identity** $W_{AUW}=W_{dry}+W_{battery}$ with a catalogue-matched
   pack mass and a held-out validation (MAE 22.2 g, MAPE 7.45 %, R² 0.969).
2. The **exact enumeration** of the FPV-DB propulsion × battery catalogue with measured-thrust
   TWR, interpolated hover current, C-rate and prop-clearance constraints, used as the ground
   truth against which six metaheuristics are scored.
3. The **constraint-aware decoder** shared by all heuristics (scenario-valid groups, cell-matched
   battery pools).
4. The explicit separation of **prediction** (ten regressors) from **optimization** (catalogue
   search), and the refusal to optimise on surrogate outputs.

Nothing in sections A–E is claimed as our own. Where a number is borrowed (e.g. TWR ≥ 2,
η = 0.80), it is labelled as a design requirement or assumption, not as a measured value.
