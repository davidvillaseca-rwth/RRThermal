# RRThermal — Contactless Respiratory-Rate Detection from Thermal Video

Student project for **Image Processing and Handling** (Uniklinik RWTH Aachen).
Full brief: [contactless_vital_sign_detection_project_brief.pdf](contactless_vital_sign_detection_project_brief.pdf)
Code, data layout and how to run things: [DOCUMENTATION.md](DOCUMENTATION.md)

| | |
|---|---|
| **Primary focus** | Contactless estimation of respiratory rate (RR) from radiometric thermal video |
| **Setting** | Simulated emergency-department (ED) handover or admission |
| **Project type** | Semester-long, offline proof of concept |
| **Core result** | Reproducible processing pipeline and quantitative comparison with a synchronized reference |
| **Current status** | Face detection works on one public recording (SIM1, participant T001). A respiratory-rate prototype is tested on synthetic breathing only; no real respiratory-rate results yet. Details in [DOCUMENTATION.md](DOCUMENTATION.md) |

## Background

Handover between emergency medical services and the ED has to transfer clinically relevant information quickly while other work goes on in parallel. Vital signs are central to that information, but measuring them takes time, attention and physical contact. The long-term idea is to acquire selected patient parameters automatically and contactlessly, so they are ready as structured information during the handover — supporting, not replacing, clinical assessment.

Respiratory rate is the starting point. Thermal cameras can see the temperature changes around the nose and mouth caused by inhaled and exhaled air, as well as upper-body breathing movement. Prior clinical and algorithmic work (see [References](#references)) shows this is feasible, and also shows that automatic target selection, robust ROI tracking, motion handling and signal-quality assessment are essential.

## Objective

Build and evaluate an **offline image-processing workflow** that:

1. Detects people in thermal video and identifies the **principal patient**.
2. Tracks a suitable facial or upper-body **region of interest (ROI)**.
3. Extracts a **respiratory signal** from the ROI time series.
4. Estimates **respiratory rate** per measurement window.
5. Compares estimates against a **synchronized reference** measurement.

The data is recorded in the institute's simulation laboratory with consenting volunteers in a scripted handover/admission scenario designed by the team.

### Out of scope

- Not a certified medical device or clinically deployable monitor.
- No real patients — simulated scenarios and volunteers only.
- Heart rate and skin temperature are optional extensions only. A thermal camera measures skin-surface radiation, **not** core body temperature.

## Research questions

**Main question:** Can a thermal-video-based workflow automatically identify and track the principal patient and estimate respiratory rate with sufficient technical accuracy and availability in a controlled, simulated ED handover or admission?

**Secondary questions**

1. **Patient identification and tracking** — How reliably can the principal patient be selected and tracked when other people are in the scene?
2. **Respiratory-signal extraction** — Which anatomical region and signal-processing approach give a stable respiratory signal under the chosen recording conditions?
3. **Accuracy and robustness** — How closely does the estimated RR agree with the reference, and how is it affected by movement, speaking, distance and head orientation?
4. **Extension potential** — Which further measurements (e.g. heart rate, facial skin temperature) look technically feasible for later work?

## Outcomes

**Primary outcome:** agreement between thermal-derived RR and the synchronized reference, reported with e.g. mean absolute error, bias and limits of agreement (Bland–Altman), plus the **proportion of windows with a valid estimate**.

**Secondary outcomes**

- Identification and temporal tracking of the intended patient, with transparent criteria for who the "principal person" is.
- Performance under prespecified scenario variations, with a structured description of failure cases and missing estimates.
- A reproducible dataset description, executable pipeline and documented recording protocol.

### Minimum successful result

An offline pipeline that processes radiometric thermal recordings, automatically identifies and tracks at least one principal participant over a defined sequence, extracts a respiratory signal, estimates RR for predefined windows, and compares the estimates quantitatively with a synchronized reference. Code and recording protocol must be documented and reproducible.

## Work plan

| # | Work package | Key activities |
|---|---|---|
| 01 | Literature review & scope | Review contactless vital signs, thermal respiration monitoring, person/face tracking, signal extraction, agreement analysis. Fix scope, success criteria, minimum result, optional extensions. |
| 02 | Algorithmic concept & project plan | Justify the approach for detection, principal-patient selection, ROI tracking, signal extraction and RR estimation. Define work packages, responsibilities, milestones, risks and fallbacks. |
| 03 | Simulation & data-acquisition design | Camera position, field of view, distance, reference-signal acquisition. Standardized script (roles, sequence, posture, movement, speaking phases, test conditions). Synchronization, storage and documentation of thermal, timing and reference data. |
| 04 | Data acquisition | Pilot and refine the setup in the simulation lab. Record volunteers, keeping development/calibration recordings separate from evaluation recordings. |
| 05 | Implementation & evaluation | Implement the workflow with a justified tech stack, reproducible from raw recordings. Evaluate patient selection, tracking availability, RR accuracy and failure cases. |
| 06 | Synthesis & reporting | Summarize feasibility, separating demonstrated findings from future clinical claims. Discuss limitations and transferability. Deliver the final presentation and an archive with code, environment info, data/recording docs, evaluation outputs and an outlook. |

**Optional extension** (only after the minimum result): one clearly defined topic such as motion robustness, multi-person scenes, heart-rate estimation, facial skin-temperature analysis, RGB–thermal fusion, or near-real-time processing.

## Deliverables

- Executable, documented processing pipeline
- Recording protocol and dataset description
- Evaluation outputs (error/agreement metrics, valid-window rate, failure-case analysis)
- Environment information for reproducibility
- Final scientific presentation

## References

1. Troyer L, Brady W. Barriers to effective EMS to emergency department information transfer at patient handover: A systematic review. *Am J Emerg Med.* 2020;38:1494–1503. [doi:10.1016/j.ajem.2020.04.036](https://doi.org/10.1016/j.ajem.2020.04.036)
2. Pires Alves R, van Meulen F, Overeem S, Zinger S, Stuijk S. Thermal Cameras for Continuous and Contactless Respiration Monitoring. *Sensors.* 2024;24(24):8118. [doi:10.3390/s24248118](https://doi.org/10.3390/s24248118)
3. Aldred A, et al. Application of thermography to estimate respiratory rate in the emergency room: The journal Temperature toolbox. *Temperature.* 2023;10:159–165. [doi:10.1080/23328940.2022.2099215](https://doi.org/10.1080/23328940.2022.2099215)
4. Chan P, et al. Estimation of respiratory rate using infrared video in an inpatient population: an observational study. *J Clin Monit Comput.* 2020;34:1275–1284. [doi:10.1007/s10877-019-00437-2](https://doi.org/10.1007/s10877-019-00437-2)
5. Cho Y, Julier SJ, Marquardt N, Bianchi-Berthouze N. Robust tracking of respiratory rate in high-dynamic range scenes using mobile thermal imaging. *Biomed Opt Express.* 2017;8(10):4480–4503. [doi:10.1364/BOE.8.004480](https://doi.org/10.1364/BOE.8.004480)
6. Pereira CB, Czaplik M, Blazek V, Leonhardt S, Teichmann D. Monitoring of Cardiorespiratory Signals Using Thermal Imaging: A Pilot Study on Healthy Human Subjects. *Sensors.* 2018;18(5):1541. [doi:10.3390/s18051541](https://doi.org/10.3390/s18051541)
7. Yang F, He S, Sadanand S, Yusuf A, Bolic M. Contactless Measurement of Vital Signs Using Thermal and RGB Cameras: A Study of COVID-19-Related Health Monitoring. *Sensors.* 2022;22(2):627. [doi:10.3390/s22020627](https://doi.org/10.3390/s22020627)
8. Alhaskir M, et al. Feasibility of LLM-Assisted Communication Feedback in Simulation-Based Medical Education: A Pilot Study. *Stud Health Technol Inform.* 2026;336:2140–2144. [doi:10.3233/SHTI260639](https://doi.org/10.3233/SHTI260639)
