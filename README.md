# In Silico Multi-Omic Dissection of the Trophoblastic PPAR-γ Regulon and Machine Learning Validation Identify Pioglitazone as a Candidate Therapy for Early-Onset Placental Insufficiency

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Official computational repository for the study:  
**"In silico multi-omic dissection of the trophoblastic PPAR-γ regulon and machine learning validation identify pioglitazone as a candidate therapy for early-onset placental insufficiency"**  
*Department of Woman, Child, and General and Specialized Surgery & Department of Experimental Medicine, University of Campania "Luigi Vanvitelli", Naples, Italy.*

---

## 📌 Study Overview

Early-onset placental insufficiency—manifesting clinically as severe early preeclampsia (PE) and fetal growth restriction (FGR)—is an orphan obstetrical syndrome (ORPHA:439167) lacking disease-modifying pharmacological interventions. This project provides a fully automated, end-to-end computational pipeline integrating multi-omic discovery, orthogonal next-generation sequencing validation, single-cell transcriptomics, structural molecular docking, explainable deep learning, and real-world pharmacovigilance surveillance to evaluate the repurposing potential of the PPAR-γ agonist **Pioglitazone**.

### Key Pipeline Components:
1. **Whole-Transcriptome Microarray Discovery (GSE75010, n = 157):** Identification of a 654-gene signature (|log2FC| > 0.3, Benjamini-Hochberg FDR q < 0.05) comparing early-onset PE (<34 weeks, n = 49) with gestational-age-matched preterm controls (<37 weeks, n = 43).
2. **Systems Biology & Upstream Master Regulators:** Functional enrichment via Gene Ontology (GO: Biological Process 2023) and ChIP-X Enrichment Analysis (ChEA 2022; identifying HIF1A and NFKB1).
3. **Perturbational Drug Repurposing Benchmarking (DSigDB / LINCS L1000):** Head-to-head comparison demonstrating superior reverse connectivity of PPAR-γ agonists over low-dose aspirin, sildenafil, and statins.
4. **Orthogonal Next-Generation RNA-Seq Validation (GSE148241, n = 43):** Cross-platform validation of signature genes (*SOD1* downregulation, *FLT1*, *ENG*, and *PLIN2* upregulation).
5. **Cross-Platform Deep Learning with Explainable AI:** A PyTorch multilayer perceptron trained on discovery data achieving an AUC of 0.87 in 5-fold cross-validation and an AUC of 0.95 in blind zero-shot testing on the independent RNA-seq cohort, evaluated through Permutation Feature Importance.
6. **Single-Cell Trophoblastic Expression Mapping:** Mapping core regulon components across placental subpopulations from human single-cell references, localizing maximal *PPARG* and *SOD1* expression to invasive extravillous trophoblasts (EVTs).
7. **Promoter Architecture & 3D Molecular Docking:** Identification of dual functional PPRE consensus motifs in the human *SOD1* promoter and stereochemical confirmation of Pioglitazone binding within the PPAR-γ catalytic pocket (PDB ID: 2XKW; ΔG = -9.4 kcal/mol).
8. **Placental NF-κB Interactome & Multivariate Logistic Regression:** Confirmation of the independent clinical predictive value of the placental inflammatory score adjusted for maternal BMI, parity, and gestational age.
9. **Multi-Source Pharmacovigilance & Safety Surveillance:** Live API retrieval from FDA FAERS (>25 million reports), EudraVigilance/VigiAccess summaries, and published clinical teratology literature confirming the absence of structural embryopathy signals.

---

## 📁 Repository Structure

```text
placental-pparg-pioglitazone/
├── .gitignore               # Ignored cache, checkpoints, and raw data archives
├── LICENSE                  # MIT open-source license
├── README.md                # Comprehensive project documentation
├── requirements.txt         # Required Python packages and dependencies
├── run_pipeline.py          # Master reproducible Python execution script
├── figures/                 # Pre-generated publication-ready figures (300 DPI)
│   ├── Figure1_MultiOmic_Landscape_and_Targets.png
│   ├── Figure2_Independent_RNAseq_Validation.png
│   ├── Figure3_DeepLearning_and_ExplainableAI.png
│   ├── Figure4_SingleCell_and_Mechanism.png
│   ├── Figure5_SOD1_Promoter_PPRE.png
│   ├── Figure6_Molecular_Docking_and_Pharmacogenomics.png
│   ├── Figure7_Pharmacovigilance_and_Safety.png
│   └── Figure8_NFKB_Regulon_and_Activity.png
└── tables/                  # Generated clinical, sequencing, and statistical tables
    ├── Table1_Cohort_Characteristics.csv
    ├── Table2_Validation_Cohort_Characteristics.csv
    ├── Table3_Global_Pharmacovigilance_Matrix.csv
    └── Table4_Regression_Analysis.csv

🚀 Quick Start & Execution
1. Clone Repository
git clone https://github.com/Sirico87/placental-pparg-pioglitazone.git
cd placental-pparg-pioglitazone
2. Set Up Environment & Install Dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
3. Run the Automated Pipeline
python run_pipeline.py
✉️ Contact & Inquiries
Angelo Sirico, MD, PhD
Department of Woman, Child, and General and Specialized Surgery
University of Campania "Luigi Vanvitelli", Naples, Italy
Email: angelo.sirico@unicampania.it