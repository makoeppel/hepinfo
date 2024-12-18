<img src="docs/_static/HEPINFO_logo.svg" alt="HEPINFO_logo" width="500"/>

HEPINFO is a framework featuring models designed for decorrelating physics data using axol1tl with mutual information. The repository is under active development, and the name "miaxol1tl" is currently a working title.

## Introduction

This repository contains the code supporting the poster [**Adaptive Machine Learning on FPGAs: Bridging Simulated and Real-World Data in High-Energy Physics**](https://indico.nikhef.nl/event/4875/contributions/20369/) presented at the **EuCAIFCon24** conference. The code is designed to facilitate machine learning-based decorrelation techniques in high-energy physics, focusing on simulated and real-world datasets.

## Data

To use the repository, follow these steps to set up the required datasets:

### Flavours of Physics Dataset
1. Download the data from the [Flavours of Physics: Finding τ → μμμ competition](https://www.kaggle.com/competitions/flavours-of-physics).

### ttbar / Dijet Data Generation
1. Navigate to the `data` directory:
   ```bash
   cd data
   ```
2. Build the Docker image (adapted from [FastSimulation](https://github.com/schmittc/FastSimulation)):
   - Linux:
     ```bash
     docker build -f Dockerfile -t fastsim:latest .
     ```
   - M1/2/3 Mac (the image builds but running madgraph does not work):
     ```bash
     docker build --platform linux/x86_64 -f Dockerfile -t fastsim:latest .
     ```
4. Start the Docker container:
   ```bash
   docker run -it --rm -v $(pwd):/scratch fastsim:latest
   cd scratch
   ```
5. Generate enough pileup events:
   - **Note:** You will need `N` pileup events for each real event, where `N` can range from 20 to 200, depending on the target LHC run conditions. More details can be found in the [pileup documentation](https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/PileUp).
   - Update the pileup command file:
     ```bash
     cp /opt/delphes/examples/Pythia8/generatePileUp.cmnd generatePileup100k.cmnd
     ```
     Edit `Main:numberOfEvents` to a higher value (e.g., 100k or more).
   - Run the pileup generation:
     ```bash
     DelphesPythia8 /opt/delphes/cards/converter_card.tcl generatePileup100k.cmnd MinBias100k.root
     ```
   - Convert the ROOT file to a pileup format:
     ```bash
     root2pileup MinBias.pileup MinBias100k.root
     ```

6. Generate the processes
So after this stage you have a file with 100k events. Now let’s move on to generating the actual events. Generate the processes -- you want both dijet and ttbar:

**madgraph_dijet.script:**
```bash
import model sm
define p = g u c d s u~ c~ d~ s~
define l+ = e+ mu+ ta+
define l- = e- mu- ta-
define vl = ve vm vt
define vl~ = ve~ vm~ vt~
define l = l+ l-
define ln=vl vl~
generate p p > j j
output dijet_process_42
launch dijet_process_42
done
set nevents 100000
set gseed 42
done
```

**madgraph_ttbar.script:**
```bash
import model sm
define p = g u c d s u~ c~ d~ s~
define l+ = e+ mu+ ta+
define l- = e- mu- ta-
define vl = ve vm vt
define vl~ = ve~ vm~ vt~
define l = l+ l-
define ln=vl vl~
generate p p > t t~
output ttbar_process_42
launch ttbar_process_42
done
set nevents 100000
set gseed 42
done
```
Create these files in the docker container and run (in the following only for the dijet): `/opt/MG5_aMC_v2_7_2/bin/mg5_aMC madgraph_dijet.script`.
This produces a lhe file which is needed as input for the next step.

6. Running Delphes:

To produce the final root files run `cp /opt/delphes/cards/delphes_card_ATLAS_PileUp.tcl delphes_card_ATLAS_PileUp.tcl` (adjust therein the path to the previously generated pileup file and the card can be also CMS).
Now adjust the path to the lhe file in the file `pythia_card` (and also the number of events to match the number of events in the lhe file).

**pythia_card (has to be created in the docker):**
```bash
! 1) Settings used in the main program.

Main:numberOfEvents = 10000         ! number of events to generate
Main:timesAllowErrors = 3          ! how many aborts before run stops

! 2) Settings related to output in init(), next() and stat().

Init:showChangedSettings = on      ! list changed settings
Init:showChangedParticleData = off ! list changed particle data
Next:numberCount = 10000             ! print message every n events
Next:numberShowInfo = 1            ! print event information n times
Next:numberShowProcess = 1         ! print process record n times
Next:numberShowEvent = 1           ! print event record n times

! Adjust tau decays
15:onMode  = off
15:onIfAny = 11 13

! 3) Set the input LHE file

Beams:frameType = 4
Beams:LHEF = /scratch/WZ_process/Events/run_01/unweighted_events.lhe.gz
```

In my case this would be `dijet_process/Events/run_01/unweighted_events.lhe.gz`. Then run `DelphesPythia8 delphes_card_ATLAS_PileUp.tcl pythia_card output.root`
This produces an output file with 20k dijet events (with a mean pileup of 50 events - that’s also something you can change in the file delphes_card_ATLAS_PileUp.tcl).





## Code for τ → 3μ

The primary code for analyzing the τ → 3μ process is detailed in the `notebooks/tau_3mu.ipynb` notebook. For the VHDL simulation of the Bernoulli layer, an example testbench is available in the `tb` folder.

---

**Contributions:** Contributions and collaborations are welcome. Please open an issue or submit a pull request to suggest improvements or report issues.

**License:** This project is licensed under the MIT License. See the `LICENSE` file for details.

---

We hope this repository aids in advancing decorrelation techniques and adaptive machine learning in high-energy physics.

