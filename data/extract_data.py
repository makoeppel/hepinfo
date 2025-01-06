import uproot
import numpy as np
from tqdm import tqdm


normal_data = []
abnormal_data = []

def append_data(PT, Eta, Phi, num, current_event):
    for i in range(num):
        if len(PT) - 1 >= i:
            current_event.append(PT[i])
            current_event.append(Eta[i])
            current_event.append(Phi[i])
        else:
            current_event.append(0)
            current_event.append(0)
            current_event.append(0)

def read_data(name, normal=True):
    # we have 57 variables: MET, 4 electrons, 4 muons and 10 jets
    # these are 19 objects, times 3 parameters -> 57 vars
    # in addition we read the pile-up as Vertex_size -> 58 vars
    # vertex_size,misMET,misEta,misPhi,e0PT,e0Eta,e0Phi,...,m0PT,m0Eta,m0Phi,...,j0PT,j0Eta,j0Phi,...
    print("Read", name)
    with uproot.open(name) as file:
        vertex_size = np.array(file["Delphes"]["Vertex_size"].array())
        misMET = file["Delphes"]["MissingET/MissingET.MET"].array()
        misEta = file["Delphes"]["MissingET/MissingET.Eta"].array()
        misPhi = file["Delphes"]["MissingET/MissingET.Phi"].array()

        ePT = file["Delphes"]["Electron/Electron.PT"].array()
        eEta = file["Delphes"]["Electron/Electron.Eta"].array()
        ePhi = file["Delphes"]["Electron/Electron.Phi"].array()

        mPT = file["Delphes"]["Muon/Muon.PT"].array()
        mEta = file["Delphes"]["Muon/Muon.Eta"].array()
        mPhi = file["Delphes"]["Muon/Muon.Phi"].array()

        jPT = file["Delphes"]["Jet/Jet.PT"].array()
        jEta = file["Delphes"]["Jet/Jet.Eta"].array()
        jPhi = file["Delphes"]["Jet/Jet.Phi"].array()

        for values in tqdm(zip(vertex_size,misMET,misEta,misPhi,ePT,eEta,ePhi,mPT,mEta,mPhi,jPT,jEta,jPhi), total=len(vertex_size)):
            vertex_sizei,misMETi,misEtai,misPhii,ePTi,eEtai,ePhii,mPTi,mEtai,mPhii,jPTi,jEtai,jPhii = values
            current_event = []
            current_event.append(vertex_sizei)
            current_event.append(misMETi[0])
            current_event.append(misEtai[0])
            current_event.append(misPhii[0])
            append_data(ePTi,eEtai,ePhii,4,current_event)
            append_data(mPTi,mEtai,mPhii,4,current_event)
            append_data(jPTi,jEtai,jPhii,10,current_event)
            if normal: normal_data.append(current_event)
            if not normal: abnormal_data.append(current_event)

read_data("ttbar.root")
read_data("dijet.root")
read_data("higgs.root", False)

np.save("normal_data", np.array(normal_data))
np.save("abnormal_data", np.array(abnormal_data))
