import numpy as np

class BranchGenParticles:
    def __init__(self,file):
        self.file = file
        self.length = len(file["Particle.Status"].array())
        self.Status = file["Particle.Status"].array()
        self.PID = file["Particle.PID"].array()
        self.M1 = file["Particle.M1"].array()
        self.M2 = file["Particle.M2"].array()
        self.D1 = file["Particle.D1"].array()
        self.D2  = file["Particle.D2"].array()
        self.PT = file["Particle.PT"].array()
        self.Eta =  file["Particle.Eta"].array()
        self.Phi = file["Particle.Phi"].array()
        self.Mass = file["Particle.Mass"].array()
        self.Charge = file["Particle.Charge"].array()
        self.Labels = ["Status", "PID" , "M1", "M2", "D1", "D2", "PT", "Eta", "Phi", "Mass","Charge"]
        
    def length_At(self, i):
        return len(self.Status[i])
    def Status_At(self, i):
        return self.Status[i]
    def PID_At(self, i):
        return self.PID[i]
    def M1_At(self, i):
        return self.M1[i]
    def M2_At(self, i):
        return self.M2[i]
    def D1_At(self, i):
        return self.D1[i]
    def D2_At(self, i):
        return self.D2[i]
    def PT_At(self, i):
        return self.PT[i]
    def Eta_At(self, i):
        return self.Eta[i]
    def Phi_At(self, i):
        return self.Phi[i]
    def Mass_At(self, i):
        return self.Mass[i]
    def Charge_At(self, i):
        return self.Charge[i]
    
    
    

class BranchJet:
    def __init__(self,file):
        self.file = file
        self.length = len(file["Jet.PT"].array())
        self.PT = file["Jet.PT"].array()
        self.Eta =  file["Jet.Eta"].array()
        self.Phi = file["Jet.Phi"].array()
        self.Mass = file["Jet.Mass"].array()
        self.Charge = file["Jet.Charge"].array()
        self.BTag = file["Jet.BTag"].array()
        
    def PT_At(self, i):
        return self.PT[i]
    def Eta_At(self, i):
        return self.Eta[i]
    def Phi_At(self, i):
        return self.Phi[i]
    def Mass_At(self, i):
        return self.Mass[i]
    def Charge_At(self, i):
        return self.Charge[i]
    def BTag_At(self, i):
        return self.BTag[i]

    
class BranchParticleFlowJet10:
    def __init__(self,file):
        self.file = file
        self.length = len(file["ParticleFlowJet10.PT"].array())
        self.PT = file["ParticleFlowJet10.PT"].array()
        self.Eta =  file["ParticleFlowJet10.Eta"].array()
        self.Phi = file["ParticleFlowJet10.Phi"].array()
        self.Mass = file["ParticleFlowJet10.Mass"].array()
        self.Charge = file["ParticleFlowJet10.Charge"].array()
        self.BTag = file["ParticleFlowJet10.BTag"].array()
        
    def PT_At(self, i):
        return self.PT[i]
    def Eta_At(self, i):
        return self.Eta[i]
    def Phi_At(self, i):
        return self.Phi[i]
    def Mass_At(self, i):
        return self.Mass[i]
    def Charge_At(self, i):
        return self.Charge[i]
    def BTag_At(self, i):
        return self.BTag[i]


    
class BrachElectron:
    def __init__(self,file):
        self.file = file
        self.length = len(file["Electron.PT"].array())
        self.PT = file["Electron.PT"].array()
        self.Eta =  file["Electron.Eta"].array()
        self.Phi = file["Electron.Phi"].array()
        
    def PT_At(self, i):
        return self.PT[i]
    def Eta_At(self, i):
        return self.Eta[i]
    def Phi_At(self, i):
        return self.Phi[i]
    
class BranchMuon:
    def __init__(self,file):
        self.file = file
        self.length = len(file["Muon.PT"].array())
        self.PT = file["Muon.PT"].array()
        self.Eta =  file["Muon.Eta"].array()
        self.Phi = file["Muon.Phi"].array()
        
    def PT_At(self, i):
        return self.PT[i]
    def Eta_At(self, i):
        return self.Eta[i]
    def Phi_At(self, i):
        return self.Phi[i]
    
class BranchMissingET:
    def __init__(self,file):
        self.file = file
        self.length = len(file["MissingET.MET"].array())
        self.MET = file["MissingET.MET"].array()
        self.Eta =  file["MissingET.Eta"].array()
        self.Phi = file["MissingET.Phi"].array()
        
    def MET_At(self, i):
        return self.MET[i]
    def Eta_At(self, i):
        return self.Eta[i]
    def Phi_At(self, i):
        return self.Phi[i]
    
class Event_Weight:
    def __init__(self,file):
        self.file = file
        self.length = len(file["Event.Weight"].array())
        self.Event_Weight = np.array(file["Event.Weight"].array())
        
    def Event_Weight_At(self, i):
        return self.Event_Weight[i]