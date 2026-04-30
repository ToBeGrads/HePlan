# Example connection in MainWindow.__init__()
self.loader_widget = PatientLoaderWidget()
self.loader_widget.data_loaded.connect(self.on_patient_loaded)

def on_patient_loaded(self, patient_data: PatientData):
    self.current_patient = patient_data
    self.mri_viewer.set_volume(patient_data)      # Module 2
    self.ai_module.set_volume(patient_data)       # Module 3
    self.registration.set_reference(patient_data) # Module 4
    self.planning.set_patient_data(patient_data)  # Module 5