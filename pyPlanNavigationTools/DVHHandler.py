import numpy as np 
import matplotlib.pyplot as plt 
import os
from pyPlanNavigationTools.DVHHandler import DosimetricResults
import pydicom
import glob
import copy
from scipy.interpolate import interp1d


class DwellsData:
	def __init__(self, patient_path, i_plan_selected=0, cvt_catheters=None, gmco_folder_name="gMCO_OUTPUT_FILES"):
		self.nPlans = 0
		self.cvt_catheters = cvt_catheters
		self.patient_path = patient_path
		self.i_plan_selected = i_plan_selected
		if isinstance(gmco_folder_name, str):
			if self.cvt_catheters is None:
				self.dwell_times_file = os.path.join(self.patient_path, gmco_folder_name, "dwell_times.bin")
				self.dwell_positions_file = os.path.join(self.patient_path, gmco_folder_name, "dwells.txt")
			else:
				self.dwell_times_file = os.path.join(self.patient_path, gmco_folder_name, 
					"CVT_{}_catheters".format(cvt_catheters), "dwell_times.bin")
				self.dwell_positions_file = os.path.join(self.patient_path, gmco_folder_name, 
					"CVT_{}_catheters".format(cvt_catheters), "dwells.txt")
			self.set_dwell_times()
			self.set_dwell_positions()
		elif isinstance(gmco_folder_name, list):
			for folder in gmco_folder_name:
				self.dwell_times_file = os.path.join(folder, "dwell_times.bin")
				self.dwell_positions_file = os.path.join(folder, "dwells.txt")
				self.set_dwell_times()
			self.set_dwell_positions()

	def set_dwell_times(self):
		if self.nPlans == 0:
			f = open(self.dwell_times_file, 'r')
			self.nPlans = int(np.fromfile(f, count=1, dtype=np.int32))
			self.n_act = int(np.fromfile(f, count=1, dtype=np.int32))
			self.dwell_times = np.fromfile(f, count=-1, dtype=np.float32)
			f.close()
			self.dwell_times = self.dwell_times.reshape(self.nPlans, self.n_act)
		else:
			f = open(self.dwell_times_file, 'r')
			nPlans = int(np.fromfile(f, count=1, dtype=np.int32))
			n_act = int(np.fromfile(f, count=1, dtype=np.int32))
			if n_act != self.n_act:
				raise RuntimeError("Wrong number of active dwells.")
			dwell_times = np.fromfile(f, count=-1, dtype=np.float32)
			f.close()
			dwell_times = dwell_times.reshape(nPlans, n_act)
			self.dwell_times = np.vstack((self.dwell_times, dwell_times))
			self.nPlans += nPlans

	def set_dwell_positions(self):
		dwell_positions = np.loadtxt(self.dwell_positions_file)
		self.dwells_data = np.zeros((self.n_act, dwell_positions.shape[1]))
		# Keep active positions only
		self.dwells_data[:, :-1] = dwell_positions[np.where(dwell_positions[:, -1] == 1)[0], :-1]
		# Get the dwell times of the selected plan
		self.dwells_data[:, -1] = self.dwell_times[self.i_plan_selected, :]

	def save_dwell_times_to_txt(self, filename):
		fdwells = open(filename, 'w')
		fdwells.write("______________________________________________________________________")
		fdwells.write("\n\n")
		fdwells.write("                              IPSA                                    ")
		fdwells.write("\n______________________________________________________________________\n\n")
		for dwell in self.dwells_data:
			rel_pos = dwell[-2]
			dt = dwell[-1]
			CFdata = float(dwell[0])
			fdwells.write('%6.6f'%CFdata+'\t')
			CFdata = float(dwell[1])
			fdwells.write('%6.6f'%CFdata+'\t')
			CFdata = float(dwell[2])
			fdwells.write('%6.6f'%CFdata+'\t')
			fdwells.write('%d\t'%dwell[-3])
			fdwells.write('%6.6f\t'%rel_pos)
			fdwells.write('%6.2f\n'%dt)
		fdwells.close()


	def saveDwellTimes(self, RP_file, prescription, export_path=None, normalize_dwell_weights=False,
		normalization_point=None):
		print("Exporting plan {} to DICOM...".format(self.i_plan_selected))
		self.RP_file = pydicom.dcmread(RP_file)	
		for fractionGroup in self.RP_file.FractionGroupSequence:
			for setup in fractionGroup.ReferencedBrachyApplicationSetupSequence:
				setup["300a", "00a4"].value = prescription

		if normalize_dwell_weights:
			normalization_point = [str(10*normalization_point[0]), 
								   str(10*normalization_point[1]),
								   str(10*normalization_point[2])]
			self.RP_file["3007", "1000"] = pydicom.DataElement(["3007", "1000"], "DS", prescription)
			try:
				self.RP_file.DoseReferenceSequence.clear()
			except:
				pass
			self.RP_file.DoseReferenceSequence = []
			dataset = pydicom.Dataset()
			dataset.add_new(["300a", "0012"], "IS", 1) # DoseReferenceNumber
			dataset.add_new(["300a", "0014"], "CS", "COORDINATES") # DoseReferenceStructureType
			dataset.add_new(["300a", "0018"], "DS", normalization_point) # DoseReferencePointCoordinates
			dataset.DoseReferenceType = "UNSPECIFIED"
			dataset.DoseReferenceDescription = "gMCO"
			dataset.TargetPrescriptionDose = prescription
			self.RP_file.DoseReferenceSequence.append(dataset)
			self.RP_file["3005", "1008"] = pydicom.DataElement(["3005", "1008"], "CS", "BRACHY_OPT")
			self.RP_file["3007", "100e"] = pydicom.DataElement(["3007", "100e"], "DS", 100.0)
			self.RP_file["3007", "100f"] = pydicom.DataElement(["3007", "100f"], "DS", 1)
			self.RP_file["3007", "1011"] = pydicom.DataElement(["3007", "1011"], "DS", 0)
			self.RP_file["3007", "1013"] = pydicom.DataElement(["3007", "1013"], "DS", 1.0)
			self.RP_file["3007", "1014"] = pydicom.DataElement(["3007", "1014"], "LO", "NORMALIZED_ON_POINTS")

			if isinstance(self.RP_file["300f", "1000"].value, bytes):
				tmp = pydicom.values.convert_SQ(self.RP_file["300f", "1000"].value, True, True)
			else:
				tmp = self.RP_file["300f", "1000"]
			last_roi_number = 0
			is_normalization_point_found = False
			index_to_remove = None
			for item in tmp:
				for i, structureSetROI in enumerate(item.StructureSetROISequence):
					referencedFrameOfReferenceUID = copy.deepcopy(structureSetROI["3006", "0024"])
					last_roi_number = int(structureSetROI["3006", "0022"].value)
					if ["3006", "0038"] in structureSetROI:
						point_set = structureSetROI["3006", "0038"].value
						if isinstance(point_set, bytes):
							point_set = point_set.decode('utf-8')
						if point_set == "POINT_SET_Patient":
							is_normalization_point_found = True
							# structureSetROI.clear()
							# del structureSetROI
							# last_roi_number -= 1
							# index_to_remove = i

			# if index_to_remove is not None:
			# 	for item in tmp:
			# 		item.StructureSetROISequence.pop(index_to_remove)

			dataset_in = pydicom.Dataset()
			dataset_in.add_new(["3005", "1002"], "CS", "FALSE")
			dataset_in.add_new(["3005", "1004"], "DS", 1)
			dataset_in.add_new(["3005", "1006"], "DS", 128)
			dataset_in.add_new(["3005", "1008"], "DS", 0)
			dataset_in.add_new(["3005", "100a"], "CS", "FALSE")
			dataset_in.add_new(["3005", "100c"], "CS", "FALSE")
			dataset_in.add_new(["3005", "100e"], "CS", "TRUE")
			dataset_in.add_new(["3005", "1012"], "DS", 0)
			dataset_in.add_new(["3005", "1015"], "CS", "TRUE")

			dataset_tmp = pydicom.Dataset()
			normalization_point_roi_number = last_roi_number + 1
			dataset_tmp.add_new(["3005", "0010"], "LO", "MDS NORDION OTP ANATOMY MODELLING")
			dataset_tmp.add_new(["3006", "0022"], "IS", normalization_point_roi_number)
			dataset_tmp.add_new(["3006", "0026"], "LO", "gMCO")
			dataset_tmp.add_new(["3006", "0036"], "CS", "MANUAL")
			dataset_tmp.add_new(["3006", "0038"], "LO", "POINT_SET_Patient")
			dataset_tmp.add_new(["3007", "1025"], "CS", "TRUE")
			dataset_tmp["3006", "0024"] = referencedFrameOfReferenceUID
			dataset_tmp.add_new(["3005", "1000"], "SQ", pydicom.sequence.Sequence([dataset_in]))

			for item in tmp:
				item.StructureSetROISequence.append(dataset_tmp)

			# for item in tmp:
			# 	if is_normalization_point_found:
			# 		item.RTROIObservationsSequence.pop(normalization_point_roi_number)

			for item in tmp:
				dataset = pydicom.Dataset()
				dataset.add_new(["3006", "0082"], "IS", normalization_point_roi_number)
				dataset.add_new(["3006", "0084"], "IS", normalization_point_roi_number)
				dataset.add_new(["3006", "00a4"], "CS", "PATIENT")
				dataset.add_new(["3006", "00a6"], "PN", "")
				item.RTROIObservationsSequence.append(dataset)

			# index_to_remove = None
			# for item in tmp:
			# 	for i, contour in enumerate(item.ROIContourSequence):
			# 		for contour_item in contour.ContourSequence:
			# 			if contour_item.ContourGeometricType == "POINT":
			# 				contour.clear()
			# 				del contour
			# 				index_to_remove = i
			# 				break
			# if index_to_remove is not None:
			# 	for item in tmp:	
			# 		item.ROIContourSequence.pop(index_to_remove)

			dataset_norm_point = pydicom.Dataset()
			dataset_norm_point.add_new(["3006", "0042"], "CS", "POINT")
			dataset_norm_point.add_new(["3006", "0046"], "IS", 1)
			dataset_norm_point.add_new(["3006", "0048"], "IS", 1)
			dataset_norm_point.add_new(["3006", "0050"], "DS", normalization_point)
			dataset_norm_point.add_new(["3007", "0010"], "LO", "NUCLETRON")
			dataset_norm_point.add_new(["3007", "1015"], "DS", 1.00000)
			dataset_norm_point.add_new(["3007", "1016"], "CS", "gMCO")
			dataset_norm_point.add_new(["3007", "1022"], "DS", 1.00000)
			dataset_norm_point.add_new(["3007", "1025"], "CS", "TRUE")
			dataset = pydicom.Dataset()
			dataset.add_new(["0021", "0010"], "LO", "NUCLETRON")
			dataset.add_new(["3006", "002a"], "IS", ["242", "236", "0"])
			dataset.add_new(["3006", "0040"], "SQ", pydicom.sequence.Sequence([dataset_norm_point]))
			dataset.add_new(["3006", "0084"], "IS", normalization_point_roi_number)
			for item in tmp:	
				item.ROIContourSequence.append(dataset)
			self.RP_file["300f", "1000"] = pydicom.DataElement(["300f", "1000"], "SQ", tmp)

		if normalize_dwell_weights:
			normalization_factor = np.max(self.dwells_data[:, -1])
		else:
			normalization_factor = 100
		total_time = 0	
		for applicationSetup in self.RP_file.ApplicationSetupSequence:
			for i_channel, channel in enumerate(applicationSetup.ChannelSequence):
				channel_control_points_indices = np.where(self.dwells_data[:, -3] == channel.ChannelNumber)[0]
				# Number of channel control points is always twice the number of channel dwell positons
				n_control_points = 2 * len(channel_control_points_indices)
				# the channel total time is the sum of all dwell times
				channel_total_time = np.sum(self.dwells_data[channel_control_points_indices, -1])
				# Cumulative dwell time weights for all dwell positions
				if channel_total_time != 0.0:
					if normalize_dwell_weights:
						cumulative_time_weights =\
						np.cumsum(self.dwells_data[channel_control_points_indices, -1] / normalization_factor)
					else:
						cumulative_time_weights =\
						normalization_factor * np.cumsum(self.dwells_data[channel_control_points_indices, -1] / channel_total_time)
				else:
					cumulative_time_weights = np.zeros(len(channel_control_points_indices))
				total_time += channel_total_time
				# Overwrite current DICOM data with the current data
				if n_control_points != 0:
					channel.FinalCumulativeTimeWeight = cumulative_time_weights[-1]
					channel.ChannelTotalTime = channel_total_time
					channel.NumberOfControlPoints = n_control_points
					channel.BrachyControlPointSequence = []
					cumulative_time_weight = 0
					j_control_point = 0
					for i_control_point in range(int(n_control_points / 2)):
						control_point_relative_position = self.dwells_data[channel_control_points_indices[i_control_point], -2]
						dummy_control_point_1 = pydicom.dataset.Dataset()
						dummy_control_point_1.ControlPointIndex = j_control_point
						dummy_control_point_1.ControlPointRelativePosition = control_point_relative_position
						dummy_control_point_1.ControlPoint3DPosition =\
						[str(10*self.dwells_data[channel_control_points_indices[i_control_point], 0]),
						str(10*self.dwells_data[channel_control_points_indices[i_control_point], 1]),
						str(10*self.dwells_data[channel_control_points_indices[i_control_point], 2])]
						dummy_control_point_1.CumulativeTimeWeight = cumulative_time_weight
						channel.BrachyControlPointSequence.append(dummy_control_point_1)
						j_control_point += 1

						dummy_control_point_2 = pydicom.dataset.Dataset()
						dummy_control_point_2.ControlPointIndex = j_control_point
						dummy_control_point_2.ControlPointRelativePosition =\
						dummy_control_point_1.ControlPointRelativePosition
						dummy_control_point_2.ControlPoint3DPosition =\
						dummy_control_point_1.ControlPoint3DPosition
						cumulative_time_weight = cumulative_time_weights[i_control_point]
						dummy_control_point_2.CumulativeTimeWeight = cumulative_time_weight
						channel.BrachyControlPointSequence.append(dummy_control_point_2)
						j_control_point += 1
				else:
					channel.ChannelTotalTime = 0
					# try:
					del channel.FinalCumulativeTimeWeight
					# except AttributeError:
					# 	pass
					channel.NumberOfControlPoints = 2
					channel.BrachyControlPointSequence = []

					dummy_control_point_1 = pydicom.dataset.Dataset()
					dummy_control_point_1.ControlPointIndex = 0
					dummy_control_point_1.ControlPointRelativePosition = 0
					dummy_control_point_1.CumulativeTimeWeight = None
					channel.BrachyControlPointSequence.append(dummy_control_point_1)

					dummy_control_point_2 = pydicom.dataset.Dataset()
					dummy_control_point_2.ControlPointIndex = 1
					dummy_control_point_2.ControlPointRelativePosition = 0
					dummy_control_point_2.CumulativeTimeWeight = None
					channel.BrachyControlPointSequence.append(dummy_control_point_2)
				# print(channel)
				# print("=========================================")
				# print()
		if export_path is None:
			RP_exported_filename = os.path.join(self.patient_path, "RTPLAN_gMCO_planID_{}.dcm".format(self.i_plan_selected))
		else:
			RP_exported_filename = os.path.join(export_path, "RTPLAN_gMCO_planID_{}.dcm".format(self.i_plan_selected))
		self.RP_file.SoftwareVersions = "gMCO + {}".format(self.RP_file.SoftwareVersions)
		# self.RP_file["300a", "0002"] =\
		# pydicom.DataElement(["300a", "0002"], "LO", "gMCO_opt_{}".format(self.i_plan_selected))
		# self.RP_file["300a", "0003"] =\
		# pydicom.DataElement(["300a", "0003"], "LO", "gMCO_opt_{}".format(self.i_plan_selected))
		if not normalize_dwell_weights:
			pydicom.DataElement(["300a", "0004"], "LO", "gMCO_opt_{}".format(self.i_plan_selected))
			
		f = open(os.path.join(self.patient_path, "plan_id.txt"), 'w')	
		f.write(str(self.i_plan_selected))
		f.close()

		for item in self.RP_file.SourceSequence:
			reference_air_kerma_rate = float(item["300a", "022a"].value)
		for item in self.RP_file.ApplicationSetupSequence:
			item["300a", "0250"].value = reference_air_kerma_rate * total_time / 3600
		self.RP_file.SeriesInstanceUID = pydicom.uid.generate_uid(prefix='1.2.826.0.1.3680043.10.424.')
		self.RP_file.SOPInstanceUID = pydicom.uid.generate_uid(prefix='1.2.826.0.1.3680043.10.424.')
		self.RP_file.save_as(RP_exported_filename)
		print("Done! Data save to {}".format(RP_exported_filename))
		return RP_exported_filename

class Catheter():
	def __init__(self, id, n_act=0):
		self.id = id
		self.n_act = n_act
		self.n_dp = None # Number of dwell positions
		self.active_dwell_pos = None # Coordinates of the active dwell positions from optimization
		self.dwell_times = None # The dwell times from optimization
		self.dwell_pos = None # Coordinates of all the dwell positions
		self.total_time = 0
		self.max_time = 0
		self.std_time = 0
		self.channel_relative_dwell_pos = None

	def set_catheter(self, dwell_pos):
		self.dwell_pos = dwell_pos[:, :3]
		self.dwell_pos = np.round(self.dwell_pos, 4)
		self.n_dp = self.dwell_pos.shape[0]
		self.dwell_times = np.full(self.n_dp, 0, dtype=np.float64)
		self.active_dwell_pos = np.full(self.n_dp, 0, dtype=bool)
		self.channel_relative_dwell_pos = dwell_pos[:, -1]
		try:
			self.zx_interpolator = interp1d(self.dwell_pos[:, 2], self.dwell_pos[:, 0], 
					bounds_error=False, fill_value=np.nan)
		except ValueError:
			self.zx_interpolator = None
		try:
			self.zy_interpolator = interp1d(self.dwell_pos[:, 2], self.dwell_pos[:, 1], 
				bounds_error=False, fill_value=np.nan)
		except ValueError:
			self.zx_interpolator = None

	def interpolate_xy_given_z(self, z):
		if self.zx_interpolator is not None and self.zy_interpolator is not None:
			return self.zx_interpolator(z), self.zy_interpolator(z)
		else:
			print("Warning! interpolation for dwell position is nan!")
			return np.nan

	def get_closest_dwell_from_z(self, z):
		if self.dwell_pos is not None:
			iDwell = np.argmin(np.abs(self.dwell_pos[:, 2] - z))
			closest_dwell_pos = np.copy(self.dwell_pos[iDwell])
			is_zero = self.dwell_times[iDwell] == 0.0
			if z > np.max(self.dwell_pos[:, 2]) or z < np.min(self.dwell_pos[:, 2]):
				closest_dwell_pos.fill(np.nan)
			return closest_dwell_pos, is_zero
		else:
			return np.nan, is_zero

	def set_active_dwells(self, active_dwell_pos):
		if len(active_dwell_pos) == len(self.active_dwell_pos):
			self.active_dwell_pos = active_dwell_pos
		else:
			raise RuntimeError("set_active_dwells is wrong!")

	def set_channel_relative_dwell_pos(self, channel_relative_dwell_pos):
		if len(channel_relative_dwell_pos) == len(self.channel_relative_dwell_pos):
			self.channel_relative_dwell_pos = channel_relative_dwell_pos
		else:
			raise RuntimeError("set_active_dwells is wrong!")

	def get_max_time(self):
		return np.max(self.dwell_times)

	def get_total_time(self):
		return np.sum(self.dwell_times)

	def get_std_time(self):
		iActiveDwells = np.where(self.active_dwell_pos == 1)[0]
		if len(iActiveDwells) == 0:
			return 0
		else:
			return np.std(self.dwell_times[iActiveDwells])

class Catheter_setup():
	def __init__(self):
		self.n_cat = 0
		self.dwell_step = 0
		self.n_act = 0
		self.n_plans = 0
		self.dwell_times = None

	def set_active_dwells(self, filename):
		dwells_data_gmco = np.loadtxt(filename)
		for catheter in self.setup:
			ind_cat = np.where(catheter.id == dwells_data_gmco[:, 6])[0]
			catheter.set_active_dwells(dwells_data_gmco[ind_cat, -1])
			catheter.set_channel_relative_dwell_pos(dwells_data_gmco[ind_cat, -2])
		self.set_dwell_step()

	def set_dwell_step(self):
		dwell_steps = []
		for catheter in self.setup:
			if len(catheter.channel_relative_dwell_pos) > 1:
				dwell_steps.append(np.abs(catheter.channel_relative_dwell_pos[0] - catheter.channel_relative_dwell_pos[1]))
		dwell_steps_unique = np.unique(dwell_steps)
		if len(dwell_steps_unique) == 1:
			self.dwell_step = dwell_steps_unique[0]
		else:
			idx_max_occurence = None
			max_occurence = 0
			for i, dwell_step in enumerate(dwell_steps_unique):
				n_occurences = dwell_steps.count(dwell_step)
				if n_occurences > max_occurence:
					max_occurence = n_occurences
					idx_max_occurence = i
			self.dwell_step = dwell_steps_unique[idx_max_occurence]

	def load_catheters(self, filename, usecols=(0,1,2,3,6), skiprows=5):
		# load dwells.txt, contains all dwell pos
		self.setup = []
		dwells = np.loadtxt(filename, skiprows=skiprows, usecols=usecols)
		id_cat = np.unique(dwells[:, 3])
		self.n_cat = len(id_cat) # Number of catheters placed
		for iCat in id_cat:
			iDwells = np.where(dwells[:, 3] == iCat)[0] #all dwells 
			#print(iDwells)
			if len(iDwells) == 0:
				continue
			catheter = Catheter(int(iCat), len(iDwells))
			catheter.set_catheter(dwell_pos=dwells[iDwells, :])
			self.setup.append(catheter)
		self.set_dwell_step()

	def load_dwell_times(self, filename):
		if self.n_plans == 0:
			f = open(filename, 'r')
			self.n_plans = int(np.fromfile(f, count=1, dtype=np.int32))
			self.n_act = int(np.fromfile(f, count=1, dtype=np.int32))
			self.dwell_times = np.fromfile(f, count=-1, dtype=np.float32)
			f.close()
			self.dwell_times = self.dwell_times.reshape(self.n_plans, self.n_act)
		else:
			f = open(filename, 'r')
			n_plans = int(np.fromfile(f, count=1, dtype=np.int32))
			n_act = int(np.fromfile(f, count=1, dtype=np.int32))
			if n_act != self.n_act:
				raise RuntimeError("Cannot set dwell times because of wrong number of active dwells!")
			dwell_times = np.fromfile(f, count=-1, dtype=np.float32)
			f.close()
			dwell_times = dwell_times.reshape(n_plans, n_act)
			self.dwell_times = np.vstack((self.dwell_times, dwell_times))
			self.n_plans += n_plans

	def set_dwell_times(self, i_plan_selected):
		plan_dwell_times = self.dwell_times[i_plan_selected]
		cumul_active_dwell = 0
		for catheter in self.setup:
			for iDwell in range(catheter.n_dp):
				try:
					if catheter.active_dwell_pos[iDwell]:
						catheter.dwell_times[iDwell] = plan_dwell_times[cumul_active_dwell]
						cumul_active_dwell += 1
				except:
					print(i_plan_selected)
			
	def get_catheter(self, iCat):
		if (iCat  > 0 ) and (iCat < len(self.setup)+1):
			return self.setup[iCat - 1]
		else:
			raise Exception('Catheter index out of range')

	def get_total_time(self):
		total_time = 0
		for catheter in self.setup:
			total_time += catheter.get_total_time()
		return total_time

	def save_dwells_to_txt(self, filename):
		with open(filename, "w") as fdwells:
			fdwells.write("______________________________________________________________________")
			fdwells.write("\n\n")
			fdwells.write("                              IPSA                                    ")
			fdwells.write("\n______________________________________________________________________\n\n")
			for catheter in self.setup:
				for i_dwell_position in range(catheter.dwell_pos.shape[0]):
					fdwells.write("{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{:.2f}\n"\
					.format(catheter.dwell_pos[i_dwell_position, 0],
					catheter.dwell_pos[i_dwell_position, 1],
					catheter.dwell_pos[i_dwell_position, 2],
					catheter.id,
					"inactive",
					"unfrozen",
					catheter.channel_relative_dwell_pos[i_dwell_position]))


