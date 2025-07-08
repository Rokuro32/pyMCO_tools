#!/usr/bin/env python3
"""
Script complet pour extraire et traiter les données DICOM de radiothérapie
Utilise les fonctions existantes du code importDicomDataForOpt.py avec corrections
"""

import pydicom
import numpy as np
import os
import sys
from datetime import datetime

# Import des fonctions existantes
from pyPlanNavigationTools.importDicomDataForOpt import (
    get_catheters,
    get_dwells_of_clinical_plan,
    get_dwell_positions_for_optimization,
    get_dose_points
)

class DicomExtractor:
    """Classe principale utilisant les fonctions existantes avec corrections"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
    
    def get_anatomy_contours_fixed(self, RSfile, output_path, add_id=False, replace_organ_name=None):
        """Version corrigée de get_anatomy_contours avec replace_organ_name défini"""
        
        if replace_organ_name is None:
            replace_organ_name = {
                "PTV": "TARGET",
                "PROSTATE": "TARGET",
                "VESSIE": "BLADDER",
                "BOWEL": "INTESTINS",
                "SMALL_BOWEL": "INTESTINS"
            }
        
        d = pydicom.read_file(RSfile)
        id_patient = d.PatientName
        file = open(os.path.join(output_path, 'anatomy.txt'), 'w')
        file.write("______________________________________________________________________")
        file.write("\n\n")
        file.write("                              IPSA                                    ")
        file.write("\n______________________________________________________________________\n\n")
        
        ROI = []
        for roi in d.StructureSetROISequence: 
            roiName = str(roi.ROIName).upper()
            roiName = roiName.replace(" ", "_")
            roiNameTmp = ""
            for char in roiName:
                if char == "_" or char.isalpha() or char.isnumeric():
                    roiNameTmp += char
            roiName = roiNameTmp
            
            # Application des remplacements
            for organ_name_original, organ_name in replace_organ_name.items():
                if roiName == organ_name_original:
                    roiName = organ_name
                    break
                    
            if add_id:
                roiName += "_ID{}".format(roi.ROINumber)
            ROI.append(roiName)

        tmp_all_organ_contours = {}
        for c, C in enumerate(d.ROIContourSequence):
            tmp_all_organ_contours[ROI[c]] = []
            if "ContourSequence" in C.dir():
                for cont in C.ContourSequence:
                    tmp_contour = []
                    Cdata = cont.ContourData
                    Np = cont.NumberOfContourPoints
                    for p in range(Np):
                        p3 = 3*p
                        tmp_contour.append([float(Cdata[p3])/10., 
                            float(Cdata[p3+1])/10., float(Cdata[p3+2])/10.])
                    
                    # Validation du contour avec Shapely si disponible
                    try:
                        from shapely.geometry import Polygon
                        tmp_polygon = Polygon(tmp_contour)
                        if not tmp_polygon.is_valid:
                            if self.verbose:
                                print(f"Warning! Contour invalide pour {ROI[c]}, contour #{cont.ContourNumber}")
                        tmp_ext = list(tmp_polygon.exterior.coords)
                        tmp_contour = np.array([[coord[0], coord[1], coord[2]] for coord in tmp_ext])
                    except (ImportError, Exception):
                        # Si Shapely n'est pas disponible ou échoue, utiliser les points originaux
                        tmp_contour = np.array(tmp_contour)
                    
                    tmp_all_organ_contours[ROI[c]].append(tmp_contour)
        
        # Sauvegarde dans le fichier
        for organ, contours in tmp_all_organ_contours.items():
            nPoints = 0
            for contour in contours:
                nPoints += len(contour)
            if nPoints == 0:
                if self.verbose:
                    print(f"Skipping organ {organ} because it does not have contours!")
                continue
            file.write(organ+"\n")
            file.write(str(nPoints)+"\n")
            for contour in contours:
                for point in contour:
                    file.write('%6.6f\t'%point[0])
                    file.write('%6.6f\t'%point[1])
                    file.write('%6.6f\t'%point[2]+"\n")
        file.close()
        
        return len(tmp_all_organ_contours), sum(sum(len(contour) for contour in contours) 
                                               for contours in tmp_all_organ_contours.values())
    
    def get_catheters_and_anatomy_for_optimization_fixed(self, RPfile, RSfile, output_path, 
                                                        imaging_modality="CT", split_target=True,
                                                        source_step_size=3.0, generate_anatomy_file=True,
                                                        generate_interpolated_anatomy=True):
        """Version corrigée de get_catheters_and_anatomy_for_optimization avec interpolation"""
        
        # Générer les positions d'arrêt pour l'optimisation
        dwell_step = get_dwell_positions_for_optimization(
            RPfile, output_path, 
            imaging_modality=imaging_modality, 
            source_step_size=source_step_size
        )

        if generate_anatomy_file:
            # Dictionnaire de remplacement des noms d'organes
            replace_organ_name = {
                "PTV": "TARGET",
                "PROSTATE": "TARGET",
                "VESSIE": "BLADDER",
                "BOWEL": "INTESTINS",
                "SMALL_BOWEL": "INTESTINS"
            }
            
            n_organs, n_points = self.get_anatomy_contours_fixed(
                RSfile, output_path, replace_organ_name=replace_organ_name
            )
            
            if self.verbose:
                print(f"    Anatomy: {n_organs} organes, {n_points} points extraits")
            
            # Génération de l'anatomy interpolé si demandé
            if generate_interpolated_anatomy:
                try:
                    self.generate_interpolated_anatomy(output_path)
                    if self.verbose:
                        print(f"    Anatomy interpolé généré")
                except Exception as e:
                    if self.verbose:
                        print(f"    ⚠️  Erreur interpolation anatomy: {e}")
        
        # Division en lobes si demandée (fonction simplifiée pour éviter les dépendances)
        if split_target:
            if self.verbose:
                print("    Division en lobes non implémentée dans cette version")
        
        return float(dwell_step)
    
    def generate_interpolated_anatomy(self, output_path):
        """Génère un fichier anatomy_interpolated.txt en utilisant les fonctions existantes"""
        
        try:
            # Import des classes nécessaires depuis votre code
            from pyPlanNavigationTools.doseDistributionsHandler import Anatomy
            
            # Charger l'anatomy depuis le fichier généré
            anatomy_file = os.path.join(output_path, "anatomy.txt")
            if not os.path.exists(anatomy_file):
                raise FileNotFoundError("Le fichier anatomy.txt doit être généré d'abord")
            
            # Configuration des couleurs pour les organes
            RTSTRUCT_colors = {
                "TARGET": "red", 
                "BLADDER": "orange", 
                "RECTUM": "green", 
                "URETHRA": "gold",
                "INTESTINS": "blue",
                "BOWEL": "blue",
                "SKIN": "gray"
            }
            
            if self.verbose:
                print(f"      Chargement anatomy depuis {anatomy_file}")
            
            # Charger l'anatomy avec votre classe existante
            anatomy = Anatomy(anatomy_file, RTSTRUCT=RTSTRUCT_colors, unit="cm")
            
            if self.verbose:
                print(f"      Interpolation pour {len(anatomy.organs)} organes")
            
            # Interpoler l'anatomy avec les paramètres par défaut
            anatomy.interpolate_anatomy(
                grid_spacing=0.1,           # Espacement de grille en cm
                n_slice_between_planes=1,   # 1 slice entre chaque plan
                smooth_contours=False,      # Pas de lissage par défaut
                organs_to_interpolate=[]    # Tous les organes
            )
            
            # Sauvegarder le fichier interpolé
            interpolated_file = os.path.join(output_path, "anatomy_interpolated.txt")
            anatomy.save_to_txt_file(interpolated_file.replace("_interpolated.txt", ""))
            
            # Le fichier anatomy_interpolated.txt est automatiquement généré par save_to_txt_file
            if os.path.exists(interpolated_file):
                if self.verbose:
                    print(f"      ✓ Fichier anatomy_interpolated.txt généré")
                return interpolated_file
            else:
                if self.verbose:
                    print(f"      ⚠️  Le fichier interpolé n'a pas été créé automatiquement")
                return None
                
        except ImportError as e:
            if self.verbose:
                print(f"      ⚠️  Import error pour l'interpolation: {e}")
            # Fallback: créer une version simple sans interpolation
            return self.create_simple_interpolated_anatomy(output_path)
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️  Erreur lors de l'interpolation: {e}")
            return None
    
    def create_simple_interpolated_anatomy(self, output_path):
        """Crée une version simple d'anatomy interpolé sans dépendances complexes"""
        
        anatomy_file = os.path.join(output_path, "anatomy.txt")
        interpolated_file = os.path.join(output_path, "anatomy_interpolated.txt")
        
        if not os.path.exists(anatomy_file):
            return None
        
        try:
            # Lire le fichier anatomy original
            with open(anatomy_file, 'r') as f:
                lines = f.readlines()
            
            # Parser les données
            organs_data = {}
            i = 5  # Skip header
            
            while i < len(lines):
                line = lines[i].strip()
                if line and line[0].isalpha():  # Nom d'organe
                    organ_name = line
                    i += 1
                    if i < len(lines):
                        try:
                            n_points = int(lines[i].strip())
                            i += 1
                            
                            # Lire les points
                            points = []
                            for j in range(n_points):
                                if i + j < len(lines):
                                    point_line = lines[i + j].strip().split('\t')
                                    if len(point_line) >= 3:
                                        points.append([float(point_line[0]), 
                                                     float(point_line[1]), 
                                                     float(point_line[2])])
                            
                            organs_data[organ_name] = points
                            i += n_points
                        except:
                            i += 1
                else:
                    i += 1
            
            # Interpolation simple: doubler les points en interpolant entre les tranches
            interpolated_organs = {}
            for organ_name, points in organs_data.items():
                if len(points) < 2:
                    interpolated_organs[organ_name] = points
                    continue
                
                # Grouper par coordonnée Z
                z_groups = {}
                for point in points:
                    z = round(point[2], 3)
                    if z not in z_groups:
                        z_groups[z] = []
                    z_groups[z].append(point)
                
                # Interpoler entre les coupes Z
                interpolated_points = []
                z_coords = sorted(z_groups.keys())
                
                for i, z in enumerate(z_coords):
                    # Ajouter les points originaux
                    interpolated_points.extend(z_groups[z])
                    
                    # Interpoler vers la coupe suivante si elle existe
                    if i < len(z_coords) - 1:
                        next_z = z_coords[i + 1]
                        if abs(next_z - z) > 0.2:  # Si l'écart est > 2mm
                            # Créer une coupe intermédiaire
                            mid_z = (z + next_z) / 2
                            current_points = z_groups[z]
                            next_points = z_groups[next_z]
                            
                            # Interpolation simple entre les coupes
                            for j, curr_pt in enumerate(current_points):
                                if j < len(next_points):
                                    next_pt = next_points[j]
                                    mid_point = [
                                        (curr_pt[0] + next_pt[0]) / 2,
                                        (curr_pt[1] + next_pt[1]) / 2,
                                        mid_z
                                    ]
                                    interpolated_points.append(mid_point)
                
                interpolated_organs[organ_name] = interpolated_points
            
            # Écrire le fichier interpolé
            with open(interpolated_file, 'w') as f:
                f.write("______________________________________________________________________\n")
                f.write("\n")
                f.write("                              IPSA                                    \n")
                f.write("______________________________________________________________________\n\n")
                
                for organ_name, points in interpolated_organs.items():
                    if len(points) > 0:
                        f.write(f"{organ_name}\n")
                        f.write(f"{len(points)}\n")
                        for point in points:
                            f.write(f"{point[0]:6.6f}\t{point[1]:6.6f}\t{point[2]:6.6f}\n")
            
            if self.verbose:
                print(f"      ✓ Interpolation simple créée: {len(interpolated_organs)} organes")
            
            return interpolated_file
            
        except Exception as e:
            if self.verbose:
                print(f"      ❌ Erreur interpolation simple: {e}")
            return None
    
    def extract_single_patient(self, rtstruct_file=None, rtplan_file=None, 
                              output_dir=None, patient_name=None,
                              imaging_modality="US", source_step_size=3.0,
                              split_target=False):
        """
        Extrait les données d'un patient unique en utilisant les fonctions corrigées
        """
        
        if not rtstruct_file and not rtplan_file:
            raise ValueError("Au moins un fichier RTSTRUCT ou RTPLAN doit être fourni")
        
        if self.verbose:
            print(f"\n=== EXTRACTION PATIENT: {patient_name or 'Inconnu'} ===")
        
        results = {
            'patient_name': patient_name,
            'files_processed': [],
            'files_generated': [],
            'errors': [],
            'parameters': {
                'imaging_modality': imaging_modality,
                'source_step_size': source_step_size,
                'split_target': split_target
            }
        }
        
        # Détermination du dossier de sortie
        if output_dir is None:
            output_dir = f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if patient_name:
            output_dir = os.path.join(output_dir, patient_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraction complète avec les fonctions corrigées si les deux fichiers sont disponibles
        if rtstruct_file and rtplan_file:
            try:
                if self.verbose:
                    print(f"  Utilisation des fonctions corrigées...")
                
                dwell_step = self.get_catheters_and_anatomy_for_optimization_fixed(
                    rtplan_file, 
                    rtstruct_file, 
                    output_dir, 
                    imaging_modality=imaging_modality,
                    split_target=split_target,
                    source_step_size=source_step_size, 
                    generate_anatomy_file=True,
                    generate_interpolated_anatomy=True  # Génération de l'anatomy interpolé
                )
                
                results['optimization'] = {
                    'dwell_step': dwell_step,
                    'files_generated': self._get_generated_optimization_files(output_dir)
                }
                results['files_processed'].extend([rtplan_file, rtstruct_file])
                
                if self.verbose:
                    print(f"✓ Fichiers d'optimisation générés (dwell_step: {dwell_step} mm)")
                    
            except Exception as e:
                error_msg = f"Erreur génération fichiers optimisation: {e}"
                results['errors'].append(error_msg)
                if self.verbose:
                    print(f"❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
        
        # Extraction séparée des points d'arrêt si seulement RTPLAN
        elif rtplan_file:
            try:
                dwell_results = self.extract_dwell_points_only(rtplan_file, output_dir)
                results['dwell_points'] = dwell_results
                results['files_processed'].append(rtplan_file)
                if self.verbose:
                    print(f"✓ Points d'arrêt extraits: {dwell_results['n_dwell_points']} points")
            except Exception as e:
                error_msg = f"Erreur extraction RTPLAN: {e}"
                results['errors'].append(error_msg)
                if self.verbose:
                    print(f"❌ {error_msg}")
        
        # Extraction séparée des contours si seulement RTSTRUCT  
        elif rtstruct_file:
            try:
                anatomy_results = self.extract_anatomy_only(rtstruct_file, output_dir, 
                                                           generate_interpolated=True)
                results['anatomy'] = anatomy_results
                results['files_processed'].append(rtstruct_file)
                if self.verbose:
                    print(f"✓ Contours anatomiques extraits: {len(anatomy_results['organs'])} organes")
            except Exception as e:
                error_msg = f"Erreur extraction RTSTRUCT: {e}"
                results['errors'].append(error_msg)
                if self.verbose:
                    print(f"❌ {error_msg}")
        
        # Extraction des cathéters pour modalité CT
        if rtplan_file and imaging_modality == "CT":
            try:
                if self.verbose:
                    print(f"  Extraction cathéters (modalité CT)...")
                n_cat, cat_data = get_catheters(rtplan_file, output_dir)
                results['catheters'] = {
                    'n_catheters': n_cat,
                    'catheter_data': cat_data,
                    'file_generated': os.path.join(output_dir, "catheters.txt")
                }
                if self.verbose:
                    print(f"✓ Cathéters extraits: {n_cat} cathéters")
            except Exception as e:
                error_msg = f"Erreur extraction cathéters: {e}"
                results['errors'].append(error_msg)
                if self.verbose:
                    print(f"❌ {error_msg}")
        
        # Extraction des points de dose si disponible
        if rtplan_file:
            try:
                if self.verbose:
                    print(f"  Extraction points de dose...")
                get_dose_points(rtplan_file, output_dir)
                dose_points_file = os.path.join(output_dir, "dose_points.txt")
                if os.path.exists(dose_points_file):
                    results['dose_points_file'] = dose_points_file
                    if self.verbose:
                        print(f"✓ Points de dose extraits")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Points de dose non disponibles: {e}")
        
        # Génération d'informations détaillées sur le patient
        if rtplan_file:
            try:
                patient_info = self.extract_patient_info(rtplan_file)
                results['patient_info'] = patient_info
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Informations patient non extraites: {e}")
        
        # Génération du rapport
        self.generate_report(results, output_dir)
        
        return results
    
    def extract_dwell_points_only(self, rtplan_file, output_dir):
        """Extrait seulement les points d'arrêt en utilisant les fonctions existantes"""
        
        if self.verbose:
            print(f"  Extraction points d'arrêt: {os.path.basename(rtplan_file)}")
        
        # Utilisation de la fonction existante
        n_cat, dwell_data = get_dwells_of_clinical_plan(rtplan_file, output_dir, save_prescription=True)
        
        # Extraction des informations supplémentaires
        d = pydicom.read_file(rtplan_file)
        patient_info = self.extract_patient_info_from_dicom(d)
        
        # Calcul des statistiques
        total_time = np.sum(dwell_data[:, -1]) if len(dwell_data) > 0 else 0
        
        results = {
            'patient_info': patient_info,
            'n_catheters': n_cat,
            'n_dwell_points': len(dwell_data),
            'total_time': total_time,
            'dwell_data': dwell_data,
            'files_generated': [
                os.path.join(output_dir, "dwells_planned.txt"),
                os.path.join(output_dir, "S_k.txt"),
                os.path.join(output_dir, "prescription.txt")
            ]
        }
        
        return results
    
    def extract_anatomy_only(self, rtstruct_file, output_dir, generate_interpolated=True):
        """Extrait seulement les contours anatomiques en utilisant les fonctions existantes"""
        
        if self.verbose:
            print(f"  Extraction contours anatomiques: {os.path.basename(rtstruct_file)}")
        
        # Dictionnaire de remplacement des noms d'organes
        replace_organ_name = {
            "PTV": "TARGET",
            "PROSTATE": "TARGET",
            "VESSIE": "BLADDER",
            "BOWEL": "INTESTINS",
            "SMALL_BOWEL": "INTESTINS"
        }
        
        # Utilisation de la fonction corrigée
        n_organs, total_points = self.get_anatomy_contours_fixed(
            rtstruct_file, output_dir, add_id=False, 
            replace_organ_name=replace_organ_name
        )
        
        # Génération de l'anatomy interpolé si demandé
        interpolated_file = None
        if generate_interpolated:
            try:
                interpolated_file = self.generate_interpolated_anatomy(output_dir)
                if self.verbose and interpolated_file:
                    print(f"    ✓ Anatomy interpolé généré")
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️  Erreur interpolation: {e}")
        
        # Lecture du fichier généré pour extraire les informations
        anatomy_file = os.path.join(output_dir, "anatomy.txt")
        organs, total_points = self.parse_anatomy_file(anatomy_file)
        
        # Extraction des informations patient
        d = pydicom.read_file(rtstruct_file)
        patient_info = self.extract_patient_info_from_dicom(d)
        
        files_generated = [anatomy_file]
        if interpolated_file and os.path.exists(interpolated_file):
            files_generated.append(interpolated_file)
        
        results = {
            'patient_info': patient_info,
            'organs': organs,
            'total_points': total_points,
            'files_generated': files_generated,
            'interpolated_generated': interpolated_file is not None
        }
        
        return results
    
    def extract_patient_info(self, rtplan_file):
        """Extrait les informations patient du fichier RTPLAN"""
        d = pydicom.read_file(rtplan_file)
        return self.extract_patient_info_from_dicom(d)
    
    def extract_patient_info_from_dicom(self, dicom_data):
        """Extrait les informations patient d'un objet DICOM"""
        patient_info = {
            "patient_name": str(getattr(dicom_data, 'PatientName', 'Inconnu')),
            "patient_id": str(getattr(dicom_data, 'PatientID', 'Inconnu')),
            "study_date": str(getattr(dicom_data, 'StudyDate', 'Inconnu')),
            "modality": str(getattr(dicom_data, 'Modality', 'Inconnu'))
        }
        
        # Informations spécifiques RTPLAN
        if dicom_data.Modality == "RTPLAN":
            patient_info.update({
                "plan_name": str(getattr(dicom_data, 'RTPlanName', 'Inconnu')),
                "plan_date": str(getattr(dicom_data, 'RTPlanDate', 'Inconnu')),
                "brachy_treatment_type": str(getattr(dicom_data, 'BrachyTreatmentType', 'Inconnu'))
            })
            
            # Prescription
            try:
                for item_fgs in dicom_data.FractionGroupSequence:
                    for item in item_fgs.ReferencedBrachyApplicationSetupSequence:
                        patient_info["prescription"] = item["300a", "00a4"].value
                        break
            except:
                patient_info["prescription"] = None
            
            # Force kerma
            try:
                for source in dicom_data.SourceSequence:
                    patient_info["air_kerma_strength"] = source["300a", "022a"].value
                    break
            except:
                patient_info["air_kerma_strength"] = None
        
        return patient_info
    
    def parse_anatomy_file(self, anatomy_file):
        """Parse le fichier anatomy.txt pour extraire les informations"""
        if not os.path.exists(anatomy_file):
            return [], 0
        
        organs = []
        total_points = 0
        
        with open(anatomy_file, 'r') as f:
            lines = f.readlines()
        
        i = 5  # Skip header
        while i < len(lines):
            line = lines[i].strip()
            if line and line[0].isalpha():  # Nom d'organe
                organs.append(line)
                i += 1
                if i < len(lines):
                    try:
                        n_points = int(lines[i].strip())
                        total_points += n_points
                        i += n_points + 1
                    except:
                        i += 1
            else:
                i += 1
        
        return organs, total_points
    
    def _get_generated_optimization_files(self, output_dir):
        """Retourne la liste des fichiers d'optimisation générés"""
        possible_files = [
            "anatomy.txt",
            "anatomy_interpolated.txt",  # Ajout du fichier interpolé
            "dwells.txt", 
            "dwells_planned.txt",
            "S_k.txt",
            "prescription.txt",
            "catheters.txt"
        ]
        
        generated_files = []
        for filename in possible_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                generated_files.append(filepath)
        
        return generated_files
    
    def generate_report(self, results, output_dir):
        """Génère un rapport de synthèse"""
        report_file = os.path.join(output_dir, "extraction_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("=== RAPPORT D'EXTRACTION DICOM ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Patient: {results.get('patient_name', 'Inconnu')}\n\n")
            
            # Informations patient si disponibles
            if 'patient_info' in results:
                pi = results['patient_info']
                f.write("INFORMATIONS PATIENT:\n")
                f.write(f"  Nom: {pi.get('patient_name', 'N/A')}\n")
                f.write(f"  ID: {pi.get('patient_id', 'N/A')}\n")
                if 'plan_name' in pi:
                    f.write(f"  Plan: {pi['plan_name']}\n")
                if 'prescription' in pi and pi['prescription']:
                    f.write(f"  Prescription: {pi['prescription']} Gy\n")
                if 'brachy_treatment_type' in pi:
                    f.write(f"  Type traitement: {pi['brachy_treatment_type']}\n")
                f.write("\n")
            
            # Paramètres d'extraction
            if 'parameters' in results:
                params = results['parameters']
                f.write("PARAMÈTRES D'EXTRACTION:\n")
                f.write(f"  Modalité d'imagerie: {params['imaging_modality']}\n")
                f.write(f"  Pas de source: {params['source_step_size']} mm\n")
                f.write(f"  Division cible: {params['split_target']}\n")
                f.write("\n")
            
            # Fichiers traités
            f.write("FICHIERS TRAITÉS:\n")
            for file_path in results.get('files_processed', []):
                f.write(f"  - {os.path.basename(file_path)}\n")
            f.write("\n")
            
            # Résultats optimisation (mode complet)
            if 'optimization' in results:
                opt = results['optimization']
                f.write("EXTRACTION COMPLÈTE:\n")
                f.write(f"  Pas de source utilisé: {opt['dwell_step']} mm\n")
                f.write(f"  Fichiers générés: {len(opt['files_generated'])}\n")
                for file_path in opt['files_generated']:
                    f.write(f"    ✓ {os.path.basename(file_path)}\n")
                f.write("\n")
            
            # Résultats dwell points seulement
            if 'dwell_points' in results:
                dp = results['dwell_points']
                f.write("POINTS D'ARRÊT:\n")
                f.write(f"  Cathéters actifs: {dp['n_catheters']}\n")
                f.write(f"  Points d'arrêt: {dp['n_dwell_points']}\n")
                f.write(f"  Temps total: {dp['total_time']:.2f}s\n")
                f.write("\n")
            
            # Résultats anatomy seulement
            if 'anatomy' in results:
                anat = results['anatomy']
                f.write("CONTOURS ANATOMIQUES:\n")
                f.write(f"  Organes: {len(anat['organs'])}\n")
                f.write(f"  Points totaux: {anat['total_points']}\n")
                f.write("  Liste des organes:\n")
                for organ in anat['organs']:
                    f.write(f"    - {organ}\n")
                f.write("\n")
            
            # Cathéters (modalité CT)
            if 'catheters' in results:
                cat = results['catheters']
                f.write("CATHÉTERS (CT):\n")
                f.write(f"  Nombre de cathéters: {cat['n_catheters']}\n")
                f.write(f"  Fichier généré: {os.path.basename(cat['file_generated'])}\n")
                f.write("\n")
            
            # Points de dose
            if 'dose_points_file' in results:
                f.write("POINTS DE DOSE:\n")
                f.write(f"  Fichier généré: {os.path.basename(results['dose_points_file'])}\n")
                f.write("\n")
            
            # Erreurs
            if results.get('errors'):
                f.write("ERREURS:\n")
                for error in results['errors']:
                    f.write(f"  ❌ {error}\n")
                f.write("\n")
        
        if self.verbose:
            print(f"✓ Rapport généré: {report_file}")
    
    def process_batch(self, input_folder, output_folder, stored_by_studies=True,
                     imaging_modality="US", source_step_size=3.0, split_target=False):
        """Traite un lot de patients avec les fonctions existantes"""
        
        if self.verbose:
            print(f"=== TRAITEMENT EN LOT ===")
            print(f"Dossier d'entrée: {input_folder}")
            print(f"Dossier de sortie: {output_folder}")
            print(f"Modalité: {imaging_modality}, Pas source: {source_step_size}mm")
        
        # Filtrer et lister les patients
        all_items = os.listdir(input_folder)
        patients = [item for item in all_items 
                    if os.path.isdir(os.path.join(input_folder, item)) 
                    and not item.startswith('.')]
        
        if self.verbose:
            print(f"Patients trouvés: {len(patients)}")
        
        batch_results = []
        success_count = 0
        error_count = 0
        
        for patient in patients:
            try:
                if self.verbose:
                    print(f"\n--- Patient: {patient} ---")
                
                # Trouver les fichiers DICOM
                rtstruct_file, rtplan_file = self.find_dicom_files(
                    input_folder, patient, stored_by_studies
                )
                
                if not rtstruct_file and not rtplan_file:
                    if self.verbose:
                        print(f"❌ Aucun fichier DICOM trouvé pour {patient}")
                    error_count += 1
                    continue
                
                # Traiter le patient
                patient_output_dir = os.path.join(output_folder, patient)
                results = self.extract_single_patient(
                    rtstruct_file, rtplan_file, patient_output_dir, patient,
                    imaging_modality=imaging_modality,
                    source_step_size=source_step_size,
                    split_target=split_target
                )
                
                batch_results.append(results)
                if results.get('errors'):
                    error_count += 1
                else:
                    success_count += 1
                
                if self.verbose:
                    if results.get('errors'):
                        print(f"⚠️  Patient {patient} traité avec erreurs")
                        for error in results['errors']:
                            print(f"     - {error}")
                    else:
                        print(f"✓ Patient {patient} traité avec succès")
                    
            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"❌ Erreur pour patient {patient}: {e}")
        
        # Rapport final
        if self.verbose:
            print(f"\n=== RÉSUMÉ DU TRAITEMENT EN LOT ===")
            print(f"Patients traités avec succès: {success_count}")
            print(f"Patients en erreur: {error_count}")
            print(f"Total: {len(patients)}")
        
        return batch_results
    
    def find_dicom_files(self, input_folder, patient, stored_by_studies):
        """Trouve les fichiers RTSTRUCT et RTPLAN pour un patient"""
        
        rtstruct_file = None
        rtplan_file = None
        
        patient_path = os.path.join(input_folder, patient)
        
        if stored_by_studies:
            # Structure: patient/study/series/files
            all_items = os.listdir(patient_path)
            studies = [item for item in all_items 
                       if os.path.isdir(os.path.join(patient_path, item)) 
                       and not item.startswith('.')]
            
            for study in studies:
                study_path = os.path.join(patient_path, study)
                all_items = os.listdir(study_path)
                series = [item for item in all_items 
                          if os.path.isdir(os.path.join(study_path, item)) 
                          and not item.startswith('.')]
                
                for serie in series:
                    serie_path = os.path.join(study_path, serie)
                    all_files = os.listdir(serie_path)
                    dicom_files = [f for f in all_files if not f.startswith('.')]
                    
                    for dicom_file in dicom_files:
                        file_path = os.path.join(serie_path, dicom_file)
                        try:
                            d = pydicom.read_file(file_path)
                            if d.Modality == "RTSTRUCT":
                                rtstruct_file = file_path
                            elif d.Modality == "RTPLAN":
                                rtplan_file = file_path
                        except:
                            continue
        else:
            # Structure: patient/files
            all_files = os.listdir(patient_path)
            dicom_files = [f for f in all_files 
                           if not f.startswith('.') 
                           and os.path.isfile(os.path.join(patient_path, f))]
            
            for dicom_file in dicom_files:
                file_path = os.path.join(patient_path, dicom_file)
                try:
                    d = pydicom.read_file(file_path)
                    if d.Modality == "RTSTRUCT":
                        rtstruct_file = file_path
                    elif d.Modality == "RTPLAN":
                        rtplan_file = file_path
                except:
                    continue
        
        return rtstruct_file, rtplan_file

if __name__ == "__main__":
    # =================================================================
    # CONFIGURATION - Modifiez ces chemins selon votre environnement
    # =================================================================
    
    # OPTION 1: Patient unique
    SINGLE_PATIENT_MODE = False  # Mettre à True pour traiter un seul patient
    RTSTRUCT_FILE = "/path/to/struct.dcm"  # Fichier RTSTRUCT
    RTPLAN_FILE = "/path/to/plan.dcm"      # Fichier RTPLAN  
    PATIENT_NAME = "Patient_001"           # Nom du patient
    
    # OPTION 2: Traitement en lot (recommandé)
    BATCH_MODE = True  # Mettre à True pour traitement en lot
    
    # Structure des dossiers d'entrée
    INPUT_FOLDER = "/Users/xavierarata/gMCO_robust/patient_data"
    STORED_BY_STUDIES = True
    OUTPUT_FOLDER = "/Users/xavierarata/gMCO_robust/output/gMCO_input"
    
    # Paramètres d'extraction
    IMAGING_MODALITY = "US"        # "US" ou "CT"
    SOURCE_STEP_SIZE = 3.0         # Taille du pas en mm
    SPLIT_TARGET = False           # Diviser la cible en lobes
    VERBOSE = True                 # Affichage détaillé
    
    # =================================================================
    # EXÉCUTION AUTOMATIQUE
    # =================================================================
    
    extractor = DicomExtractor(verbose=VERBOSE)
    
    try:
        if SINGLE_PATIENT_MODE:
            # Mode patient unique
            if VERBOSE:
                print("=== MODE PATIENT UNIQUE ===")
                print(f"RTSTRUCT: {RTSTRUCT_FILE}")
                print(f"RTPLAN: {RTPLAN_FILE}")
                print(f"Sortie: {OUTPUT_FOLDER}")
            
            if not os.path.exists(RTSTRUCT_FILE) and not os.path.exists(RTPLAN_FILE):
                print("❌ Erreur: Au moins un fichier RTSTRUCT ou RTPLAN doit exister")
                sys.exit(1)
            
            rtstruct = RTSTRUCT_FILE if os.path.exists(RTSTRUCT_FILE) else None
            rtplan = RTPLAN_FILE if os.path.exists(RTPLAN_FILE) else None
            
            results = extractor.extract_single_patient(
                rtstruct, rtplan, OUTPUT_FOLDER, PATIENT_NAME,
                imaging_modality=IMAGING_MODALITY,
                source_step_size=SOURCE_STEP_SIZE,
                split_target=SPLIT_TARGET
            )
            
            if VERBOSE:
                print(f"\n✓ Patient {PATIENT_NAME} traité avec succès!")
                if results.get('errors'):
                    print("⚠️  Erreurs rencontrées:")
                    for error in results['errors']:
                        print(f"   - {error}")
        
        elif BATCH_MODE:
            # Mode traitement en lot
            if VERBOSE:
                print("=== MODE TRAITEMENT EN LOT ===")
                print(f"Dossier d'entrée: {INPUT_FOLDER}")
                print(f"Dossier de sortie: {OUTPUT_FOLDER}")
                print(f"Structure par études: {STORED_BY_STUDIES}")
                print(f"Modalité: {IMAGING_MODALITY}")
                print(f"Pas de source: {SOURCE_STEP_SIZE} mm")
            
            if not os.path.exists(INPUT_FOLDER):
                print(f"❌ Erreur: Le dossier d'entrée {INPUT_FOLDER} n'existe pas")
                sys.exit(1)
            
            batch_results = extractor.process_batch(
                INPUT_FOLDER, OUTPUT_FOLDER, STORED_BY_STUDIES,
                imaging_modality=IMAGING_MODALITY,
                source_step_size=SOURCE_STEP_SIZE,
                split_target=SPLIT_TARGET
            )
            
            # Statistiques finales
            success_count = len([r for r in batch_results if not r.get('errors')])
            error_count = len(batch_results) - success_count
            
            if VERBOSE:
                print(f"\n=== STATISTIQUES FINALES ===")
                print(f"✓ Patients traités avec succès: {success_count}")
                print(f"❌ Patients avec erreurs: {error_count}")
                print(f"📊 Total traité: {len(batch_results)}")
                
                # Détail des fichiers générés
                total_optimization = sum(1 for r in batch_results if 'optimization' in r)
                total_anatomy = sum(1 for r in batch_results if 'anatomy' in r)
                total_dwells = sum(1 for r in batch_results if 'dwell_points' in r)
                total_catheters = sum(1 for r in batch_results if 'catheters' in r)
                
                print(f"\n📁 Fichiers générés:")
                print(f"   - Extraction complète: {total_optimization} patients")
                print(f"   - Contours seulement: {total_anatomy} patients")
                print(f"   - Points d'arrêt seulement: {total_dwells} patients")
                print(f"   - Cathéters (CT): {total_catheters} patients")
        
        else:
            print("❌ Erreur: Aucun mode sélectionné (SINGLE_PATIENT_MODE ou BATCH_MODE)")
            sys.exit(1)
        
        if VERBOSE:
            print(f"\n🎉 Extraction terminée avec succès!")
            print(f"📂 Résultats disponibles dans: {OUTPUT_FOLDER}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Extraction interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        if VERBOSE:
            import traceback
            traceback.print_exc()
        sys.exit(1)