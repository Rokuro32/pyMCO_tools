#!/usr/bin/env python3
"""
Script pour convertir dwell_times.bin en array NumPy et expliquer le parsing en détail.
Utilise la classe DwellsData pour validation et comparaison.
"""

import numpy as np
import os
from pyPlanNavigationTools.DwellsData import DwellsData

def parse_dwell_times_binary(filename="dwell_times.bin", verbose=True):
    """
    Parse le fichier dwell_times.bin et explique chaque étape.
    
    Args:
        filename: Nom du fichier binaire
        verbose: Si True, affiche les détails du parsing
        
    Returns:
        tuple: (n_plans, n_active_dwells, dwell_times_array)
    """
    
    if verbose:
        print("="*70)
        print("PARSING DÉTAILLÉ DU FICHIER dwell_times.bin")
        print("="*70)
        print(f"\nFichier: {filename}")
        print(f"Taille du fichier: {os.path.getsize(filename)} bytes")
    
    # Ouvrir le fichier en mode binaire
    with open(filename, 'rb') as f:
        # ÉTAPE 1: Lire le nombre de plans (premier int32)
        # Position dans le fichier: bytes 0-3
        n_plans_bytes = f.read(4)  # 4 bytes pour un int32
        n_plans = int(np.frombuffer(n_plans_bytes, dtype=np.int32)[0])
        
        if verbose:
            print("\nÉTAPE 1 - Lecture du nombre de plans:")
            print(f"  - Bytes lus (hex): {n_plans_bytes.hex()}")
            print(f"  - Valeur décodée: {n_plans} plans")
            print(f"  - Position actuelle dans le fichier: {f.tell()} bytes")
        
        # ÉTAPE 2: Lire le nombre de dwells actifs (deuxième int32)
        # Position dans le fichier: bytes 4-7
        n_active_bytes = f.read(4)
        n_active_dwells = int(np.frombuffer(n_active_bytes, dtype=np.int32)[0])
        
        if verbose:
            print("\nÉTAPE 2 - Lecture du nombre de dwells actifs:")
            print(f"  - Bytes lus (hex): {n_active_bytes.hex()}")
            print(f"  - Valeur décodée: {n_active_dwells} dwells actifs")
            print(f"  - Position actuelle dans le fichier: {f.tell()} bytes")
        
        # ÉTAPE 3: Calculer le nombre total de valeurs float32 à lire
        total_floats = n_plans * n_active_dwells
        bytes_to_read = total_floats * 4  # 4 bytes par float32
        
        if verbose:
            print("\nÉTAPE 3 - Préparation pour la lecture des temps:")
            print(f"  - Nombre total de valeurs float32: {total_floats}")
            print(f"  - Bytes à lire: {bytes_to_read}")
            print(f"  - Structure attendue: matrice {n_plans} x {n_active_dwells}")
        
        # ÉTAPE 4: Lire tous les temps d'un coup
        # Position: à partir du byte 8
        remaining_bytes = f.read()
        
        if verbose:
            print("\nÉTAPE 4 - Lecture des données:")
            print(f"  - Bytes restants dans le fichier: {len(remaining_bytes)}")
            print(f"  - Correspond à {len(remaining_bytes) / 4} valeurs float32")
        
        # Convertir en array numpy
        dwell_times_flat = np.frombuffer(remaining_bytes, dtype=np.float32)
        
        if verbose:
            print(f"\nÉTAPE 5 - Conversion en array NumPy:")
            print(f"  - Shape de l'array plat: {dwell_times_flat.shape}")
            print(f"  - Premières 10 valeurs: {dwell_times_flat[:10]}")
            print(f"  - Min/Max global: {np.min(dwell_times_flat):.3f} / {np.max(dwell_times_flat):.3f}")
        
        # ÉTAPE 6: Reshape en matrice 2D
        dwell_times_array = dwell_times_flat.reshape(n_plans, n_active_dwells)
        
        if verbose:
            print(f"\nÉTAPE 6 - Reshape en matrice 2D:")
            print(f"  - Shape finale: {dwell_times_array.shape}")
            print(f"  - Type de données: {dwell_times_array.dtype}")
    
    return n_plans, n_active_dwells, dwell_times_array


def analyze_dwell_times_structure(dwell_times_array, max_plans=5):
    """
    Analyse la structure de l'array des dwell times.
    
    Args:
        dwell_times_array: Array numpy des dwell times
        max_plans: Nombre maximum de plans à analyser en détail
    """
    print("\n" + "="*70)
    print("ANALYSE DE LA STRUCTURE DES DONNÉES")
    print("="*70)
    
    n_plans, n_active_dwells = dwell_times_array.shape
    
    # Statistiques globales
    print("\nSTATISTIQUES GLOBALES:")
    print(f"  - Nombre total de valeurs: {dwell_times_array.size}")
    print(f"  - Valeurs > 0: {np.sum(dwell_times_array > 0)} ({100*np.sum(dwell_times_array > 0)/dwell_times_array.size:.1f}%)")
    print(f"  - Temps total tous plans: {np.sum(dwell_times_array):.2f} s")
    print(f"  - Temps moyen par plan: {np.mean(np.sum(dwell_times_array, axis=1)):.2f} s")
    
    # Analyse par plan
    print(f"\nANALYSE DES {min(max_plans, n_plans)} PREMIERS PLANS:")
    print("Plan | Dwells>0 | % actifs | Temps total | Temps max | Temps moyen")
    print("-"*70)
    
    for plan_id in range(min(max_plans, n_plans)):
        plan_times = dwell_times_array[plan_id]
        non_zero = np.sum(plan_times > 0)
        percent_active = 100 * non_zero / n_active_dwells
        total_time = np.sum(plan_times)
        max_time = np.max(plan_times) if non_zero > 0 else 0
        mean_time = np.mean(plan_times[plan_times > 0]) if non_zero > 0 else 0
        
        print(f"{plan_id:4d} | {non_zero:8d} | {percent_active:8.1f} | {total_time:11.2f} | {max_time:9.2f} | {mean_time:11.2f}")
    
    # Distribution des temps
    print("\nDISTRIBUTION DES TEMPS (tous plans confondus):")
    all_times = dwell_times_array.flatten()
    non_zero_times = all_times[all_times > 0]
    
    if len(non_zero_times) > 0:
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        print("Percentile | Valeur (s)")
        print("-"*25)
        for p in percentiles:
            val = np.percentile(non_zero_times, p)
            label = "Min" if p == 0 else ("Max" if p == 100 else f"{p}%")
            print(f"{label:10s} | {val:10.2f}")


def compare_with_dwellsdata(patient_path, dwell_times_array, plan_id=0):
    """
    Compare les résultats avec ceux obtenus via DwellsData.
    
    Args:
        patient_path: Chemin vers le dossier patient
        dwell_times_array: Array numpy parsé manuellement
        plan_id: Plan à comparer
    """
    print("\n" + "="*70)
    print(f"COMPARAISON AVEC DwellsData (Plan {plan_id})")
    print("="*70)
    
    # Charger via DwellsData
    dwells_data = DwellsData(patient_path, i_plan_selected=plan_id)
    
    # Extraire les temps du plan via notre parsing
    our_times = dwell_times_array[plan_id]
    
    # Extraire les temps via DwellsData
    dwellsdata_times = dwells_data.dwell_times[plan_id]
    
    print(f"\nCOMPARAISON DES ARRAYS:")
    print(f"  - Shape de notre array: {our_times.shape}")
    print(f"  - Shape de DwellsData: {dwellsdata_times.shape}")
    print(f"  - Arrays identiques: {np.array_equal(our_times, dwellsdata_times)}")
    
    if np.array_equal(our_times, dwellsdata_times):
        print("  ✓ Les arrays sont identiques!")
    else:
        print("  ✗ Les arrays diffèrent!")
        diff = np.abs(our_times - dwellsdata_times)
        print(f"  - Différence max: {np.max(diff)}")
        print(f"  - Nombre de différences: {np.sum(diff > 0)}")
    
    # Comparer les temps dans dwells_data
    dwells_data_times = dwells_data.dwells_data[:, -1]
    
    print(f"\nCOMPARAISON AVEC dwells_data (positions + temps):")
    print(f"  - Nombre de dwells dans dwells_data: {len(dwells_data_times)}")
    print(f"  - Temps total dans dwells_data: {np.sum(dwells_data_times):.2f} s")
    print(f"  - Temps total dans notre array: {np.sum(our_times):.2f} s")
    
    # Vérifier que les temps non-zéro correspondent
    our_nonzero = our_times[our_times > 0]
    dwells_nonzero = dwells_data_times[dwells_data_times > 0]
    
    print(f"\nTEMPS NON-ZÉRO:")
    print(f"  - Notre array: {len(our_nonzero)} valeurs")
    print(f"  - dwells_data: {len(dwells_nonzero)} valeurs")
    
    if len(our_nonzero) == len(dwells_nonzero):
        print("  ✓ Même nombre de dwells actifs")
        if np.allclose(sorted(our_nonzero), sorted(dwells_nonzero)):
            print("  ✓ Les temps correspondent (après tri)")
        else:
            print("  ✗ Les temps ne correspondent pas")


def save_arrays(dwell_times_array, output_dir="./numpy_arrays"):
    """
    Sauvegarde les arrays dans différents formats.
    
    Args:
        dwell_times_array: Array numpy à sauvegarder
        output_dir: Dossier de sortie
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*70)
    print("SAUVEGARDE DES ARRAYS")
    print("="*70)
    
    # Format NumPy (.npy)
    npy_file = os.path.join(output_dir, "dwell_times.npy")
    np.save(npy_file, dwell_times_array)
    print(f"\n✓ Sauvé en format NumPy: {npy_file}")
    print(f"  - Pour charger: array = np.load('{npy_file}')")
    
    # Format texte (.txt)
    txt_file = os.path.join(output_dir, "dwell_times.txt")
    np.savetxt(txt_file, dwell_times_array, fmt='%.6f')
    print(f"\n✓ Sauvé en format texte: {txt_file}")
    print(f"  - Format: une ligne par plan")
    
    # Format CSV avec en-têtes
    csv_file = os.path.join(output_dir, "dwell_times.csv")
    with open(csv_file, 'w') as f:
        # En-tête
        f.write("plan_id," + ",".join([f"dwell_{i}" for i in range(dwell_times_array.shape[1])]) + "\n")
        # Données
        for i, row in enumerate(dwell_times_array):
            f.write(f"{i}," + ",".join([f"{v:.6f}" for v in row]) + "\n")
    print(f"\n✓ Sauvé en format CSV: {csv_file}")
    print(f"  - Avec en-têtes pour chaque dwell")
    
    # Sauver aussi un résumé
    summary_file = os.path.join(output_dir, "dwell_times_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("RÉSUMÉ DES DWELL TIMES\n")
        f.write("="*50 + "\n\n")
        f.write(f"Nombre de plans: {dwell_times_array.shape[0]}\n")
        f.write(f"Nombre de dwells actifs par plan: {dwell_times_array.shape[1]}\n")
        f.write(f"Temps total tous plans: {np.sum(dwell_times_array):.2f} s\n\n")
        
        f.write("Statistiques par plan:\n")
        f.write("Plan | Dwells>0 | Temps total | Temps max\n")
        f.write("-"*45 + "\n")
        
        for i in range(dwell_times_array.shape[0]):
            plan_times = dwell_times_array[i]
            non_zero = np.sum(plan_times > 0)
            total = np.sum(plan_times)
            max_time = np.max(plan_times) if non_zero > 0 else 0
            f.write(f"{i:4d} | {non_zero:8d} | {total:11.2f} | {max_time:9.2f}\n")
    
    print(f"\n✓ Sauvé le résumé: {summary_file}")


# Script principal
if __name__ == "__main__":
    # Chemins
    patient_path = "/Users/xavierarata/gMCO_robust"
    dwell_times_file = "/Users/xavierarata/gMCO_robust/gMCO_OUTPUT_FILES/dwell_times.bin"
    
    # 1. Parser le fichier binaire avec explications détaillées
    n_plans, n_active_dwells, dwell_times_array = parse_dwell_times_binary(dwell_times_file, verbose=True)
    
    # 2. Analyser la structure des données
    analyze_dwell_times_structure(dwell_times_array, max_plans=10)
    
    # 3. Comparer avec DwellsData
    compare_with_dwellsdata(patient_path, dwell_times_array, plan_id=3)
    
    # 4. Sauvegarder les arrays
    save_arrays(dwell_times_array)
    
    # 5. Exemple d'utilisation
    print("\n" + "="*70)
    print("EXEMPLE D'UTILISATION")
    print("="*70)
    
    print("\n# Pour charger et utiliser l'array:")
    print("import numpy as np")
    print("dwell_times = np.load('numpy_arrays/dwell_times.npy')")
    print(f"# Shape: {dwell_times_array.shape}")
    print()
    print("# Accéder aux temps du plan 3:")
    print("plan_3_times = dwell_times[3]")
    print(f"# {len(dwell_times_array[3])} dwells, temps total: {np.sum(dwell_times_array[3]):.2f} s")
    print()
    print("# Trouver les dwells actifs du plan 3:")
    print("active_indices = np.where(plan_3_times > 0)[0]")
    print("active_times = plan_3_times[active_indices]")
    print(f"# {np.sum(dwell_times_array[3] > 0)} dwells actifs")