import numpy as np
from datasets.scannetpp_constants import SCANNETPP_CLASS_LABLES, ARCHITECTURAL_ELEMENTS

"""

Mean AP: 0.014
Mean AP25: 0.027
Mean AP50: 0.052

Mean AP: 0.006
Mean AP25: 0.013
Mean AP50: 0.023

"""

# Path to your input file
file_path = "/work/courses/3dv/20/OpenArchitect3D/Mask3D/results.txt"

# Initialize lists to store the scores
ap_scores = []
ap25_scores = []
ap50_scores = []

# Read the file and parse the values
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or 'average' in line.lower():
            continue  # skip empty lines, comments, and average line

        parts = line.split(":")
        #print(parts)
        
        #if len(parts) < 4:
        #    continue  # invalid line

        #label = parts[0]
        #scores = parts[1].split("            ")#[1:]
        try:
            label = parts[0].strip()
            scores = parts[1].split("            ")
            if len(scores) == 1:
                scores = parts[1].split("          ")
        except:
            print("## Found error ##")
            #print(label)
            #print(parts)
        #if label not in ARCHITECTURAL_ELEMENTS:
        scores = scores[1:]
        print(label, scores)
        try:
            ap = float(scores[0]) if scores[0].lower() != 'nan' else np.nan
            ap25 = float(scores[1]) if scores[1].lower() != 'nan' else np.nan
            ap50 = float(scores[2]) if scores[2].lower() != 'nan' else np.nan

            ap_scores.append(ap)
            ap25_scores.append(ap25)
            ap50_scores.append(ap50)

        except ValueError:
            continue  # skip invalid lines

# Convert to numpy arrays for easier handling
ap_scores = np.array(ap_scores)
ap25_scores = np.array(ap25_scores)
ap50_scores = np.array(ap50_scores)

# Compute averages, ignoring NaNs
mean_ap = np.nanmean(ap_scores)
mean_ap25 = np.nanmean(ap25_scores)
mean_ap50 = np.nanmean(ap50_scores)

# Print results
print(f"Mean AP: {mean_ap:.3f}")
print(f"Mean AP25: {mean_ap25:.3f}")
print(f"Mean AP50: {mean_ap50:.3f}")