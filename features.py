import numpy as np

def get_angle(p1, p2, p3):
    """
    Calculates the 3D angle at vertex p2 between vectors p2->p1 and p2->p3.
    Returns the angle in radians.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def get_hand_scale(landmarks):
    """
    Computes the size of the hand. We define scale as the distance
    between the wrist (0) and the middle finger MCP joint (9).
    """
    return np.linalg.norm(landmarks[9] - landmarks[0])

def get_tip_distances(landmarks, scale):
    """
    Computes the Euclidean distance between all pairs of finger tips
    (Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20),
    normalized by the hand scale.
    Returns 10 distance features.
    """
    tips = [4, 8, 12, 16, 20]
    distances = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            t1 = landmarks[tips[i]]
            t2 = landmarks[tips[j]]
            dist = np.linalg.norm(t1 - t2)
            # Normalize by scale to ensure scale invariance
            distances.append(dist / scale if scale > 0 else dist)
    return distances

def extract_hand_features(landmarks_flat, include_coords=True):
    """
    Extracts geometric features from a flat 63-landmark list (x0,y0,z0 ... x20,y20,z20).
    Features calculated:
    - 15 joint angles (MCP base angle, PIP angle, DIP angle for each of the 5 fingers)
    - 10 normalized fingertip distances
    - (Optional) 63 scale-normalized relative coordinates
    
    Returns a 1D numpy array of features.
    """
    # Reshape to (21, 3)
    landmarks = np.array(landmarks_flat).reshape(21, 3)
    
    # 1. Compute scale
    scale = get_hand_scale(landmarks)
    
    # 2. Compute 15 joint angles
    angles = []
    finger_triples = [
        # Thumb: wrist->joint1->joint2, joint1->joint2->joint3, joint2->joint3->tip
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        # Index
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        # Middle
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        # Ring
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        # Pinky
        (0, 17, 18), (17, 18, 19), (18, 19, 20)
    ]
    for p1_idx, p2_idx, p3_idx in finger_triples:
        angles.append(get_angle(landmarks[p1_idx], landmarks[p2_idx], landmarks[p3_idx]))
        
    # 3. Compute 10 normalized tip-to-tip distances
    tip_dists = get_tip_distances(landmarks, scale)
    
    # Combine features
    geom_features = angles + tip_dists
    
    if include_coords:
        # Subtract wrist (landmark 0) to make coordinate representation translation-invariant
        wrist = landmarks[0]
        rel_coords = landmarks - wrist
        # Normalize relative coordinates by scale
        norm_coords = rel_coords / scale if scale > 0 else rel_coords
        flat_norm_coords = norm_coords.flatten()
        return np.concatenate([geom_features, flat_norm_coords])
    else:
        return np.array(geom_features)
