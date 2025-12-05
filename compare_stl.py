import trimesh
import numpy as np
import matplotlib.pyplot as plt

def get_profile(mesh, num_slices=200):
    """
    Tworzy profil promieniowy wzdłuż osi Z.
    Dzieli wysokość na num_slices i dla każdej wysokości mierzy promień od środka X.
    Zwraca: (z_values, radii)
    """
    z_min, z_max = mesh.bounds[:,2]
    zs = np.linspace(z_min, z_max, num_slices)
    radii = []

    verts = mesh.vertices
    x_center = (mesh.bounds[0,0] + mesh.bounds[1,0]) / 2.0

    for z in zs:
        delta = (z_max - z_min)/num_slices
        mask = np.abs(verts[:,2] - z) < delta
        slice_verts = verts[mask]
        if len(slice_verts) == 0:
            radii.append(0)
            continue
        x_vals = slice_verts[:,0]
        radius = np.max(np.abs(x_vals - x_center))
        radii.append(radius)
    return zs, np.array(radii)

def compare_stl_profile(file1, file2, num_slices=200):
    mesh1 = trimesh.load(file1)
    mesh2 = trimesh.load(file2)

    vol1, vol2 = mesh1.volume, mesh2.volume
    vol_diff = abs(vol1 - vol2)/max(vol1, vol2)*100
    print(f"Objętość: {vol1:.2f} mm³ vs {vol2:.2f} mm³ (różnica: {vol_diff:.2f}%)")

    zs1, r1 = get_profile(mesh1, num_slices)
    zs2, r2 = get_profile(mesh2, num_slices)

    # Synchronizacja wysokości
    z_min = max(zs1.min(), zs2.min())
    z_max = min(zs1.max(), zs2.max())
    zs_common = np.linspace(z_min, z_max, num_slices)

    r1_interp = np.interp(zs_common, zs1, r1)
    r2_interp = np.interp(zs_common, zs2, r2)

    mean_diff = np.mean(np.abs(r1_interp - r2_interp))
    max_diff = np.max(np.abs(r1_interp - r2_interp))
    print(f"Średnia różnica promieni: {mean_diff:.3f} mm")
    print(f"Maksymalna różnica promieni: {max_diff:.3f} mm")

    tol = 1.0
    score = 100 * np.clip(1 - mean_diff/tol, 0, 1)
    print(f"Szacowana zgodność: {score:.2f}%")

    plt.figure(figsize=(6,4))
    plt.plot(zs_common, r1_interp, label="Model 1")
    plt.plot(zs_common, r2_interp, label="Model 2")
    plt.xlabel("Z [mm]")
    plt.ylabel("Promień [mm]")
    plt.title("Porównanie profili wzdłuż osi Z")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_stl_profile("butelka_scaled2.stl", "butelka2.stl", num_slices=200)