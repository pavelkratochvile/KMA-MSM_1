import random
import math
import matplotlib.pyplot as plt
import numpy as np

def uvnitr_oblasti(x:float, y:float) -> bool:
    return (0 < x < 1) and (0 < y < x)

def f(x:float, y:float) -> float:
    if not uvnitr_oblasti(x, y):
        return 0
    return 6*(x -y)

def generuj_sdruzene(n_vzorku: int, x_min: float, x_max: float, y_min: float, y_max: float, M: float):
    data_x = []
    data_y = []

    while len(data_x) < n_vzorku:
        x_kandidat = random.uniform(x_min, x_max)
        y_kandidat = random.uniform(y_min, y_max)
        
        if not uvnitr_oblasti(x_kandidat, y_kandidat):
            continue

        u = random.uniform(0, M)
        
        if u <= f(x_kandidat, y_kandidat):
            data_x.append(x_kandidat)
            data_y.append(y_kandidat)

    return data_x, data_y

if __name__ == "__main__":
    n = 1000000
    # Obdélník ohraničující oblast a konstanta M (horní odhad maxima f(x, y))
    X, Y = generuj_sdruzene(n, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, M=6.0)

    # -------------------------------- BODOVÉ ODHADY --------------------------------

    # Výběrové průměry (odhad středních hodnot E[X], E[Y])
    prumer_x = sum(X) / n
    prumer_y = sum(Y) / n

    # Výběrové rozptyly s^2 (děleno n - 1 pro nestranný odhad)
    var_x = sum((x - prumer_x) ** 2 for x in X) / (n - 1)
    var_y = sum((y - prumer_y) ** 2 for y in Y) / (n - 1)

    # Výběrová kovariance cov(X, Y)
    cov_xy = sum((x - prumer_x) * (y - prumer_y) for x, y in zip(X, Y)) / (n - 1)

    print(f"Počet vzorků: {n}")
    print(f"Odhad E[X]: {prumer_x:.4f}")
    print(f"Odhad E[Y]: {prumer_y:.4f}")
    print(f"Odhad Var(X): {var_x:.4f}")
    print(f"Odhad Var(Y): {var_y:.4f}")
    print(f"Odhad Cov(X, Y): {cov_xy:.4f}")

    #------------------------------- INTERVALOVÉ ODHADY --------------------------------

    # 95% intervalový odhad pro střední hodnoty
    z = 1.69 

    s_x = math.sqrt(var_x)
    s_y = math.sqrt(var_y)

    delta_x = z*(s_x/math.sqrt(n))
    delta_y = z*(s_y/math.sqrt(n))

    print(f"Intervalový odhad E[X] je {prumer_x:.4f} +- {delta_x:.4f}")
    print(f"Intervalový odhad E[Y] je {prumer_y:.4f} +- {delta_y:.4f}")
        
    # ------------------------------- GRAFICKÉ ZOBRAZENÍ --------------------------------
    x_arr = np.array(X)
    y_arr = np.array(Y)

    # --- Výpočet ECDF ---
    # Seřazená data představují body na ose X, osa Y roste po krocích 1/n
    x_sorted = np.sort(x_arr)
    y_sorted = np.sort(y_arr)
    ecdf_y_vals = np.linspace(1 / n, 1.0, n)

    # Podpůrná osa pro teoretické křivky
    t = np.linspace(0, 1, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Histogram X vs teoretická hustota f_X(x) = 3x^2
    axes[0, 0].hist(x_arr, bins=40, density=True, alpha=0.6, color="royalblue", edgecolor="black", label="Histogram dat")
    axes[0, 0].plot(t, 3 * t**2, "r-", linewidth=2.5, label="Teorie: $f_X(x) = 3x^2$")
    axes[0, 0].set_title("Histogram veličiny X")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Hustota")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    # 2. ECDF X vs teoretická CDF F_X(x) = x^3
    axes[0, 1].step(x_sorted, ecdf_y_vals, where="post", color="royalblue", label="Empirická CDF")
    axes[0, 1].plot(t, t**3, "r--", linewidth=2.5, label="Teorie: $F_X(x) = x^3$")
    axes[0, 1].set_title("Empirická distribuční funkce X")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("$F(x)$")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    # 3. Histogram Y vs teoretická hustota f_Y(y) = 3(1-y)^2
    axes[1, 0].hist(y_arr, bins=40, density=True, alpha=0.6, color="darkorange", edgecolor="black", label="Histogram dat")
    axes[1, 0].plot(t, 3 * (1 - t)**2, "r-", linewidth=2.5, label="Teorie: $f_Y(y) = 3(1-y)^2$")
    axes[1, 0].set_title("Histogram veličiny Y")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("Hustota")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    # 4. ECDF Y vs teoretická CDF F_Y(y) = 1 - (1-y)^3
    axes[1, 1].step(y_sorted, ecdf_y_vals, where="post", color="darkorange", label="Empirická CDF")
    axes[1, 1].plot(t, 1 - (1 - t)**3, "r--", linewidth=2.5, label="Teorie: $F_Y(y) = 1 - (1-y)^3$")
    axes[1, 1].set_title("Empirická distribuční funkce Y")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("$F(y)$")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()