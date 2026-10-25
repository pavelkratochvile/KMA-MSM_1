"""
Úloha 1 – KMA/MSM: generování a charakteristiky dvourozměrné náhodné veličiny
Autor:  Pavel Kratochvíle, Michal Malík
Datum:  24. 10. 2026

Popis:
    Sdružená hustota f(x, y) = 6 (x - y) na oblasti 0 < y < x < 1.
    Program generuje náhodný výběr metodou rejection sampling, odhaduje
    střední hodnoty, rozptyly a kovarianci, počítá intervalové odhady
    středních hodnot a porovnává data s teoretickými funkcemi.
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Přehled: která část kódu řeší který krok zadání
#   ÚKOL 1 – konstanta c ............ funkce f (c = 6 spočteno ručně z normování)
#   ÚKOL 2 – marginály f_X, f_Y, F_X, F_Y ... vykresli_marginaly() (+ křivky ve vykresli())
#   ÚKOL 3 – teoretické charakteristiky ..... slovník TEORIE
#   ÚKOL 4 – generování dat ................. generuj_sdruzene()
#   ÚKOL 5 – odhady a intervaly ............. bodove_odhady(), interval_stredni_hodnoty()
#   ÚKOL 6 – porovnání s teorií ............. vykresli(), ks_statistika()
#   ÚKOL 7 – závislost na rozsahu výběru .... cyklus přes n v main()

# hodnota z normálního rozdělení pro 95% interval spolehlivosti
Z_975 = 1.96

# ÚKOL 3: teoretické hodnoty středních hodnot, rozptylů a kovariance
# (spočtené ručně z marginálních hustot, slouží k porovnání s odhady z dat)
TEORIE = {
    "E[X]": 3 / 4,
    "E[Y]": 1 / 4,
    "Var(X)": 3 / 80,
    "Var(Y)": 3 / 80,
    "Cov(X,Y)": 1 / 80,
}


# ÚKOL 1 / 4: oblast D = {0 < y < x < 1}, mimo ni je hustota nulová
def uvnitr_oblasti(x: float, y: float) -> bool:
    return (0 < x < 1) and (0 < y < x)


# ÚKOL 1: sdružená hustota f(x, y) = c (x - y); konstanta c = 6 vyšla
# z podmínky normování (integrál f přes oblast D se rovná 1)
def f(x: float, y: float) -> float:
    if not uvnitr_oblasti(x, y):
        return 0
    return 6*(x - y)


# ÚKOL 4: generování dat metodou rejection sampling
# (M musí být alespoň maximum hustoty, zde max f = 6 v bodě (1, 0))
def generuj_sdruzene(n_vzorku: int, x_min: float, x_max: float, y_min: float, y_max: float, M: float):
    data_x = []
    data_y = []

    while len(data_x) < n_vzorku:
        # kandidát rovnoměrně z obdélníku obsahujícího oblast D
        x_kandidat = random.uniform(x_min, x_max)
        y_kandidat = random.uniform(y_min, y_max)

        # kandidáty mimo oblast rovnou zahodíme (hustota je tam nulová)
        if not uvnitr_oblasti(x_kandidat, y_kandidat):
            continue

        # náhodná výška pod konstantou M; bod přijmeme, leží-li pod grafem hustoty
        u = random.uniform(0, M)

        if u <= f(x_kandidat, y_kandidat):
            data_x.append(x_kandidat)
            data_y.append(y_kandidat)

    return data_x, data_y


# ÚKOL 5: bodové odhady (výběrový průměr, rozptyl a kovariance)
def bodove_odhady(X, Y) -> dict:
    n = len(X)
    prumer_x = sum(X) / n
    prumer_y = sum(Y) / n
    # rozptyl a kovariance dělené n - 1 (nestranné odhady)
    var_x = sum((x - prumer_x) ** 2 for x in X) / (n - 1)
    var_y = sum((y - prumer_y) ** 2 for y in Y) / (n - 1)
    cov_xy = sum((x - prumer_x) * (y - prumer_y)
                 for x, y in zip(X, Y)) / (n - 1)
    return {"E[X]": prumer_x, "E[Y]": prumer_y,
            "Var(X)": var_x, "Var(Y)": var_y, "Cov(X,Y)": cov_xy}


# ÚKOL 5: 95% intervalový odhad střední hodnoty ve tvaru průměr ± z * s / sqrt(n)
def interval_stredni_hodnoty(prumer: float, var: float, n: int):
    # delta = poloviční šířka intervalu (z * směrodatná chyba průměru)
    delta = Z_975 * math.sqrt(var) / math.sqrt(n)
    return prumer - delta, prumer + delta, delta


# ÚKOL 6: Kolmogorovova-Smirnovova statistika D_n = max |F_n(x) - F(x)|,
# číselná míra shody empirické a teoretické distribuční funkce
def ks_statistika(data, teoreticka_cdf) -> float:
    n = len(data)
    x_sorted = np.sort(np.array(data))
    F = teoreticka_cdf(x_sorted)
    i = np.arange(1, n + 1)
    # empirická CDF ve skocích nabývá hodnot (i-1)/n a i/n, porovnáme obě strany skoku
    d_plus = np.max(i / n - F)
    d_minus = np.max(F - (i - 1) / n)
    return float(max(d_plus, d_minus))


# ------------------------------- GRAFICKÉ ZOBRAZENÍ --------------------------------
# ÚKOL 2: graf teoretických marginálních hustot a distribučních funkcí
# f_X(x) = 3x^2,  F_X(x) = x^3,  f_Y(y) = 3(1-y)^2,  F_Y(y) = 1 - (1-y)^3
# (vzorce vyšly integrací sdružené hustoty přes druhou proměnnou)
def vykresli_marginaly():
    # osa přesahuje oblast (0, 1), aby bylo vidět, že hustota je mimo ni nulová
    # a distribuční funkce je vlevo 0 a vpravo 1
    t = np.linspace(-0.25, 1.25, 600)
    uvnitr = (t > 0) & (t < 1)

    f_x = np.where(uvnitr, 3 * t ** 2, 0.0)
    F_x = np.clip(t, 0, 1) ** 3
    f_y = np.where(uvnitr, 3 * (1 - t) ** 2, 0.0)
    F_y = 1 - (1 - np.clip(t, 0, 1)) ** 3

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].plot(t, f_x, color="royalblue", lw=2.5, label="$f_X(x)=3x^2$")
    ax[0, 0].set(title="Marginální hustota X", xlabel="x", ylabel="$f_X(x)$")

    ax[0, 1].plot(t, F_x, color="royalblue", lw=2.5, label="$F_X(x)=x^3$")
    ax[0, 1].set(title="Marginální distribuční funkce X", xlabel="x",
                 ylabel="$F_X(x)$")

    ax[1, 0].plot(t, f_y, color="darkorange", lw=2.5,
                  label="$f_Y(y)=3(1-y)^2$")
    ax[1, 0].set(title="Marginální hustota Y", xlabel="y", ylabel="$f_Y(y)$")

    ax[1, 1].plot(t, F_y, color="darkorange", lw=2.5,
                  label="$F_Y(y)=1-(1-y)^3$")
    ax[1, 1].set(title="Marginální distribuční funkce Y", xlabel="y",
                 ylabel="$F_Y(y)$")

    for a in ax.flat:
        a.grid(True, linestyle="--", alpha=0.5)
        a.legend()
    plt.tight_layout()
    plt.show()


# ÚKOL 2 + 6: histogramy a empirické distribuční funkce z dat (ÚKOL 6)
# vykreslené spolu s teoretickými marginálami (ÚKOL 2, červené křivky)
def vykresli(X, Y):
    n = len(X)
    x_arr, y_arr = np.array(X), np.array(Y)
    # hodnoty empirické distribuční funkce: po seřazení dat roste po krocích 1/n
    ecdf = np.linspace(1 / n, 1.0, n)
    t = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # histogram X vs. teoretická hustota f_X(x) = 3x^2
    ax[0, 0].hist(x_arr, bins=25, density=True, alpha=0.6,
                  color="royalblue", edgecolor="black", label="Histogram dat")
    ax[0, 0].plot(t, 3 * t ** 2, "r-", lw=2.5, label="Teorie: $f_X(x)=3x^2$")
    ax[0, 0].set(title="Histogram veličiny X", xlabel="x", ylabel="Hustota")

    # ECDF X vs. teoretická distribuční funkce F_X(x) = x^3
    ax[0, 1].step(np.sort(x_arr), ecdf, where="post", color="royalblue",
                  label="Empirická CDF")
    ax[0, 1].plot(t, t ** 3, "r--", lw=2.5, label="Teorie: $F_X(x)=x^3$")
    ax[0, 1].set(title="Empirická distribuční funkce X", xlabel="x",
                 ylabel="$F(x)$")

    # histogram Y vs. teoretická hustota f_Y(y) = 3(1-y)^2
    ax[1, 0].hist(y_arr, bins=25, density=True, alpha=0.6,
                  color="darkorange", edgecolor="black", label="Histogram dat")
    ax[1, 0].plot(t, 3 * (1 - t) ** 2, "r-", lw=2.5,
                  label="Teorie: $f_Y(y)=3(1-y)^2$")
    ax[1, 0].set(title="Histogram veličiny Y", xlabel="y", ylabel="Hustota")

    # ECDF Y vs. teoretická distribuční funkce F_Y(y) = 1 - (1-y)^3
    ax[1, 1].step(np.sort(y_arr), ecdf, where="post", color="darkorange",
                  label="Empirická CDF")
    ax[1, 1].plot(t, 1 - (1 - t) ** 3, "r--", lw=2.5,
                  label="Teorie: $F_Y(y)=1-(1-y)^3$")
    ax[1, 1].set(title="Empirická distribuční funkce Y", xlabel="y",
                 ylabel="$F(y)$")

    for a in ax.flat:
        a.grid(True, linestyle="--", alpha=0.5)
        a.legend()
    plt.tight_layout()
    plt.show()


def main():
    random.seed(42)  # reprodukovatelnost výsledků
    n = 1000

    # --------- ÚKOL 2: graf teoretických marginálních funkcí (před generováním dat) ---------
    vykresli_marginaly()

    # --------- ÚKOL 4: vygenerování náhodného výběru -------------
    X, Y = generuj_sdruzene(n, 0.0, 1.0, 0.0, 1.0, M=6.0)

    # --------- ÚKOL 5: bodové odhady a jejich porovnání s teorií (ÚKOL 3) -------------
    odhady = bodove_odhady(X, Y)
    print(f"Počet vzorků: {n}")
    for nazev, hodnota in odhady.items():
        print(f"{nazev:9s} odhad = {hodnota:.4f}   teorie = {TEORIE[nazev]:.4f}")

    # --------- ÚKOL 5: intervalové odhady středních hodnot + kontrola pokrytí teorie -------------
    for nazev, var_nazev in (("E[X]", "Var(X)"), ("E[Y]", "Var(Y)")):
        dolni, horni, _ = interval_stredni_hodnoty(odhady[nazev],
                                                   odhady[var_nazev], n)
        pokryva = dolni <= TEORIE[nazev] <= horni
        print(f"95% interval {nazev}: ({dolni:.4f}; {horni:.4f}), "
              f"pokrývá teorii: {pokryva}")

    # --------- ÚKOL 6: číselná míra shody ECDF s teoretickou CDF (graf je na konci main) -------------
    d_x = ks_statistika(X, lambda x: x ** 3)
    d_y = ks_statistika(Y, lambda y: 1 - (1 - y) ** 3)
    print(f"KS statistika D_n pro X: {d_x:.4f}")
    print(f"KS statistika D_n pro Y: {d_y:.4f}")

    # --------- ÚKOL 7: závislost přesnosti odhadů na rozsahu výběru ------------------
    # pro každé n se vygeneruje nový výběr a sleduje se chyba odhadu,
    # šířka intervalu (očekávaný pokles ~ 1/sqrt(n)) a míra shody D_n
    print("\nZávislost na rozsahu výběru n")
    print(f"{'n':>6} {'E[X]':>8} {'chyba E[X]':>11} {'šířka E[X]':>11} "
          f"{'E[Y]':>8} {'chyba E[Y]':>11} {'šířka E[Y]':>11} "
          f"{'D_n(X)':>8} {'D_n(Y)':>8}")
    for n_i in [250, 500, 1000, 5000]:
        Xi, Yi = generuj_sdruzene(n_i, 0.0, 1.0, 0.0, 1.0, M=6.0)
        odh = bodove_odhady(Xi, Yi)
        _, _, delta_x = interval_stredni_hodnoty(odh["E[X]"], odh["Var(X)"], n_i)
        _, _, delta_y = interval_stredni_hodnoty(odh["E[Y]"], odh["Var(Y)"], n_i)
        chyba_x = abs(odh["E[X]"] - TEORIE["E[X]"])
        chyba_y = abs(odh["E[Y]"] - TEORIE["E[Y]"])
        dn_x = ks_statistika(Xi, lambda x: x ** 3)
        dn_y = ks_statistika(Yi, lambda y: 1 - (1 - y) ** 3)
        # šířka intervalu = 2 * delta (celá šířka, ne poloviční)
        print(f"{n_i:>6} {odh['E[X]']:>8.4f} {chyba_x:>11.4f} {2 * delta_x:>11.4f} "
              f"{odh['E[Y]']:>8.4f} {chyba_y:>11.4f} {2 * delta_y:>11.4f} "
              f"{dn_x:>8.4f} {dn_y:>8.4f}")

    # --------- ÚKOL 2 + 6: grafy marginál, histogramů a ECDF ---------
    vykresli(X, Y)


if __name__ == "__main__":
    main()
