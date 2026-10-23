"""Band-Likelihood Gradient Boosting (BLGB): modelo original del proyecto.

Idea
----
La clase positiva se define por la pertenencia de una variable continua a una banda,
``y = 1{L ≤ Q(t+h) ≤ U}``. Los clasificadores convencionales solo ven la etiqueta binaria
y descartan la información de *cuán lejos* de la banda quedó cada día. BLGB modela la
distribución condicional del cambio logarítmico del caudal,

    z = log(Q(t+h) / Q_ref),    z | x ~ N(μ(x), σ(x)²),

donde ``Q_ref`` es un caudal observado en el instante de la predicción. La banda,
transformada al mismo espacio, depende de cada observación: ``l = log(L/Q_ref)``,
``u = log(U/Q_ref)``. La probabilidad de la clase positiva es

    p(x) = Φ((u − μ)/σ) − Φ((l − μ)/σ).

μ y s = log σ son dos ensambles aditivos de árboles de regresión que se ajustan por
boosting de segundo orden sobre una pérdida híbrida:

    ℒ = (1/n) Σ_i [ −y_i log p_i − (1 − y_i) log(1 − p_i) ]  +  λ (1/n) Σ_i [ ½ r_i² + s_i ],

con ``r_i = (z_i − μ_i)/σ_i``. El primer término es la log-verosimilitud del evento de
interés (la tarea); el segundo, la log-verosimilitud gaussiana del caudal futuro, que
aporta supervisión continua. Para λ > 0 la suma es una regla de puntuación
estrictamente propia (capítulo 5 del libro), de modo que su minimizador poblacional es
la distribución condicional verdadera.

Implementación
--------------
Todo el núcleo está programado con numpy: discretización por cuantiles, construcción de
árboles por niveles con histogramas de gradiente y hessiano, ganancia de segundo orden,
búsqueda lineal de Armijo que garantiza el descenso monótono de la pérdida de
entrenamiento, y parada temprana con un bloque de validación cronológico.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

_SQRT2PI = np.sqrt(2.0 * np.pi)
_EPS = 1e-7


def _phi(x):
    return np.exp(-0.5 * x * x) / _SQRT2PI


# ---------------------------------------------------------------------------
# Árbol de regresión por histogramas (segundo orden)
# ---------------------------------------------------------------------------
class _Arbol:
    """Árbol binario guardado en arreglos planos.

    Atributos por nodo: ``feat`` (−1 en las hojas), ``umbral`` (índice de bin: van a la
    izquierda los bins ≤ umbral), ``izq``, ``der`` y ``valor``.
    """

    __slots__ = ("feat", "umbral", "izq", "der", "valor")

    def __init__(self):
        self.feat, self.umbral, self.izq, self.der, self.valor = [], [], [], [], []

    def _nuevo(self, valor: float) -> int:
        for lista, v in ((self.feat, -1), (self.umbral, 0), (self.izq, -1), (self.der, -1),
                         (self.valor, valor)):
            lista.append(v)
        return len(self.valor) - 1

    def congelar(self):
        self.feat = np.asarray(self.feat, dtype=np.int32)
        self.umbral = np.asarray(self.umbral, dtype=np.int32)
        self.izq = np.asarray(self.izq, dtype=np.int32)
        self.der = np.asarray(self.der, dtype=np.int32)
        self.valor = np.asarray(self.valor, dtype=np.float64)
        return self

    def predecir(self, Xb: np.ndarray) -> np.ndarray:
        nodo = np.zeros(len(Xb), dtype=np.int32)
        activos = self.feat[nodo] >= 0
        while activos.any():
            i = np.flatnonzero(activos)
            n = nodo[i]
            va_izq = Xb[i, self.feat[n]] <= self.umbral[n]
            nodo[i] = np.where(va_izq, self.izq[n], self.der[n])
            activos = self.feat[nodo] >= 0
        return self.valor[nodo]


def construir_arbol(Xb: np.ndarray, g: np.ndarray, h: np.ndarray, n_bins: int, profundidad: int,
                    min_hess: float, reg_lambda: float, gamma: float,
                    columnas: np.ndarray) -> _Arbol:
    """Crece un árbol por niveles maximizando la ganancia de segundo orden.

    Para un nodo con sumas (G, H) dividido en (G_L, H_L) y (G_R, H_R):

        ganancia = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ,

    y el valor de cada hoja es −G/(H+λ), el paso de Newton regularizado. Los histogramas de
    todos los nodos de un nivel se calculan con una sola llamada a ``np.bincount``.

    Complejidad por nivel: O(n·k + m·k·B), con k columnas, B bins y m nodos del nivel.
    """
    n = len(g)
    k = len(columnas)
    sub = Xb[:, columnas].astype(np.int64)            # (n, k)
    offs = (np.arange(k, dtype=np.int64) * n_bins)[None, :]
    arbol = _Arbol()
    raiz = arbol._nuevo(-g.sum() / (h.sum() + reg_lambda))
    nodo_de = np.zeros(n, dtype=np.int64)              # nodo actual de cada observación
    frontera = [raiz]
    for _ in range(profundidad):
        if not frontera:
            break
        pos = {nd: j for j, nd in enumerate(frontera)}
        local = np.full(len(arbol.valor), -1, dtype=np.int64)
        for nd, j in pos.items():
            local[nd] = j
        loc = local[nodo_de]
        vivos = loc >= 0
        if not vivos.any():
            break
        m = len(frontera)
        idx = (loc[vivos, None] * (k * n_bins) + offs + sub[vivos]).ravel()
        tam = m * k * n_bins
        G = np.bincount(idx, weights=np.repeat(g[vivos], k), minlength=tam).reshape(m, k, n_bins)
        H = np.bincount(idx, weights=np.repeat(h[vivos], k), minlength=tam).reshape(m, k, n_bins)
        GL, HL = np.cumsum(G, axis=2), np.cumsum(H, axis=2)
        Gt, Ht = GL[:, :, -1:], HL[:, :, -1:]
        GR, HR = Gt - GL, Ht - HL
        with np.errstate(divide="ignore", invalid="ignore"):
            ganancia = 0.5 * (GL ** 2 / (HL + reg_lambda) + GR ** 2 / (HR + reg_lambda)
                              - Gt ** 2 / (Ht + reg_lambda)) - gamma
        ganancia[(HL < min_hess) | (HR < min_hess)] = -np.inf
        ganancia[:, :, -1] = -np.inf                   # el último bin no separa nada
        plano = ganancia.reshape(m, -1)
        mejor = plano.argmax(axis=1)
        nueva_frontera = []
        for j, nd in enumerate(frontera):
            gmax = plano[j, mejor[j]]
            if not np.isfinite(gmax) or gmax <= 0:
                continue
            f_loc, b = divmod(int(mejor[j]), n_bins)
            gl, hl = GL[j, f_loc, b], HL[j, f_loc, b]
            gr, hr = GR[j, f_loc, b], HR[j, f_loc, b]
            iz = arbol._nuevo(-gl / (hl + reg_lambda))
            de = arbol._nuevo(-gr / (hr + reg_lambda))
            arbol.feat[nd], arbol.umbral[nd] = int(columnas[f_loc]), b
            arbol.izq[nd], arbol.der[nd] = iz, de
            en_nodo = nodo_de == nd
            a_izq = en_nodo & (Xb[:, columnas[f_loc]] <= b)
            nodo_de[en_nodo] = de
            nodo_de[a_izq] = iz
            nueva_frontera += [iz, de]
        frontera = nueva_frontera
    return arbol.congelar()


# ---------------------------------------------------------------------------
# Pérdida híbrida, gradientes y hessianos
# ---------------------------------------------------------------------------
def probabilidad_banda(mu, s, l, u):
    """p = Φ((u−μ)/σ) − Φ((l−μ)/σ) y los términos intermedios."""
    sig = np.exp(s)
    a, b = (l - mu) / sig, (u - mu) / sig
    p = np.clip(ndtr(b) - ndtr(a), _EPS, 1 - _EPS)
    return p, a, b, sig


def perdida_hibrida(mu, s, l, u, y, z, lam):
    """Pérdida media: entropía cruzada de la banda + λ · NLL gaussiana del cambio logarítmico."""
    p, _, _, sig = probabilidad_banda(mu, s, l, u)
    bce = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    if z is None or lam == 0:
        return float(bce.mean())
    r = (z - mu) / sig
    return float(bce.mean() + lam * (0.5 * r * r + s).mean())


def gradientes(mu, s, l, u, y, z, lam):
    """Gradientes y hessianos (Gauss-Newton) respecto de μ y s = log σ.

    Término de banda, con dp/dμ = (φ(a) − φ(b))/σ y dp/ds = aφ(a) − bφ(b):
        g = (p − y)/(p(1 − p)) · dp/dθ,   h = (dp/dθ)²/(p(1 − p)).
    Término gaussiano (información de Fisher):
        g_μ = −(z − μ)/σ²,  h_μ = 1/σ²,   g_s = 1 − r²,  h_s = 2.
    """
    p, a, b, sig = probabilidad_banda(mu, s, l, u)
    pa, pb = _phi(a), _phi(b)
    dp_mu = (pa - pb) / sig
    dp_s = a * pa - b * pb
    w = 1.0 / (p * (1 - p))
    g_mu, g_s = (p - y) * w * dp_mu, (p - y) * w * dp_s
    h_mu, h_s = w * dp_mu ** 2, w * dp_s ** 2
    if z is not None and lam > 0:
        r = (z - mu) / sig
        g_mu = g_mu - lam * r / sig
        h_mu = h_mu + lam / sig ** 2
        g_s = g_s + lam * (1 - r * r)
        h_s = h_s + 2 * lam
    return g_mu, h_mu + 1e-9, g_s, h_s + 1e-9


# ---------------------------------------------------------------------------
# Estimador compatible con scikit-learn
# ---------------------------------------------------------------------------
class BandLikelihoodBoosting(ClassifierMixin, BaseEstimator):
    """Clasificador de banda por boosting distribucional con pérdida híbrida.

    Parameters
    ----------
    banda_inf, banda_sup : float
        Límites L y U de la banda (m³/s).
    col_referencia : int or str
        Columna de X a partir de la cual se obtiene el caudal de referencia observado en t.
    exponente_referencia : float
        Q_ref = X[:, col_referencia] ** exponente. Con la columna ``calamar_cuad`` (Q(t)²) y
        exponente 0.5 se recupera exactamente el caudal del día de la predicción.
    lambda_hibrido : float
        Peso λ ≥ 0 de la verosimilitud gaussiana. λ = 0 entrena solo con la etiqueta binaria.
    n_estimadores : int
        Número máximo de rondas de boosting (cada ronda ajusta un árbol para μ y otro para s).
    tasa : float
        Paso inicial η de la búsqueda lineal.
    profundidad : int
        Profundidad máxima de los árboles.
    min_hess : float
        Suma mínima de hessianos por hoja (análogo de ``min_child_weight``).
    reg_lambda : float
        Regularización L2 de los valores de hoja.
    gamma : float
        Ganancia mínima para aceptar una división.
    submuestreo, columnas : float
        Fracciones de filas y de columnas usadas en cada ronda.
    n_bins : int
        Número de bins por variable (≤ 255).
    sigma_min, sigma_max : float
        Cotas de σ; garantizan que la pérdida esté acotada inferiormente.
    heterocedastico : bool
        Si es False, σ es constante (componente retirado en la ablación).
    paciencia : int
        Rondas sin mejora en validación antes de detenerse; 0 desactiva la parada temprana.
    fraccion_validacion : float
        Fracción final (cronológica) del entrenamiento usada para la parada temprana.
    reajustar : bool
        Tras la parada temprana, reentrena con todos los datos y el número de rondas hallado.
    busqueda_lineal : bool
        Si es False, cada ronda usa el paso fijo η sin verificar el descenso (componente retirado
        en la ablación; se pierde la garantía de monotonía).
    random_state : int
        Semilla del submuestreo.
    """

    def __init__(self, banda_inf=8461.29, banda_sup=11447.63, col_referencia=0,
                 exponente_referencia=1.0, lambda_hibrido=1.0, n_estimadores=300, tasa=0.1, profundidad=4,
                 min_hess=5.0, reg_lambda=1.0, gamma=0.0, submuestreo=0.8, columnas=0.8,
                 n_bins=64, sigma_min=0.01, sigma_max=3.0, heterocedastico=True,
                 paciencia=30, fraccion_validacion=0.15, reajustar=True, busqueda_lineal=True,
                 random_state=42):
        self.banda_inf = banda_inf
        self.banda_sup = banda_sup
        self.col_referencia = col_referencia
        self.exponente_referencia = exponente_referencia
        self.lambda_hibrido = lambda_hibrido
        self.n_estimadores = n_estimadores
        self.tasa = tasa
        self.profundidad = profundidad
        self.min_hess = min_hess
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.submuestreo = submuestreo
        self.columnas = columnas
        self.n_bins = n_bins
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.heterocedastico = heterocedastico
        self.paciencia = paciencia
        self.fraccion_validacion = fraccion_validacion
        self.reajustar = reajustar
        self.busqueda_lineal = busqueda_lineal
        self.random_state = random_state

    # -- utilidades -------------------------------------------------------
    def _indice_referencia(self, X):
        if isinstance(self.col_referencia, str):
            nombres = getattr(self, "feature_names_in_", None)
            if nombres is None or self.col_referencia not in list(nombres):
                raise ValueError(f"col_referencia '{self.col_referencia}' no está en X.")
            return list(nombres).index(self.col_referencia)
        return int(self.col_referencia)

    def _validar_hiper(self):
        if not self.banda_inf < self.banda_sup or self.banda_inf <= 0:
            raise ValueError("Se requiere 0 < banda_inf < banda_sup.")
        if self.lambda_hibrido < 0:
            raise ValueError("lambda_hibrido debe ser no negativo.")
        if not 0 < self.tasa <= 1:
            raise ValueError("tasa debe estar en (0, 1].")
        if self.profundidad < 1 or self.n_estimadores < 1:
            raise ValueError("profundidad y n_estimadores deben ser positivos.")
        if not 2 <= self.n_bins <= 255:
            raise ValueError("n_bins debe estar entre 2 y 255.")

    def _binarizar(self, X):
        Xb = np.empty(X.shape, dtype=np.uint8)
        for j, bordes in enumerate(self.bordes_):
            col = np.where(np.isnan(X[:, j]), self.medianas_[j], X[:, j])
            Xb[:, j] = np.searchsorted(bordes, col, side="left")
        return Xb

    def _limites(self, qref):
        q = np.maximum(qref, 1e-6)
        return np.log(self.banda_inf / q), np.log(self.banda_sup / q)

    def _clip_s(self, s):
        return np.clip(s, np.log(self.sigma_min), np.log(self.sigma_max))

    # -- ajuste -----------------------------------------------------------
    def fit(self, X, y, q_futuro=None):
        """Ajusta el modelo.

        Parameters
        ----------
        X : array-like (n, p)
            Predictores en orden cronológico; incluye la columna de referencia.
        y : array-like (n,)
            Etiqueta binaria de pertenencia a la banda.
        q_futuro : array-like (n,), optional
            Caudal futuro observado. Si se omite, solo se usa la etiqueta (λ efectivo = 0).
        """
        self._validar_hiper()
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X, y = check_X_y(X, y, force_all_finite="allow-nan", dtype=np.float64)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("BLGB requiere exactamente dos clases.")
        y = (y == self.classes_[1]).astype(np.float64)
        self.n_features_in_ = X.shape[1]
        j = self._indice_referencia(X)
        qref = X[:, j] ** self.exponente_referencia
        if np.any(~np.isfinite(qref)) or np.any(qref <= 0):
            raise ValueError("La columna de referencia debe ser positiva y sin faltantes.")
        z = None
        if q_futuro is not None:
            q_futuro = np.asarray(q_futuro, dtype=np.float64)
            if q_futuro.shape[0] != X.shape[0] or np.any(q_futuro <= 0):
                raise ValueError("q_futuro debe ser positivo y de la misma longitud que X.")
            z = np.log(q_futuro / qref)

        # Discretización por cuantiles (aprendida solo con los datos de ajuste)
        self.medianas_ = np.nanmedian(X, axis=0)
        self.medianas_ = np.where(np.isnan(self.medianas_), 0.0, self.medianas_)
        cuantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        Xi = np.where(np.isnan(X), self.medianas_, X)
        self.bordes_ = [np.unique(np.quantile(Xi[:, c], cuantiles)) for c in range(X.shape[1])]
        Xb = self._binarizar(X)
        l, u = self._limites(qref)

        n = len(y)
        if self.paciencia and 0 < self.fraccion_validacion < 0.5:
            corte = int(n * (1 - self.fraccion_validacion))
            idx_tr, idx_va = np.arange(corte), np.arange(corte, n)
        else:
            idx_tr, idx_va = np.arange(n), np.array([], dtype=int)

        estado = self._boosting(Xb, y, z, l, u, idx_tr, idx_va, self.n_estimadores)
        if self.reajustar and len(idx_va) and estado["mejor_iter"] > 0:
            estado = self._boosting(Xb, y, z, l, u, np.arange(n), np.array([], dtype=int),
                                    estado["mejor_iter"])
        self.mu0_, self.s0_ = estado["mu0"], estado["s0"]
        self.arboles_mu_, self.arboles_s_ = estado["arboles_mu"], estado["arboles_s"]
        self.pasos_ = estado["pasos"]
        self.historial_ = estado["historial"]
        self.n_iter_ = len(self.arboles_mu_)
        return self

    def _boosting(self, Xb, y, z, l, u, idx_tr, idx_va, n_rondas):
        rng = np.random.default_rng(self.random_state)
        lam = self.lambda_hibrido if z is not None else 0.0
        p_feat = Xb.shape[1]
        ztr = None if z is None else z[idx_tr]
        if z is not None:
            mu0, s0 = float(np.mean(ztr)), float(np.log(max(np.std(ztr), self.sigma_min)))
        else:
            # Sin caudal futuro: σ inicial unitaria y μ centrado en la banda
            mu0, s0 = float(np.mean((l[idx_tr] + u[idx_tr]) / 2)), 0.0
        s0 = float(self._clip_s(np.array([s0]))[0])
        mu = np.full(len(y), mu0)
        s = np.full(len(y), s0)
        arboles_mu, arboles_s, pasos, historial = [], [], [], []
        mejor_va, mejor_iter, sin_mejora = np.inf, 0, 0

        def L(idx, m, ss):
            return perdida_hibrida(m[idx], ss[idx], l[idx], u[idx], y[idx],
                                   None if z is None else z[idx], lam)

        perdida_actual = L(idx_tr, mu, s)
        for t in range(n_rondas):
            filas = idx_tr
            if self.submuestreo < 1:
                filas = np.sort(rng.choice(idx_tr, max(2, int(len(idx_tr) * self.submuestreo)),
                                           replace=False))
            cols = np.sort(rng.choice(p_feat, max(1, int(round(p_feat * self.columnas))),
                                      replace=False))
            g_mu, h_mu, g_s, h_s = gradientes(mu[filas], s[filas], l[filas], u[filas], y[filas],
                                              None if z is None else z[filas], lam)
            a_mu = construir_arbol(Xb[filas], g_mu, h_mu, self.n_bins, self.profundidad,
                                   self.min_hess, self.reg_lambda, self.gamma, cols)
            d_mu = a_mu.predecir(Xb)
            if self.heterocedastico:
                a_s = construir_arbol(Xb[filas], g_s, h_s, self.n_bins, self.profundidad,
                                      self.min_hess, self.reg_lambda, self.gamma, cols)
                d_s = a_s.predecir(Xb)
            else:
                a_s, d_s = None, np.zeros(len(y))

            # Búsqueda lineal de Armijo: el paso se reduce a la mitad hasta que la pérdida
            # de entrenamiento disminuye; si no disminuye tras 8 intentos, la ronda se rechaza.
            g_full = gradientes(mu[idx_tr], s[idx_tr], l[idx_tr], u[idx_tr], y[idx_tr],
                                None if z is None else z[idx_tr], lam)
            pendiente = (np.dot(g_full[0], d_mu[idx_tr]) + np.dot(g_full[2], d_s[idx_tr])) / len(idx_tr)
            eta, aceptado = self.tasa, False
            for _ in range(8 if self.busqueda_lineal else 1):
                mu_n, s_n = mu + eta * d_mu, self._clip_s(s + eta * d_s)
                nueva = L(idx_tr, mu_n, s_n)
                if not self.busqueda_lineal or nueva <= perdida_actual + 1e-4 * eta * min(pendiente, 0.0):
                    aceptado = True
                    break
                eta *= 0.5
            if not aceptado:
                break
            mu, s, perdida_actual = mu_n, s_n, nueva
            arboles_mu.append(a_mu)
            arboles_s.append(a_s)
            pasos.append(eta)
            reg = {"iter": t + 1, "perdida_train": perdida_actual, "paso": eta}
            if len(idx_va):
                p_va = probabilidad_banda(mu[idx_va], s[idx_va], l[idx_va], u[idx_va])[0]
                yv = y[idx_va]
                bce_va = float(-(yv * np.log(p_va) + (1 - yv) * np.log1p(-p_va)).mean())
                reg["bce_validacion"] = bce_va
                if bce_va < mejor_va - 1e-6:
                    mejor_va, mejor_iter, sin_mejora = bce_va, t + 1, 0
                else:
                    sin_mejora += 1
                    if sin_mejora >= self.paciencia:
                        historial.append(reg)
                        break
            historial.append(reg)
        if len(idx_va):
            arboles_mu, arboles_s, pasos = (arboles_mu[:mejor_iter], arboles_s[:mejor_iter],
                                            pasos[:mejor_iter])
        else:
            mejor_iter = len(arboles_mu)
        return {"mu0": mu0, "s0": s0, "arboles_mu": arboles_mu, "arboles_s": arboles_s,
                "pasos": pasos, "historial": historial, "mejor_iter": mejor_iter}

    # -- predicción -------------------------------------------------------
    def _parametros(self, X):
        check_is_fitted(self, "arboles_mu_")
        X = check_array(X, force_all_finite="allow-nan", dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} columnas; se esperaban {self.n_features_in_}.")
        Xb = self._binarizar(X)
        mu = np.full(len(X), self.mu0_)
        s = np.full(len(X), self.s0_)
        for eta, a_mu, a_s in zip(self.pasos_, self.arboles_mu_, self.arboles_s_):
            mu += eta * a_mu.predecir(Xb)
            if a_s is not None:
                s = self._clip_s(s + eta * a_s.predecir(Xb))
        qref = X[:, self._indice_referencia(X)] ** self.exponente_referencia
        return mu, s, qref

    def predict_proba(self, X):
        """Probabilidad de que el caudal futuro quede dentro de la banda."""
        mu, s, qref = self._parametros(X)
        l, u = self._limites(qref)
        p = probabilidad_banda(mu, s, l, u)[0]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def predict_distribucion(self, X, nivel: float = 0.8):
        """Mediana e intervalo de predicción del caudal futuro (m³/s).

        Returns
        -------
        dict
            ``mediana``, ``inferior`` y ``superior`` del intervalo central de nivel ``nivel``.
        """
        from scipy.stats import norm
        mu, s, qref = self._parametros(X)
        zq = norm.ppf(0.5 + nivel / 2)
        sig = np.exp(s)
        return {"mediana": qref * np.exp(mu), "inferior": qref * np.exp(mu - zq * sig),
                "superior": qref * np.exp(mu + zq * sig), "mu": mu, "sigma": sig}
